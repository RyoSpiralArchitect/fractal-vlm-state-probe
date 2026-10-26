from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
from itertools import combinations, product
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.cli.summarize_graded_cache_study import (
    summarize_decompositions,
    verify_decomposition,
    verify_transfer,
)
from fractal_vlm_state_probe.cli.summarize_map_seed_cache_study import verify_aggregate
from fractal_vlm_state_probe.graded_permutation import array_sha256
from fractal_vlm_state_probe.map_seed_input_baseline import VIEWS, validate_config
from fractal_vlm_state_probe.padding_policy import _checked_artifact
from fractal_vlm_state_probe.permutation_maps import ANCHOR, MAPS, NEW_MAPS
from fractal_vlm_state_probe.stimulus import write_json


def verify_interaction(saved, cells, view):
    if view == "raw_rgb":
        expected = (
            cells["jj"].astype(np.int16)
            + cells["mm"].astype(np.int16)
            - cells["jm"].astype(np.int16)
            - cells["mj"].astype(np.int16)
        ).astype(np.float64) / 255
    elif view == "processor_pixel_values":
        expected = (
            cells["jj"].astype(np.float64)
            + cells["mm"].astype(np.float64)
            - cells["jm"].astype(np.float64)
            - cells["mj"].astype(np.float64)
        )
    else:
        raise ValueError("unknown input coordinates")
    if (
        saved.shape != expected.shape
        or saved.dtype != np.float64
        or not np.isfinite(saved).all()
    ):
        raise ValueError("input interaction shape/dtype/finiteness differs")
    error = float(np.max(np.abs(saved - expected)))
    if error > 1e-12:
        raise ValueError("input interaction differs from independent cell arithmetic")
    return error


def verify_scores(result, vectors, metadata):
    matrix, primary, directions = (
        result[k] for k in ("map_matrix", "primary_tests", "directional_decompositions")
    )
    if len(matrix) != 32 or {
        (r["view"], r["reference_map"], r["test_map"]) for r in matrix
    } != set(product(VIEWS, MAPS, MAPS)):
        raise ValueError("input matrix coverage differs")
    cosines = sum(
        verify_transfer(
            r,
            metadata,
            vectors[r["view"]][r["reference_map"]],
            vectors[r["view"]][r["test_map"]],
        )
        for r in matrix
    )
    if len(primary) != 4 or {(r["view"], r["regime"]) for r in primary} != set(
        product(VIEWS, ("same_map", "cross_map"))
    ):
        raise ValueError("input primary family coverage differs")
    aggregate = sum(
        verify_aggregate(r, [m for m in matrix if m["view"] == r["view"]], metadata)
        for r in primary
    )
    order = sorted(
        primary,
        key=lambda r: r["exact_block_test"]["p_greater"] if r["available"] else 1,
    )
    previous = 0
    for rank, row in enumerate(order):
        p = row["exact_block_test"]["p_greater"] if row["available"] else 1
        previous = max(previous, min(1, (4 - rank) * p))
        if (
            row["holm_family_size"] != 4
            or abs(previous - row["holm_p_greater"]) > 1e-14
        ):
            raise ValueError("input Holm family differs")
    if len(directions) != 384 or {
        (r["view"], r["reference_condition"], r["compared_condition"], r["pair_id"])
        for r in directions
    } != {
        (v, a, b, label)
        for v in VIEWS
        for a, b in combinations(MAPS, 2)
        for label in metadata
    }:
        raise ValueError("input direction coverage differs")
    for r in directions:
        label = r["pair_id"]
        if any(r[k] != value for k, value in metadata[label].items()) or r["phase"] != (
            "reference" if metadata[label]["replicate"] in (1, 2) else "test"
        ):
            raise ValueError("input direction metadata differs")
        verify_decomposition(
            r,
            vectors[r["view"]][r["reference_condition"]][label],
            vectors[r["view"]][r["compared_condition"]][label],
        )
    gaps = result["same_cross_gaps"]
    if len(gaps) != 2 or {g["view"] for g in gaps} != set(VIEWS):
        raise ValueError("input regime gap coverage differs")
    for gap in gaps:
        same, cross = (
            next(
                r for r in primary if r["view"] == gap["view"] and r["regime"] == regime
            )
            for regime in ("same_map", "cross_map")
        )
        if same["available"] and cross["available"]:
            if (
                set(gap["family_margin_differences"]) != set(same["family_margins"])
                or abs(
                    gap["same_minus_cross_mean_margin"]
                    - same["mean_family_margin"]
                    + cross["mean_family_margin"]
                )
                > 1e-12
                or any(
                    abs(
                        gap["family_margin_differences"][f]
                        - same["family_margins"][f]
                        + cross["family_margins"][f]
                    )
                    > 1e-12
                    for f in same["family_margins"]
                )
            ):
                raise ValueError("input regime gap differs")
        elif (
            gap["same_minus_cross_mean_margin"] is not None
            or gap["family_margin_differences"] is not None
        ):
            raise ValueError("unavailable input gap must remain null")
    return {
        "matrix_cosines": cosines,
        "aggregate_cosines": aggregate,
        "synchronized_exact_tests": 4,
        "directional_records": 384,
    }


def verify_equivariance(result, vectors, maps):
    expected = {(label, key) for label in vectors[ANCHOR] for key in NEW_MAPS}
    rows = result["raw_equivariance"]["inverse_checks"]
    if (
        len(rows) != 96
        or {(r["pair_id"], r["map"]) for r in rows} != expected
        or not all(r["exact_inverse_equal"] for r in rows)
    ):
        raise ValueError("raw inverse denominator differs")
    for label in vectors[ANCHOR]:
        canonical = {}
        for key in MAPS:
            value = vectors[key][label]
            restored = value.copy()
            content = np.empty((76800, 3), dtype=np.float64)
            content[maps[key]] = value[40:280].reshape(-1, 3)
            restored[40:280] = content.reshape(240, 320, 3)
            canonical[key] = restored
        if any(
            not np.array_equal(canonical[key], canonical[ANCHOR]) for key in NEW_MAPS
        ):
            raise ValueError("independent raw inverse differs")
    diagonal = {
        r["reference_map"]: {
            s["label"]: s["reference_cosines"] for s in r["test_seed_scores"]
        }
        for r in result["map_matrix"]
        if r["view"] == "raw_rgb"
        and r["reference_map"] == r["test_map"]
        and r["available"]
    }
    error = (
        None
        if len(diagonal) != 4
        else max(
            abs(value - diagonal[key][label][f])
            for key in NEW_MAPS
            for label, values in diagonal[ANCHOR].items()
            for f, value in values.items()
        )
    )
    saved = result["raw_equivariance"]["diagonal_cosine_max_abs_error"]
    if (error is None) != (saved is None) or (
        error is not None and (error > 1e-12 or abs(error - saved) > 1e-14)
    ):
        raise ValueError("raw same-map isometry check differs")
    return 96


def publish(root, output, *, figures=True):
    result, frozen = (
        _read(root / "study_summary.json"),
        _read(root / "frozen_specification.json"),
    )
    validate_config(frozen["config"])
    if (root / "execution_failure.json").exists():
        raise ValueError("failed input execution requires a separate audit")
    for name, field in (
        ("frozen_specification.json", "frozen_specification_sha256"),
        ("frozen_references.json", "frozen_references_sha256"),
        ("cells.json", "cells_sha256"),
        ("vectors.json", "vectors_sha256"),
    ):
        if _sha(root / name) != result[field]:
            raise ValueError(f"input measurement receipt changed: {name}")
    repo = Path(__file__).resolve().parents[3]
    for path, digest in frozen["code_sha256"].items():
        if _sha(repo / path) != digest:
            raise ValueError(f"registered implementation changed: {path}")
    for path, digest in frozen["source_sha256"].items():
        if _sha(Path(path)) != digest:
            raise ValueError(f"input source changed: {path}")
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    if runtime != frozen["runtime"]:
        raise ValueError("input verification runtime changed")
    metadata = frozen["hierarchy_metadata"]
    cell_rows, vector_rows, refs = (
        _read(root / n)
        for n in ("cells.json", "vectors.json", "frozen_references.json")
    )
    if (
        len(cell_rows) != 512
        or {(r["map"], r["pair_id"], r["cell"]) for r in cell_rows}
        != set(product(MAPS, metadata, ("mm", "jj", "mj", "jm")))
        or len(vector_rows) != 256
        or {(r["view"], r["map"], r["pair_id"]) for r in vector_rows}
        != set(product(VIEWS, MAPS, metadata))
        or len(refs) != 128
        or refs != [r for r in vector_rows if r["phase"] == "reference"]
    ):
        raise ValueError("input cell/vector/reference coverage differs")
    for r in [*cell_rows, *vector_rows]:
        if r["phase"] != (
            "reference" if metadata[r["pair_id"]]["replicate"] in (1, 2) else "test"
        ):
            raise ValueError("input reference/test phase differs")
    vectors = {v: {k: {} for k in MAPS} for v in VIEWS}
    for row in vector_rows:
        path = Path(row["path"])
        if _sha(path) != row["sha256"]:
            raise ValueError("saved input vector changed")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            list(value.shape) != row["shape"]
            or str(value.dtype) != row["dtype"]
            or not np.isclose(
                np.sqrt(np.sum(value * value)), row["l2_norm"], rtol=1e-12, atol=1e-12
            )
        ):
            raise ValueError("input vector shape/dtype/norm differs")
        vectors[row["view"]][row["map"]][row["pair_id"]] = value
    maps = {}
    for key, receipt in frozen["maps"].items():
        if _sha(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("input map changed")
        maps[key] = np.load(receipt["path"], allow_pickle=False)
    import torch
    from mlx_vlm.utils import load_processor

    torch.set_num_threads(1)
    snapshot = Path(frozen["processor_snapshot"])
    expected = frozen["processor_provenance"]
    if (
        snapshot.name != expected["snapshot_revision"]
        or _sha(snapshot / "preprocessor_config.json")
        != expected["preprocessor_config_sha256"]
    ):
        raise ValueError("processor snapshot/config differs")
    processor = load_processor(snapshot, add_detokenizer=False, trust_remote_code=True)
    cls = type(processor.image_processor)
    if (
        f"{cls.__module__}.{cls.__name__}" != expected["implementation"]
        or _sha(Path(inspect.getfile(cls))) != expected["implementation_sha256"]
    ):
        raise ValueError("processor implementation differs")
    source = {r["pair_id"]: r for r in _read(Path(frozen["input_summary"]))["records"]}
    cell_lookup = {(r["map"], r["pair_id"], r["cell"]): r for r in cell_rows}
    errors = {v: 0.0 for v in VIEWS}
    for key in MAPS:
        for label in metadata:
            arrays = {v: {} for v in VIEWS}
            for cell in ("mm", "jj", "mj", "jm"):
                row = cell_lookup[(key, label, cell)]
                original = source[label]["maps"][key]
                expected = original["cells"][cell]
                if (
                    row["artifact"] != original["artifacts"][cell]
                    or not row["raw_exact"]
                    or not row["processor_exact"]
                ):
                    raise ValueError("cell source receipt differs")
                rgb, _ = _checked_artifact(row["artifact"])
                if (
                    array_sha256(rgb) != row["canvas_rgb_bytes_sha256"]
                    or array_sha256(rgb) != expected["canvas_rgb_bytes_sha256"]
                ):
                    raise ValueError("raw input bytes differ")
                payload = processor(
                    images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
                )
                pixels = np.asarray(payload["pixel_values"])
                audit_path = Path(row["cache_input_audit_path"])
                if _sha(audit_path) != row["cache_input_audit_sha256"]:
                    raise ValueError("cache input audit changed")
                audit = _read(audit_path)
                if (
                    pixels.shape != (1, 3, 1024, 1024)
                    or pixels.dtype != np.float32
                    or not np.isfinite(pixels).all()
                    or array_sha256(pixels) != row["processor_pixels_sha256"]
                    or array_sha256(pixels) != expected["pixel_values_sha256"]
                    or array_sha256(pixels)
                    != audit["effective"]["pixel_values"]["sha256"]
                    or {
                        k: np.asarray(v).tolist()
                        for k, v in payload.items()
                        if k != "pixel_values"
                    }
                    != expected["processor_non_pixel_payload"]
                ):
                    raise ValueError("independent processor reproduction differs")
                arrays["raw_rgb"][cell], arrays["processor_pixel_values"][cell] = (
                    rgb,
                    pixels,
                )
            for view in VIEWS:
                errors[view] = max(
                    errors[view],
                    verify_interaction(vectors[view][key][label], arrays[view], view),
                )
            del arrays
        print(f"verified all input cells and interactions: {key}", flush=True)
    verification = verify_scores(result, vectors, metadata)
    verification.update(
        raw_inverse_checks=verify_equivariance(result, vectors["raw_rgb"], maps),
        raw_cells_reloaded=512,
        processor_cells_reproduced=512,
        interaction_vectors_recomputed=256,
        maximum_interaction_oracle_error=errors,
    )
    if (
        result["input_cells"] != 512
        or result["interaction_vectors"] != 256
        or result["new_cache_forwards"] != 0
        or result["model_weights_loaded"]
    ):
        raise ValueError("input-only execution denominator differs")
    if output.exists():
        raise FileExistsError("use a fresh input publication directory")
    output.mkdir(parents=True)
    ledgers = {
        "cells.jsonl": cell_rows,
        "vectors.jsonl": vector_rows,
        "directional_decompositions.jsonl": result["directional_decompositions"],
    }
    for name, rows in ledgers.items():
        with (output / name).open("w") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
    summary = {k: v for k, v in result.items() if k != "directional_decompositions"}
    summary.update(
        analysis_kind="fixed_fraction_map_seed_input_baseline_publication",
        frozen_specification=frozen,
        source_summary_sha256=_sha(root / "study_summary.json"),
        exporter_sha256=_sha(Path(__file__)),
        verification_helper_sha256={
            name: _sha(Path(__file__).with_name(name))
            for name in (
                "summarize_graded_cache_study.py",
                "summarize_map_seed_cache_study.py",
                "summarize_padding_cache_study.py",
            )
        },
        independent_verification=verification,
        directional_summary=summarize_decompositions(
            result["directional_decompositions"]
        ),
        ledgers={
            name: {"rows": len(rows), "sha256": _sha(output / name)}
            for name, rows in ledgers.items()
        },
        claim_boundaries=[
            "The input follow-up was registered after cache results on the same reused input-selected images and maps.",
            "Raw same-map agreement follows coordinate-permutation isometry, not learned invariance or independent replication.",
            "Four input primary tests have their own Holm family; significance differences from cache do not test processing gain.",
            "All matrix views, directional decompositions, cross/same ratios and input/cache differences are descriptive.",
            "No new cache forward, semantic extraction, persistent state or causal mediation is measured.",
        ],
    )
    if figures:
        plot_results(summary, _read(Path(frozen["cache_publication"])), output)
        summary["figures"] = {
            name: _sha(output / name)
            for name in (
                "input_map_matrix.png",
                "input_map_matrix.pdf",
                "input_cache_margins.png",
                "input_cache_margins.pdf",
            )
        }
    write_json(output / "summary.json", summary)
    return summary


def plot_results(summary, cache, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    labels, titles = ["H", "16", "17", "18"], ["Raw RGB", "Processor pixel values"]
    limit = max(
        (abs(r["mean_family_margin"]) for r in summary["map_matrix"] if r["available"]),
        default=1e-6,
    )
    limit = max(limit, 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    for ax, view, title in zip(axes, VIEWS, titles):
        grid = np.full((4, 4), np.nan)
        for row in summary["map_matrix"]:
            if row["view"] == view and row["available"]:
                grid[MAPS.index(row["test_map"]), MAPS.index(row["reference_map"])] = (
                    row["mean_family_margin"]
                )
        artist = ax.imshow(
            grid, cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        )
        ax.set(
            xticks=range(4),
            yticks=range(4),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Reference map",
            ylabel="Test map",
            title=title,
        )
        for i, j in product(range(4), repeat=2):
            value = grid[i, j]
            ax.text(
                j,
                i,
                f"{value:.4f}" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
                color="white" if abs(value) > limit * 0.65 else "black",
                fontsize=10,
            )
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.20, top=0.82, wspace=0.35)
    fig.colorbar(
        artist, cax=fig.add_axes([0.89, 0.25, 0.02, 0.50]), label="Mean family margin"
    )
    fig.suptitle("Fixed 3/4 maps: correspondence already present in inputs")
    fig.text(
        0.5,
        0.05,
        "H: historical anchor; 16/17/18: new maps. All 32 matrix views are descriptive.",
        ha="center",
        fontsize=9,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"input_map_matrix.{extension}", dpi=160)
    plt.close(fig)
    series = [
        ("Raw RGB", summary["primary_tests"], "raw_rgb", None),
        ("Processor", summary["primary_tests"], "processor_pixel_values", None),
    ]
    for target, name in (
        ("layer_001_keys", "L1 keys"),
        ("layer_012_keys", "L12 keys"),
        ("layer_023_values", "L23 values"),
    ):
        for view, region in (("image_tokens", "image"), ("post_image", "post")):
            series.append((f"{name} / {region}", cache["primary_tests"], view, target))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for regime, offset, color, label in (
        ("same_map", -0.17, "#2376a8", "Same map"),
        ("cross_map", 0.17, "#c75d37", "Cross map"),
    ):
        values = [
            next(
                r["mean_family_margin"] if r["available"] else np.nan
                for r in rows
                if r["view"] == view
                and r["regime"] == regime
                and (target is None or r["target"] == target)
            )
            for _, rows, view, target in series
        ]
        ax.barh(
            np.arange(len(series)) + offset,
            values,
            height=0.30,
            color=color,
            label=label,
        )
    ax.set(
        yticks=range(len(series)),
        yticklabels=[s[0] for s in series],
        xlabel="Mean family margin (different coordinate systems)",
    )
    ax.invert_yaxis()
    ax.axhline(1.5, color="0.6", linewidth=0.8)
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Input and cache correspondence: descriptive comparison")
    fig.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.18)
    fig.text(
        0.5,
        0.04,
        "Separate input/cache test families. Score differences do not establish causal processing gain.",
        ha="center",
        fontsize=9,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"input_cache_margins.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Independently verify and publish input map correspondence."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.run_root.resolve(), args.output_root.resolve())
    print(json.dumps(result["independent_verification"], indent=2), flush=True)


if __name__ == "__main__":
    main()
