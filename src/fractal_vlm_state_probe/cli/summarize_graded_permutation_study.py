from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.stimulus import sha256_file, write_json

CELLS = ("mm", "jj", "mj", "jm")


def read(path):
    return json.loads(Path(path).read_text())


def cycle_oracle(full, order, count):
    selected = np.zeros(len(full), dtype=bool)
    selected[order[:count]] = True
    result = np.arange(len(full))
    # Independently follow successors, without cycle extraction or contraction.
    for position in np.flatnonzero(selected):
        successor = full[position]
        while not selected[successor]:
            successor = full[successor]
        result[position] = successor
    return result


def verify_gate(cells, gate):
    centroids = np.array([cells[c]["processor"]["spectral_centroid"] for c in CELLS])
    hf = np.array([cells[c]["processor"]["high_frequency_energy_ratio"] for c in CELLS])
    valid = bool(
        np.isfinite(centroids).all()
        and np.isfinite(hf).all()
        and np.all(centroids > 1e-12)
        and np.all((hf >= 0) & (hf <= 1))
        and all(cells[c]["processor"]["luminance_std"] > 1e-12 for c in CELLS)
    )
    error = float(np.max(np.abs(centroids / centroids.mean() - 1))) if valid else None
    spread = float(np.ptp(hf)) if valid else None
    moments = all(
        cells[c]["marginal_audit"]["content_pixels_unchanged"]
        and cells[c]["marginal_audit"]["padding_pixels_equal_registered_fill"]
        and cells[c]["marginal_audit"]["mean_max_abs_error"] <= 1e-12
        and cells[c]["marginal_audit"]["variance_max_abs_error"] <= 1e-12
        for c in CELLS
    )
    marginal = all(cells[c]["registered_expanded_palette_verified"] for c in CELLS)
    shapes = len({tuple(cells[c]["pixel_values_shape"]) for c in CELLS}) == 1
    accepted = bool(
        valid and moments and marginal and shapes and error <= 0.05 and spread <= 0.02
    )
    if (
        gate["accepted"] != accepted
        or gate["nondegenerate_finite_spectra"] != valid
        or gate["content_and_fill_verified"] != moments
        or gate["centroid_relative_tolerance"] != 0.05
        or gate["hf_absolute_tolerance"] != 0.02
        or gate["max_centroid_relative_error"] != error
        or gate["hf_max_pairwise_absolute_difference"] != spread
        or gate["registered_expanded_rgb_multisets_match"] != marginal
        or gate["processor_shapes_equal"] != shapes
        or gate["common_centroid_target"]
        != (float(centroids.mean()) if valid else None)
        or gate["common_hf_target"] != (float(hf.mean()) if valid else None)
    ):
        raise ValueError("independent four-cell gate differs")


def load_pixels(artifact):
    path = Path(artifact["manifest_path"])
    if sha256_file(path) != artifact["manifest_sha256"]:
        raise ValueError("image manifest hash differs")
    manifest = read(path)
    if len(manifest["frames"]) != 1:
        raise ValueError("expected one frame")
    frame = path.parent / manifest["frames"][0]["path"]
    if (
        sha256_file(frame) != manifest["frames"][0]["sha256"]
        or sha256_file(frame) != artifact["frame_sha256"]
    ):
        raise ValueError("image bytes differ from receipt")
    with Image.open(frame) as image:
        return np.asarray(image.convert("RGB")).copy()


def sorted_rgb_codes(rgb):
    pixels = rgb.reshape(-1, 3).astype(np.uint32)
    return np.sort((pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2])


def verify_pixels(canvas, baseline, mapping, color, row):
    original, interior = baseline[40:280], canvas[40:280]
    if canvas.shape != (320, 320, 3) or baseline.shape != canvas.shape:
        raise ValueError("canvas shape differs")
    if not np.array_equal(interior.reshape(-1, 3), original.reshape(-1, 3)[mapping]):
        raise ValueError("saved pixels differ from registered permutation")
    if not np.array_equal(
        interior.reshape(-1, 3)[np.argsort(mapping)], original.reshape(-1, 3)
    ):
        raise ValueError("inverse pixels differ")
    if not np.all(canvas[:40] == color) or not np.all(canvas[280:] == color):
        raise ValueError("padding changed")
    codes = sorted_rgb_codes(canvas)
    if not np.array_equal(codes, sorted_rgb_codes(baseline)):
        raise ValueError("expanded joint RGB counts changed")
    sorted_rgb = np.column_stack(
        ((codes >> 16) & 255, (codes >> 8) & 255, codes & 255)
    ).astype(np.uint8)
    if (
        hashlib.sha256(sorted_rgb.tobytes()).hexdigest()
        != row["canvas_rgb_multiset_sha256"]
    ):
        raise ValueError("recorded RGB multiset hash differs")
    if (
        hashlib.sha256(canvas.tobytes()).hexdigest() != row["canvas_rgb_bytes_sha256"]
        or hashlib.sha256(interior.tobytes()).hexdigest()
        != row["content_rgb_bytes_sha256"]
    ):
        raise ValueError("recorded pixel hash differs")
    if (
        float(np.any(interior != original, axis=2).mean())
        != row["changed_rgb_site_fraction"]
    ):
        raise ValueError("changed RGB site fraction differs")
    c, x, fill = (
        interior.astype(np.float64) / 255,
        canvas.astype(np.float64) / 255,
        color / 255,
    )
    mu, var = c.mean(axis=(0, 1)), c.var(axis=(0, 1))
    expected_mean = 0.75 * mu + 0.25 * fill
    expected_var = 0.75 * var + 0.75 * 0.25 * (mu - fill) ** 2
    audit = row["marginal_audit"]
    if (
        audit["fill_rgb_u8"] != color.tolist()
        or audit["content_fraction"] != 0.75
        or audit["added_padding_pixels"] != 25600
        or audit["original_pixel_count"] != 76800
        or audit["canvas_pixel_count"] != 102400
        or not audit["content_pixels_unchanged"]
        or not audit["padding_pixels_equal_registered_fill"]
        or not row["registered_expanded_palette_verified"]
        or not row["inverse_roundtrip_exact"]
        or audit["expected_canvas_rgb_mean"] != expected_mean.tolist()
        or audit["expected_canvas_rgb_variance"] != expected_var.tolist()
        or audit["mean_max_abs_error"]
        != float(np.max(np.abs(x.mean(axis=(0, 1)) - expected_mean)))
        or audit["variance_max_abs_error"]
        != float(np.max(np.abs(x.var(axis=(0, 1)) - expected_var)))
    ):
        raise ValueError("recorded marginal audit differs")
    integer = canvas.astype(np.uint64)
    oracle_mean = integer.sum(axis=(0, 1)) / (102400 * 255)
    oracle_var = (integer * integer).sum(axis=(0, 1)) / (
        102400 * 255**2
    ) - oracle_mean**2
    return {
        "mean": float(np.max(np.abs(oracle_mean - expected_mean))),
        "variance": float(np.max(np.abs(oracle_var - expected_var))),
    }


def publish(source_path, output, *, figures=True):
    study = read(source_path)
    root = source_path.parent
    frozen = read(root / "frozen_specification.json")
    if (
        frozen != study["frozen_specification"]
        or sha256_file(root / "frozen_specification.json")
        != study["frozen_specification_sha256"]
    ):
        raise ValueError("frozen input specification changed")
    repo = Path(__file__).resolve().parents[3]
    for path, digest in frozen["code_sha256"].items():
        if sha256_file(repo / path) != digest:
            raise ValueError(f"registered code changed: {path}")
    for key, digest in frozen["source_sha256"].items():
        if sha256_file(Path(frozen["source_paths"][key])) != digest:
            raise ValueError("historical receipt changed")
    if (root / "execution_failure.json").exists():
        raise ValueError("execution failure retained; do not publish as completed")
    full = np.load(frozen["source_paths"]["full_permutation"], allow_pickle=False)
    order = np.load(root / "selection_order.npy", allow_pickle=False)
    if sha256_file(root / "selection_order.npy") != frozen[
        "selection_order_sha256"
    ] or not np.array_equal(order, np.random.default_rng(20261008).permutation(76800)):
        raise ValueError("selection order differs")
    mappings, prior_support = {}, set()
    for key, receipt in frozen["maps"].items():
        path = Path(receipt["path"])
        mapping = np.load(path, allow_pickle=False)
        if sha256_file(path) != receipt["sha256"] or not np.array_equal(
            mapping, cycle_oracle(full, order, receipt["selected_count"])
        ):
            raise ValueError("graded mapping does not reproduce independently")
        support = set(np.flatnonzero(mapping != np.arange(76800)))
        if not prior_support <= support <= set(order[: receipt["selected_count"]]):
            raise ValueError("graded mapping support is not nested")
        prior_support = support
        mappings[key] = mapping
    prior = {
        r["pair_id"]: r
        for r in read(frozen["source_paths"]["padding_receipt"])["records"]
    }
    if (
        len(study["records"]) != 32
        or {r["pair_id"] for r in study["records"]} != set(prior)
        or len(mappings) != 7
    ):
        raise ValueError("input denominator differs")
    rows, compact, errors = [], [], []
    endpoint_checks = 0
    for record in study["records"]:
        pair = record["pair_id"]
        old = prior[pair]
        baseline = old["conditions"]["original/palette_mean"]
        source_pixels = {c: load_pixels(a) for c, a in baseline["artifacts"].items()}
        item = {
            k: record[k]
            for k in ("pair_id", "pairing_family", "broad_class", "replicate")
        }
        if any(item[k] != old[k] for k in item) or set(record["levels"]) != set(
            mappings
        ):
            raise ValueError("source hierarchy or level denominator differs")
        item["levels"] = {}
        for key, level in record["levels"].items():
            verify_gate(level["cells"], level["gate"])
            item["levels"][key] = {"gate": level["gate"]}
            for cell in CELLS:
                row = level["cells"][cell]
                canvas = load_pixels(level["artifacts"][cell])
                donor = "mm" if cell in ("mm", "jm") else "jj"
                color = np.rint(
                    source_pixels[donor][40:280].sum(axis=(0, 1), dtype=np.uint64)
                    / 76800
                ).astype(np.uint8)
                errors.append(
                    verify_pixels(
                        canvas, source_pixels[cell], mappings[key], color, row
                    )
                )
                if (
                    not row["non_pixel_payload_unchanged"]
                    or row["processor_non_pixel_payload"]
                    != baseline["cells"][cell]["processor_non_pixel_payload"]
                    or row["processor_non_pixel_payload"]["image_sizes"] != [[320, 320]]
                ):
                    raise ValueError("non-pixel processor metadata differs")
                if row["pixel_values_shape"] != [3, 1024, 1024]:
                    raise ValueError("qualified processor pixel shape differs")
                if key in ("selected_00_of_08", "selected_08_of_08"):
                    state = (
                        "original"
                        if key == "selected_00_of_08"
                        else "shared_pixel_permutation"
                    )
                    endpoint = old["conditions"][f"{state}/palette_mean"]
                    if (
                        not np.array_equal(
                            canvas, load_pixels(endpoint["artifacts"][cell])
                        )
                        or not row["endpoint_pixel_and_stats_reproduced"]
                        or any(
                            row[k] != endpoint["cells"][cell][k]
                            for k in (
                                "processor",
                                "raw",
                                "pixel_values_sha256",
                                "pixel_values_dtype",
                                "pixel_values_shape",
                            )
                        )
                        or level["gate"] != endpoint["gate"]
                    ):
                        raise ValueError("historical endpoint differs")
                    endpoint_checks += 1
                elif row["endpoint_pixel_and_stats_reproduced"] is not None:
                    raise ValueError("an intermediate level is not an endpoint replay")
                rows.append(
                    {
                        "pair_id": pair,
                        "level": key,
                        "cell": cell,
                        **row,
                        "artifact": level["artifacts"][cell],
                    }
                )
        compact.append(item)
        print(f"verified input pixels and gates: {pair}", flush=True)
    if (
        len(rows) != 896
        or endpoint_checks != 256
        or study["input_cells"] != 896
        or study["new_cache_forwards"] != 0
        or study["model_weights_loaded"]
    ):
        raise ValueError("completed denominators differ")
    levels = {}
    families = {r["pairing_family"] for r in compact}
    for key, saved in study["levels"].items():
        accepted = [r for r in compact if r["levels"][key]["gate"]["accepted"]]
        complete = sorted(
            f
            for f in families
            if {r["replicate"] for r in accepted if r["pairing_family"] == f}
            == {1, 2, 3, 4}
        )
        if (
            saved["accepted_pair_ids"] != [r["pair_id"] for r in accepted]
            or saved["accepted_blocks"] != len(accepted)
            or saved["complete_four_seed_families"] != complete
            or saved["all_input_blocks_accepted"] != (len(accepted) == 32)
        ):
            raise ValueError("input acceptance summary differs")
        stats = {}
        for field in (
            "max_centroid_relative_error",
            "hf_max_pairwise_absolute_difference",
            "common_centroid_target",
            "common_hf_target",
        ):
            values = [r["levels"][key]["gate"][field] for r in compact]
            stats[field] = {
                "min": min(values),
                "median": float(np.median(values)),
                "max": max(values),
            }
        level_rows = [r for r in rows if r["level"] == key]
        levels[key] = {
            **saved,
            **stats,
            "changed_rgb_site_fraction": {
                "min": min(r["changed_rgb_site_fraction"] for r in level_rows),
                "median": float(
                    np.median([r["changed_rgb_site_fraction"] for r in level_rows])
                ),
                "max": max(r["changed_rgb_site_fraction"] for r in level_rows),
            },
            "moment_gate_failures": sum(
                not r["levels"][key]["gate"]["content_and_fill_verified"]
                for r in compact
            ),
        }
    if output.exists():
        raise FileExistsError("use a fresh publication directory")
    output.mkdir(parents=True)
    ledger = output / "cells.jsonl"
    with ledger.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    summary = {
        "schema_version": 1,
        "analysis_kind": "graded_permutation_input_snapshot",
        "date": study["date"],
        "registered_design_commit": frozen["registered_design_commit"],
        "source_report_sha256": sha256_file(source_path),
        "code_exporter_sha256": sha256_file(Path(__file__)),
        "frozen_specification": frozen,
        "frozen_specification_sha256": study["frozen_specification_sha256"],
        "block_count": 32,
        "input_cells": len(rows),
        "levels": levels,
        "records": compact,
        "cell_ledger": {
            "path": ledger.name,
            "rows": len(rows),
            "sha256": sha256_file(ledger),
        },
        "verification": {
            "independently_reconstructed_maps": len(mappings),
            "exact_expanded_rgb_histograms": len(rows),
            "inverse_pixel_roundtrips": len(rows),
            "unchanged_padding_and_metadata": len(rows),
            "historical_endpoint_checks": endpoint_checks,
            "independently_recomputed_gates": 224,
        },
        "separate_integer_moment_oracle": {
            field: {
                "max_abs_error": max(r[field] for r in errors),
                "over_1e_minus12_cells": sum(r[field] > 1e-12 for r in errors),
            }
            for field in ("mean", "variance")
        },
        "full_panel_input_eligible_levels": [
            k for k, v in levels.items() if v["all_input_blocks_accepted"]
        ],
        "new_cache_forwards": 0,
        "model_weights_loaded": False,
        "new_cache_results": "NOT_MEASURED",
        "previous_selected_cache_counts_unchanged": {
            "source_cells": 872,
            "tensor_sidecars": 2472,
            "factorial_analyses": 618,
        },
        "claim_boundaries": study["claim_boundaries"],
        "execution_failures": [],
    }
    if figures:
        make_figures(summary, study, output)
        summary["figures"] = {
            name: sha256_file(output / name)
            for name in (
                "graded_input_audit.png",
                "graded_input_audit.pdf",
                "graded_example.png",
            )
        }
    write_json(output / "summary.json", summary)
    return summary


def make_figures(summary, study, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(summary["levels"])
    x = [summary["frozen_specification"]["maps"][k]["selected_fraction"] for k in keys]
    labels = ["0", "1/8", "1/4", "1/2", "3/4", "7/8", "1"]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    counts = [summary["levels"][k]["accepted_blocks"] for k in keys]
    axes[0, 0].plot(x, counts, "o-", color="#176b58")
    for position, count in zip(x, counts):
        axes[0, 0].annotate(
            f"{count}/32",
            (position, count),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axes[0, 0].set(
        title="Complete four-cell input gates",
        ylabel="Accepted blocks",
        ylim=(-2, 37),
        yticks=[0, 8, 16, 24, 32],
    )
    for ax, field, title, threshold, color in (
        (
            axes[0, 1],
            "max_centroid_relative_error",
            "Within-level centroid mismatch",
            0.05,
            "#286d99",
        ),
        (
            axes[1, 0],
            "hf_max_pairwise_absolute_difference",
            "Within-level HF spread",
            0.02,
            "#ad4d62",
        ),
        (
            axes[1, 1],
            "common_centroid_target",
            "Centroid targets also change across levels",
            None,
            "#715239",
        ),
    ):
        stats = [summary["levels"][k][field] for k in keys]
        ax.fill_between(
            x,
            [s["min"] for s in stats],
            [s["max"] for s in stats],
            color=color,
            alpha=0.15,
            label="32-block range",
        )
        ax.plot(
            x, [s["median"] for s in stats], "o-", color=color, label="Block median"
        )
        if threshold is not None:
            ax.axhline(
                threshold,
                color="#666666",
                linestyle="--",
                linewidth=1,
                label="Frozen threshold",
            )
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0, 1].set_yscale("log")
    for ax in axes.ravel():
        ax.set_xticks(x, labels)
        ax.set_xlabel("Selected-site fraction (not changed-RGB fraction)")
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Fixed palette-mean padding and exact RGB marginals | input-only", fontsize=15
    )
    fig.savefig(output / "graded_input_audit.png", dpi=160)
    fig.savefig(output / "graded_input_audit.pdf")
    plt.close(fig)
    anchor = next(
        r for r in study["records"] if r["pair_id"] == "geometry_checker_hex_r1"
    )
    fig, axes = plt.subplots(2, 7, figsize=(16, 5.6), layout="constrained")
    for row, cell in enumerate(("mm", "jj")):
        for col, (key, label) in enumerate(zip(keys, labels)):
            axes[row, col].imshow(load_pixels(anchor["levels"][key]["artifacts"][cell]))
            axes[row, col].set_title(f"{cell.upper()} | {label}", fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle(
        "Fixed example: geometry_checker_hex_r1 | entire padded canvases", fontsize=14
    )
    fig.savefig(output / "graded_example.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Independently audit and publish the graded input study."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    summary = publish(args.source, args.output_root, figures=not args.no_figures)
    print(json.dumps(summary["verification"], indent=2))
    print(f"wrote {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
