from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cli.run_controlled_overlap import (
    REPO,
    load_maps,
    qualified_processor,
    qualified_sources,
    read,
)
from fractal_vlm_state_probe.cli.summarize_graded_permutation_study import (
    load_pixels,
    verify_gate,
    verify_pixels,
)
from fractal_vlm_state_probe.cli.summarize_source_coordinate_overlap import (
    oracle_partition,
)
from fractal_vlm_state_probe.image_stats import _frame_image_stats
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def close(actual, expected, label, tolerance=1e-12):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.max(np.abs(a - b), initial=0) > tolerance
    ):
        raise ValueError(f"independent verification differs: {label}")


def oracle_summary(scores, labels, families, metadata):
    rows, family_values, correct, tied = [], {f: [] for f in families}, 0, 0
    broad = {
        f: next(v["broad_class"] for v in metadata.values() if v["pairing_family"] == f)
        for f in families
    }
    for label, values in zip(labels, scores):
        family = metadata[label]["pairing_family"]
        other = [
            float(values[i])
            for i, f in enumerate(families)
            if f != family and broad[f] == broad[family]
        ]
        own = float(values[families.index(family)])
        margin = own - sum(other) / len(other)
        rows.append(margin)
        family_values[family].append(margin)
        winners = [f for f, v in zip(families, values) if abs(v - max(values)) <= 1e-12]
        tied += len(winners) != 1
        correct += winners == [family]
    margins = {f: sum(v) / len(v) for f, v in family_values.items()}
    return rows, margins, sum(margins.values()) / len(margins), correct, tied


def verify_summary(saved, scores, labels, families, metadata):
    rows, margins, mean, correct, tied = oracle_summary(
        scores, labels, families, metadata
    )
    if (
        saved["test_seed_count"] != 16
        or [r["label"] for r in saved["test_seed_margins"]] != labels
        or set(saved["family_margins"]) != set(families)
    ):
        raise ValueError("summary denominator differs")
    for index, (row, label) in enumerate(zip(saved["test_seed_margins"], labels)):
        if any(row[k] != v for k, v in metadata[label].items()):
            raise ValueError("summary metadata differs")
        family = metadata[label]["pairing_family"]
        own = scores[index, families.index(family)]
        close(row["own_family_cosine_contribution"], own, "own contribution")
        close(
            row["other_family_cosine_contribution"],
            own - rows[index],
            "other contribution",
        )
    close([r["margin"] for r in saved["test_seed_margins"]], rows, "test margins")
    close(
        [saved["family_margins"][f] for f in families],
        list(margins.values()),
        "family margins",
    )
    close(saved["mean_family_margin"], mean, "mean margin")
    if (
        saved["retrieval_correct_count"] != correct
        or saved["retrieval_tied_count"] != tied
        or saved["positive_family_margin_count"] != sum(v > 0 for v in margins.values())
        or saved["negative_family_margin_count"] != sum(v < 0 for v in margins.values())
    ):
        raise ValueError("retrieval or sign count differs")


def oracle_score(reference, test, labels, families, metadata, categories=None):
    def unit(path):
        a = np.load(path, allow_pickle=False).ravel()
        if not np.isfinite(a).all():
            raise ValueError("nonfinite interaction")
        norm = float(np.sqrt(np.einsum("i,i->", a, a)))
        return a / norm if norm else None

    refs = []
    for family in families:
        members = sorted(
            k
            for k, v in metadata.items()
            if v["pairing_family"] == family and v["replicate"] in (1, 2)
        )
        a, b = [unit(reference[k]) for k in members]
        if a is None or b is None:
            return None
        combined = a + b
        norm = np.sqrt(np.einsum("i,i->", combined, combined))
        if norm <= 1e-12:
            return None
        refs.append(combined / norm)
    tests = [unit(test[k]) for k in labels]
    if any(v is None for v in tests):
        return None
    a, b = np.stack(tests), np.stack(refs)
    scores = np.einsum("ij,kj->ik", a, b)
    components = None
    if categories is not None:
        components = {}
        codes = np.repeat(categories.ravel(), 3)
        for i, name in enumerate(
            (
                "shared_fixed_content",
                "shared_moved_content",
                "noncoincident_content",
                "padding",
            )
        ):
            selected = codes == i
            components[name] = np.einsum("ij,kj->ik", a[:, selected], b[:, selected])
    return scores, components


def oracle_bootstrap(margins, spec):
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([spec["seed"]]))
    )
    output = []
    for _ in range(spec["draws"]):
        masks = generator.integers(0, 4, 4)
        families = generator.integers(0, 8, 8)
        sources = generator.integers(0, 2, (8, 2))
        total = np.zeros((2, 3))
        for mask in masks:
            destinations = generator.integers(0, 2, 2)
            block = np.take(margins[:, mask], destinations, axis=1).mean(axis=1)
            for index, family in enumerate(families):
                total += (
                    np.take(block[:, :, family], sources[index], axis=-1).mean(axis=-1)
                    / 32
                )
        output.append(np.column_stack((total[:, 0], total[:, 2] - total[:, 0])))
    return np.quantile(np.stack(output), spec["quantiles"], axis=0).transpose(1, 2, 0)


def publish(root, output):
    if output.exists():
        raise FileExistsError("use a fresh publication root")
    study, analysis = (
        read(root / "input_summary.json"),
        read(root / "analysis_summary.json"),
    )
    frozen = study["frozen_specification"]
    config = frozen["config"]
    if (
        (root / "execution_failure.json").exists()
        or sha256_file(root / "input_summary.json") != analysis["input_summary_sha256"]
        or read(root / "frozen_specification.json") != frozen
    ):
        raise ValueError("failed or changed input measurement")
    for filename, digest in frozen["code_sha256"].items():
        if sha256_file(REPO / filename) != digest:
            raise ValueError("registered measurement code changed")
    for filename, digest in frozen["source_sha256"].items():
        if sha256_file(Path(filename)) != digest:
            raise ValueError("historical source changed")
    for file, digest in (
        ("vectors.json", study["vectors_sha256"]),
        ("calibration.json", study["calibration_sha256"]),
        ("frozen_specification.json", study["frozen_specification_sha256"]),
        ("scores.jsonl", analysis["scores_sha256"]),
    ):
        if sha256_file(root / file) != digest:
            raise ValueError("input/score receipt changed")
    if sha256_file(Path(frozen["map_report"])) != frozen["map_report_sha256"]:
        raise ValueError("map report changed")
    report, maps = load_maps(Path(frozen["map_report"]))
    baseline, prior, sources, metadata, _ = qualified_sources()
    if metadata != frozen["hierarchy_metadata"]:
        raise ValueError("metadata changed")
    processor, normalization = qualified_processor(baseline)
    checks = read(root / "calibration.json")["checks"]
    expected_checks = {
        (r["pair_id"], c): r["maps"]["historical_3q"]["cells"][c]["pixel_values_sha256"]
        for r in prior["records"]
        for c in ("mm", "jj", "mj", "jm")
    }
    if (
        len(checks) != 128
        or len({(r["pair_id"], r["cell"]) for r in checks}) != 128
        or any(
            not r["exact"]
            or r["pixel_values_sha256"] != expected_checks[(r["pair_id"], r["cell"])]
            for r in checks
        )
    ):
        raise ValueError("historical calibration denominator differs")
    vector_rows = read(root / "vectors.json")["vectors"]
    vectors = {(r["view"], r["map"], r["pair_id"]): r for r in vector_rows}
    expected_vectors = {
        (v, k, label)
        for v in config["views"]
        for k in maps
        for label, m in metadata.items()
        if k.endswith("reference") or m["replicate"] in (3, 4)
    }
    if set(vectors) != expected_vectors or len(vector_rows) != 1024:
        raise ValueError("vector coverage differs")
    for row in vector_rows:
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("vector bytes changed")
    records = study["records"]
    keys = [(r["pair_id"], r["map"]) for r in records]
    if len(keys) != 896 or set(keys) != {
        (label, key) for label in metadata for key in maps
    }:
        raise ValueError("input cell denominator differs")
    cells, errors, replay_count = [], [], 0
    old_cells = {}
    for label in metadata:
        source = sources[label]["conditions"]["original/palette_mean"]
        old_cells[label] = {c: load_pixels(a) for c, a in source["artifacts"].items()}
    for index, record in enumerate(records):
        label, key = record["pair_id"], record["map"]
        if (
            any(record[k] != v for k, v in metadata[label].items())
            or set(record["cells"]) != {"mm", "jj", "mj", "jm"}
            or set(record["artifacts"]) != set(record["cells"])
        ):
            raise ValueError("cell hierarchy differs")
        verify_gate(record["cells"], record["gate"])
        raw, pixels_by_cell = {}, {}
        for cell, row in record["cells"].items():
            canvas = load_pixels(record["artifacts"][cell])
            source = sources[label]["conditions"]["original/palette_mean"]["cells"][
                cell
            ]
            verify_pixels(
                canvas,
                old_cells[label][cell],
                maps[key],
                np.array(source["marginal_audit"]["fill_rgb_u8"]),
                row,
            )
            payload = processor(
                images=Image.fromarray(canvas), text=["<image>"], return_tensors="np"
            )
            pixels = np.asarray(payload["pixel_values"])
            other = {
                k: np.asarray(v).tolist()
                for k, v in payload.items()
                if k != "pixel_values"
            }
            if (
                hashlib.sha256(pixels.tobytes()).hexdigest()
                != row["pixel_values_sha256"]
                or other != row["processor_non_pixel_payload"]
                or other != source["processor_non_pixel_payload"]
            ):
                raise ValueError("independent processor replay differs")
            decoded = np.moveaxis(
                pixels[0].astype(np.float64)
                * np.asarray(normalization["image_std"])[:, None, None]
                + np.asarray(normalization["image_mean"])[:, None, None],
                0,
                -1,
            )
            if (
                _frame_image_stats(decoded) != row["processor"]
                or _frame_image_stats(canvas.astype(np.float64) / 255) != row["raw"]
            ):
                raise ValueError("pixel-derived statistics differ")
            raw[cell], pixels_by_cell[cell] = (
                canvas.astype(np.int16),
                pixels.astype(np.float64),
            )
            cells.append(
                {
                    "pair_id": label,
                    "map": key,
                    "cell": cell,
                    **metadata[label],
                    "artifact": record["artifacts"][cell],
                    **row,
                }
            )
            replay_count += 1
        if key.endswith("reference") or metadata[label]["replicate"] in (3, 4):
            expected_raw = (raw["jj"] + raw["mm"] - raw["jm"] - raw["mj"]).astype(
                np.float64
            ) / 255
            expected_pixels = (
                pixels_by_cell["mm"]
                + pixels_by_cell["jj"]
                - pixels_by_cell["mj"]
                - pixels_by_cell["jm"]
            )
            for view, expected in zip(config["views"], (expected_raw, expected_pixels)):
                row = vectors[(view, key, label)]
                saved = np.load(row["path"], allow_pickle=False)
                close(saved, expected, "integer/raw or promoted processor interaction")
                errors.append(float(np.max(np.abs(saved - expected))))
                if (
                    list(saved.shape) != row["shape"]
                    or str(saved.dtype) != row["dtype"]
                ):
                    raise ValueError("vector layout receipt differs")
                close(
                    row["norm"], np.linalg.norm(saved), "vector norm", tolerance=1e-10
                )
        if index % 28 == 27:
            print(f"independent pixels and vectors {index + 1}/896 blocks", flush=True)
    counts = {
        k: {
            "accepted_blocks": sum(
                r["gate"]["accepted"] for r in records if r["map"] == k
            ),
            "total_blocks": 32,
        }
        for k in maps
    }
    eligible = all(v["accepted_blocks"] == 32 for v in counts.values())
    if (
        counts != study["maps"]
        or eligible != study["cache_panel_eligible"]
        or eligible != analysis["cache_panel_eligible"]
        or study["input_cells"] != 3584
        or study["processor_calls"] != 3712
        or study["historical_processor_checks"] != 128
        or study["new_model_forwards"] != 0
        or study["model_weights_loaded"]
        or analysis["new_model_forwards"] != 0
        or analysis["new_p_values"] != 0
        or analysis["confirmation_cohort"] != "not_generated_not_sealed"
    ):
        raise ValueError("completion counts or eligibility differ")
    rows = [
        json.loads(line) for line in (root / "scores.jsonl").read_text().splitlines()
    ]
    expected_keys = {(v, k) for v in config["views"] for k in maps}
    if (
        len(rows) != 56
        or {(r["view"], r["test_map"]) for r in rows} != expected_keys
        or analysis["score_views"] != 56
    ):
        raise ValueError("score denominator differs")
    families = sorted({m["pairing_family"] for m in metadata.values()})
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    cube = np.full((2, 4, 2, 3, 16, 8), np.nan)
    max_score_error = 0.0
    for row in rows:
        view, key = row["view"], row["test_map"]
        recipe = report["maps"][key]["recipe"]
        reference = recipe["reference"]
        if row["reference_map"] != reference or any(
            row[k] != v for k, v in recipe.items()
        ):
            raise ValueError("score recipe differs")
        a = {
            label: r["path"]
            for (v, k, label), r in vectors.items()
            if v == view and k == reference
        }
        b = {
            label: r["path"]
            for (v, k, label), r in vectors.items()
            if v == view and k == key
        }
        result = oracle_score(
            a,
            b,
            labels,
            families,
            metadata,
            oracle_partition(maps[reference], maps[key]) if view == "raw_rgb" else None,
        )
        if row["available"] != (result is not None):
            raise ValueError("score availability differs")
        if result is None:
            continue
        scores, parts = result
        if row["labels"] != labels or row["families"] != families:
            raise ValueError("score labels differ")
        close(row["scores"], scores, "full cosine scores")
        max_score_error = max(
            max_score_error, float(np.max(np.abs(scores - row["scores"])))
        )
        verify_summary(row["summary"], scores, labels, families, metadata)
        if parts is not None:
            if set(row["component_scores"]) != set(parts):
                raise ValueError("component denominator differs")
            for c in parts:
                close(row["component_scores"][c], parts[c], "raw additive scores")
            close(sum(parts.values()), scores, "raw reconstruction")
        if recipe["q"] is not None:
            cube[
                config["views"].index(view),
                recipe["mask"],
                recipe["destination"],
                config["overlap_sixteenths"].index(recipe["q"]),
            ] = scores
    if len(analysis["aggregates"]) != 6 or {
        (r["view"], r["q_sixteenths"]) for r in analysis["aggregates"]
    } != {(v, q) for v in config["views"] for q in (0, 1, 4)}:
        raise ValueError("aggregate coverage differs")
    for row in analysis["aggregates"]:
        vi, qi = (
            config["views"].index(row["view"]),
            (0, 1, 4).index(row["q_sixteenths"]),
        )
        available = bool(np.isfinite(cube[vi, :, :, qi]).all())
        if row["available"] != available:
            raise ValueError("aggregate missingness differs")
        if available:
            verify_summary(
                row, cube[vi, :, :, qi].mean(axis=(0, 1)), labels, families, metadata
            )
            if row["view"] == "raw_rgb":
                selected = [
                    r
                    for r in rows
                    if r["view"] == "raw_rgb" and r["q"] == row["q_sixteenths"]
                ]
                if set(row["components"]) != set(selected[0]["component_scores"]):
                    raise ValueError("aggregate component coverage differs")
                for c in row["components"]:
                    verify_summary(
                        row["components"][c],
                        np.mean([r["component_scores"][c] for r in selected], axis=0),
                        labels,
                        families,
                        metadata,
                    )
    bootstrap_verified = False
    if np.isfinite(cube).all():
        margins = np.empty((2, 4, 2, 3, 8, 2))
        for index in np.ndindex(2, 4, 2, 3):
            seed_margins = oracle_summary(cube[index], labels, families, metadata)[0]
            for fi, family in enumerate(families):
                margins[index + (fi,)] = [
                    v
                    for k, v in zip(labels, seed_margins)
                    if metadata[k]["pairing_family"] == family
                ]
        if (
            sha256_file(root / "paired_family_seed_margins.npy")
            != analysis["paired_margins_sha256"]
        ):
            raise ValueError("paired margin receipt changed")
        close(
            np.load(root / "paired_family_seed_margins.npy", allow_pickle=False),
            margins,
            "paired margin cube",
        )
        ranges = oracle_bootstrap(margins, config["bootstrap"])
        close(analysis["bootstrap"]["ranges"], ranges, "paired bootstrap")
        if any(
            analysis["bootstrap"][k] != config["bootstrap"][k]
            for k in ("draws", "quantiles", "interpretation")
        ):
            raise ValueError("bootstrap specification differs")
        if len(analysis["primary_endpoints"]) != 4 or {
            (r["view"], r["endpoint"]) for r in analysis["primary_endpoints"]
        } != {(v, e) for v in config["views"] for e in config["primary_endpoints"]}:
            raise ValueError("endpoint coverage differs")
        for row in analysis["primary_endpoints"]:
            vi, ei = (
                config["views"].index(row["view"]),
                config["primary_endpoints"].index(row["endpoint"]),
            )
            values = (
                margins[vi, :, :, 0]
                if ei == 0
                else margins[vi, :, :, 2] - margins[vi, :, :, 0]
            )
            close(row["mean"], values.mean(), "endpoint mean")
            close(row["mask_values"], values.mean(axis=(1, 2, 3)), "endpoint masks")
            close(
                [row["family_values"][f] for f in families],
                values.mean(axis=(0, 1, 3)),
                "endpoint families",
            )
            close(row["exploratory_range"], ranges[vi, ei], "endpoint range")
            if (
                not row["available"]
                or row["p_value"] is not None
                or row["equivalence_tested"]
            ):
                raise ValueError("endpoint claim contract differs")
        bootstrap_verified = True
    elif (
        any(r["available"] for r in analysis["primary_endpoints"])
        or analysis["bootstrap"] is not None
    ):
        raise ValueError("incomplete paired panel acquired endpoints")
    lookup = {(r["pair_id"], r["map"]): r["gate"] for r in records}
    drift = []
    for label in metadata:
        for m in range(4):
            for d in range(2):
                zero = lookup[(label, f"m{m}_q0_d{d}")]
                for q in (1, 4):
                    other = lookup[(label, f"m{m}_q{q}_d{d}")]
                    drift.append(
                        {
                            "pair_id": label,
                            "mask": m,
                            "destination": d,
                            "q_sixteenths": q,
                            "centroid_target_relative_change_from_zero": other[
                                "common_centroid_target"
                            ]
                            / zero["common_centroid_target"]
                            - 1,
                            "hf_target_change_from_zero": other["common_hf_target"]
                            - zero["common_hf_target"],
                        }
                    )
    output.mkdir(parents=True)
    ledgers = {
        "cells.jsonl": cells,
        "vectors.jsonl": vector_rows,
        "scores.jsonl": rows,
        "between_level_spectra.jsonl": drift,
    }
    receipts = {}
    for name, records_out in ledgers.items():
        path = output / name
        path.write_text(
            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records_out)
        )
        receipts[name] = {"rows": len(records_out), "sha256": sha256_file(path)}
    verification = {
        "coordinate_maps": len(maps),
        "input_cells": len(cells),
        "four_cell_gates": len(records),
        "vector_recomputations": len(errors),
        "maximum_vector_oracle_abs_error": max(errors),
        "full_score_matrices": len(rows),
        "maximum_score_oracle_abs_error": max_score_error,
        "bootstrap_verified": bootstrap_verified,
        "verification_processor_calls": replay_count,
    }
    summary = {
        **analysis,
        "analysis_kind": "controlled_overlap_phase1_publication",
        "frozen_specification": frozen,
        "source_analysis_sha256": sha256_file(root / "analysis_summary.json"),
        "map_construction": report,
        "input_gate_counts": counts,
        "input_cells": study["input_cells"],
        "measurement_processor_calls": study["processor_calls"],
        "historical_processor_checks": 128,
        "independent_verification": verification,
        "ledgers": receipts,
        "exporter_sha256": sha256_file(Path(__file__)),
        "claim_boundaries": [
            "Development only; all existing sources reused; no confirmation cohort released.",
            "Bootstrap ranges are exploratory, conditional on fixed references and do not guarantee coverage.",
            "No p-values, equivalence tests, causal gain, model forwards or cache observations are added.",
            "Within-condition input gates do not imply between-level spectral matching.",
        ],
    }
    plot_results(summary, output)
    summary["figures"] = {p.name: sha256_file(p) for p in sorted(output.glob("*.png"))}
    write_json(output / "summary.json", summary)
    print(json.dumps(verification, indent=2), flush=True)


def plot_results(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, view, title, color in zip(
        axes,
        ("raw_rgb", "processor_pixel_values"),
        ("Raw RGB", "Processor pixels"),
        ("#2376a8", "#c75d37"),
    ):
        rows = [r for r in summary["aggregates"] if r["view"] == view]
        if not all(r["available"] for r in rows):
            ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center")
            continue
        x = [100 * r["q_sixteenths"] / 16 for r in rows]
        ax.plot(x, [r["mean_family_margin"] for r in rows], marker="o", color=color)
        for f in rows[0]["family_margins"]:
            ax.plot(
                x,
                [r["family_margins"][f] for r in rows],
                color=color,
                alpha=0.25,
                linewidth=0.7,
            )
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set(
            title=title,
            xlabel="Shared fixed content (%)",
            ylabel="Mean family margin",
            xticks=[0, 6.25, 25],
        )
    fig.suptitle("Controlled overlap: all eight families retained")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.text(
        0.5,
        0.02,
        "Four masks x two destinations; reused development images. No model forward.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output / "overlap_profiles.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, row in enumerate(summary["primary_endpoints"]):
        if row["available"]:
            lo, hi = row["exploratory_range"]
            ax.plot(
                [lo, hi], [i, i], color="#2376a8" if i < 2 else "#c75d37", linewidth=3
            )
            ax.scatter(row["mean"], i, color="black", zorder=3)
    ax.set(
        yticks=range(4),
        yticklabels=[
            "Raw: zero overlap",
            "Raw: high - zero",
            "Processor: zero overlap",
            "Processor: high - zero",
        ],
        xlabel="Mean margin or paired margin difference",
    )
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title("Four registered endpoints; exploratory bootstrap ranges")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(
        0.5,
        0.03,
        "Fixed references; percentile tails 0.00625 / 0.99375. Coverage is not guaranteed.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output / "endpoint_ranges.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    publish(args.run_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
