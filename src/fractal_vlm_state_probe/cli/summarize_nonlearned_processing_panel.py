from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cli.analyze_nonlearned_processing_panel import (
    load_capture,
    load_features,
    read_jsonl,
    write_ledger,
)
from fractal_vlm_state_probe.cli.run_controlled_overlap import (
    BASELINE,
    REPO,
    qualified_processor,
    read,
)
from fractal_vlm_state_probe.cli.run_nonlearned_processing_calibration import (
    assert_hashes,
    sha256_array,
)
from fractal_vlm_state_probe.cli.summarize_controlled_overlap import (
    close,
    oracle_score,
    oracle_summary,
    verify_summary,
)
from fractal_vlm_state_probe.nonlearned_processing import FEATURES, STARTS
from fractal_vlm_state_probe.nonlearned_processing_calibration import (
    synthetic_calibration,
    oracle_features,
)
from fractal_vlm_state_probe.nonlearned_processing_panel import VIEWS, feature_key
from fractal_vlm_state_probe.padding_policy import _checked_artifact
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def oracle_ranges(margins, spec):
    generator = np.random.default_rng(np.random.SeedSequence([spec["seed"]]))
    available = np.isfinite(np.take(margins, [0, 2], axis=3)).all(axis=(1, 2, 3, 4, 5))
    draws = []
    for _ in range(spec["draws"]):
        masks, families = generator.integers(0, 4, 4), generator.integers(0, 8, 8)
        seeds = generator.integers(0, 2, (8, 2))
        total = np.zeros((12, 3))
        for mask in masks:
            destinations = generator.integers(0, 2, 2)
            block = np.take(margins[:, mask], destinations, axis=1).mean(axis=1)
            for j, family in enumerate(families):
                total += (
                    np.take(block[:, :, family], seeds[j], axis=-1).mean(axis=-1) / 32
                )
        draws.append(np.column_stack((total[:, 0], total[:, 2] - total[:, 0])))
    draws = np.stack(draws)
    return [
        np.quantile(draws[:, i], spec["quantiles"], axis=0).T if ok else None
        for i, ok in enumerate(available)
    ]


def archive_key(row):
    return "__".join(row[k] for k in ("start", "feature", "map", "pair_id"))


def verify_analysis(root, frozen, cells):
    output = root / "analysis"
    if (output / "analysis_failure.json").exists():
        raise ValueError("failed analysis cannot be published")
    summary = read(output / "summary.json")
    if (
        summary["status"] != "COMPLETE"
        or summary["capture_summary_sha256"]
        != sha256_file(root / "capture_summary.json")
        or any(
            summary[k] != 0
            for k in ("new_model_forwards", "new_processor_calls", "new_p_values")
        )
        or summary["confirmation_cohort"] != "not_generated_not_sealed"
    ):
        raise ValueError("analysis completion or claim boundary differs")
    ledgers = {}
    for name, receipt in summary["ledgers"].items():
        if sha256_file(output / name) != receipt["sha256"]:
            raise ValueError("analysis ledger changed")
        ledgers[name] = read_jsonl(output / name)
        if len(ledgers[name]) != receipt["rows"]:
            raise ValueError("analysis ledger denominator differs")
    if {k: len(v) for k, v in ledgers.items()} != {
        "vectors.jsonl": 6144,
        "scores.jsonl": 336,
        "directions.jsonl": 3072,
    }:
        raise ValueError("planned analysis denominator differs")
    metadata = frozen["hierarchy_metadata"]
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    features = {}
    for row in cells:
        features.setdefault((row["map"], row["pair_id"]), {})[row["cell"]] = (
            load_features(root, row)
        )
    vector_rows = ledgers["vectors.jsonl"]
    vectors = {
        (r["start"], r["feature"], r["map"], r["pair_id"]): r for r in vector_rows
    }
    if len(vectors) != 6144 or set(vectors) != {
        (s, f, m, k) for s, f in VIEWS for m, k in features
    }:
        raise ValueError("vector identity coverage differs")
    portable = {}
    zero_raw = 0
    for row in vector_rows:
        key = (row["map"], row["pair_id"])
        a = features[key]
        view = feature_key(row["start"], row["feature"])
        expected = (a["jj"][view] - a["mj"][view]) - (a["jm"][view] - a["mm"][view])
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError("interaction vector changed")
        saved = np.load(path, allow_pickle=False)
        if (
            saved.dtype != np.float64
            or saved.shape != expected.shape
            or not np.array_equal(saved, expected)
            or row["shape"] != list(saved.shape)
            or row["dtype"] != "float64"
        ):
            raise ValueError("paired interaction reconstruction differs")
        norm = float(np.sqrt(np.einsum("i,i->", expected, expected)))
        close(row["norm"], norm, "vector norm", tolerance=1e-12)
        if row["available"] != (norm != 0):
            raise ValueError("vector availability differs")
        bound = (
            4
            * np.sqrt(expected.size)
            * frozen["config"]["calibration"]["oracle_absolute_tolerance"]
        )
        close(row["conservative_feature_oracle_l2_bound"], bound, "numerical bound")
        if not np.isclose(
            row["norm_to_oracle_bound_ratio"], norm / bound, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("numerical sensitivity diagnostic differs")
        if (row["start"], row["feature"]) == ("raw_rgb", "global_moments"):
            if np.any(expected):
                raise ValueError("raw negative control failed")
            zero_raw += 1
        public_key = archive_key(row)
        if public_key in portable:
            raise ValueError("duplicate portable vector key")
        portable[public_key] = saved
    if zero_raw != 512 or summary["raw_global_moment_zero_blocks"] != zero_raw:
        raise ValueError("raw zero interaction count differs")
    maps = read(REPO / frozen["config"]["phase1_publication"])["map_construction"][
        "maps"
    ]
    score_rows = ledgers["scores.jsonl"]
    if {(r["start"], r["feature"], r["test_map"]) for r in score_rows} != {
        (s, f, m) for s, f in VIEWS for m in maps
    }:
        raise ValueError("score identity coverage differs")
    cube = np.full((12, 4, 2, 3, 16, 8), np.nan)
    maximum = 0.0
    for row in score_rows:
        s, f, m = row["start"], row["feature"], row["test_map"]
        recipe = maps[m]["recipe"]
        if (
            any(row[k] != v for k, v in recipe.items())
            or row["reference_map"] != recipe["reference"]
        ):
            raise ValueError("score recipe differs")
        paths = {
            key: {
                label: v["path"]
                for (ss, ff, mm, label), v in vectors.items()
                if (ss, ff, mm) == (s, f, key)
            }
            for key in {m, recipe["reference"]}
        }
        result = oracle_score(
            paths[recipe["reference"]], paths[m], labels, families, metadata
        )
        if row["available"] != (result is not None):
            raise ValueError("score availability differs")
        if result is None:
            if "scores" in row or not row.get("reason"):
                raise ValueError("unavailable score was imputed")
            continue
        scores = result[0]
        if row["labels"] != labels or row["families"] != families:
            raise ValueError("score labels differ")
        close(row["scores"], scores, "independent cosine matrix")
        maximum = max(
            maximum, float(np.max(np.abs(scores - np.asarray(row["scores"]))))
        )
        verify_summary(row["summary"], scores, labels, families, metadata)
        if recipe["q"] is not None:
            cube[
                VIEWS.index((s, f)),
                recipe["mask"],
                recipe["destination"],
                (0, 1, 4).index(recipe["q"]),
            ] = scores
    aggregates = summary["aggregates"]
    if len(aggregates) != 36 or {
        (r["start"], r["feature"], r["q_sixteenths"]) for r in aggregates
    } != {(s, f, q) for s, f in VIEWS for q in (0, 1, 4)}:
        raise ValueError("overlap aggregate coverage differs")
    for row in aggregates:
        values = cube[
            VIEWS.index((row["start"], row["feature"])),
            :,
            :,
            (0, 1, 4).index(row["q_sixteenths"]),
        ]
        if row["available"] != bool(np.isfinite(values).all()):
            raise ValueError("aggregate availability differs")
        if row["available"]:
            verify_summary(row, values.mean(axis=(0, 1)), labels, families, metadata)
    margins = np.full((12, 4, 2, 3, 8, 2), np.nan)
    for index in np.ndindex(12, 4, 2, 3):
        if np.isfinite(cube[index]).all():
            values = oracle_summary(cube[index], labels, families, metadata)[0]
            for fi, family in enumerate(families):
                margins[index + (fi,)] = [
                    v
                    for k, v in zip(labels, values)
                    if metadata[k]["pairing_family"] == family
                ]
    if sha256_file(output / "paired_margins.npy") != summary["paired_margins_sha256"]:
        raise ValueError("paired margins changed")
    saved = np.load(output / "paired_margins.npy", allow_pickle=False)
    if (
        saved.shape != margins.shape
        or not np.array_equal(np.isnan(saved), np.isnan(margins))
        or np.isinf(saved).any()
    ):
        raise ValueError("paired margin missingness differs")
    close(saved[np.isfinite(margins)], margins[np.isfinite(margins)], "paired margins")
    spec = frozen["config"]["planned_panel"]["bootstrap"]
    ranges = oracle_ranges(margins, spec)
    if any(summary["bootstrap"][k] != v for k, v in spec.items()):
        raise ValueError("bootstrap specification differs")
    endpoints = summary["primary_endpoints"]
    names = ("zero_overlap_margin", "high_minus_zero_margin")
    if len(endpoints) != 24 or {
        (e["start"], e["feature"], e["endpoint"]) for e in endpoints
    } != {(s, f, e) for s, f in VIEWS for e in names}:
        raise ValueError("twenty-four endpoint coverage differs")
    for vi, interval in enumerate(ranges):
        if summary["bootstrap"]["available_views"][vi] != (interval is not None):
            raise ValueError("bootstrap availability differs")
        if interval is None:
            if summary["bootstrap"]["ranges"][vi] is not None:
                raise ValueError("unavailable range was imputed")
        else:
            close(summary["bootstrap"]["ranges"][vi], interval, "independent bootstrap")
    for row in endpoints:
        vi = VIEWS.index((row["start"], row["feature"]))
        ei = names.index(row["endpoint"])
        if (
            row["available"] != (ranges[vi] is not None)
            or row["p_value"] is not None
            or row["equivalence_tested"]
        ):
            raise ValueError("endpoint availability/claim differs")
        if not row["available"]:
            if any(
                k in row
                for k in ("mean", "family_values", "mask_values", "exploratory_range")
            ):
                raise ValueError("unavailable endpoint was imputed")
            continue
        values = (
            margins[vi, :, :, 0]
            if ei == 0
            else margins[vi, :, :, 2] - margins[vi, :, :, 0]
        )
        close(row["mean"], values.mean(), "primary mean")
        close(row["mask_values"], values.mean(axis=(1, 2, 3)), "mask means")
        if set(row["family_values"]) != set(families):
            raise ValueError("endpoint family coverage differs")
        close(
            [row["family_values"][f] for f in families],
            values.mean(axis=(0, 1, 3)),
            "family means",
        )
        close(row["exploratory_range"], ranges[vi][ei], "endpoint range")
    directions = ledgers["directions.jsonl"]
    if {
        (
            r["start"],
            r["feature"],
            r["mask"],
            r["destination"],
            r["pair_id"],
            r["q_sixteenths"],
        )
        for r in directions
    } != {
        (s, f, m, d, k, q)
        for s, f in VIEWS
        for m in range(4)
        for d in range(2)
        for k in labels
        for q in (1, 4)
    }:
        raise ValueError("direction identity coverage differs")
    for row in directions:
        s, f, m, d, k, q = (
            row[x]
            for x in (
                "start",
                "feature",
                "mask",
                "destination",
                "pair_id",
                "q_sixteenths",
            )
        )
        a = portable[
            archive_key(
                {"start": s, "feature": f, "map": f"m{m}_q0_d{d}", "pair_id": k}
            )
        ]
        b = portable[
            archive_key(
                {"start": s, "feature": f, "map": f"m{m}_q{q}_d{d}", "pair_id": k}
            )
        ]
        na, nb = float(np.sqrt(np.sum(a * a))), float(np.sqrt(np.sum(b * b)))
        expected = {
            "zero_norm": na,
            "other_norm": nb,
            "cosine": float(np.sum(a * b) / (na * nb)) if na and nb else None,
            "norm_ratio": nb / na if na else None,
            "relative_l2_difference": float(np.sqrt(np.sum((b - a) ** 2)) / na)
            if na
            else None,
        }
        for key, value in expected.items():
            if value is None:
                if row[key] is not None:
                    raise ValueError("undefined direction statistic was imputed")
            else:
                close(row[key], value, "paired direction")
    return (
        summary,
        ledgers,
        portable,
        {
            "vector_arrays_verified": 6144,
            "score_matrices_verified": 336,
            "direction_records_verified": 3072,
            "bootstrap_verified": True,
            "maximum_score_oracle_abs_error": maximum,
        },
    )


def verify_processor(processor, artifact, expected, config):
    rgb, _ = _checked_artifact(artifact)
    payload = processor(
        images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
    )
    pixels = np.asarray(payload["pixel_values"])
    other = {
        k: np.asarray(v).tolist() for k, v in payload.items() if k != "pixel_values"
    }
    if (
        rgb.dtype != np.uint8
        or list(rgb.shape) != config["raw_shape_hwc"]
        or pixels.dtype != np.float32
        or list(pixels.shape) != config["processor_shape_nchw"]
        or not np.isfinite(pixels).all()
        or sha256_array(pixels) != expected["pixel_values_sha256"]
        or other != expected["processor_non_pixel_payload"]
    ):
        raise ValueError("independent processor regeneration differs")
    return rgb, pixels


def publish(root, output):
    if any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        raise ValueError(
            "offline flags required for independent processor regeneration"
        )
    if output.exists():
        raise FileExistsError("fresh verification/publication root required; no resume")
    capture, frozen, cells = load_capture(root)
    output.mkdir(parents=True)
    attempted = {"calibration": 0, "panel": 0}
    try:
        if (
            shutil.disk_usage(output).free
            < frozen["config"]["planned_panel"]["minimum_free_disk_gib"] * 2**30
        ):
            raise ValueError("insufficient independent verification disk reserve")
        analysis, ledgers, portable, statistics = verify_analysis(root, frozen, cells)
        config = frozen["config"]
        synthetic, arrays = synthetic_calibration(config)
        if synthetic != read(root / "synthetic_calibration.json"):
            raise ValueError("synthetic calibration replay differs")
        with np.load(root / "synthetic_features.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(
                not np.array_equal(saved[k], a) for k, a in arrays.items()
            ):
                raise ValueError("synthetic array replay differs")
        pre = read(root / "preflight_receipt.json")
        pre_root = Path(pre["path"])
        if (
            sha256_file(pre_root / "preflight.json") != pre["sha256"]
            or read(pre_root / "frozen_specification.json") != frozen
            or read(pre_root / "preflight.json")["status"] != "PASS"
            or read(pre_root / "preflight.json")["new_processor_calls"] != 0
            or read(pre_root / "preflight.json")["new_model_forwards"] != 0
            or read(pre_root / "preflight.json")["processor_instantiated"]
            or read(pre_root / "preflight.json")["model_weights_loaded"]
        ):
            raise ValueError("preflight provenance differs")
        processor, _ = qualified_processor(read(BASELINE))
        calibration = []
        for job in frozen["processor_jobs"]:
            attempted["calibration"] += 1
            _, pixels = verify_processor(processor, job["artifact"], job, config)
            calibration.append(
                {
                    "pair_id": job["pair_id"],
                    "cell": job["cell"],
                    "pixel_values_sha256": sha256_array(pixels),
                    "exact": True,
                }
            )
        if len(calibration) != 128:
            raise ValueError("verification calibration budget differs")
        print("independent processor calibration 128/128 exact", flush=True)
        source = read(REPO / config["phase1_root"] / "input_summary.json")
        lookup = {(r["map"], r["pair_id"]): r for r in source["records"]}
        tolerance = config["calibration"]["oracle_absolute_tolerance"]
        verified = []
        maximum = 0.0
        for i, row in enumerate(cells):
            if (
                shutil.disk_usage(output).free
                < config["planned_panel"]["minimum_free_disk_gib"] * 2**30
            ):
                raise ValueError("independent verification disk reserve breached")
            attempted["panel"] += 1
            write_json(
                output / "verification_progress.json",
                {"attempted": attempted, "current_cell": row["path"]},
            )
            block = lookup[(row["map"], row["pair_id"])]
            rgb, pixels = verify_processor(
                processor,
                block["artifacts"][row["cell"]],
                block["cells"][row["cell"]],
                config,
            )
            saved = load_features(root, row)
            errors = {}
            for start, values in zip(STARTS, (rgb, pixels)):
                expected = oracle_features(values, start, config)
                for feature in FEATURES:
                    key = feature_key(start, feature)
                    error = float(np.max(np.abs(saved[key] - expected[feature])))
                    if error > tolerance:
                        raise ValueError(
                            f"independent real-image feature differs: {row['path']}/{key}/{error}"
                        )
                    errors[key] = error
                    maximum = max(maximum, error)
            checked = {
                "ordinal": row["ordinal"],
                "map": row["map"],
                "pair_id": row["pair_id"],
                "cell": row["cell"],
                "pixel_values_sha256": sha256_array(pixels),
                "feature_oracle_abs_errors": errors,
                "passed": True,
            }
            verified.append(checked)
            with (output / "feature_checks.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(checked, separators=(",", ":"), allow_nan=False) + "\n"
                )
            if (i + 1) % 64 == 0:
                print(f"independent feature cells {i + 1}/2048", flush=True)
        if attempted != {"calibration": 128, "panel": 2048} or len(verified) != 2048:
            raise ValueError("verification processor budget differs")
        assert_hashes(frozen)
        public_vectors = [
            {**row, "public_array_key": archive_key(row)}
            for row in ledgers["vectors.jsonl"]
        ]
        public_ledgers = {
            **ledgers,
            "vectors.jsonl": public_vectors,
            "cells.jsonl": cells,
            "calibration.jsonl": read_jsonl(root / "calibration.jsonl"),
            "verification_calibration.jsonl": calibration,
        }
        receipts = {
            name: write_ledger(output / name, rows)
            for name, rows in public_ledgers.items()
        }
        receipts["feature_checks.jsonl"] = {
            "rows": len(verified),
            "sha256": sha256_file(output / "feature_checks.jsonl"),
        }
        np.savez_compressed(output / "interaction_vectors.npz", **portable)
        statistics.update(
            historical_processor_regenerations=128,
            experimental_processor_regenerations=2048,
            real_image_feature_oracle_pairs=24576,
            maximum_feature_oracle_abs_error=maximum,
            verification_processor_calls=2176,
            verification_model_forwards=0,
            full_processor_pixels_stored=False,
        )
        summary = {
            **analysis,
            "analysis_kind": "nonlearned_processing_panel_publication",
            "frozen_specification": frozen,
            "capture_summary": capture,
            "source_analysis_sha256": sha256_file(root / "analysis/summary.json"),
            "ledgers": receipts,
            "interaction_archive_sha256": sha256_file(
                output / "interaction_vectors.npz"
            ),
            "independent_verification": statistics,
            "context_input_endpoints": read(REPO / config["phase1_publication"])[
                "primary_endpoints"
            ],
            "context_cache_endpoints": read(REPO / config["phase2_publication"])[
                "primary_endpoints"
            ],
            "claim_boundaries": [
                "Reused-data development, not untouched confirmation.",
                "All twelve views and twenty-four endpoints retained, including unavailable directions.",
                "Descriptive profile comparisons are not sufficiency, equivalence, semantics or causal learning effects.",
                "Processor regeneration and feature-oracle verification add no new independent sources or model forwards.",
            ],
        }
        plot(summary, output)
        summary["figures"] = {
            p.name: sha256_file(p) for p in sorted(output.glob("*.png"))
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(statistics, indent=2), flush=True)
        return summary
    except Exception as error:
        write_json(
            output / "verification_failure.json",
            {
                "schema_version": 1,
                "type": type(error).__name__,
                "message": str(error),
                "attempted_processor_calls": attempted,
                "primary_results_withheld": True,
            },
        )
        raise


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(11, 17), sharex=True, sharey="row")
    for fi, feature in enumerate(FEATURES):
        for si, start in enumerate(STARTS):
            ax = axes[fi, si]
            rows = [
                r
                for r in summary["aggregates"]
                if (r["start"], r["feature"]) == (start, feature)
            ]
            if all(r["available"] for r in rows):
                for family in rows[0]["family_margins"]:
                    ax.plot(
                        [0, 6.25, 25],
                        [r["family_margins"][family] for r in rows],
                        color="0.65",
                        alpha=0.5,
                        linewidth=0.8,
                    )
                ax.plot(
                    [0, 6.25, 25],
                    [r["mean_family_margin"] for r in rows],
                    "o-",
                    color="#2376a8" if si else "#25865b",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Unavailable direction",
                    ha="center",
                    transform=ax.transAxes,
                )
            ax.axhline(0, color="0.4", linewidth=0.7)
            ax.set(
                title=f"{start} / {feature}",
                xticks=[0, 6.25, 25],
                xlabel="Shared fixed content (%)",
                ylabel="Family margin",
            )
    fig.suptitle("Non-learned processing: every registered feature path retained")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.text(
        0.5,
        0.01,
        "Common row scales; eight families shown. Development inputs; no fitted baseline selection.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output / "baseline_overlap_profiles.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, endpoint, title in zip(
        axes,
        ("zero_overlap_margin", "high_minus_zero_margin"),
        ("Zero-overlap margin", "25% minus zero margin"),
    ):
        rows = [r for r in summary["primary_endpoints"] if r["endpoint"] == endpoint]
        for i, row in enumerate(rows):
            if row["available"]:
                ax.plot(row["exploratory_range"], [i, i], color="#2376a8", linewidth=2)
                ax.scatter(row["mean"], i, color="black", s=18)
            else:
                ax.text(0, i, "Unavailable", fontsize=8)
        ax.set(
            yticks=range(12),
            yticklabels=[
                f"{'raw' if s == 'raw_rgb' else 'proc'} / {f}" for s, f in VIEWS
            ],
            title=title,
            xlabel="Margin or paired difference",
        )
        ax.axvline(0, color="0.4", linewidth=0.7)
        ax.invert_yaxis()
    fig.suptitle(
        "Twenty-four registered endpoints: exploratory ranges, not equivalence tests"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(
        0.5,
        0.02,
        "Fixed references; tails 0.025/24 and 1-0.025/24, unchanged for unavailable views. Coverage not guaranteed.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output / "baseline_endpoint_ranges.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    publish(args.run_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
