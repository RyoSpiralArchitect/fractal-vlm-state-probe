from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_tensor_artifact import load_cache_tensor_artifact
from fractal_vlm_state_probe.cli.analyze_controlled_overlap_cache_panel import (
    load_capture,
)
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.cli.summarize_controlled_overlap import (
    close,
    oracle_score,
    oracle_summary,
    verify_summary,
)
from fractal_vlm_state_probe.cli.summarize_controlled_overlap_cache_calibration import (
    verify_pair,
)
from fractal_vlm_state_probe.cli.summarize_map_seed_cache_study import verify_factorial
from fractal_vlm_state_probe.controlled_overlap_cache_panel import VIEWS
from fractal_vlm_state_probe.stimulus import write_json


def oracle_ranges(margins, spec):
    available = np.isfinite(np.take(margins, [0, 2], axis=3)).all(axis=(1, 2, 3, 4, 5))
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([spec["seed"]]))
    )
    draws = []
    for _ in range(spec["draws"]):
        masks, families = generator.integers(0, 4, 4), generator.integers(0, 8, 8)
        seeds = generator.integers(0, 2, (8, 2))
        total = np.zeros((6, 3))
        for mask in masks:
            destinations = generator.integers(0, 2, 2)
            block = np.take(margins[:, mask], destinations, axis=1).mean(axis=1)
            for index, family in enumerate(families):
                total += (
                    np.take(block[:, :, family], seeds[index], axis=-1).mean(axis=-1)
                    / 32
                )
        draws.append(np.column_stack((total[:, 0], total[:, 2] - total[:, 0])))
    draws = np.stack(draws)
    return [
        np.quantile(draws[:, vi], spec["quantiles"], axis=0).T
        if available[vi]
        else None
        for vi in range(6)
    ]


def verify_panel_inputs(root, registry, frozen):
    phase1 = _read(REPO / frozen["config"]["phase1_root"] / "input_summary.json")
    lookup = {(r["map"], r["pair_id"]): r for r in phase1["records"]}
    rows, responses = [], Counter()
    for relative, receipt in registry.items():
        if receipt["kind"] != "panel":
            continue
        path = root / relative
        run, old = _read(path), _read(Path(receipt["historical_run"]))
        audit = _read(path.with_suffix(".input_audit.json"))
        before_inputs = _read(
            Path(receipt["historical_run"]).with_suffix(".input_audit.json")
        )["effective"]
        cell = receipt["cell"]
        measured = lookup[(receipt["map"], receipt["pair_id"])]
        expected, artifact = measured["cells"][cell], measured["artifacts"][cell]
        event, historical = run["stream_events"][0], old["stream_events"][0]
        if (
            len(run["stream_events"]) != 1
            or run.get("dry_run")
            or run["reproducibility"]["seed"] != 20260604
            or any(run[k] != old[k] for k in ("model_id", "context_policy", "runtime"))
            or any(event[k] != historical[k] for k in ("prompt", "cache_token_layout"))
            or event["assistant_text"] != "The image"
            or [s["token_id"] for s in event["generation"]["steps"]]
            != [785, 2168, 2168]
            or event["frame_sha256s"] != [artifact["frame_sha256"]]
        ):
            raise ValueError("independent panel source/suffix/layout check failed")
        context = {
            "manifest_path": artifact["manifest_path"],
            "manifest_sha256": artifact["manifest_sha256"],
            "frozen_specification_sha256": _sha(root / "frozen_specification.json"),
        }
        current = audit["effective"]
        if (
            audit["context"] != context
            or audit["override_image_sizes"] is not None
            or audit["native"] != current
            or set(current) != set(before_inputs)
            or any(
                current[k] != before_inputs[k] for k in current if k != "pixel_values"
            )
            or current["pixel_values"]["sha256"] != expected["pixel_values_sha256"]
            or any(
                current["pixel_values"][k] != before_inputs["pixel_values"][k]
                for k in ("dtype", "shape")
            )
        ):
            raise ValueError("independent pixel-only input check failed")
        if len(event["cache_tensor_artifacts"]) != 3 or {
            f"{a['layer_index']}:{a['tensor']}" for a in event["cache_tensor_artifacts"]
        } != {"1:keys", "12:keys", "23:values"}:
            raise ValueError("independent target coverage differs")
        for a in event["cache_tensor_artifacts"]:
            values = load_cache_tensor_artifact(path, a)
            if (
                a["source_dtype"] != "mlx.core.bfloat16"
                or a["dtype"] != "float32"
                or values.shape != (1, 2, 397, 64)
                or not np.isfinite(values).all()
            ):
                raise ValueError("independent tensor storage differs")
        responses[event["assistant_text"]] += 1
        rows.append(
            {
                **receipt,
                "actual_prepared_inputs": current,
                "actual_source_response": event["assistant_text"],
                "generation_step_ids": [785, 2168, 2168],
                "independent_pixel_only_check": True,
            }
        )
    if len(rows) != 2048:
        raise ValueError("independent panel cell coverage differs")
    return rows, dict(responses)


def publish(root, output):
    if output.exists():
        raise FileExistsError("fresh publication root required")
    capture, frozen, registry = load_capture(root)
    analysis_root = root / "analysis"
    if (analysis_root / "analysis_failure.json").exists():
        raise ValueError("failed analysis cannot be published as verified")
    analysis = _read(analysis_root / "summary.json")
    if (
        analysis["capture_summary_sha256"] != _sha(root / "capture_summary.json")
        or analysis["new_model_forwards"] != 0
        or analysis["new_p_values"] != 0
        or analysis["confirmation_cohort"] != "not_generated_not_sealed"
    ):
        raise ValueError("analysis source or claim boundary differs")
    for name, digest in capture["receipts"].items():
        if _sha(root / name) != digest:
            raise ValueError("capture publication receipt changed")
    if (
        _read(root / "runtime_identity.json") != frozen["runtime_identity"]
        or _read(root / "model_snapshot.json") != frozen["model_snapshot"]
    ):
        raise ValueError("actual snapshot/runtime differs")
    preflight_receipt = _read(root / "preflight_receipt.json")
    preflight_root = Path(preflight_receipt["path"])
    preflight = _read(preflight_root / "preflight.json")
    if (
        _sha(preflight_root / "preflight.json") != preflight_receipt["sha256"]
        or _read(preflight_root / "frozen_specification.json") != frozen
        or preflight["status"] != "PASS"
        or preflight["new_model_forwards"] != 0
        or preflight["model_weights_loaded"]
    ):
        raise ValueError("preflight differs")
    calibration = _read(root / "calibration.json")
    checks = {(c["pair_id"], c["cell"], c["target"]): c for c in calibration["checks"]}
    calibrated = 0
    for relative, receipt in registry.items():
        if receipt["kind"] != "calibration":
            continue
        result, _, _ = verify_pair(
            Path(receipt["historical_run"]),
            root / relative,
            receipt,
            frozen,
            _sha(root / "frozen_specification.json"),
        )
        for target, array in result.items():
            row = checks[(receipt["pair_id"], receipt["cell"], target)]
            if any(
                row[k] != array[k]
                for k in ("before_array_sha256", "after_array_sha256")
            ):
                raise ValueError("calibration array receipt differs")
            calibrated += 1
    if calibrated != 384:
        raise ValueError("calibration independent denominator differs")
    cells, responses = verify_panel_inputs(root, registry, frozen)
    ledger_rows = {}
    for name, receipt in analysis["ledgers"].items():
        path = analysis_root / name
        if _sha(path) != receipt["sha256"]:
            raise ValueError("analysis ledger changed")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != receipt["rows"]:
            raise ValueError("analysis ledger denominator differs")
        ledger_rows[name] = rows
    if {k: len(v) for k, v in ledger_rows.items()} != {
        "vectors.jsonl": 3072,
        "factorials.jsonl": 1536,
        "scores.jsonl": 168,
        "directions.jsonl": 1536,
    }:
        raise ValueError("planned analysis coverage differs")
    metadata = frozen["hierarchy_metadata"]
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    vector_rows = ledger_rows["vectors.jsonl"]
    vectors = {
        (v["target"], v["region"], v["map"], v["pair_id"]): v for v in vector_rows
    }
    blocks = {(j["map"], j["pair_id"]) for j in frozen["panel_jobs"]}
    if len(vectors) != 3072 or set(vectors) != {
        (t, r, k, label) for t, r in VIEWS for k, label in blocks
    }:
        raise ValueError("vector identity coverage differs")
    for row in vector_rows:
        if _sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("saved interaction changed")
    factorial_rows = ledger_rows["factorials.jsonl"]
    if {(r["target"], r["map"], r["pair_id"]) for r in factorial_rows} != {
        (t, k, label) for t in ("1:keys", "12:keys", "23:values") for k, label in blocks
    }:
        raise ValueError("factorial identity coverage differs")
    vector_error, pre_image_zero = 0.0, 0
    argmax_image, energy_above_09 = 0, 0
    dominance = {r: Counter() for r in ("image_tokens", "all_effective")}
    for row in factorial_rows:
        path = Path(row["path"])
        if _sha(path) != row["sha256"]:
            raise ValueError("factorial report changed")
        report = _read(path)
        arrays = {}
        phase = (
            "reference" if metadata[row["pair_id"]]["replicate"] in (1, 2) else "test"
        )
        for cell, source in report["cells"].items():
            source_path = (
                root / f"panel/{phase}/{row['map']}/{row['pair_id']}_{cell}.json"
            )
            if source["source_path"] != str(source_path):
                raise ValueError("factorial source mapping differs")
            arrays[cell] = load_cache_tensor_artifact(
                source_path, source["cache_tensor_artifact"]
            ).astype(np.float64)
        interaction = verify_factorial(report, arrays)
        regions = {r["region"]: r for r in report["regions"]}
        argmax_image += (
            42
            <= regions["all_effective"]["effects"]["interaction"][
                "argmax_sequence_position"
            ]
            < 298
        )
        for region, counts in dominance.items():
            shares = regions[region]["balanced_contrast_energy"]["energy_shares"]
            if all(v is None for v in shares.values()):
                counts["unavailable"] += 1
            else:
                largest = max(shares.values())
                winners = [k for k, v in shares.items() if v == largest]
                counts[winners[0] if len(winners) == 1 else "tie"] += 1
        zero = all(
            np.array_equal(arrays["mm"][:, :, :42], arrays[c][:, :, :42])
            for c in ("jj", "mj", "jm")
        )
        if row["pre_image_all_effects_zero"] != zero:
            raise ValueError("pre-image zero diagnostic differs")
        pre_image_zero += zero
        fraction = report["interaction_partition"]["image_energy_fraction"]
        energy_above_09 += fraction is not None and fraction > 0.9
        if fraction is None:
            if row["interaction_image_energy_fraction"] is not None:
                raise ValueError("undefined image energy fraction differs")
        else:
            close(
                row["interaction_image_energy_fraction"],
                fraction,
                "image energy fraction",
            )
        for region, (start, stop) in (
            ("image_tokens", (42, 298)),
            ("post_image", (298, 397)),
        ):
            item = vectors[(row["target"], region, row["map"], row["pair_id"])]
            saved = np.load(item["path"], allow_pickle=False)
            expected = interaction[:, :, start:stop]
            close(saved, expected, "factorial interaction vector")
            vector_error = max(vector_error, float(np.max(np.abs(saved - expected))))
            if list(saved.shape) != item["shape"] or str(saved.dtype) != item["dtype"]:
                raise ValueError("interaction layout differs")
            close(
                item["norm"],
                float(np.sqrt(np.sum(expected * expected))),
                "interaction norm",
                tolerance=1e-10,
            )
    if (
        analysis["pre_image_all_effects_zero_count"] != pre_image_zero
        or analysis["factorial_target_analyses"] != 1536
    ):
        raise ValueError("factorial aggregate differs")
    maps = _read(REPO / frozen["config"]["phase1_publication"])["map_construction"][
        "maps"
    ]
    scores = ledger_rows["scores.jsonl"]
    if {(r["target"], r["region"], r["test_map"]) for r in scores} != {
        (t, r, key) for t, r in VIEWS for key in maps
    }:
        raise ValueError("score identity coverage differs")
    cube = np.full((6, 4, 2, 3, 16, 8), np.nan)
    score_error = 0.0
    for row in scores:
        target, region, key = row["target"], row["region"], row["test_map"]
        recipe = maps[key]["recipe"]
        if (
            any(row[k] != v for k, v in recipe.items())
            or row["reference_map"] != recipe["reference"]
        ):
            raise ValueError("score recipe differs")

        def paths(k):
            return {
                label: v["path"]
                for (t, r, m, label), v in vectors.items()
                if (t, r, m) == (target, region, k)
            }

        result = oracle_score(
            paths(recipe["reference"]), paths(key), labels, families, metadata
        )
        if row["available"] != (result is not None):
            raise ValueError("score availability differs")
        if result is None:
            continue
        values = result[0]
        if row["labels"] != labels or row["families"] != families:
            raise ValueError("score labels differ")
        close(row["scores"], values, "full cache cosine")
        score_error = max(score_error, float(np.max(np.abs(values - row["scores"]))))
        verify_summary(row["summary"], values, labels, families, metadata)
        if recipe["q"] is not None:
            cube[
                VIEWS.index((target, region)),
                recipe["mask"],
                recipe["destination"],
                (0, 1, 4).index(recipe["q"]),
            ] = values
    aggregates = analysis["aggregates"]
    if len(aggregates) != 18 or {
        (r["target"], r["region"], r["q_sixteenths"]) for r in aggregates
    } != {(t, r, q) for t, r in VIEWS for q in (0, 1, 4)}:
        raise ValueError("aggregate identity coverage differs")
    for row in aggregates:
        values = cube[
            VIEWS.index((row["target"], row["region"])),
            :,
            :,
            (0, 1, 4).index(row["q_sixteenths"]),
        ]
        if row["available"] != bool(np.isfinite(values).all()):
            raise ValueError("aggregate availability differs")
        if row["available"]:
            verify_summary(row, values.mean(axis=(0, 1)), labels, families, metadata)
    margins = np.full((6, 4, 2, 3, 8, 2), np.nan)
    for index in np.ndindex(6, 4, 2, 3):
        if np.isfinite(cube[index]).all():
            seed_values = oracle_summary(cube[index], labels, families, metadata)[0]
            for fi, family in enumerate(families):
                margins[index + (fi,)] = [
                    v
                    for k, v in zip(labels, seed_values)
                    if metadata[k]["pairing_family"] == family
                ]
    if (
        _sha(analysis_root / "paired_family_seed_margins.npy")
        != analysis["paired_margins_sha256"]
    ):
        raise ValueError("paired margins changed")
    saved = np.load(
        analysis_root / "paired_family_seed_margins.npy", allow_pickle=False
    )
    if saved.shape != margins.shape or not np.array_equal(
        np.isnan(saved), np.isnan(margins)
    ):
        raise ValueError("paired margin missingness differs")
    close(saved[np.isfinite(margins)], margins[np.isfinite(margins)], "paired margins")
    spec = frozen["config"]["analysis"]["bootstrap"]
    ranges = oracle_ranges(margins, spec)
    for k in ("draws", "seed", "quantiles", "interpretation"):
        if analysis["bootstrap"][k] != spec[k]:
            raise ValueError("bootstrap specification differs")
    endpoints = analysis["primary_endpoints"]
    if len(endpoints) != 12 or {
        (r["target"], r["region"], r["endpoint"]) for r in endpoints
    } != {
        (t, r, e)
        for t, r in VIEWS
        for e in ("zero_overlap_margin", "high_minus_zero_margin")
    }:
        raise ValueError("endpoint coverage differs")
    for vi, interval in enumerate(ranges):
        if analysis["bootstrap"]["available_views"][vi] != (interval is not None):
            raise ValueError("bootstrap availability differs")
        if interval is None:
            if analysis["bootstrap"]["ranges"][vi] is not None:
                raise ValueError("unavailable bootstrap was reported")
        else:
            close(
                analysis["bootstrap"]["ranges"][vi],
                interval,
                "independent paired bootstrap",
            )
    for row in endpoints:
        vi = VIEWS.index((row["target"], row["region"]))
        ei = ("zero_overlap_margin", "high_minus_zero_margin").index(row["endpoint"])
        if (
            row["available"] != (ranges[vi] is not None)
            or row["p_value"] is not None
            or row["equivalence_tested"]
        ):
            raise ValueError("endpoint claim/availability differs")
        if not row["available"]:
            continue
        values = (
            margins[vi, :, :, 0]
            if ei == 0
            else margins[vi, :, :, 2] - margins[vi, :, :, 0]
        )
        close(row["mean"], values.mean(), "primary mean")
        close(row["mask_values"], values.mean(axis=(1, 2, 3)), "primary mask means")
        if set(row["family_values"]) != set(families):
            raise ValueError("endpoint family coverage differs")
        close(
            [row["family_values"][f] for f in families],
            values.mean(axis=(0, 1, 3)),
            "primary family means",
        )
        close(row["exploratory_range"], ranges[vi][ei], "primary range")
    directions = ledger_rows["directions.jsonl"]
    if {
        (
            r["target"],
            r["region"],
            r["mask"],
            r["destination"],
            r["pair_id"],
            r["q_sixteenths"],
        )
        for r in directions
    } != {
        (t, r, m, d, label, q)
        for t, r in VIEWS
        for m in range(4)
        for d in range(2)
        for label in labels
        for q in (1, 4)
    }:
        raise ValueError("direction coverage differs")
    for row in directions:
        t, r, m, d, label, q = (
            row[k]
            for k in (
                "target",
                "region",
                "mask",
                "destination",
                "pair_id",
                "q_sixteenths",
            )
        )
        a = np.load(
            vectors[(t, r, f"m{m}_q0_d{d}", label)]["path"], allow_pickle=False
        ).ravel()
        b = np.load(
            vectors[(t, r, f"m{m}_q{q}_d{d}", label)]["path"], allow_pickle=False
        ).ravel()
        na, nb = np.sqrt(np.sum(a * a)), np.sqrt(np.sum(b * b))
        expected = {
            "zero_norm": na,
            "other_norm": nb,
            "cosine": np.sum(a * b) / (na * nb) if na and nb else None,
            "norm_ratio": nb / na if na else None,
            "relative_l2_difference": np.sqrt(np.sum((b - a) ** 2)) / na
            if na
            else None,
        }
        for k, v in expected.items():
            if v is None:
                if row[k] is not None:
                    raise ValueError("undefined direction statistic differs")
            else:
                close(
                    row[k],
                    v,
                    "direction diagnostic",
                    tolerance=1e-10 if k.endswith("norm") else 1e-12,
                )
    output.mkdir(parents=True)
    receipts = {}
    for name, rows in {
        **ledger_rows,
        "cells.jsonl": cells,
        "calibration.jsonl": calibration["checks"],
    }.items():
        p = output / name
        p.write_text(
            "".join(
                json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n"
                for r in rows
            )
        )
        receipts[name] = {"rows": len(rows), "sha256": _sha(p)}
    verification = {
        "calibration_tensor_pairs": calibrated,
        "panel_source_cells": len(cells),
        "factorial_targets": 1536,
        "interaction_vectors": 3072,
        "score_matrices": 168,
        "direction_records": 1536,
        "maximum_interaction_oracle_abs_error": vector_error,
        "maximum_score_oracle_abs_error": score_error,
        "bootstrap_verified": True,
        "verification_model_forwards": 0,
    }
    summary = {
        **analysis,
        "analysis_kind": "controlled_overlap_cache_panel_publication",
        "frozen_specification": frozen,
        "capture_summary": capture,
        "source_analysis_sha256": _sha(analysis_root / "summary.json"),
        "source_response_counts": responses,
        "ledgers": receipts,
        "independent_verification": verification,
        "localization_diagnostics": {
            "factorial_targets": 1536,
            "pre_image_all_effects_zero_count": pre_image_zero,
            "interaction_argmax_image_count": argmax_image,
            "image_energy_fraction_above_0_9_count": energy_above_09,
            "balanced_dominance_by_region": {k: dict(v) for k, v in dominance.items()},
            "interpretation": "dependent_selected_target_diagnostics_not_replication_units",
        },
        "input_primary_endpoints": _read(REPO / frozen["config"]["phase1_publication"])[
            "primary_endpoints"
        ],
        "exporter_sha256": _sha(Path(__file__)),
        "verification_helper_sha256": {
            str(Path(inspect_module).relative_to(REPO)): _sha(Path(inspect_module))
            for inspect_module in (
                REPO
                / "src/fractal_vlm_state_probe/cli/summarize_map_seed_cache_study.py",
                REPO
                / "src/fractal_vlm_state_probe/cli/summarize_controlled_overlap.py",
                REPO
                / "src/fractal_vlm_state_probe/cli/summarize_controlled_overlap_cache_calibration.py",
            )
        },
        "claim_boundaries": [
            "Registered reused-data development, not untouched confirmation.",
            "Ranges are exploratory; no new p-values or equivalence/absence claims.",
            "Overlap changes spatial arrangements; between-level spectra are not identical.",
            "Input/cache score differences are not causal gain, persistence or semantic steering.",
        ],
    }
    plot(summary, output)
    summary["figures"] = {p.name: _sha(p) for p in sorted(output.glob("*.png"))}
    write_json(output / "summary.json", summary)
    print(json.dumps(verification, indent=2), flush=True)


def plot(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True)
    for ax, (target, region) in zip(axes.ravel(), VIEWS):
        rows = [
            r
            for r in summary["aggregates"]
            if (r["target"], r["region"]) == (target, region)
        ]
        x = [0, 6.25, 25]
        color = "#2376a8" if region == "image_tokens" else "#c75d37"
        if all(r["available"] for r in rows):
            for f in rows[0]["family_margins"]:
                ax.plot(
                    x,
                    [r["family_margins"][f] for r in rows],
                    color="0.55",
                    alpha=0.35,
                    linewidth=0.8,
                )
            ax.plot(x, [r["mean_family_margin"] for r in rows], "o-", color=color)
        else:
            ax.text(
                0.5,
                0.5,
                "Incomplete secondary profile",
                transform=ax.transAxes,
                ha="center",
            )
        ax.axhline(0, color="0.3", linewidth=0.7)
        ax.set(
            title=f"L{target.replace(':', ' ')} / {region.replace('_', ' ')}",
            xticks=x,
            xlabel="Shared fixed content (%)",
            ylabel="Family margin",
        )
    fig.suptitle("Controlled overlap in cache: all eight families retained")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.text(
        0.5,
        0.015,
        "Common y scale. Four masks x two destinations; development images, fixed references.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output / "cache_overlap_profiles.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, endpoint, title in zip(
        axes,
        ("zero_overlap_margin", "high_minus_zero_margin"),
        ("Zero-overlap margin", "25% minus zero margin"),
    ):
        rows = [r for r in summary["primary_endpoints"] if r["endpoint"] == endpoint]
        for i, row in enumerate(rows):
            if row["available"]:
                ax.plot(
                    row["exploratory_range"],
                    [i, i],
                    color="#2376a8" if i % 2 == 0 else "#c75d37",
                    linewidth=2,
                )
                ax.scatter(row["mean"], i, color="black", s=20)
            else:
                ax.text(0, i, "Unavailable", fontsize=8)
        ax.set(
            yticks=range(6),
            yticklabels=[
                f"L{t.replace(':', ' ')} / {'image' if r == 'image_tokens' else 'post'}"
                for t, r in VIEWS
            ],
            title=title,
            xlabel="Mean margin or paired difference",
        )
        ax.axvline(0, color="0.4", linewidth=0.7)
        ax.invert_yaxis()
    fig.suptitle("Twelve registered endpoints: exploratory bootstrap ranges")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.text(
        0.5,
        0.025,
        "Fixed references; tails 0.025/12 and 1-0.025/12. Coverage is not guaranteed.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(output / "cache_endpoint_ranges.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    publish(args.run_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
