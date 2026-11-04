from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fractal_vlm_state_probe.cli.analyze_nonlearned_processing_panel import (
    read_jsonl,
    write_ledger,
)
from fractal_vlm_state_probe.cli.analyze_representation_panel import load_capture, scan
from fractal_vlm_state_probe.cli.run_representation_panel import (
    REPO,
    cal,
    cal_pub,
    read,
    recalibration_parity,
    sha,
)
from fractal_vlm_state_probe.cli.summarize_controlled_overlap import (
    close,
    oracle_bootstrap,
    oracle_summary,
    verify_summary,
)
from fractal_vlm_state_probe.cli.summarize_controlled_overlap_cache_calibration import (
    verify_pair,
)
from fractal_vlm_state_probe.representation_panel import ENDPOINTS, SITES
from fractal_vlm_state_probe.stimulus import write_json


def verify_statistics(frozen, analysis, ledgers, saved_margins, replay):
    vectors, scores, directions, cube, labels, families = replay
    cfg, meta = frozen["release"], frozen["hierarchy_metadata"]
    tol = cfg["score_absolute_tolerance"]

    def norm_check(a, b):
        if not np.isclose(a, b, atol=tol, rtol=cfg["norm_relative_tolerance"]):
            raise ValueError("independent vector norm differs")

    for original, independent in zip(ledgers["vectors.jsonl"], vectors, strict=True):
        if {k: v for k, v in original.items() if k != "norm"} != {
            k: v for k, v in independent.items() if k != "norm"
        }:
            raise ValueError(
                "independent full-vector bytes/identity/availability differ"
            )
        norm_check(original["norm"], independent["norm"])
    maximum = 0.0
    for a, b in zip(ledgers["scores.jsonl"], scores, strict=True):
        if any(a[k] != v for k, v in b.items() if k != "scores"):
            raise ValueError("score identity, recipe or availability differs")
        if not b["available"]:
            if "scores" in a or not a.get("reason"):
                raise ValueError("unavailable score imputed")
            continue
        matrix = np.asarray(b["scores"])
        close(a["scores"], matrix, "independent cosines", tol)
        maximum = max(maximum, float(np.max(np.abs(np.asarray(a["scores"]) - matrix))))
        verify_summary(a["summary"], matrix, labels, families, meta)
    for a, b in zip(ledgers["directions.jsonl"], directions, strict=True):
        for k, v in b.items():
            if k in ("zero_norm", "other_norm"):
                norm_check(a[k], v)
            elif (
                k in ("cosine", "norm_ratio", "relative_l2_difference")
                and v is not None
            ):
                close(a[k], v, "paired direction", tol)
            elif a[k] != v:
                raise ValueError("paired direction availability/identity differs")
    agg = analysis["aggregates"]
    if len(agg) != 6 or {(r["site"], r["q_sixteenths"]) for r in agg} != {
        (s, q) for s in SITES for q in (0, 1, 4)
    }:
        raise ValueError("aggregate coverage differs")
    for r in agg:
        a = cube[SITES.index(r["site"]), :, :, (0, 1, 4).index(r["q_sixteenths"])]
        if r["available"] != bool(np.isfinite(a).all()):
            raise ValueError("aggregate availability differs")
        if r["available"]:
            verify_summary(r, a.mean(axis=(0, 1)), labels, families, meta)
    margins = np.full((2, 4, 2, 3, 8, 2), np.nan)
    for idx in np.ndindex(2, 4, 2, 3):
        if np.isfinite(cube[idx]).all():
            values = oracle_summary(cube[idx], labels, families, meta)[0]
            for fi, family in enumerate(families):
                margins[idx + (fi,)] = [
                    v
                    for k, v in zip(labels, values)
                    if meta[k]["pairing_family"] == family
                ]
    if (
        margins.shape != saved_margins.shape
        or np.isinf(saved_margins).any()
        or not np.array_equal(np.isnan(margins), np.isnan(saved_margins))
    ):
        raise ValueError("margin missingness differs")
    good = np.isfinite(margins)
    close(saved_margins[good], margins[good], "paired margin re-derivation", tol)
    spec = cfg["bootstrap"]
    ranges = oracle_bootstrap(margins, spec)
    endpoints = analysis["primary_endpoints"]
    if (
        len(endpoints) != 4
        or {(r["site"], r["endpoint"]) for r in endpoints}
        != {(s, e) for s in SITES for e in ENDPOINTS}
        or any(analysis["bootstrap"][k] != v for k, v in spec.items())
    ):
        raise ValueError("four-endpoint family or bootstrap recipe differs")
    for r in endpoints:
        vi, ei = SITES.index(r["site"]), ENDPOINTS.index(r["endpoint"])
        available = bool(np.isfinite(np.take(margins[vi], [0, 2], axis=2)).all())
        if (
            r["available"] != available
            or analysis["bootstrap"]["available_views"][vi] != available
            or r["p_value"] is not None
            or r["equivalence_tested"]
        ):
            raise ValueError("primary availability/claim differs")
        if not available:
            if (
                any(
                    k in r
                    for k in (
                        "mean",
                        "family_values",
                        "mask_values",
                        "exploratory_range",
                    )
                )
                or analysis["bootstrap"]["ranges"][vi] is not None
            ):
                raise ValueError("unavailable primary result imputed")
            continue
        zero = margins[vi, :, :, 0]
        values = zero if ei == 0 else margins[vi, :, :, 2] - zero
        close(r["mean"], values.mean(), "endpoint mean", tol)
        close(r["mask_values"], values.mean(axis=(1, 2, 3)), "mask values", tol)
        if set(r["family_values"]) != set(families):
            raise ValueError("family coverage differs")
        close(
            [r["family_values"][f] for f in families],
            values.mean(axis=(0, 1, 3)),
            "family values",
            tol,
        )
        close(r["exploratory_range"], ranges[vi, ei], "independent bootstrap", tol)
        close(analysis["bootstrap"]["ranges"][vi], ranges[vi], "saved bootstrap", tol)
    return maximum


def plot_profiles(aggregates, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True, layout="constrained")
    xs = np.array([0, 6.25, 25])
    for ax, site, color in zip(axes, SITES, ("#216782", "#ad571d")):
        rows = sorted(
            (r for r in aggregates if r["site"] == site),
            key=lambda r: r["q_sixteenths"],
        )
        families = sorted(
            {f for r in rows if r["available"] for f in r["family_margins"]}
        )
        for f in families:
            ax.plot(
                xs,
                [r["family_margins"][f] if r["available"] else np.nan for r in rows],
                color="0.65",
                marker=".",
                lw=0.9,
            )
        ax.plot(
            xs,
            [r["mean_family_margin"] if r["available"] else np.nan for r in rows],
            color=color,
            marker="o",
            lw=2.5,
            label="Equal-family mean",
        )
        for x, r in zip(xs, rows):
            if not r["available"]:
                ax.text(x, 0, "N/A", ha="center", va="bottom")
        ax.axhline(0, color="0.35", lw=0.8, ls="--")
        ax.set(title=site.capitalize(), xticks=xs, xlabel="Shared fixed content (%)")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Within-class family margin")
    fig.suptitle(
        "Full-tensor correspondence across overlap levels\nEight development families; thin lines are family means",
        fontsize=12,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def publish(root, out):
    if out.exists():
        raise FileExistsError("fresh verification output required")
    out.mkdir(parents=True)
    try:
        capture, frozen, rows = load_capture(root)
        a_dir = root / "analysis"
        if (a_dir / "analysis_failure.json").exists():
            raise ValueError("failed analysis cannot be published")
        analysis = read(a_dir / "summary.json")
        if (
            analysis["status"] != "COMPLETE"
            or analysis["capture_summary_sha256"] != sha(root / "capture_summary.json")
            or analysis["new_model_forwards"] != 0
            or analysis["new_p_values"] != 0
            or analysis["confirmation_cohort"] != "not_generated_not_sealed"
        ):
            raise ValueError("analysis receipt/claim boundary differs")
        counts = {"vectors.jsonl": 1024, "scores.jsonl": 56, "directions.jsonl": 512}
        if set(analysis["ledgers"]) != set(counts):
            raise ValueError("analysis ledger set differs")
        ledgers = {}
        for name, n in counts.items():
            receipt = analysis["ledgers"][name]
            if receipt["rows"] != n or sha(a_dir / name) != receipt["sha256"]:
                raise ValueError("analysis ledger changed")
            ledgers[name] = read_jsonl(a_dir / name)
            if len(ledgers[name]) != n:
                raise ValueError("analysis denominator differs")
        if sha(a_dir / "paired_margins.npy") != analysis["paired_margins_sha256"]:
            raise ValueError("paired margins changed")
        if recalibration_parity(frozen, root) != read(
            root / "recalibration_parity.json"
        ):
            raise ValueError("fresh representation calibration differs")
        with TemporaryDirectory(dir=root) as tmp:
            replay_dir = Path(tmp) / "verified"
            cal_pub.publish(root / "recalibration", replay_dir)
            prior_dir = root / "recalibration_verified"
            if {p.name for p in prior_dir.iterdir()} != {
                p.name for p in replay_dir.iterdir()
            } or any(sha(p) != sha(prior_dir / p.name) for p in replay_dir.iterdir()):
                raise ValueError("fresh calibration publication replay differs")
        parity = []
        for row in rows:
            before, after = Path(row["prior_run"]), root / row["path"]
            checks, audit, _ = verify_pair(
                before,
                after,
                row,
                frozen,
                capture["receipts"]["frozen_specification.json"],
            )
            if (
                row["cache_array_sha256"]
                != {k: v["after_array_sha256"] for k, v in checks.items()}
                or row["pixel_values_sha256"]
                != audit["effective"]["pixel_values"]["sha256"]
                or audit["context"]["manifest_path"] != read(before)["manifest_path"]
            ):
                raise ValueError("panel input/cache parity mapping differs")
            parity.append(
                {k: row[k] for k in ("ordinal", "map", "pair_id", "cell")}
                | {"targets": checks}
            )
        replay = scan(root, frozen, rows, root, oracle=True)
        maximum = verify_statistics(
            frozen,
            analysis,
            ledgers,
            np.load(a_dir / "paired_margins.npy", allow_pickle=False),
            replay,
        )
        ledgers.update({"cells.jsonl": rows, "cache_parity.jsonl": parity})
        receipts = {n: write_ledger(out / n, data) for n, data in ledgers.items()}
        paths = {
            "full_inputs": REPO / frozen["config"]["phase1_publication"],
            "nonlearned": REPO / frozen["observation_config"]["baseline_publication"],
            "selected_cache": REPO / frozen["release"]["cache_publication"],
        }
        context = {}
        for key, p in paths.items():
            data = read(p)
            context[key] = {
                "publication": str(p.relative_to(REPO)),
                "sha256": sha(p),
                "primary_endpoints": data["primary_endpoints"],
                "aggregates": data["aggregates"],
            }
        figure = out / "representation_overlap_profiles.png"
        plot_profiles(analysis["aggregates"], figure)
        result = {
            **analysis,
            "analysis_kind": "representation_panel_publication",
            "frozen_specification": frozen,
            "capture_summary": capture,
            "source_analysis_sha256": sha(a_dir / "summary.json"),
            "ledgers": receipts,
            "figures": {figure.name: sha(figure)},
            "comparison_context": context,
            "recalibration_parity": read(root / "recalibration_parity.json"),
            "reference_seal": read(root / "reference_seal.json"),
            "independent_verification": {
                "packed_arrays_reexpanded": 4096,
                "full_contrast_byte_equalities": 1024,
                "score_matrices": 56,
                "direction_records": 512,
                "cache_tensor_pairs": sum(len(r["targets"]) for r in parity),
                "cache_scalar_pairs": sum(
                    v["scalar_count"] for r in parity for v in r["targets"].values()
                ),
                "maximum_score_abs_error": maximum,
                "bootstrap_replayed": True,
                "model_forwards": 0,
                "processor_calls": 0,
            },
            "full_cell_arrays_published": False,
            "experimental_new_direct_probe_cells": 0,
            "claim_boundaries": [
                "Reused development sources and fixed references, not confirmation.",
                "Different representations are not aligned coordinate systems; descriptive profile differences do not identify a training effect.",
                "Public scores support primary summary replay, not full representation reconstruction without local cell sidecars.",
                "No equivalence, causal gain, invariance or absence claim.",
            ],
        }
        cal.assert_frozen(frozen)
        write_json(out / "summary.json", result)
        print(json.dumps(result["independent_verification"], indent=2), flush=True)
        return result
    except Exception as e:
        write_json(
            out / "verification_failure.json",
            {"status": "FAILED", "message": str(e), "primary_results_withheld": True},
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    a = parser.parse_args()
    publish(a.run_root.resolve(), a.output_root.resolve())


if __name__ == "__main__":
    main()
