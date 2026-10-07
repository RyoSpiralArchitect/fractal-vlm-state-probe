"""Offline pixel/count audit, independent of the measurement and gate helpers."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def histogram(rgb):
    p = rgb.reshape(-1, 3).astype(np.uint32)
    codes = (p[:, 0] << 16) | (p[:, 1] << 8) | p[:, 2]
    values, counts = np.unique(codes, return_counts=True)
    return dict(zip(map(int, values), map(int, counts)))


def verify(root, publication, live_path, prior_path):
    summary, live, prior = (
        read(publication / "summary.json"),
        read(live_path),
        read(prior_path),
    )
    frozen_path = live_path.parent / "frozen_specification.json"
    frozen = read(frozen_path)
    assert sha(live_path) == summary["source_report_sha256"]
    assert sha(frozen_path) == summary["frozen_specification_sha256"]
    assert frozen == summary["frozen_specification"] == live["frozen_specification"]
    assert sha(prior_path) == frozen["source_receipt_sha256"]
    for path, digest in frozen["code_sha256"].items():
        assert sha(root / path) == digest
    assert (
        sha(root / "src/fractal_vlm_state_probe/cli/summarize_padding_policy_study.py")
        == summary["code_exporter_sha256"]
    )
    for name, digest in summary["figures"].items():
        assert sha(publication / name) == digest
    assert sha(publication / "cells.jsonl") == summary["cell_ledger"]["sha256"]
    rows = [
        json.loads(line)
        for line in (publication / "cells.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1152 == summary["cell_ledger"]["rows"]
    assert len({(r["pair_id"], r["condition"], r["cell"]) for r in rows}) == 1152
    sources = {r["pair_id"]: r for r in prior["records"]}
    observed = {r["pair_id"]: r for r in live["records"]}
    compact = {r["pair_id"]: r for r in summary["records"]}
    cache, groups, histograms, fills = {}, defaultdict(dict), {}, {}
    oracle_errors = {"mean": [], "variance": []}
    shams = 0

    def pixels(artifact):
        path = Path(artifact["manifest_path"])
        if path not in cache:
            assert sha(path) == artifact["manifest_sha256"]
            manifest = read(path)
            frame = path.parent / manifest["frames"][0]["path"]
            assert (
                sha(frame)
                == artifact["frame_sha256"]
                == manifest["frames"][0]["sha256"]
            )
            with Image.open(frame) as image:
                cache[path] = np.asarray(image.convert("RGB")).copy()
        return cache[path]

    for row in rows:
        pair, condition, cell = row["pair_id"], row["condition"], row["cell"]
        state, policy = condition.split("/")
        block = sources[pair]
        artifacts = (
            block["sham_artifacts"]
            if state == "original"
            else block["arms"][
                "rank_low_pass" if state == "frozen_low_pass" else state
            ]["artifacts"]
        )
        content, canvas = pixels(artifacts[cell]), pixels(row["artifact"])
        assert content.shape == (240, 320, 3) and canvas.shape == (320, 320, 3)
        assert row["content_frame_sha256"] == artifacts[cell]["frame_sha256"]
        assert row["content_manifest_sha256"] == artifacts[cell]["manifest_sha256"]
        assert np.array_equal(canvas[40:280], content)
        donor = "mm" if cell in ("mm", "jm") else "jj"
        original_palette = pixels(block["sham_artifacts"][donor])
        if policy == "black":
            fill = np.zeros(3, dtype=np.uint8)
        elif policy == "fixed_gray":
            fill = np.full(3, 128, dtype=np.uint8)
        else:
            assert policy == "palette_mean"
            fill = np.rint(
                original_palette.sum(axis=(0, 1), dtype=np.uint64) / 76800
            ).astype(np.uint8)
        assert np.all(canvas[:40] == fill) and np.all(canvas[280:] == fill)
        audit = row["marginal_audit"]
        assert fill.tolist() == audit["fill_rgb_u8"]
        assert (
            audit["content_fraction"],
            audit["original_pixel_count"],
            audit["canvas_pixel_count"],
            audit["added_padding_pixels"],
        ) == (0.75, 76800, 102400, 25600)
        original_hist = histogram(original_palette)
        assert histogram(content) == original_hist
        expanded = histogram(canvas)
        expected = dict(original_hist)
        fill_code = int(fill[0]) * 65536 + int(fill[1]) * 256 + int(fill[2])
        expected[fill_code] = expected.get(fill_code, 0) + 25600
        assert expanded == expected
        tv = 0.5 * sum(
            abs(expanded.get(k, 0) / 102400 - original_hist.get(k, 0) / 76800)
            for k in expanded.keys() | original_hist.keys()
        )
        assert abs(tv - audit["joint_rgb_total_variation_from_content"]) < 1e-12
        # Reproduce the frozen float64 gate; integer sums below are a separate oracle.
        c, x, f = (
            content.astype(np.float64) / 255,
            canvas.astype(np.float64) / 255,
            fill / 255,
        )
        mu, var = c.mean(axis=(0, 1)), c.var(axis=(0, 1))
        expected_mean = 0.75 * mu + 0.25 * f
        expected_var = 0.75 * var + 0.75 * 0.25 * (mu - f) ** 2
        assert expected_mean.tolist() == audit["expected_canvas_rgb_mean"]
        assert expected_var.tolist() == audit["expected_canvas_rgb_variance"]
        assert (
            float(np.max(np.abs(x.mean(axis=(0, 1)) - expected_mean)))
            == audit["mean_max_abs_error"]
            <= 1e-12
        )
        assert (
            float(np.max(np.abs(x.var(axis=(0, 1)) - expected_var)))
            == audit["variance_max_abs_error"]
            <= 1e-12
        )
        integer = canvas.astype(np.uint64)
        mean = integer.sum(axis=(0, 1)) / (102400 * 255)
        variance = (integer * integer).sum(axis=(0, 1)) / (102400 * 255**2) - mean**2
        oracle_errors["mean"].append(float(np.max(np.abs(mean - expected_mean))))
        oracle_errors["variance"].append(float(np.max(np.abs(variance - expected_var))))
        hist_key = (pair, condition, donor)
        assert histograms.get(hist_key, expanded) == expanded
        histograms[hist_key] = expanded
        fill_key = (pair, policy, donor)
        assert fills.get(fill_key, fill.tolist()) == fill.tolist()
        fills[fill_key] = fill.tolist()
        assert row["pixel_values_shape"] == [3, 1024, 1024]
        saved = observed[pair]["conditions"][condition]
        assert row == {
            "pair_id": pair,
            "condition": condition,
            "cell": cell,
            **saved["cells"][cell],
            "artifact": saved["artifacts"][cell],
        }
        if policy == "black":
            sham = row["black_pad_sham"]
            assert (
                sham["pixel_values_bitwise_equal"]
                and not sham["whole_processor_payload_equal"]
            )
            assert sham["baseline_pixel_values_sha256"] == row["pixel_values_sha256"]
            before, after = (
                sham["baseline_non_pixel_payload"],
                row["processor_non_pixel_payload"],
            )
            assert [
                k
                for k in sorted(before.keys() | after.keys())
                if before.get(k) != after.get(k)
            ] == ["image_sizes"]
            assert before["image_sizes"] == [[320, 240]] and after["image_sizes"] == [
                [320, 320]
            ]
            assert sham["changed_non_pixel_payload_keys"] == ["image_sizes"]
            shams += 1
        groups[(pair, condition)][cell] = row

    counts = Counter()
    for (pair, condition), cells in groups.items():
        assert set(cells) == {"mm", "jj", "mj", "jm"}
        c = np.array([r["processor"]["spectral_centroid"] for r in cells.values()])
        h = np.array(
            [r["processor"]["high_frequency_energy_ratio"] for r in cells.values()]
        )
        assert np.all(np.isfinite(c)) and np.all(c > 0) and np.all(np.isfinite(h))
        assert all(r["processor"]["luminance_std"] > 0 for r in cells.values())
        error, spread = (
            float(np.max(np.abs(c - c.mean()) / c.mean())),
            float(h.max() - h.min()),
        )
        accepted = error <= 0.05 and spread <= 0.02
        gate = compact[pair]["conditions"][condition]["gate"]
        assert gate == observed[pair]["conditions"][condition]["gate"]
        assert accepted == gate["accepted"]
        assert abs(error - gate["max_centroid_relative_error"]) < 1e-14
        assert abs(spread - gate["hf_max_pairwise_absolute_difference"]) < 1e-14
        assert (
            gate["centroid_relative_tolerance"] == 0.05
            and gate["hf_absolute_tolerance"] == 0.02
        )
        counts[condition] += accepted
    for condition, status in summary["conditions"].items():
        assert counts[condition] == status["accepted_blocks"]
        accepted_ids = [
            p
            for p, k in groups
            if k == condition and compact[p]["conditions"][k]["gate"]["accepted"]
        ]
        assert accepted_ids == status["accepted_pair_ids"]
        families = defaultdict(set)
        for pair in accepted_ids:
            families[compact[pair]["pairing_family"]].add(compact[pair]["replicate"])
        assert (
            sorted(f for f, seeds in families.items() if seeds == {1, 2, 3, 4})
            == status["complete_four_seed_families"]
        )
    assert len(groups) == 288 and shams == 384
    diagnostics = {
        name: {
            "max_abs_error": max(errors),
            "cells_above_1e_12": sum(e > 1e-12 for e in errors),
            "total_cells": len(errors),
        }
        for name, errors in oracle_errors.items()
    }
    return {
        "schema_version": 1,
        "analysis_kind": "padding_policy_independent_artifact_verification",
        "scope": "independent offline artifact audit; no processor or model forward",
        "summary_sha256": sha(publication / "summary.json"),
        "cell_ledger_sha256": sha(publication / "cells.jsonl"),
        "verifier_sha256": sha(__file__),
        "verified_cells": len(rows),
        "independently_recomputed_four_cell_gates": len(groups),
        "integer_rgb_histogram_mixtures_exact": True,
        "original_float64_moment_gate_reproduced": True,
        "black_pixel_sham_receipts_verified": shams,
        "accepted_blocks_by_condition": dict(counts),
        "integer_sum_moment_oracle": diagnostics,
        "numeric_caveat": "Stored float64-reduction moments have rounding drift relative to integer-sum moments. This additional diagnostic does not change the frozen moment gate or measured inputs; exact integer RGB histogram identities hold.",
        "result": "PASS_WITH_FLOAT64_REDUCTION_CAVEAT"
        if any(d["cells_above_1e_12"] for d in diagnostics.values())
        else "PASS",
        "new_model_forwards": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--live-summary", type=Path, required=True)
    parser.add_argument("--prior-receipt", type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("run this assertion-based offline verifier without -O")
    result = verify(
        Path(__file__).resolve().parents[1],
        args.publication,
        args.live_summary,
        args.prior_receipt,
    )
    (args.publication / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
