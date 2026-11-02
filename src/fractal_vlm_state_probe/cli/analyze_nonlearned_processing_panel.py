from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.analyze_controlled_overlap import pair_scores
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO, read
from fractal_vlm_state_probe.cli.run_nonlearned_processing_calibration import (
    assert_hashes,
)
from fractal_vlm_state_probe.controlled_overlap_cache_panel import direction
from fractal_vlm_state_probe.nonlearned_processing import factorial_interaction
from fractal_vlm_state_probe.nonlearned_processing_panel import (
    VIEWS,
    cell_path,
    feature_key,
    reference_seal,
    summarize_scores,
    validate_features,
    vector_diagnostics,
)
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_features(root, receipt):
    path = root / receipt["path"]
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("feature sidecar hash differs")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    validate_features(arrays)
    return arrays


def load_capture(root):
    if (root / "execution_failure.json").exists():
        raise ValueError("failed capture cannot acquire primary results")
    capture, frozen = (
        read(root / "capture_summary.json"),
        read(root / "frozen_specification.json"),
    )
    if (
        capture["status"] != "COMPLETE"
        or capture["source_cells"] != 2048
        or capture["feature_arrays"] != 24576
        or capture["new_model_forwards"] != 0
        or capture["model_weights_loaded"]
        or not capture["same_processor_instance_for_calibration_and_panel"]
        or capture["calibration_processor_calls"] != 128
        or capture["panel_processor_calls"] != 2048
    ):
        raise ValueError("capture count/protocol boundary differs")
    for name, digest in capture["receipts"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("capture receipt changed")
    assert_hashes(frozen)
    rows = read_jsonl(root / "cells.jsonl")
    if (
        len(rows) != 2048
        or [r["path"] for r in rows] != [cell_path(j) for j in frozen["panel_jobs"]]
        or [r["ordinal"] for r in rows] != list(range(129, 2177))
    ):
        raise ValueError("cell identity/order/coverage differs")
    refs = reference_seal(rows[:256], frozen["hierarchy_metadata"])
    if read(root / "reference_features.json") != refs or read(
        root / "reference_seal.json"
    ) != {
        "reference_cells": 256,
        "completed_cells_at_seal": 384,
        "first_test_ordinal": 385,
        "sha256": sha256_file(root / "reference_features.json"),
    }:
        raise ValueError("reference seal differs")
    checks = read_jsonl(root / "calibration.jsonl")
    jobs = frozen["processor_jobs"]
    if len(checks) != 128 or len(jobs) != 128:
        raise ValueError("calibration denominator differs")
    for row, job in zip(checks, jobs):
        if not row["exact"] or any(
            row[k] != job[k]
            for k in (
                "pair_id",
                "cell",
                "pixel_values_sha256",
                "processor_non_pixel_payload",
            )
        ):
            raise ValueError("historical processor calibration differs")
    phase1 = read(REPO / frozen["config"]["phase1_root"] / "input_summary.json")
    lookup = {(r["map"], r["pair_id"]): r for r in phase1["records"]}
    for row, job in zip(rows, frozen["panel_jobs"]):
        if any(row[k] != v for k, v in job.items()) or row["feature_arrays"] != 12:
            raise ValueError("cell job metadata differs")
        source = lookup[(row["map"], row["pair_id"])]
        artifact, expected = (
            source["artifacts"][row["cell"]],
            source["cells"][row["cell"]],
        )
        if any(
            row[k] != artifact[k]
            for k in ("manifest_path", "manifest_sha256", "frame_sha256")
        ) or any(
            row[k] != expected[k]
            for k in ("pixel_values_sha256", "processor_non_pixel_payload")
        ):
            raise ValueError("cell source provenance differs")
        load_features(root, row)
    return capture, frozen, rows


def write_ledger(path, rows):
    path.write_text(
        "".join(
            json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n" for r in rows
        )
    )
    return {"rows": len(rows), "sha256": sha256_file(path)}


def analyze(root):
    output = root / "analysis"
    if output.exists():
        raise FileExistsError("fresh analysis required; retain previous output")
    capture, frozen, rows = load_capture(root)
    output.mkdir()
    try:
        config, metadata = frozen["config"], frozen["hierarchy_metadata"]
        tolerance = config["calibration"]["oracle_absolute_tolerance"]
        phase1 = read(REPO / config["phase1_publication"])
        maps = phase1["map_construction"]["maps"]
        blocks = {}
        for row in rows:
            blocks.setdefault((row["map"], row["pair_id"]), {})[row["cell"]] = row
        if len(blocks) != 512:
            raise ValueError("factorial block count differs")
        paths = {feature_key(s, f): {m: {} for m in maps} for s, f in VIEWS}
        vectors, zero_raw = [], 0
        for (key, label), cell_rows in blocks.items():
            arrays = {c: load_features(root, r) for c, r in cell_rows.items()}
            for start, feature in VIEWS:
                view = feature_key(start, feature)
                value = factorial_interaction({c: a[view] for c, a in arrays.items()})
                diagnostic = vector_diagnostics(value, tolerance)
                if (start, feature) == ("raw_rgb", "global_moments"):
                    if np.any(value):
                        raise ValueError(
                            "raw multiset interaction did not cancel exactly"
                        )
                    zero_raw += 1
                path = output / "vectors" / view / key / f"{label}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, value)
                paths[view][key][label] = path
                vectors.append(
                    {
                        "start": start,
                        "feature": feature,
                        "map": key,
                        "pair_id": label,
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        **diagnostic,
                    }
                )
        labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
        families = sorted({m["pairing_family"] for m in metadata.values()})
        cube = np.full((12, 4, 2, 3, 16, 8), np.nan)
        scores = []
        for vi, (start, feature) in enumerate(VIEWS):
            view = paths[feature_key(start, feature)]
            for key, item in maps.items():
                recipe = item["recipe"]
                scored = pair_scores(view[recipe["reference"]], view[key], metadata)
                scores.append(
                    {
                        "start": start,
                        "feature": feature,
                        "test_map": key,
                        "reference_map": recipe["reference"],
                        **recipe,
                        **scored,
                    }
                )
                if scored["available"] and recipe["q"] is not None:
                    cube[
                        vi,
                        recipe["mask"],
                        recipe["destination"],
                        (0, 1, 4).index(recipe["q"]),
                    ] = scored["scores"]
        aggregates, endpoints, bootstrap, margins = summarize_scores(
            cube, labels, families, metadata, config
        )
        np.save(output / "paired_margins.npy", margins)
        directions = []
        for start, feature in VIEWS:
            view = paths[feature_key(start, feature)]
            for m in range(4):
                for d in range(2):
                    for label in labels:
                        a = np.load(view[f"m{m}_q0_d{d}"][label], allow_pickle=False)
                        for q in (1, 4):
                            b = np.load(
                                view[f"m{m}_q{q}_d{d}"][label], allow_pickle=False
                            )
                            directions.append(
                                {
                                    "start": start,
                                    "feature": feature,
                                    "mask": m,
                                    "destination": d,
                                    "pair_id": label,
                                    "q_sixteenths": q,
                                    **direction(a, b),
                                }
                            )
        if (
            len(vectors) != 6144
            or len(scores) != 336
            or len(directions) != 3072
            or zero_raw != 512
        ):
            raise ValueError("registered analysis coverage differs")
        ledgers = {
            name: write_ledger(output / name, data)
            for name, data in (
                ("vectors.jsonl", vectors),
                ("scores.jsonl", scores),
                ("directions.jsonl", directions),
            )
        }
        result = {
            "schema_version": 1,
            "analysis_kind": "nonlearned_processing_correspondence",
            "status": "COMPLETE",
            "capture_summary_sha256": sha256_file(root / "capture_summary.json"),
            "ledgers": ledgers,
            "paired_margins_sha256": sha256_file(output / "paired_margins.npy"),
            "aggregates": aggregates,
            "primary_endpoints": endpoints,
            "bootstrap": bootstrap,
            "raw_global_moment_zero_blocks": zero_raw,
            "new_model_forwards": 0,
            "new_processor_calls": 0,
            "new_p_values": 0,
            "confirmation_cohort": "not_generated_not_sealed",
        }
        write_json(output / "summary.json", result)
        for e in endpoints:
            print(json.dumps(e), flush=True)
        return result
    except Exception as error:
        write_json(
            output / "analysis_failure.json",
            {
                "schema_version": 1,
                "type": type(error).__name__,
                "message": str(error),
                "primary_results_withheld": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root.resolve())


if __name__ == "__main__":
    main()
