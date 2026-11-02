from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cli.run_controlled_overlap import (
    BASELINE,
    REPO,
    qualified_processor,
    read,
)
from fractal_vlm_state_probe.cli.run_nonlearned_processing_calibration import (
    assert_hashes,
    sha256_array,
    specification as base_specification,
)
from fractal_vlm_state_probe.controlled_overlap_cache_panel import ordered_jobs
from fractal_vlm_state_probe.nonlearned_processing import STARTS, extract_features
from fractal_vlm_state_probe.nonlearned_processing_calibration import (
    synthetic_calibration,
)
from fractal_vlm_state_probe.nonlearned_processing_panel import (
    cell_path,
    feature_key,
    reference_seal,
    validate_features,
)
from fractal_vlm_state_probe.padding_policy import _checked_artifact
from fractal_vlm_state_probe.stimulus import sha256_file, write_json

RELEASE = REPO / "configs/nonlearned_processing_panel_release_v1.json"
CODE = [
    "src/fractal_vlm_state_probe/nonlearned_processing_panel.py",
    "src/fractal_vlm_state_probe/cli/run_nonlearned_processing_panel.py",
    "src/fractal_vlm_state_probe/cli/analyze_nonlearned_processing_panel.py",
    "src/fractal_vlm_state_probe/cli/summarize_nonlearned_processing_panel.py",
    "scripts/run_nonlearned_processing_panel.py",
    "scripts/analyze_nonlearned_processing_panel.py",
    "scripts/summarize_nonlearned_processing_panel.py",
    "configs/nonlearned_processing_panel_release_v1.json",
]


def specification():
    frozen, release = base_specification(), read(RELEASE)
    original = REPO / "configs/nonlearned_processing_phase3_v1.json"
    prior_path = REPO / release["calibration_publication"]
    prior = read(prior_path)
    if (
        sha256_file(original) != release["phase3_config_sha256"]
        or sha256_file(prior_path) != release["calibration_publication_sha256"]
        or prior["status"] != "PASS"
        or release["retry_or_resume"]
        or release["panel_processor_calls"] != 2176
        or release["verification_processor_calls"] != 2176
    ):
        raise ValueError("registered panel release differs")
    for name, digest in prior["frozen_specification"]["code_sha256"].items():
        if sha256_file(REPO / name) != digest:
            raise ValueError("qualified calibration code changed")
    frozen["source_sha256"][str(prior_path)] = sha256_file(prior_path)
    for name, receipt in prior["ledgers"].items():
        p = prior_path.parent / name
        if sha256_file(p) != receipt["sha256"]:
            raise ValueError("prior calibration ledger changed")
        frozen["source_sha256"][str(p)] = receipt["sha256"]
    for name in CODE:
        digest = sha256_file(REPO / name)
        stored = subprocess.check_output(
            ["git", "show", f"{frozen['registered_design_commit']}:{name}"], cwd=REPO
        )
        if hashlib.sha256(stored).hexdigest() != digest:
            raise ValueError("commit full panel implementation before measurement")
        frozen["code_sha256"][name] = digest
    frozen.update(
        analysis_kind="nonlearned_processing_panel_registration",
        release=release,
        panel_jobs=ordered_jobs(frozen["hierarchy_metadata"]),
    )
    return frozen


def process_cell(processor, artifact, expected, config):
    rgb, _ = _checked_artifact(artifact)
    if list(rgb.shape) != config["raw_shape_hwc"] or rgb.dtype != np.uint8:
        raise ValueError("raw source shape/dtype differs")
    payload = processor(
        images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
    )
    pixels = np.asarray(payload["pixel_values"])
    other = {
        k: np.asarray(v).tolist() for k, v in payload.items() if k != "pixel_values"
    }
    if (
        pixels.dtype != np.float32
        or list(pixels.shape) != config["processor_shape_nchw"]
        or not np.isfinite(pixels).all()
        or sha256_array(pixels) != expected["pixel_values_sha256"]
        or other != expected["processor_non_pixel_payload"]
    ):
        raise ValueError("actual processor pixels or non-pixel payload differs")
    return rgb, pixels, other


def append_json(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def run(mode, root, preflight_root=None):
    if any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        raise ValueError("both offline flags required")
    if root.exists():
        raise FileExistsError("fresh root required; no retry/resume")
    root.mkdir(parents=True)
    attempted = {"calibration": 0, "panel": 0}
    checks, journal = [], []
    try:
        frozen = specification()
        config = frozen["config"]
        if (
            shutil.disk_usage(root).free
            < config["planned_panel"]["minimum_free_disk_gib"] * 2**30
        ):
            raise ValueError("insufficient registered panel disk reserve")
        write_json(root / "frozen_specification.json", frozen)
        digest = sha256_file(root / "frozen_specification.json")
        if mode == "preflight":
            result = {
                "schema_version": 1,
                "analysis_kind": "nonlearned_processing_panel_preflight",
                "status": "PASS",
                "frozen_specification_sha256": digest,
                "new_processor_calls": 0,
                "new_model_forwards": 0,
                "processor_instantiated": False,
                "model_weights_loaded": False,
            }
            write_json(root / "preflight.json", result)
            return result
        if mode != "run" or preflight_root is None:
            raise ValueError("explicit run and preflight required")
        preflight = read(preflight_root / "preflight.json")
        if (
            preflight["status"] != "PASS"
            or preflight["new_processor_calls"] != 0
            or preflight["processor_instantiated"]
            or preflight["new_model_forwards"] != 0
            or preflight["model_weights_loaded"]
            or (preflight_root / "execution_failure.json").exists()
            or read(preflight_root / "frozen_specification.json") != frozen
            or sha256_file(preflight_root / "frozen_specification.json")
            != preflight["frozen_specification_sha256"]
        ):
            raise ValueError("qualified preflight differs")
        write_json(
            root / "preflight_receipt.json",
            {
                "path": str(preflight_root),
                "sha256": sha256_file(preflight_root / "preflight.json"),
            },
        )
        synthetic, arrays = synthetic_calibration(config)
        np.savez(root / "synthetic_features.npz", **arrays)
        write_json(root / "synthetic_calibration.json", synthetic)
        processor, _ = qualified_processor(read(BASELINE))
        for job in frozen["processor_jobs"]:
            attempted["calibration"] += 1
            write_json(
                root / "progress.json",
                {"stage": "calibration", "attempted": attempted, "current_job": job},
            )
            _, pixels, other = process_cell(processor, job["artifact"], job, config)
            check = {
                "pair_id": job["pair_id"],
                "cell": job["cell"],
                "pixel_values_sha256": sha256_array(pixels),
                "processor_non_pixel_payload": other,
                "exact": True,
            }
            checks.append(check)
            append_json(root / "calibration.jsonl", check)
        if len(checks) != 128:
            raise ValueError("fresh calibration coverage differs")
        print("historical processor calibration 128/128 exact", flush=True)
        study = read(REPO / config["phase1_root"] / "input_summary.json")
        source = {(r["map"], r["pair_id"]): r for r in study["records"]}
        for i, job in enumerate(frozen["panel_jobs"]):
            if i == 256:
                refs = reference_seal(journal, frozen["hierarchy_metadata"])
                write_json(root / "reference_features.json", refs)
                write_json(
                    root / "reference_seal.json",
                    {
                        "reference_cells": 256,
                        "completed_cells_at_seal": 384,
                        "first_test_ordinal": 385,
                        "sha256": sha256_file(root / "reference_features.json"),
                    },
                )
                print("256 reference cells sealed before test capture", flush=True)
            if (
                shutil.disk_usage(root).free
                < config["planned_panel"]["minimum_free_disk_gib"] * 2**30
            ):
                raise ValueError("disk reserve breached during capture")
            attempted["panel"] += 1
            write_json(
                root / "progress.json",
                {"stage": "panel", "attempted": attempted, "current_job": job},
            )
            row = source[(job["map"], job["pair_id"])]
            artifact, expected = (
                row["artifacts"][job["cell"]],
                row["cells"][job["cell"]],
            )
            rgb, pixels, other = process_cell(processor, artifact, expected, config)
            features = {}
            for start, values in zip(STARTS, (rgb, pixels)):
                features.update(
                    {
                        feature_key(start, f): a
                        for f, a in extract_features(values, start, config).items()
                    }
                )
            validate_features(features)
            relative = cell_path(job)
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **features)
            receipt = {
                **job,
                "ordinal": i + 129,
                "path": relative,
                "sha256": sha256_file(path),
                "manifest_path": artifact["manifest_path"],
                "manifest_sha256": artifact["manifest_sha256"],
                "frame_sha256": artifact["frame_sha256"],
                "pixel_values_sha256": sha256_array(pixels),
                "processor_non_pixel_payload": other,
                "feature_arrays": 12,
            }
            journal.append(receipt)
            append_json(root / "cells.jsonl", receipt)
            if (i + 1) % 64 == 0:
                print(f"feature cells {i + 1}/2048", flush=True)
        if len(journal) != 2048 or attempted != {"calibration": 128, "panel": 2048}:
            raise ValueError("experimental denominator differs")
        assert_hashes(frozen)
        result = {
            "schema_version": 1,
            "analysis_kind": "nonlearned_processing_panel_capture",
            "status": "COMPLETE",
            "source_cells": len(journal),
            "feature_arrays": 24576,
            "calibration_processor_calls": 128,
            "panel_processor_calls": 2048,
            "new_model_forwards": 0,
            "model_weights_loaded": False,
            "same_processor_instance_for_calibration_and_panel": True,
            "confirmation_cohort": "not_generated_not_sealed",
            "receipts": {
                name: sha256_file(root / name)
                for name in (
                    "frozen_specification.json",
                    "preflight_receipt.json",
                    "synthetic_features.npz",
                    "synthetic_calibration.json",
                    "calibration.jsonl",
                    "reference_features.json",
                    "reference_seal.json",
                    "cells.jsonl",
                )
            },
        }
        write_json(root / "capture_summary.json", result)
        return result
    except Exception as error:
        write_json(
            root / "execution_failure.json",
            {
                "schema_version": 1,
                "type": type(error).__name__,
                "message": str(error),
                "attempted": attempted,
                "completed_calibration": len(checks),
                "completed_panel_cells": len(journal),
                "primary_results_withheld": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.mode,
                args.output_root.resolve(),
                args.preflight_root.resolve() if args.preflight_root else None,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
