from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_controlled_overlap import read
from fractal_vlm_state_probe.cli.run_nonlearned_processing_calibration import (
    assert_hashes,
)
from fractal_vlm_state_probe.nonlearned_processing import FEATURES, STARTS
from fractal_vlm_state_probe.nonlearned_processing_calibration import (
    fixture,
    oracle_features,
    oracle_local_features,
)
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def publish(root, output):
    if output.exists():
        raise FileExistsError("fresh publication directory required")
    if (root / "execution_failure.json").exists():
        raise ValueError("failed calibration remains withheld")
    result, frozen, synthetic = (
        read(root / name)
        for name in (
            "summary.json",
            "frozen_specification.json",
            "synthetic_calibration.json",
        )
    )
    if (
        result["status"] != "PASS"
        or result["new_model_forwards"] != 0
        or result["model_weights_loaded"]
        or result["experimental_source_cells_scored"] != 0
        or result["experimental_panel_released"]
        or result["encoder_or_projector_released"]
        or result["confirmation_cohort"] != "not_generated_not_sealed"
        or result["registered_design_commit"] != frozen["registered_design_commit"]
    ):
        raise ValueError("calibration status or release boundary differs")
    for name, digest in result["receipts"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("calibration output receipt changed")
    assert_hashes(frozen)
    spec, config = frozen["config"]["calibration"], frozen["config"]
    before = read(root / "preflight_receipt.json")
    preflight_root = Path(before["path"])
    preflight = read(preflight_root / "preflight.json")
    if (
        sha256_file(preflight_root / "preflight.json") != before["sha256"]
        or read(preflight_root / "frozen_specification.json") != frozen
        or sha256_file(preflight_root / "frozen_specification.json")
        != preflight["frozen_specification_sha256"]
        or (preflight_root / "execution_failure.json").exists()
        or preflight["status"] != "PASS"
        or preflight["new_processor_calls"] != 0
        or preflight["new_model_forwards"] != 0
        or preflight["processor_instantiated"]
        or preflight["model_weights_loaded"]
        or preflight["input_blocks_verified"] != 896
        or preflight["input_cells_verified"] != 3584
    ):
        raise ValueError("no-model/no-processor preflight differs")
    cases = {
        (start, name, side, feature)
        for start in STARTS
        for name, side in (
            [(n, spec["fixture_side"]) for n in spec["fixtures"]]
            + [(spec["full_size_fixture"], 320 if start == "raw_rgb" else 1024)]
        )
        for feature in FEATURES
    }
    checks = synthetic["feature_checks"]
    if (
        len(checks) != 84
        or {(c["start"], c["fixture"], c["side"], c["feature"]) for c in checks}
        != cases
    ):
        raise ValueError("synthetic oracle coverage differs")
    arrays = np.load(root / "synthetic_features.npz", allow_pickle=False)
    if set(arrays.files) != {c["array_key"] for c in checks} | {
        "raw_multiset_zero_interaction"
    }:
        raise ValueError("saved synthetic feature coverage differs")
    by_key = {c["array_key"]: c for c in checks}
    verified, maximum = [], 0.0
    for start, name, side in sorted({k[:3] for k in cases}):
        source = fixture(name, side, start, spec["synthetic_seed"])
        expected = oracle_features(source, start, config)
        for feature in FEATURES:
            key = f"{start}__{name}__{side}__{feature}"
            row, actual = by_key[key], arrays[key]
            wanted = expected[feature]
            error = float(np.max(np.abs(actual - wanted)))
            if (
                actual.shape != wanted.shape
                or str(actual.dtype) != "float64"
                or list(actual.shape) != row["shape"]
                or not np.isfinite(actual).all()
                or not row["passed"]
                or error > spec["oracle_absolute_tolerance"]
                or abs(error - row["max_abs_error"]) > 1e-14
            ):
                raise ValueError("independent saved feature verification differs")
            maximum = max(maximum, error)
            verified.append(
                {
                    **row,
                    "independent_max_abs_error": error,
                    "array_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                }
            )
    negative = arrays["raw_multiset_zero_interaction"]
    if (
        negative.shape != (6,)
        or negative.dtype != np.float64
        or np.any(negative)
        or synthetic["raw_multiset_negative_control"]
        != {
            "exact_zero": True,
            "available": False,
            "norm": 0.0,
            "reason": "exact_zero_direction_unavailable",
        }
    ):
        raise ValueError("undefined raw negative control differs")
    linear = synthetic["linear_contrast_checks"]
    if len(linear) != 6 or {(r["start"], r["feature"]) for r in linear} != {
        (s, f) for s in STARTS for f in ("grid_mean", "small_mean", "large_mean")
    }:
        raise ValueError("linear calibration coverage differs")
    for start in STARTS:
        cells = {
            k: fixture("noise", 32, start, spec["synthetic_seed"] + i + 1)
            for i, k in enumerate(("mm", "jj", "mj", "jm"))
        }
        images = {
            k: (
                v.astype(float) / 255
                if start == "raw_rgb"
                else np.transpose(v[0].astype(float), (1, 2, 0))
            )
            for k, v in cells.items()
        }
        z = images["mm"] + images["jj"] - images["mj"] - images["jm"]
        features = {
            k: oracle_local_features(v, config["windows"][start], config["grid"])
            for k, v in images.items()
        }
        on_z = oracle_local_features(z, config["windows"][start], config["grid"])
        for row in (r for r in linear if r["start"] == start):
            f = row["feature"]
            contrast = (
                features["mm"][f]
                + features["jj"][f]
                - features["mj"][f]
                - features["jm"][f]
            )
            error = float(np.max(np.abs(contrast - on_z[f])))
            if (
                error > spec["oracle_absolute_tolerance"]
                or not 0 <= row["max_abs_error"] <= spec["oracle_absolute_tolerance"]
            ):
                raise ValueError("linear commutation oracle differs")
    nonlinear = synthetic["nonlinear_order_checks"]
    if (
        len(nonlinear) != 4
        or {(r["start"], r["feature"]) for r in nonlinear}
        != {(s, f) for s in STARTS for f in ("small_energy", "large_energy")}
        or any(abs(r["max_abs_order_difference"] - 0.625) > 1e-12 for r in nonlinear)
    ):
        raise ValueError("closed-form nonlinear order witness differs")
    processor = [
        json.loads(line)
        for line in (root / "processor_checks.jsonl").read_text().splitlines()
    ]
    jobs = frozen["processor_jobs"]
    if (
        len(processor) != 128
        or len(jobs) != 128
        or len({(r["pair_id"], r["cell"]) for r in processor}) != 128
    ):
        raise ValueError("processor calibration coverage differs")
    for row, job in zip(processor, jobs):
        if (
            any(
                row[k] != job[k]
                for k in (
                    "pair_id",
                    "cell",
                    "pixel_values_sha256",
                    "processor_non_pixel_payload",
                )
            )
            or any(
                row[k] != job["artifact"][k]
                for k in ("manifest_path", "manifest_sha256")
            )
            or row["shape"] != [1, 3, 1024, 1024]
            or row["dtype"] != "float32"
            or not row["exact"]
        ):
            raise ValueError("historical processor comparison receipt differs")
    if (
        result["historical_processor_checks"] != len(processor)
        or result["new_processor_calls"] != 128
        or result["synthetic_feature_oracle_pairs"] != len(verified)
        or abs(result["maximum_feature_oracle_abs_error"] - maximum) > 1e-14
        or abs(synthetic["maximum_feature_oracle_abs_error"] - maximum) > 1e-14
    ):
        raise ValueError("calibration summary counts or error differ")
    output.mkdir(parents=True)
    ledgers = {}
    for name, rows in (
        ("feature_checks.jsonl", verified),
        ("processor_checks.jsonl", processor),
    ):
        path = output / name
        path.write_text(
            "".join(
                json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n"
                for r in rows
            )
        )
        ledgers[name] = {"rows": len(rows), "sha256": sha256_file(path)}
    arrays.close()
    shutil.copyfile(root / "synthetic_features.npz", output / "synthetic_features.npz")
    summary = {
        **result,
        "analysis_kind": "nonlearned_processing_calibration_publication",
        "frozen_specification": frozen,
        "preflight": preflight,
        "source_summary_sha256": sha256_file(root / "summary.json"),
        "synthetic_calibration": synthetic,
        "ledgers": ledgers,
        "synthetic_features_sha256": sha256_file(output / "synthetic_features.npz"),
        "independent_verification": {
            "saved_feature_arrays": len(verified),
            "raw_zero_arrays": 1,
            "maximum_feature_oracle_abs_error": maximum,
            "linear_oracle_checks": 6,
            "closed_form_nonlinear_checks": 4,
            "historical_processor_receipts": len(processor),
            "new_processor_calls": 0,
            "new_model_forwards": 0,
            "processor_pixels_reloaded": False,
        },
        "exporter_sha256": sha256_file(Path(__file__)),
        "claim_boundaries": [
            "Synthetic calibration and historical processor replay only; no baseline correspondence scores.",
            "Independent verification reloads synthetic arrays; historical pixel tensors are not stored or independently reloaded.",
            "A passing calibration does not release the planned image panel or encoder/projector paths.",
            "No learned-versus-unlearned, semantic, equivalence or confirmation conclusion.",
        ],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary["independent_verification"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    publish(args.run_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
