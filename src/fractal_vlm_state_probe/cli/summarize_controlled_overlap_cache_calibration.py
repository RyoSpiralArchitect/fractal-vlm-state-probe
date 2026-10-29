from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.controlled_overlap_cache import (
    independent_exact_arrays,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json


def checked_array(run_path, artifact):
    path = (run_path.parent / artifact["path"]).resolve()
    if (
        not path.is_relative_to(run_path.parent.resolve())
        or _sha(path) != artifact["sha256"]
    ):
        raise ValueError("tensor file path/hash differs")
    with np.load(path, allow_pickle=False) as saved:
        array = saved[artifact["array_key"]]
    if (
        list(array.shape) != artifact["shape"]
        or str(array.dtype) != artifact["dtype"]
        or artifact["source_dtype"] != "mlx.core.bfloat16"
        or artifact["effective_sequence_length"] != 397
    ):
        raise ValueError("tensor storage metadata differs")
    return array


def verify_pair(before_path, after_path, receipt, frozen, frozen_sha):
    old, new = _read(before_path), _read(after_path)
    audit_path = after_path.with_suffix(".input_audit.json")
    audit, old_audit = (
        _read(audit_path),
        _read(before_path.with_suffix(".input_audit.json")),
    )
    if (
        _sha(after_path) != receipt["run_sha256"]
        or _sha(audit_path) != receipt["input_audit_sha256"]
    ):
        raise ValueError("capture/audit hash changed")
    if len(old["stream_events"]) != 1 or len(new["stream_events"]) != 1:
        raise ValueError("source event denominator differs")
    a, b = old["stream_events"][0], new["stream_events"][0]
    for key in ("model_id", "context_policy", "runtime"):
        if old[key] != new[key]:
            raise ValueError("source model/protocol/runtime differs")
    if new.get("dry_run") or new["reproducibility"]["seed"] != 20260604:
        raise ValueError("invalid source execution")
    for key in ("prompt", "frame_sha256s", "cache_token_layout"):
        if a[key] != b[key]:
            raise ValueError("source input or token layout differs")
    if (
        b["assistant_text"] != "The image"
        or a["assistant_text"] != b["assistant_text"]
        or [s["token_id"] for s in b["generation"]["steps"]] != [785, 2168, 2168]
        or [s["token_id"] for s in a["generation"]["steps"]] != [785, 2168, 2168]
    ):
        raise ValueError("source suffix text/IDs differ")
    layout = b["cache_token_layout"]["cache_position_layout"]
    if (
        not layout["available"]
        or not layout["image_partition_available"]
        or layout["strategy"] != "llava_qwen2_single_image_run_replacement"
        or layout["image_position_runs"] != [{"start": 42, "end": 297, "length": 256}]
        or [
            layout[k]
            for k in (
                "pre_image_position_count",
                "image_position_count",
                "post_image_position_count",
                "cache_sequence_length",
            )
        ]
        != [42, 256, 99, 397]
    ):
        raise ValueError("effective token region differs")
    if (
        audit["native"] != audit["effective"]
        or audit["override_image_sizes"] is not None
        or audit["effective"] != old_audit["effective"]
        or old_audit["native"] != old_audit["effective"]
        or old_audit["override_image_sizes"] is not None
        or audit["effective"]["image_sizes"]["values"] != [[320, 320]]
        or audit["context"]["frozen_specification_sha256"] != frozen_sha
    ):
        raise ValueError("actual prepared inputs differ")
    manifest = Path(audit["context"]["manifest_path"])
    if (
        _sha(manifest) != audit["context"]["manifest_sha256"]
        or frozen["manifest_sha256"][str(manifest)] != _sha(manifest)
        or _read(manifest)["frames"][0]["sha256"] != b["frame_sha256s"][0]
    ):
        raise ValueError("source manifest changed")
    if receipt["tensor_artifacts"] != b["cache_tensor_artifacts"]:
        raise ValueError("tensor registry differs")
    first = {
        f"{r['layer_index']}:{r['tensor']}": r for r in a["cache_tensor_artifacts"]
    }
    second = {
        f"{r['layer_index']}:{r['tensor']}": r for r in b["cache_tensor_artifacts"]
    }
    if (
        len(a["cache_tensor_artifacts"]) != 3
        or len(b["cache_tensor_artifacts"]) != 3
        or set(first) != {"1:keys", "12:keys", "23:values"}
        or set(second) != set(first)
    ):
        raise ValueError("tensor target denominator differs")
    result = {}
    for target in first:
        before, after = (
            checked_array(before_path, first[target]),
            checked_array(after_path, second[target]),
        )
        independent_exact_arrays(before, after)
        result[target] = {
            "before_array_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
            "after_array_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
            "nonzero": bool(np.any(before)),
            "scalar_count": before.size,
        }
    return result, audit, b["assistant_text"]


def publish(root, output):
    if output.exists():
        raise FileExistsError("use a fresh publication output root")
    if (root / "execution_failure.json").exists():
        raise ValueError("failure retained; do not publish a passing calibration")
    summary, frozen, calibration = (
        _read(root / n)
        for n in ("summary.json", "frozen_specification.json", "calibration.json")
    )
    validate_config(frozen["config"])
    expected = {
        "status": "PASS",
        "calibration_source_cells": 128,
        "calibration_tensor_comparisons": 384,
        "new_model_forwards": 128,
        "new_experimental_panel_cells": 0,
        "new_direct_probe_cells": 0,
        "panel_started": False,
        "panel_automatically_released": False,
        "confirmation_cohort": "not_generated_not_sealed",
    }
    if (
        any(summary.get(k) != v for k, v in expected.items())
        or calibration["status"] != "PASS"
        or calibration["completed_cells"] != 128
    ):
        raise ValueError("calibration status, counts or release boundary differs")
    for name, field in (
        ("frozen_specification.json", "frozen_specification_sha256"),
        ("calibration.json", "calibration_sha256"),
        ("capture_registry.json", "capture_registry_sha256"),
        ("runtime_identity.json", "runtime_identity_sha256"),
        ("model_snapshot.json", "model_snapshot_sha256"),
        ("preflight_receipt.json", "preflight_receipt_sha256"),
    ):
        if _sha(root / name) != summary[field]:
            raise ValueError("calibration receipt changed")
    for group in ("code_sha256", "source_sha256", "manifest_sha256"):
        for name, digest in frozen[group].items():
            if _sha(REPO / name if group == "code_sha256" else Path(name)) != digest:
                raise ValueError("frozen source/code/manifest changed")
    if (
        _read(root / "runtime_identity.json") != frozen["runtime_identity"]
        or _read(root / "model_snapshot.json") != frozen["model_snapshot"]
    ):
        raise ValueError("qualified runtime/snapshot differs")
    preflight_receipt = _read(root / "preflight_receipt.json")
    preflight_root = Path(preflight_receipt["path"])
    preflight = _read(preflight_root / "preflight.json")
    if (
        _sha(preflight_root / "preflight.json") != preflight_receipt["sha256"]
        or (preflight_root / "execution_failure.json").exists()
        or _read(preflight_root / "frozen_specification.json") != frozen
        or _sha(preflight_root / "frozen_specification.json")
        != preflight["frozen_specification_sha256"]
        or _sha(preflight_root / "model_snapshot.json")
        != preflight["model_snapshot_sha256"]
        or preflight["status"] != "PASS"
        or preflight["new_model_forwards"] != 0
        or preflight["model_weights_loaded"]
        or preflight["panel_started"]
    ):
        raise ValueError("no-model preflight differs")
    registry = _read(root / "capture_registry.json")
    jobs = {(j["pair_id"], j["cell"]): j for j in frozen["jobs"]}
    labels = frozen["hierarchy_metadata"]
    if (
        len(frozen["jobs"]) != 128
        or set(jobs) != {(k, c) for k in labels for c in ("mm", "jj", "mj", "jm")}
        or len(labels) != 32
        or len(registry) != 128
        or set(registry) != {f"calibration/{k}_{c}.json" for k, c in jobs}
    ):
        raise ValueError("calibration source coverage differs")
    checks = {(r["pair_id"], r["cell"], r["target"]): r for r in calibration["checks"]}
    if len(calibration["checks"]) != 384 or set(checks) != {
        (k, c, t) for k, c in jobs for t in ("1:keys", "12:keys", "23:values")
    }:
        raise ValueError("comparison coverage differs")
    cells, verified, responses, scalar_count = [], [], Counter(), 0
    for (label, cell), job in jobs.items():
        relative = f"calibration/{label}_{cell}.json"
        receipt = registry[relative]
        before_path, after_path = Path(job["historical_run"]), root / relative
        if receipt["historical_run"] != str(before_path):
            raise ValueError("historical mapping differs")
        result, audit, response = verify_pair(
            before_path,
            after_path,
            receipt,
            frozen,
            summary["frozen_specification_sha256"],
        )
        if (
            audit["context"]["manifest_path"] != job["manifest_path"]
            or audit["effective"]["pixel_values"]["sha256"]
            != job["pixel_values_sha256"]
        ):
            raise ValueError("prepared source mapping differs")
        responses[response] += 1
        cells.append(
            {
                "pair_id": label,
                "cell": cell,
                **labels[label],
                "before_path": str(before_path),
                "after_path": str(after_path),
                **receipt,
                "prepared_inputs": audit["effective"],
                "source_response": response,
                "generated_step_ids": [785, 2168, 2168],
                "independent_input_and_layout_exact": True,
            }
        )
        for target, arrays in result.items():
            row = checks[(label, cell, target)]
            if (
                row["before_path"] != str(before_path)
                or row["after_path"] != str(after_path)
                or row["before_array_sha256"] != arrays["before_array_sha256"]
                or row["after_array_sha256"] != arrays["after_array_sha256"]
                or not all(
                    row[k] for k in ("bitwise_equal", "shape_equal", "dtype_equal")
                )
                or row["max_abs_difference"] != 0
                or row["rms_difference"] != 0
                or row["relative_l2_difference"] != (0 if arrays["nonzero"] else None)
            ):
                raise ValueError("reported tensor comparison differs")
            if arrays["nonzero"]:
                if (
                    row["cosine"] is None
                    or not np.isfinite(row["cosine"])
                    or abs(row["cosine"] - 1) > 1e-12
                ):
                    raise ValueError("identity cosine differs")
            elif row["cosine"] is not None:
                raise ValueError("zero tensor cosine must be unavailable")
            verified.append({**row, "independent_tensor_bytes_exact": True})
            scalar_count += arrays["scalar_count"]
    if (root / "panel").exists():
        raise ValueError("unexpected experimental panel output")
    output.mkdir(parents=True)
    ledgers = {}
    for name, rows in (("cells.jsonl", cells), ("comparisons.jsonl", verified)):
        path = output / name
        path.write_text(
            "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
        )
        ledgers[name] = {"rows": len(rows), "sha256": _sha(path)}
    published = {
        **summary,
        "analysis_kind": "controlled_overlap_cache_calibration_publication",
        "source_summary_sha256": _sha(root / "summary.json"),
        "frozen_specification": frozen,
        "preflight": preflight,
        "registered_design_commit": frozen["registered_design_commit"],
        "ledgers": ledgers,
        "source_response_counts": dict(responses),
        "independent_verification": {
            "source_cells": 128,
            "tensor_comparisons": 384,
            "scalar_pairs": scalar_count,
            "prepared_input_equalities": 128,
            "maximum_absolute_difference": 0,
            "verification_model_forwards": 0,
            "method": "reload_npz_check_hash_dtype_finiteness_and_uint8_array_equality",
        },
        "exporter_sha256": _sha(Path(__file__)),
        "claim_boundaries": [
            "Historical calibration replay, not controlled-overlap cache measurement.",
            "No automatic panel release; repeat calibration in the later panel process.",
            "Experimental/cache-correspondence and direct-probe evidence counts are unchanged.",
            "No new p-values, equivalence test, causal claim or confirmation cohort.",
        ],
    }
    write_json(output / "summary.json", published)
    print(json.dumps(published["independent_verification"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    publish(args.run_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
