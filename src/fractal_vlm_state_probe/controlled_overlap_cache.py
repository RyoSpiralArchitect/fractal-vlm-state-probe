from __future__ import annotations

import numpy as np

from .cache_direction_holdout import validate_pairing_holdout_metadata

CELLS = ("mm", "jj", "mj", "jm")
TARGETS = ("1:keys", "12:keys", "23:values")


def validate_config(config):
    required = {
        "schema_version": 1,
        "study_id": "controlled_overlap_cache_phase2_v1",
        "model_id": "mlx-community/FastVLM-0.5B-bf16",
        "targets": list(TARGETS),
        "regions": ["image_tokens", "post_image"],
        "region_positions": {
            "pre_image": [0, 42],
            "image_tokens": [42, 298],
            "post_image": [298, 397],
        },
        "expected_tensor_shape": [1, 2, 397, 64],
        "expected_source_dtype": "mlx.core.bfloat16",
        "expected_saved_dtype": "float32",
        "expected_source_response": "The image",
        "expected_generation_step_ids": [785, 2168, 2168],
        "expected_image_sizes": [[320, 320]],
        "source_seed": 20260604,
        "source_max_tokens": 2,
        "temperature": 0,
        "historical_map": "historical_3q",
        "confirmation_cohort": "not_generated_not_sealed",
    }
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("registered cache contract differs")
    calibration, panel, analysis = (
        config[k] for k in ("calibration", "panel", "analysis")
    )
    expected = {
        "source_blocks": 32,
        "cells": 128,
        "tensor_comparisons": 384,
        "rule": "all_prepared_inputs_suffix_ids_layout_and_tensor_bytes_exact",
        "execution": "historical_cells_only_no_controlled_overlap_cache",
        "retry": "none_retain_partial_failure_stop_without_replacement",
        "resume": False,
        "passing_releases_panel_automatically": False,
        "repeat_before_later_panel_in_same_process": True,
        "minimum_free_disk_gib": 2,
    }
    if calibration != expected:
        raise ValueError("calibration budget or fail-closed policy differs")
    for key, value in {
        "status_at_registration": "not_implemented_not_released",
        "source_blocks": 32,
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "mask_replicates": 4,
        "destination_replicates": 2,
        "overlap_sixteenths": [0, 1, 4],
        "maps": 28,
        "input_gate_blocks": 896,
        "factorial_blocks": 512,
        "source_cells": 2048,
        "tensor_sidecars": 6144,
        "factorial_target_analyses": 1536,
        "calibration_cells_before_panel": 128,
        "calibration_tensors_before_panel": 384,
        "total_planned_source_cells_with_calibration": 2176,
        "total_planned_tensors_with_calibration": 6528,
        "reference_source_cells_to_freeze_before_test": 256,
        "minimum_free_disk_gib": 8,
        "new_confirmation_sources": 0,
    }.items():
        if panel.get(key) != value:
            raise ValueError("planned panel denominator or release differs")
    if (
        analysis["primary_endpoints"]
        != ["zero_overlap_margin", "high_minus_zero_margin"]
        or analysis["primary_view_count"] != 6
        or analysis["primary_endpoint_count"] != 12
        or analysis["score_matrices_with_diagonal_controls"] != 168
        or analysis["bootstrap"]["draws"] != 10000
        or analysis["bootstrap"]["seed"] != 2026102930
        or analysis["bootstrap"]["quantiles"] != [0.025 / 12, 1 - 0.025 / 12]
        or analysis["p_values"] != "none_no_validated_overlap_randomization_null"
        or analysis["equivalence"] != "not_attempted_no_absence_claim"
    ):
        raise ValueError("registered endpoint or uncertainty contract differs")


def planned_panel_blocks(metadata):
    validate_pairing_holdout_metadata(metadata)
    return [
        (key, label)
        for mask in range(4)
        for key in [
            f"m{mask}_reference",
            *[f"m{mask}_q{q}_d{d}" for q in (0, 1, 4) for d in range(2)],
        ]
        for label, meta in sorted(metadata.items())
        if key.endswith("reference") or meta["replicate"] in (3, 4)
    ]


def require_complete_calibration(checks, labels):
    expected = {(label, c, t) for label in labels for c in CELLS for t in TARGETS}
    actual = [(r["pair_id"], r["cell"], r["target"]) for r in checks]
    if (
        len(labels) != 32
        or len(checks) != 384
        or len(set(actual)) != 384
        or set(actual) != expected
    ):
        raise ValueError("calibration coverage incomplete or duplicated")
    for row in checks:
        if (
            not all(row[k] for k in ("bitwise_equal", "shape_equal", "dtype_equal"))
            or row["max_abs_difference"] != 0
            or row["rms_difference"] != 0
            or row["before_array_sha256"] != row["after_array_sha256"]
        ):
            raise ValueError("calibration nonidentical; no panel release")


def validate_capture(run, audit, expected_run, expected_inputs, config):
    if run["model_id"] != config["model_id"] or run.get("dry_run"):
        raise ValueError("model identity or dry-run differs")
    if (
        len(run["stream_events"]) != 1
        or run["context_policy"] != expected_run["context_policy"]
        or run["runtime"] != expected_run["runtime"]
        or run["reproducibility"]["seed"] != config["source_seed"]
    ):
        raise ValueError("source protocol or runtime differs")
    event, old = run["stream_events"][0], expected_run["stream_events"][0]
    for key in ("prompt", "frame_sha256s", "cache_token_layout"):
        if event[key] != old[key]:
            raise ValueError(f"source {key} differs")
    if (
        event["assistant_text"] != config["expected_source_response"]
        or [s["token_id"] for s in event["generation"]["steps"]]
        != config["expected_generation_step_ids"]
        or audit["override_image_sizes"] is not None
        or audit["native"] != audit["effective"]
        or audit["effective"] != expected_inputs
        or audit["effective"]["image_sizes"]["values"] != config["expected_image_sizes"]
    ):
        raise ValueError("prepared inputs or generated suffix differs")
    artifacts = event["cache_tensor_artifacts"]
    if len(artifacts) != 3 or {
        f"{a['layer_index']}:{a['tensor']}" for a in artifacts
    } != set(TARGETS):
        raise ValueError("capture target coverage differs")
    for artifact in artifacts:
        if (
            artifact["source_dtype"] != config["expected_source_dtype"]
            or artifact["dtype"] != config["expected_saved_dtype"]
            or artifact["shape"] != config["expected_tensor_shape"]
            or artifact["effective_sequence_length"] != 397
        ):
            raise ValueError("capture dtype or shape differs")


def independent_exact_arrays(before, after):
    if (
        before.shape != (1, 2, 397, 64)
        or after.shape != before.shape
        or before.dtype != np.float32
        or after.dtype != np.float32
        or not np.isfinite(before).all()
        or not np.isfinite(after).all()
        or not np.array_equal(before.view(np.uint8), after.view(np.uint8))
    ):
        raise ValueError("independent tensor-byte check failed")
