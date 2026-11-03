from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np

STAGES = ("off_before", "on", "off_after")


def validate_config(config):
    expected = {
        "schema_version": 1,
        "study_id": "representation_observation_calibration_v1",
        "scope": "historical_read_only_observation_calibration_only",
        "projector_class": "mlx_vlm.models.fastvlm.vision.CallableModuleList",
        "projector_components": ["Linear", "GELU", "Linear"],
        "source_dtype": "mlx.core.bfloat16",
        "saved_dtype": "float32",
        "stages": list(STAGES),
        "historical_source_cells": 128,
        "new_source_contexts": 384,
        "historical_tensor_comparisons": 1152,
        "on_off_tensor_comparisons": 768,
        "observed_source_cells": 128,
        "representation_arrays": 256,
        "minimum_free_disk_gib": 8,
        "retry_or_resume": False,
        "new_correspondence_scores": 0,
        "new_direct_probe_cells": 0,
        "experimental_panel_released": False,
        "confirmation_cohort": "not_generated_not_sealed",
        "future_panel_status": "not_implemented_not_released_requalify_in_same_process_before_any_panel",
        "sites": {
            "encoder": {
                "site": "mm_projector_positional_input_0_after_existing_tower_reshape",
                "shape": [1, 256, 3072],
            },
            "projector": {
                "site": "mm_projector_return_before_multimodal_insertion",
                "shape": [1, 256, 896],
            },
        },
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("registered observation contract differs")


@contextmanager
def observe_projector(projector):
    cls = type(projector)
    original = cls.__call__
    if getattr(original, "_representation_observer", False):
        raise ValueError("nested representation observation is not allowed")
    records = []

    def observed(instance, *args, **kwargs):
        result = original(instance, *args, **kwargs)
        if instance is projector:
            if len(args) != 1 or kwargs:
                raise ValueError("registered projector call signature differs")
            # Hold the existing lazy graph objects; materialize only after replay.
            records.append((args[0], result))
        return result

    observed._representation_observer = True
    cls.__call__ = observed
    try:
        yield records
        if len(records) != 1:
            raise ValueError("exactly one target projector call required")
    finally:
        cls.__call__ = original


def validate_arrays(arrays, config):
    if set(arrays) != {"encoder", "projector"}:
        raise ValueError("both complete representation sites required")
    for site, a in arrays.items():
        if (
            a.dtype != np.float32
            or list(a.shape) != config["sites"][site]["shape"]
            or not np.isfinite(a).all()
            or np.any(a.view(np.uint32) & np.uint32(65535))
        ):
            raise ValueError("representation shape/dtype/finiteness/bfloat16 differs")


def array_receipts(arrays, config):
    validate_arrays(arrays, config)
    return {
        site: {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "source_dtype": config["source_dtype"],
            "array_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "scalar_count": int(a.size),
            "nonzero": bool(np.any(a)),
        }
        for site, a in arrays.items()
    }


def require_complete(rows, jobs):
    expected = [(j["pair_id"], j["cell"], stage) for j in jobs for stage in STAGES]
    actual = [(r["pair_id"], r["cell"], r["stage"]) for r in rows]
    if len(jobs) != 128 or len(set(expected)) != 384 or actual != expected:
        raise ValueError("observation calibration identity/order/coverage differs")
    if [r["ordinal"] for r in rows] != list(range(1, 385)):
        raise ValueError("observation calibration ordinal differs")
    for row in rows:
        if not row["historical_cache_exact"] or not row["observer_restored"]:
            raise ValueError("historical identity or observer restoration failed")
        observed = row["stage"] == "on"
        if (
            row["observer_calls"] != int(observed)
            or bool(row.get("representation")) != observed
        ):
            raise ValueError("unexpected representation capture count")
