import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import (
    run_controlled_overlap_cache_calibration as runner,
)
from fractal_vlm_state_probe.controlled_overlap_cache import (
    CELLS,
    TARGETS,
    independent_exact_arrays,
    planned_panel_blocks,
    require_complete_calibration,
    validate_capture,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def config():
    return json.loads(
        (
            Path(__file__).parents[1]
            / "configs/controlled_overlap_cache_phase2_v1.json"
        ).read_text()
    )


@pytest.fixture
def metadata():
    return {
        f"f{f}_r{r}": {
            "pairing_family": f"f{f}",
            "replicate": r,
            "broad_class": "a" if f < 4 else "b",
        }
        for f in range(8)
        for r in range(1, 5)
    }


def test_registered_plan_and_capture_denominators(config, metadata):
    validate_config(config)
    blocks = planned_panel_blocks(metadata)
    assert len(blocks) == len(set(blocks)) == 512
    assert sum(k.endswith("reference") for k, _ in blocks) == 128
    assert sum(metadata[label]["replicate"] in (1, 2) for _, label in blocks) * 4 == 256
    assert len(blocks) * 4 == config["panel"]["source_cells"]
    assert len(blocks) * 4 * 3 == config["panel"]["tensor_sidecars"]
    assert all(
        k.endswith("reference") or metadata[label]["replicate"] in (3, 4)
        for k, label in blocks
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "expected_source_response", "ACK"),
        (None, "targets", ["1:keys"]),
        ("calibration", "cells", 4),
        ("calibration", "resume", True),
        ("calibration", "passing_releases_panel_automatically", True),
        ("panel", "source_cells", 3584),
        ("analysis", "primary_endpoint_count", 6),
        ("analysis", "equivalence", "absence_from_nonsignificance"),
    ],
)
def test_registration_corruption_rejected(config, section, key, value):
    changed = copy.deepcopy(config)
    (changed if section is None else changed[section])[key] = value
    with pytest.raises(ValueError):
        validate_config(changed)


@pytest.fixture
def checks(metadata):
    return [
        {
            "pair_id": label,
            "cell": c,
            "target": t,
            "bitwise_equal": True,
            "shape_equal": True,
            "dtype_equal": True,
            "max_abs_difference": 0,
            "rms_difference": 0,
            "before_array_sha256": "a",
            "after_array_sha256": "a",
        }
        for label in metadata
        for c in CELLS
        for t in TARGETS
    ]


def test_complete_calibration_requires_all_identities(checks, metadata):
    require_complete_calibration(checks, metadata)
    for changed in (checks[:-1], [*checks[:-1], checks[0]]):
        with pytest.raises(ValueError, match="coverage"):
            require_complete_calibration(changed, metadata)
    checks[0]["bitwise_equal"] = False
    with pytest.raises(ValueError, match="nonidentical"):
        require_complete_calibration(checks, metadata)


@pytest.mark.parametrize(
    "field,value",
    [("max_abs_difference", 1e-9), ("dtype_equal", False), ("after_array_sha256", "b")],
)
def test_claimed_exactness_cannot_override_numeric_or_hash_mismatch(
    checks, metadata, field, value
):
    checks[0][field] = value
    with pytest.raises(ValueError):
        require_complete_calibration(checks, metadata)


@pytest.mark.parametrize(
    "corruption", ["shape", "dtype", "value", "negative_zero", "nan"]
)
def test_independent_byte_check_rejects_nonidentity(corruption):
    before = np.zeros((1, 2, 397, 64), dtype=np.float32)
    after = before.copy()
    independent_exact_arrays(before, after)
    if corruption == "shape":
        after = after[:, :, :-1]
    elif corruption == "dtype":
        after = after.astype(np.float64)
    elif corruption == "value":
        after.flat[0] = 1e-10
    elif corruption == "negative_zero":
        after.flat[0] = -0.0
    else:
        after.flat[0] = np.nan
    with pytest.raises(ValueError):
        independent_exact_arrays(before, after)


@pytest.fixture
def capture(config):
    artifacts = [
        {
            "layer_index": int(t.split(":")[0]),
            "tensor": t.split(":")[1],
            "source_dtype": config["expected_source_dtype"],
            "dtype": "float32",
            "shape": [1, 2, 397, 64],
            "effective_sequence_length": 397,
        }
        for t in TARGETS
    ]
    run = {
        "model_id": config["model_id"],
        "dry_run": False,
        "runtime": {"mlx": "qualified"},
        "context_policy": {"source_cache_only": True},
        "reproducibility": {"seed": 20260604},
        "stream_events": [
            {
                "prompt": "registered",
                "frame_sha256s": ["image"],
                "cache_token_layout": {"fixed": True},
                "assistant_text": "The image",
                "generation": {"steps": [{"token_id": i} for i in [785, 2168, 2168]]},
                "cache_tensor_artifacts": artifacts,
            }
        ],
    }
    inputs = {
        "image_sizes": {"values": [[320, 320]]},
        "pixel_values": {"sha256": "pixels"},
    }
    audit = {
        "native": copy.deepcopy(inputs),
        "effective": copy.deepcopy(inputs),
        "override_image_sizes": None,
    }
    return run, audit, inputs


@pytest.mark.parametrize(
    "corruption",
    ["suffix", "ids", "layout", "inputs", "override", "targets", "dtype", "seed"],
)
def test_capture_contract_fails_closed(config, capture, corruption):
    run, audit, inputs = capture
    old = copy.deepcopy(run)
    validate_capture(run, audit, old, inputs, config)
    event = run["stream_events"][0]
    if corruption == "suffix":
        event["assistant_text"] = "ACK"
    elif corruption == "ids":
        event["generation"]["steps"].pop()
    elif corruption == "layout":
        event["cache_token_layout"]["fixed"] = False
    elif corruption == "inputs":
        audit["native"]["pixel_values"]["sha256"] = "different"
    elif corruption == "override":
        audit["override_image_sizes"] = [[240, 320]]
    elif corruption == "targets":
        event["cache_tensor_artifacts"].pop()
    elif corruption == "dtype":
        event["cache_tensor_artifacts"][0]["source_dtype"] = "mlx.core.float16"
    else:
        run["reproducibility"]["seed"] = 1
    with pytest.raises(ValueError):
        validate_capture(run, audit, old, inputs, config)


def test_no_model_preflight_and_failed_preflight_never_load_runtime(
    tmp_path, monkeypatch, config
):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    frozen = {"config": config, "model_snapshot": {"revision": "qualified"}}
    monkeypatch.setattr(runner, "qualify", lambda: frozen)
    monkeypatch.setattr(
        runner,
        "_freeze_model_snapshot",
        lambda model, path: write_json(path, frozen["model_snapshot"]),
    )

    def forbidden(*args):
        raise AssertionError("preflight must not load model")

    monkeypatch.setattr(runner, "_load_mlx_runtime", forbidden)
    preflight = tmp_path / "preflight"
    result = runner.run("preflight", preflight)
    assert result["new_model_forwards"] == 0 and not result["model_weights_loaded"]
    result["status"] = "FAILED"
    write_json(preflight / "preflight.json", result)
    with pytest.raises(ValueError, match="preflight"):
        runner.run("calibrate", tmp_path / "calibration", preflight)
    failure = json.loads((tmp_path / "calibration/execution_failure.json").read_text())
    assert failure["attempted_source_cells"] == 0 and not failure["panel_started"]
    with pytest.raises(FileExistsError):
        runner.run("preflight", preflight)


def test_offline_flags_are_required(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(ValueError, match="offline"):
        runner.require_offline()
