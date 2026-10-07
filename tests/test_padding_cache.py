import json
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.padding_cache import (
    changed_input_keys,
    input_fingerprint,
    override_image_sizes,
    prepared_input_audit,
    require_calibration,
    tensor_difference,
)


def payload():
    return {
        "pixel_values": np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4),
        "image_sizes": [[4, 2]],
        "input_ids": np.array([[1, -200, 2]], dtype=np.int32),
        "attention_mask": np.ones((1, 3), dtype=np.int32),
    }


@pytest.mark.parametrize("as_array", [False, True])
def test_metadata_swap_preserves_all_other_arrays_and_original(as_array):
    original = payload()
    if as_array:
        original["image_sizes"] = np.array(original["image_sizes"], dtype=np.int32)
    before = input_fingerprint(original)
    result = override_image_sizes(original, [[4, 4]])
    assert input_fingerprint(original) == before
    assert result["pixel_values"] is original["pixel_values"]
    assert type(result["image_sizes"]) is type(original["image_sizes"])
    assert changed_input_keys(before, input_fingerprint(result)) == ["image_sizes"]
    with pytest.raises(ValueError, match="shape"):
        override_image_sizes(original, [[4, 4], [4, 4]])
    with pytest.raises(ValueError, match="change image_sizes only"):
        override_image_sizes(original, [[4, 2]])


def test_real_path_audit_records_native_and_effective_and_restores_hook(tmp_path):
    original = payload()
    module = SimpleNamespace(prepare_inputs=lambda *args, **kwargs: original)
    prepare = module.prepare_inputs
    path = tmp_path / "audit.json"
    with prepared_input_audit(
        module,
        audit_path=path,
        context={"cell": "mm"},
        expected_pixel_sha256=input_fingerprint(original)["pixel_values"]["sha256"],
        expected_sizes=[[4, 2]],
        override_sizes=[[4, 4]],
    ) as calls:
        result = module.prepare_inputs()
        assert result["image_sizes"] == [[4, 4]]
        assert len(calls) == 1
        saved = json.loads(path.read_text())
        assert saved["native"]["image_sizes"]["values"] == [[4, 2]]
        assert saved["effective"]["image_sizes"]["values"] == [[4, 4]]
    assert module.prepare_inputs is prepare


@pytest.mark.parametrize(
    "failure", ["pixels", "sizes", "zero_calls", "two_calls", "forward_error"]
)
def test_audit_fails_closed_and_restores_hook(tmp_path, failure):
    original = payload()
    module = SimpleNamespace(prepare_inputs=lambda: original)
    prepare = module.prepare_inputs
    digest = (
        "bad"
        if failure == "pixels"
        else input_fingerprint(original)["pixel_values"]["sha256"]
    )
    sizes = [[8, 8]] if failure == "sizes" else [[4, 2]]
    with pytest.raises((ValueError, RuntimeError)):
        with prepared_input_audit(
            module,
            audit_path=tmp_path / "audit.json",
            context={},
            expected_pixel_sha256=digest,
            expected_sizes=sizes,
        ):
            if failure != "zero_calls":
                module.prepare_inputs()
            if failure == "two_calls":
                module.prepare_inputs()
            if failure == "forward_error":
                raise RuntimeError("synthetic forward failure")
    assert module.prepare_inputs is prepare


def test_tensor_comparison_and_calibration_do_not_relax_equality():
    a = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4)
    equal = tensor_difference(a, a.copy())
    assert equal["bitwise_equal"] and equal["max_abs_difference"] == 0
    require_calibration([equal], expected_count=1)
    b = a.copy()
    b.flat[1] = np.nextafter(b.flat[1], np.float32(2))
    different = tensor_difference(a, b)
    assert not different["bitwise_equal"] and different["max_abs_difference"] > 0
    with pytest.raises(ValueError, match="stop before panel"):
        require_calibration([different], expected_count=1)
    with pytest.raises(ValueError, match="incomplete"):
        require_calibration([equal], expected_count=2)
    assert not tensor_difference(a, a.astype(np.float64))["bitwise_equal"]
    assert tensor_difference(a, a[:, :, :1])["max_abs_difference"] is None
    assert tensor_difference(np.zeros(3), np.zeros(3))["cosine"] is None


def test_nonfinite_or_non_numeric_inputs_are_not_silently_hashed():
    with pytest.raises(ValueError, match="nonfinite"):
        input_fingerprint({"pixels": np.array([np.nan])})
    with pytest.raises(ValueError, match="unsupported"):
        input_fingerprint({"unknown": "not a tensor"})
    with pytest.raises(ValueError, match="nonfinite"):
        tensor_difference(np.array([np.inf]), np.array([np.inf]))
