import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import run_nonlearned_processing_calibration as runner
from fractal_vlm_state_probe.nonlearned_processing import (
    FEATURES,
    STARTS,
    box_mean,
    direction_status,
    extract_features,
    factorial_interaction,
    grid_pool,
)
from fractal_vlm_state_probe import nonlearned_processing_calibration as calibration
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def config():
    return json.loads(
        (
            Path(__file__).parents[1] / "configs/nonlearned_processing_phase3_v1.json"
        ).read_text()
    )


@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize(
    "name", ["zero", "constant", "ramp", "checkerboard", "impulse", "noise"]
)
def test_features_match_separate_convolution_and_pooling_oracle(config, start, name):
    values = calibration.fixture(name, 32, start, 812)
    actual = extract_features(values, start, config)
    expected = calibration.oracle_features(values, start, config)
    assert list(actual) == list(FEATURES)
    for key in FEATURES:
        assert actual[key].dtype == np.float64
        assert actual[key].shape == ((6,) if key == "global_moments" else (192,))
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-10)


def test_grid_pool_order_and_border_replication():
    grid = np.arange(8 * 8 * 3).reshape(8, 8, 3)
    enlarged = np.repeat(np.repeat(grid, 4, axis=0), 4, axis=1)
    np.testing.assert_array_equal(grid_pool(enlarged, [8, 8]), grid.ravel())
    impulse = np.zeros((8, 8, 3))
    impulse[0, 0] = 9
    mixed = box_mean(impulse, 3)
    np.testing.assert_array_equal(mixed[0, 0], 4)
    np.testing.assert_array_equal(mixed[0, 1], 2)
    np.testing.assert_array_equal(mixed[1, 1], 1)
    np.testing.assert_array_equal(box_mean(grid, 1), grid)


@pytest.mark.parametrize("width", [0, 2, -3, 3.5])
def test_even_or_invalid_windows_are_rejected(width):
    with pytest.raises(ValueError, match="positive odd"):
        box_mean(np.zeros((8, 8, 3)), width)


def test_invalid_shapes_dtypes_and_nonfinite_values_rejected(config):
    for values in (np.zeros((32, 32)), np.zeros((0, 0, 3)), np.full((8, 8, 3), np.nan)):
        with pytest.raises(ValueError):
            box_mean(values, 3)
    with pytest.raises(ValueError):
        grid_pool(np.zeros((31, 32, 3)), [8, 8])
    with pytest.raises(ValueError):
        extract_features(np.zeros((32, 32, 3)), "raw_rgb", config)
    with pytest.raises(ValueError):
        extract_features(
            np.zeros((1, 3, 32, 32), dtype=np.float64), "processor_pixel_values", config
        )
    with pytest.raises(ValueError):
        extract_features(np.zeros((32, 32, 3), dtype=np.uint8), "unknown", config)


def test_algebraic_raw_negative_control_is_exact_and_unavailable(config):
    rng = np.random.default_rng(21)
    a, b = [rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8) for _ in range(2)]
    cells = {
        "mm": a,
        "jj": b,
        "jm": rng.permutation(a.reshape(-1, 3)).reshape(a.shape),
        "mj": rng.permutation(b.reshape(-1, 3)).reshape(b.shape),
    }
    z = factorial_interaction(
        {
            k: extract_features(v, "raw_rgb", config)["global_moments"]
            for k, v in cells.items()
        }
    )
    np.testing.assert_array_equal(z, 0)
    assert direction_status(z) == {
        "available": False,
        "norm": 0.0,
        "reason": "exact_zero_direction_unavailable",
    }
    assert direction_status([1e-12])["available"]
    with pytest.raises(ValueError):
        direction_status([1e-200])
    with pytest.raises(ValueError):
        factorial_interaction({"mm": a})


def test_registered_full_size_calibration_counts_order_and_linear_checks(config):
    report, arrays = calibration.synthetic_calibration(config)
    assert len(report["feature_checks"]) == 84 and len(arrays) == 85
    assert len(report["linear_contrast_checks"]) == 6
    assert len(report["nonlinear_order_checks"]) == 4
    assert report["raw_multiset_negative_control"]["exact_zero"]
    assert not report["raw_multiset_negative_control"]["available"]
    assert report["maximum_feature_oracle_abs_error"] <= 1e-10
    assert all(
        r["max_abs_order_difference"] == pytest.approx(0.625)
        for r in report["nonlinear_order_checks"]
    )


def test_bad_oracle_stops_synthetic_calibration(config, monkeypatch):
    original = calibration.oracle_features

    def broken(*args):
        values = original(*args)
        values["grid_mean"] += 1
        return values

    monkeypatch.setattr(calibration, "oracle_features", broken)
    with pytest.raises(ValueError, match="oracle differs"):
        calibration.synthetic_calibration(config)


def test_preflight_does_not_instantiate_processor_and_failed_gate_retained(
    tmp_path, monkeypatch, config
):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    frozen = {"config": config, "code_sha256": {}, "source_sha256": {}}
    monkeypatch.setattr(runner, "specification", lambda: copy.deepcopy(frozen))

    def forbidden(*args):
        raise AssertionError("no processor before qualified preflight")

    monkeypatch.setattr(runner, "qualified_processor", forbidden)
    preflight = tmp_path / "preflight"
    result = runner.run("preflight", preflight)
    assert result["new_model_forwards"] == result["new_processor_calls"] == 0
    assert (
        not result["processor_instantiated"]
        and not result["experimental_panel_released"]
    )
    with pytest.raises(FileExistsError):
        runner.run("preflight", preflight)
    result["status"] = "FAILED"
    write_json(preflight / "preflight.json", result)
    root = tmp_path / "failed"
    with pytest.raises(ValueError, match="preflight"):
        runner.run("calibrate", root, preflight)
    failure = json.loads((root / "execution_failure.json").read_text())
    assert failure["attempted_processor_calls"] == 0
    assert failure["experimental_results_withheld"]
    assert not (root / "summary.json").exists()


def test_design_keeps_baseline_panel_and_confirmation_closed(config):
    assert (
        config["planned_panel"]["status_at_registration"]
        == "not_implemented_not_released"
    )
    assert not config["calibration"]["passing_releases_panel_automatically"]
    assert not config["calibration"]["retry_or_resume"]
    assert config["planned_panel"]["primary_endpoint_count"] == 2 * len(STARTS) * len(
        FEATURES
    )
    assert config["planned_panel"]["bootstrap"]["quantiles"] == [
        0.025 / 24,
        1 - 0.025 / 24,
    ]
    assert config["planned_panel"]["interaction_vectors"] == 512 * 12
    assert config["confirmation_cohort"] == "not_generated_not_sealed"
    assert config["new_model_forwards"] == 0
