"""Job, byte-bound and window-layout fixtures without native model execution."""

import copy
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe import qwen_calibration_contract as core
from fractal_vlm_state_probe.stat_matched_source_storage import read

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return read(ROOT / core.PLAN)


def metadata_fixture():
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    rows, shams = [], []
    receipt = {"shape": [320, 320, 3], "dtype": "uint8"}
    for p in spec["sources"]["pairs"]:
        if p["replicate"] == 5:
            for cell in ("mm", "mj", "jm", "jj"):
                rows.append(
                    {
                        "pair_id": p["pair_id"],
                        "shift": 0,
                        "cell": cell,
                        "split": "reference",
                        "identity": p["pair_id"] + "/s0/" + cell,
                        "raw": copy.deepcopy(receipt),
                    }
                )
    for name in ("zero", "constant"):
        shams.append(
            {
                "fixture": name,
                "shift": 0,
                "identity": name + "_s0",
                "raw": copy.deepcopy(receipt),
            }
        )
    return spec, rows, shams


def test_exact_development_case_order_and_no_confirmation_roles():
    spec, rows, shams = metadata_fixture()
    selected = core.jobs(spec, rows, shams)
    assert len(selected) == 34
    assert [r["case_id"] for r in selected[:32]] == [r["identity"] for r in rows]
    assert [r["fixture"] for r in selected[-2:]] == ["zero", "constant"]
    assert all("_r5/s0/" in r["case_id"] for r in selected[:32])


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "test_role", "shape", "missing_sham"]
)
def test_incomplete_or_changed_cases_rejected(damage):
    spec, rows, shams = metadata_fixture()
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif damage == "test_role":
        rows[0]["split"] = "test"
    elif damage == "shape":
        rows[0]["raw"]["shape"] = [240, 320, 3]
    else:
        shams.pop()
    with pytest.raises(ValueError):
        core.jobs(spec, rows, shams)


def test_call_and_storage_budgets_include_native_prefetch():
    p = plan()
    assert p["jobs"]["cases"] == 34
    assert p["jobs"]["model_contexts"] == 34 * 3 == 102
    assert p["protocol"]["maximum_decoder_calls_per_context"] == 2
    assert p["protocol"]["decoder_calls_upper"] == 102 * 2 == 204
    assert p["protocol"]["processor_only_calls"] == 34
    assert p["protocol"]["combined_processor_calls_upper"] == 102 + 34 == 136
    assert (
        p["limits"]["max_language_sequence"] == p["limits"]["max_prefix_sequence"] + 1
    )
    b = core.storage_bound(p)
    assert (
        b["upper_payload_bytes_from_caps"] < b["payload_allowance_bytes"] == 8 * 2**30
    )
    assert b["required_launch_free_bytes"] == 18 * 2**30
    assert b["qualified_runtime_or_joint_budget"] is False


def test_payload_and_disk_accounting_mismatch_fail_closed():
    p = plan()
    p["storage"]["payload_allowance_bytes"] = 1
    with pytest.raises(ValueError, match="payload budget"):
        core.storage_bound(p)
    p = plan()
    p["storage"]["required_launch_free_bytes"] -= 1
    with pytest.raises(ValueError, match="disk accounting"):
        core.storage_bound(p)


@pytest.mark.parametrize("h,w", [(8, 8), (10, 14), (22, 22), (32, 32)])
def test_window_oracle_matches_separate_padded_transpose_construction(h, w):
    p = plan()
    grid = np.array([[1, h, w]], dtype=np.int64)
    actual = core.window_order(grid, p)
    rows, cols, window = h // 2, w // 2, 4
    padded = np.pad(
        np.arange(rows * cols).reshape(1, rows, cols),
        ((0, 0), (0, window - rows % window), (0, window - cols % window)),
        constant_values=-100,
    )
    reference = (
        padded.reshape(
            1, padded.shape[1] // window, window, padded.shape[2] // window, window
        )
        .transpose(0, 1, 3, 2, 4)
        .ravel()
    )
    np.testing.assert_array_equal(actual, reference[reference != -100])
    np.testing.assert_array_equal(np.sort(actual), np.arange(rows * cols))


def layout_fixture():
    p = plan()
    grid = np.array([[1, 10, 14]], dtype=np.int64)
    n, m = 140, 35
    pixels = np.zeros((n, 1176), dtype=np.float32)
    pre = np.zeros((n, 1280), dtype=np.float32)
    auxiliary = np.arange(m * 2048, dtype=np.float32).reshape(m, 2048)
    final = auxiliary[np.argsort(core.window_order(grid, p))].copy()
    return p, pixels, grid, pre, auxiliary, final


def test_full_layout_and_bitwise_window_reverse():
    p, pixels, grid, pre, auxiliary, final = layout_fixture()
    assert core.validate_saved_layout(pixels, grid, pre, auxiliary, final, p) == {
        "patch_tokens": 140,
        "merged_tokens": 35,
        "window_words_exact": True,
    }


@pytest.mark.parametrize("damage", ["crop", "dtype", "nan", "window", "signed_zero"])
def test_lossy_or_reordered_observations_are_rejected(damage):
    p, pixels, grid, pre, auxiliary, final = layout_fixture()
    if damage == "crop":
        pre = pre[:-1]
    elif damage == "dtype":
        pixels = pixels.astype(np.float64)
    elif damage == "nan":
        final[0, 0] = np.nan
    elif damage == "window":
        final = auxiliary.copy()
    else:
        final[0, 0] = -0.0
    with pytest.raises(ValueError):
        core.validate_saved_layout(pixels, grid, pre, auxiliary, final, p)


@pytest.mark.parametrize(
    "grid", [[[2, 8, 8]], [[1, 7, 8]], [[1, 64, 64]], [[1, 0, 8]], [[1.0, 8, 8]]]
)
def test_unqualified_grids_rejected(grid):
    with pytest.raises(ValueError):
        core.window_order(grid, plan())


def test_preparation_does_not_supply_a_native_run_mode(tmp_path):
    p = plan()
    assert p["approved_for_native_execution"] is False
    assert p["cohort_generation_authorized"] is False
    assert p["passing_calibration_releases_cohort"] is False
    assert p["storage"]["joint_cohort_required_launch_free_bytes"] is None
    assert all(v == 0 for v in p["new_observations_at_registration"].values())
    with pytest.raises(ValueError, match="canonical"):
        core.prepare(ROOT, tmp_path)
