"""Design-contract checks only: no image/map generation or processor execution."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/stat_matched_input_calibration_v1.json"


def read(path):
    return json.loads(path.read_bytes())


def test_registration_is_not_an_execution_or_confirmation_release():
    s = read(CFG)
    assert s["registered_date"] == "2026-11-07"
    assert s["scope"] == "construction_and_synthetic_calibration_design_only"
    assert not s["execution_authorized_by_this_file"]
    assert s["execution_runner"] is None
    assert s["status_at_registration"] == "not_implemented_not_released"
    assert not s["passing_calibration_releases_panel"]
    assert set(s["new_observations_at_registration"].values()) == {0}
    assert s["confirmation"]["status"] == "not_generated_not_sealed"
    assert s["confirmation"]["cohort_manifest"] is None
    assert s["confirmation"]["sample_size"] is None
    c = s["future_calibration"]
    assert c["release_required"] and c["implementation_commit"] is None
    assert not c["model_weights_loaded"] and c["model_forwards"] == 0
    assert not c["retry_or_resume"] and not c["outcome_based_replacement"]
    assert not s["construction"]["construction_uses_image_scores"]
    assert not s["construction"]["search_or_rejection_sampling"]


def test_pinned_sources_and_all_108_existing_files_are_unchanged():
    s = read(CFG)
    for name, digest in s["pinned_artifacts"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    previous = read(ROOT / s["preserve_inherited_frozen_hashes_from"])
    assert previous["status"] == "VERIFIED"
    assert len(previous["code_sha256"]) == s["inherited_frozen_files"] == 108
    for name, digest in previous["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    historical = read(
        ROOT
        / "examples/research_notes/0063_nonlearned_processing_calibration/summary.json"
    )
    assert historical["status"] == "PASS"
    jobs = historical["frozen_specification"]["processor_jobs"]
    assert len(jobs) == 128 and "processor_non_pixel_payload" in jobs[0]
    c = s["future_calibration"]
    assert c["synthetic_non_pixel_expected"] == jobs[0]["processor_non_pixel_payload"]
    assert c["processor_call"]["text"] == ["<image>"]
    assert c["processor_call"]["return_tensors"] == "np"
    assert not c["processor_call"]["model_chat_template"]


def test_inherited_nonlinear_feature_definitions_and_numerical_tolerance():
    s = read(CFG)
    old = read(ROOT / s["inherited_feature_config"])
    f = s["feature_contract"]
    for key in ("starts", "features", "grid", "windows"):
        assert f[key] == old[key]
    assert (
        f["energy_order"]
        == "grid_pool(square(box_mean(image)))_not_box_mean(square(image))"
    )
    assert (
        f["absolute_tolerance"]
        == old["calibration"]["oracle_absolute_tolerance"]
        == 1e-10
    )
    assert f["relative_tolerance"] == 0
    assert not f["normalized_factorial_direction_equality_implied"]
    assert not s["claim_ceiling"]["preserves_all_low_level_statistics"]
    assert not s["claim_ceiling"]["destroys_all_structure"]
    assert not s["claim_ceiling"]["identifies_learning_or_semantics"]


def test_candidate_geometry_counts_and_conditional_support_reasoning():
    c = read(CFG)["construction"]
    assert c["raw_shape_hwc"] == [320, 320, 3]
    assert c["content_rows"] == [40, 280] and c["grid"] == [8, 8]
    assert c["cell_shape_hw"] == [40, 40]
    assert c["active_grid_rows"] == [1, 7] and c["active_grid_cols"] == [0, 8]
    assert [40 * i for i in c["active_grid_rows"]] == c["content_rows"]
    assert c["tile_offsets_yx"] == [[7, 7], [7, 27], [27, 7], [27, 27]]
    assert c["tile_shape_hw"] == [6, 6] and c["arrangement_shifts"] == [0, 1, 2, 3]
    assert c["maps"] == len(c["arrangement_shifts"]) == 4
    moved = 6 * 8 * 4 * 6 * 6
    assert moved == c["moved_sites_nonidentity"] == 6912
    assert 320 * 320 - moved == c["fixed_sites_nonidentity"]
    assert Fraction(moved, 240 * 320) == Fraction(*c["content_moved_fraction"])
    assert Fraction(moved, 320 * 320) == Fraction(*c["canvas_moved_fraction"])
    h = c["phase_hypothesis"]
    scale = Fraction(*h["resize_ratio"])
    radius = (
        Fraction(*h["candidate_resize_support_radius_raw"])
        + h["largest_processor_box_radius"] / scale
    )
    assert radius == Fraction(*h["combined_candidate_radius_raw"]) == Fraction(51, 8)
    assert radius < h["minimum_tile_guard_raw"] == 7
    assert 6 + 2 * radius < h["tile_translation_raw"] == 20
    assert h["tile_translation_raw"] * scale == h["tile_translation_processor"] == 64
    assert h["qualification_status"].startswith("conditional_support_reasoning_only")


def test_fixed_budget_and_negative_control_are_not_optional():
    s = read(CFG)
    f, c = s["fixtures"], s["future_calibration"]
    assert f["order"] == [
        "zero",
        "constant",
        "ramp",
        "checkerboard",
        "impulse",
        "noise",
    ]
    assert f["noise_seed"] == 2026110701
    assert f["matched_canvases"] == len(f["order"]) * f["arrays_per_fixture"] == 24
    assert (
        f["matched_canvases"] + f["sentinel"]["canvases"]
        == f["new_synthetic_canvases_planned"]
        == c["synthetic_processor_calls"]
        == 25
    )
    assert (
        c["historical_processor_checks"] + c["synthetic_processor_calls"]
        == c["total_processor_calls"]
        == 153
    )
    assert c["feature_oracle_vector_pairs"] == 25 * 2 * 6 == 300
    assert c["matched_pairs"] == 6 * 3 == 18
    assert c["matched_feature_vector_comparisons"] == 18 * 12 == 216
    assert c["joint_multiset_comparisons"] == 18 * 2 * (64 + 1) == 2340
    assert c["nonconstant_changed_array_checks"] == 4 * 3 * 2 == 24
    assert c["constant_or_zero_word_exact_shams"] == 2 * 3 * 2 == 12
    assert c["verification_processor_calls"] == 0
    assert (
        c["required_launch_free_bytes"]
        == c["reserve_bytes"]
        + c["payload_allowance_bytes"]
        + c["overhead_allowance_bytes"]
        == 9 * 2**30
    )
    assert c["reserve_bytes"] == 8 * 2**30
    assert f["sentinel"]["raw_rows"] == [40, 80] and f["sentinel"]["raw_cols"] == [
        0,
        40,
    ]
    assert f["sentinel"]["expected"] == "reject_match_not_a_failed_integrity_check"
