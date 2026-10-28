import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def scope():
    return json.loads((ROOT / "configs/preprint_scope_v1.json").read_text())


def test_scope_is_explicitly_not_a_measurement_registration():
    saved = scope()
    assert saved["schema_version"] == 1
    assert saved["artifact_kind"] == "design_only"
    assert saved["status"] == "scope_recorded_measurement_registration_pending"
    assert saved["execution_authorized_by_this_file"] is False
    assert saved["execution_runner"] is None
    assert all(value == 0 for value in saved["new_observations"].values())


def test_candidate_fixed_intersections_have_exact_content_counts():
    design = scope()["input_design"]
    assert design["status"] == "candidate_construction_not_frozen"
    size = design["content_shape_hw"][0] * design["content_shape_hw"][1]
    fixed = Fraction(*design["target_actual_fixed_fraction"])
    selected = Fraction(*design["selected_fraction"])
    assert fixed + selected == 1
    counts = []
    for parts in design["candidate_shared_fixed_fractions"]:
        overlap = Fraction(*parts)
        assert max(0, 2 * fixed - 1) <= overlap <= fixed
        assert (overlap * size).denominator == 1
        counts.append(int(overlap * size))
    assert counts == [0, 4800, 19200]
    assert design["overlap_denominator"] == "content_pixels"
    assert design["fixed_overlap_zero_implies_total_coincidence_zero"] is False
    assert design["shared_moved_coincidences_must_be_audited"] is True


def test_incomplete_map_registration_is_not_filled_with_defaults():
    design = scope()["input_design"]
    for key in (
        "construction_algorithm",
        "shared_moved_coincidence_policy",
        "construction_retry_limit",
        "map_seed_manifest",
        "mask_replicates",
        "destination_replicates",
    ):
        assert design[key] is None
    assert design["reference_map_shared_across_overlap_levels"] is True
    assert design["same_mask_destination_replicates_required"] is True


def test_confirmation_is_not_claimed_sealed_or_fresh_by_policy_alone():
    confirmation = scope()["confirmation"]
    assert confirmation["status"] == "not_generated_not_sealed"
    for key in (
        "cohort_manifest",
        "cohort_manifest_sha256",
        "sample_size",
        "registration_commit",
        "release_receipt",
    ):
        assert confirmation[key] is None
    assert (
        confirmation["existing_r1_r4_role"] == "development_only_for_new_overlap_study"
    )
    assert confirmation["adapt_design_after_confirmation_scores"] is False
    assert confirmation["family_specific_references_imply_zero_shot_transfer"] is False


def test_estimand_count_and_existing_cache_targets():
    estimands = scope()["estimands"]
    assert (
        len(estimands["input_views"]) * len(estimands["primary_input_endpoints"]) == 4
    )
    assert estimands["primary_input_endpoint_count_if_candidate_design_adopted"] == 4
    assert estimands["normalization"] == "full_vector_before_partition"
    assert estimands["cache_targets"] == [[1, "keys"], [12, "keys"], [23, "values"]]
    assert estimands["cache_regions"] == ["image", "post_image"]
    assert estimands["input_cache_ratios_are_causal_gain"] is False


def test_inference_decisions_remain_open_and_dependence_explicit():
    inference = scope()["inference"]
    for key in (
        "interval_and_resampling_specification",
        "multiplicity_family",
        "equivalence_margin",
        "precision_and_compute_budget",
    ):
        assert inference[key] is None
    assert inference["dependence_units"] == [
        "pairing_family",
        "source_seed",
        "mask_seed",
        "destination_seed",
    ]
    assert inference["four_cells_are_independent_replicates"] is False
    assert inference["absence_from_nonsignificance"] is False
    assert inference["paired_difference_requires_paired_inference"] is True


def test_existing_gate_is_referenced_without_new_spectral_equivalence_claim():
    design = scope()["input_design"]
    gate = json.loads((ROOT / design["existing_gate_config"]).read_text())
    assert gate["centroid_relative_tolerance"] == 0.05
    assert gate["hf_absolute_tolerance"] == 0.02
    assert (
        design["within_condition_gate_implies_between_level_spectral_equality"] is False
    )


def test_note_ranges_cover_scope_without_claiming_execution():
    stages = scope()["stages"]
    assert [
        n for stage in stages for n in range(stage["notes"][0], stage["notes"][1] + 1)
    ] == list(range(52, 81))
    assert stages[0]["status"] == "scope_only"
    assert all(s["status"] == "not_started" for s in stages[1:])
