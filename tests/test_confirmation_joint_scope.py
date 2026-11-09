"""Two-model scope resolution only; never load model or processor weights."""

import math
from pathlib import Path
from statistics import NormalDist

import pytest

from fractal_vlm_state_probe.stat_matched_source_storage import read, sha

ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT / "examples/research_notes/0100_two_model_confirmation_scope/decision.json"
)


def test_two_models_remain_required_and_old_preparation_stays_pinned():
    p = read(DECISION)
    assert p["status"] == "TWO_MODEL_SCOPE_RESOLVED_QUALIFICATION_PENDING"
    assert p["models"] == [
        "mlx-community/FastVLM-0.5B-bf16",
        "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    ]
    assert p["Qwen_is_optional_extension"] is False
    old = p["parent_preparation"]
    assert sha(ROOT / old["plan"]) == old["plan_sha256"]
    assert sha(ROOT / old["allocated_metadata"]) == old["allocated_metadata_sha256"]
    assert read(ROOT / old["plan"])["model_scope_decision"].startswith(
        "pending_operator"
    )
    assert old["old_pending_scope_is_historical_not_current"] is True


def test_joint_primary_family_and_paired_resampling_not_two_independent_studies():
    p = read(DECISION)
    sites, q = p["primary_candidates"], p["candidate_inference"]
    assert len(sites) == q["primary_endpoint_count_if_both_Qwen_sites_qualify"] == 4
    assert len({(s["model"], s["site"]) for s in sites}) == 4
    assert sum(s["qualified_on_development_path"] for s in sites) == 2
    alpha = q["familywise_alpha"] / 4
    assert q["per_endpoint_two_sided_coverage"] == pytest.approx(1 - alpha)
    assert q["quantiles"] == pytest.approx([alpha / 2, 1 - alpha / 2])
    assert q["normal_planning_critical_value"] == pytest.approx(
        NormalDist().inv_cdf(1 - alpha / 2)
    )
    assert (
        q["whole_test_pair_shared_resamples_across_models_sites_cells_and_arrangements"]
        is True
    )
    assert q["old_two_endpoint_precision_report_validates_joint_design"] is False
    assert q["Qwen_development_precision_estimate"] is None
    assert q["after_results_one_model_downgrade_allowed"] is False
    assert q["finite_sample_coverage_guaranteed"] is False


def test_shared_metadata_does_not_double_independent_units_or_seal_content():
    p = read(DECISION)
    c = p["shared_cohort_candidate"]
    assert c["reference_pairs"] == 16 and c["test_pairs"] == 64
    assert c["source_images"] == 2 * (c["reference_pairs"] + c["test_pairs"]) == 160
    assert (
        c["two_model_experimental_contexts_if_candidate_adopted"]
        == 2 * c["cells_per_model"]
        == 2560
    )
    assert c["independent_test_units_are_not_doubled_by_model_count"] is True
    assert c["actual_content_generated_or_sealed"] is False
    assert c["final_sample_size_for_both_models_justified"] is False
    assert "both_models" in c["before_test_source_generation"]


def test_static_Qwen_inspection_does_not_promote_unknown_runtime_contracts():
    q = read(DECISION)["Qwen_static_inspection"]
    for k in ("observed_site_shapes", "observed_site_dtypes"):
        assert q[k] is None
    assert q["config_vision_width"] == 1280 and q["config_output_width"] == 2048
    assert q["config_spatial_merge_size"] == 2
    assert q["grid_patch_order_and_window_inverse_qualified"] is False
    assert q["weights_integrity_qualified"] is False
    assert q["FastVLM_NCHW_and_1024_shape_and_input_gates_transfer_assumed"] is False
    assert q["historical_source_cache_pilot_is_visual_site_qualification"] is False
    for digest in [
        q["config_sha256"],
        q["preprocessor_config_sha256"],
        *q["installed_sources_sha256"].values(),
    ]:
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")


def test_joint_budget_unknown_is_not_filled_by_doubling_FastVLM():
    resources = read(DECISION)["resources"]
    assert resources["FastVLM_prior_required_launch_free_bytes"] == 44 * 2**30
    assert resources["prior_44GiB_applies_to_both_models"] is False
    assert resources["automatic_doubling_of_prior_budget_is_qualification"] is False
    for key in (
        "Qwen_calibration_contexts_upper",
        "Qwen_total_contexts_upper",
        "Qwen_processor_only_calls_upper",
        "Qwen_decoder_calls_upper",
        "Qwen_storage_bytes_upper",
        "joint_contexts_upper",
        "joint_required_launch_free_bytes",
    ):
        assert resources[key] is None


def test_selection_reply_never_authorizes_generation_or_execution():
    p = read(DECISION)
    gate = p["generation_and_execution_hold"]
    assert all(
        gate[k] is False
        for k in (
            "cohort_source_generation_authorized",
            "cohort_map_generation_authorized",
            "processor_calls_authorized",
            "model_contexts_authorized",
        )
    )
    assert all(type(v) is int and v == 0 for v in p["new_observations"].values())
    assert p["candidate_inference"]["p_values"] is None
    assert p["candidate_inference"]["equivalence_tested"] is False
    assert math.isfinite(p["candidate_inference"]["normal_planning_critical_value"])
