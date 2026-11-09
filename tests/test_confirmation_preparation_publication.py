"""Verify preparation artifacts without confirmation content or local captures."""

import hashlib
from pathlib import Path

import pytest

from fractal_vlm_state_probe import confirmation_preparation as core
from fractal_vlm_state_probe.stat_matched_source_storage import ledger, read, sha

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0099_confirmation_preparation"


def test_plan_summary_receipt_and_preparation_code_hashes():
    summary, receipt = (
        read(PUB / "summary.json"),
        read(PUB / "preparation_receipt.json"),
    )
    assert sha(PUB / "summary.json") == receipt["summary_sha256"]
    assert sha(PUB / "plan.json") == summary["plan_sha256"] == receipt["plan_sha256"]
    for path, digest in summary["output_sha256"].items():
        assert sha(PUB / path) == digest
    for path, digest in receipt["preparation_code_sha256"].items():
        assert sha(ROOT / path) == digest
    assert summary["status"] == "PREPARED_NOT_CALIBRATED_NOT_RELEASED"
    for key in (
        "source_images_generated",
        "coordinate_maps_generated",
        "processor_calls",
        "model_contexts",
        "confirmation_scores",
    ):
        assert summary[key] == receipt[key] == 0
    assert summary["cohort_actual_content_sealed"] is False
    assert summary["source_content_overlap_checked"] is False
    assert summary["map_candidate_processor_qualified"] is False
    assert summary["execution_authorized"] is False


def test_all_seed_metadata_reconstructs_without_stimulus_generation():
    plan = read(PUB / "plan.json")
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    saved = read(PUB / "allocated_source_metadata.json")
    assert core.allocated_pairs(spec, plan) == saved["pairs"]
    assert saved["source_images_generated"] == 0
    for p in saved["pairs"]:
        for role in ("source_a", "source_b"):
            key = f"{plan['candidate']['source_namespace']}|{p['pairing_family']}|{p['replicate']}|{role}"
            expected = 1 + int.from_bytes(
                hashlib.sha256(key.encode("ascii")).digest()[:8], "big"
            ) % (2**31 - 2)
            assert p[role]["seed"] == expected
    audit = read(PUB / "seed_audit.json")
    assert audit["historical_sources"] == 128
    assert audit["new_sources_allocated"] == 160
    assert audit["new_unique_initializers"] == 260
    assert audit["collisions"] == []
    assert audit["decoded_content_overlap_checked"] is False


def test_development_precision_report_replays_without_new_cohort():
    plan, saved = read(PUB / "plan.json"), read(PUB / "precision.json")
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    profiles = ledger(
        ROOT / "examples/research_notes/0098_reference_crossing_results",
        "profiles.jsonl",
    )
    actual = core.precision_plan(profiles, spec, plan["development_precision"])
    assert actual["centered_pair_residuals"] == saved["centered_pair_residuals"]
    assert len(actual["rows"]) == len(saved["rows"]) == 18
    for a, b in zip(actual["rows"], saved["rows"], strict=True):
        for key in a:
            assert a[key] == pytest.approx(b[key], abs=1e-12, rel=0)
    assert saved["new_map_or_reference_variance_estimated"] is False
    assert saved["exact_bootstrap_coverage_established"] is False
    assert saved["sample_size_selected_by_observed_effect_or_significance"] is False


def test_preparation_storage_and_inference_scope_are_not_released():
    plan, summary = read(PUB / "plan.json"), read(PUB / "summary.json")
    required = plan["budget_candidates"]["required_launch_free_bytes"]
    assert required == 44 * 2**30
    assert summary["future_disk_gate_met_at_preparation"] == (
        summary["free_bytes_observed"] >= required
    )
    assert summary["future_disk_gate_met_at_preparation"] is False
    assert (
        plan["model_scope_decision"] == "pending_operator_FastVLM_first_or_two_models"
    )
    assert (
        plan["candidate_inference"]["final_executable_interval_registration_required"]
        is True
    )
    assert plan["development_precision"]["new_map_variance_is_known"] is False
