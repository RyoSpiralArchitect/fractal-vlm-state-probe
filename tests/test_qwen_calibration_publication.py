"""Metadata and analytic budget replay, never native calibration."""

from pathlib import Path

from fractal_vlm_state_probe import qwen_calibration_contract as core
from fractal_vlm_state_probe.stat_matched_source_storage import read, sha

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0102_qwen_calibration_registration"


def test_preparation_receipts_and_zero_call_scope():
    receipt, summary = (
        read(PUB / "preparation_receipt.json"),
        read(PUB / "summary.json"),
    )
    assert sha(PUB / "summary.json") == receipt["summary_sha256"]
    assert sha(PUB / "plan.json") == receipt["plan_sha256"] == summary["plan_sha256"]
    for path, digest in receipt["preparation_code_sha256"].items():
        assert sha(ROOT / path) == digest
    for path, digest in summary["output_sha256"].items():
        assert sha(PUB / path) == digest
    assert summary["model_weights_loaded"] is summary["processor_instantiated"] is False
    assert summary["native_runtime_qualified"] is False
    for key in (
        "processor_calls",
        "model_contexts",
        "new_source_images",
        "new_cohort_maps",
        "confirmation_scores",
    ):
        assert summary[key] == 0
    assert summary["existing_raw_files_hash_checked"] == 34
    assert summary["raw_arrays_decoded"] == 0


def test_budget_replays_and_cannot_be_read_as_joint_release():
    p, b = read(PUB / "plan.json"), read(PUB / "budget.json")
    expected = core.storage_bound(p)
    assert all(b[k] == v for k, v in expected.items())
    assert b["upper_payload_bytes_from_caps"] == 5139318144
    assert b["required_launch_free_bytes"] == 18 * 2**30
    assert p["storage"]["joint_cohort_required_launch_free_bytes"] is None
    assert p["protocol"]["decoder_calls_upper"] == 204
    assert (
        p["approved_for_native_execution"] is p["cohort_generation_authorized"] is False
    )


def test_all_existing_development_jobs_are_retained_without_new_sources():
    jobs = read(PUB / "jobs.json")["jobs"]
    assert len(jobs) == len({j["case_id"] for j in jobs}) == 34
    assert all("_r5/s0/" in j["case_id"] for j in jobs[:32])
    assert len({j["family"] for j in jobs[:32]}) == 8
    assert [j["fixture"] for j in jobs[-2:]] == ["zero", "constant"]
    assert all(
        j["raw"]["dtype"] == "uint8" and j["raw"]["shape"] == [320, 320, 3]
        for j in jobs
    )
