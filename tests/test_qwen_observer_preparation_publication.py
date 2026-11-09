"""Software/capacity receipt checks only, without native model calls."""

from pathlib import Path

from fractal_vlm_state_probe.stat_matched_source_storage import read, sha

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "examples/research_notes/0101_qwen_observer_preparation/summary.json"


def test_observer_and_parent_scope_hashes_match():
    p = read(SUMMARY)
    assert sha(ROOT / p["parent_scope"]) == p["parent_scope_sha256"]
    for path, digest in p["software_sha256"].items():
        assert sha(ROOT / path) == digest
    assert p["status"] == "SOFTWARE_PREPARED_NATIVE_CALIBRATION_NOT_RELEASED"


def test_new_capacity_does_not_satisfy_unknown_joint_threshold():
    p = read(SUMMARY)["capacity_observation"]
    assert p["free_bytes"] >= p["FastVLM_only_threshold_bytes"] == 44 * 2**30
    assert p["FastVLM_only_threshold_met"] is True
    assert p["joint_two_model_threshold_bytes"] is None
    assert p["joint_threshold_qualified"] is False
    assert p["runtime_launch_preflight"] is False
    assert p["cleanup_or_deletion_performed"] is False


def test_software_receipt_never_promotes_native_or_confirmation_gates():
    p = read(SUMMARY)
    for key in (
        "native_OFF_ON_OFF_parity_qualified",
        "Qwen_processor_input_gates_qualified",
        "Qwen_precision_justified",
        "joint_storage_budget_fixed",
        "complete_native_calibration_registration",
        "model_weights_loaded",
        "cohort_generation_released",
        "native_execution_released",
    ):
        assert p[key] is False
    assert p["native_observed_shapes_and_dtypes"] is None
    for key in (
        "new_source_images",
        "new_cohort_maps",
        "processor_calls",
        "model_contexts",
        "confirmation_scores",
    ):
        assert p[key] == 0
