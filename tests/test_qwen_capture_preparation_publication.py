"""Receipt replay for software preparation, not native calibration evidence."""

from pathlib import Path

from fractal_vlm_state_probe.qwen_calibration_contract import PLAN
from fractal_vlm_state_probe.stat_matched_source_storage import read, sha

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0103_qwen_capture_verifier_preparation"


def test_software_receipt_hashes_and_immutable_calibration_plan():
    receipt = read(PUB / "summary.json")
    assert receipt["plan_sha256"] == sha(ROOT / PLAN)
    for path, digest in receipt["software_sha256"].items():
        assert sha(ROOT / path) == digest
    assert receipt["focused_software_tests_passed"] == 82


def test_software_receipt_preserves_unobserved_and_unreleased_status():
    receipt = read(PUB / "summary.json")
    assert receipt["status"] == "SOFTWARE_TESTED_NATIVE_NOT_RELEASED"
    for name in ("offline_native_preflight_performed", "model_weights_loaded", "processor_instantiated",
                 "native_runtime_qualified", "native_execution_released", "cohort_generation_authorized"):
        assert receipt[name] is False
    for name in ("native_processor_calls", "native_model_contexts", "new_source_images",
                 "new_cohort_maps", "confirmation_scores"):
        assert receipt[name] == 0
    assert receipt["joint_cohort_required_launch_free_bytes"] is None
