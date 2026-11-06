import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "examples/research_notes/0082_first_token_prompt_preflight_stop"


def test_storage_failure_cannot_be_counted_as_a_zero_effect_or_experiment():
    summary = json.loads((PUBLIC / "summary.json").read_bytes())
    cfg = json.loads((ROOT / "configs/first_token_prompt_panel_v1.json").read_bytes())
    snapshot = json.loads((PUBLIC / "storage_snapshot.json").read_bytes())
    assert summary["status"] == "NOT_RUN" and summary["effect_estimate"] is None
    assert (
        summary["new_contexts"]
        == summary["new_calibration_contexts"]
        == summary["new_panel_contexts"]
        == 0
    )
    assert (
        summary["completed_pairs"] == 0
        and summary["planned_pairs"] == 32
        and summary["planned_contexts"] == 352
    )
    assert (
        not summary["model_weights_loaded"]
        and not summary["portable_score_archive_created"]
    )
    assert summary["audit_model_calls"] == summary["audit_score_vectors_read"] == 0
    failure = summary["failure"]
    assert failure["mode"] == "preflight" and failure["status"] == "FAILED"
    assert (
        failure["attempted"]
        == failure["completed"]
        == failure["calibrated_rows"]
        == failure["panel_contexts"]
        == 0
    )
    assert (
        failure["message"] == "insufficient registered disk reserve"
        and not failure["retry"]
    )
    original = (json.dumps(failure, indent=2, ensure_ascii=False) + "\n").encode()
    assert hashlib.sha256(original).hexdigest() == summary["failure_sha256"]
    assert snapshot == summary["storage_snapshot"]
    assert (
        hashlib.sha256((PUBLIC / "storage_snapshot.json").read_bytes()).hexdigest()
        == summary["storage_snapshot_sha256"]
    )
    assert (
        snapshot["free_at_failure_bytes"] is None and snapshot["observed_after_failure"]
    )
    assert snapshot["required_bytes"] == summary["required_free_bytes"] == 12041764864
    assert snapshot["shortfall_bytes"] == max(
        0, snapshot["required_bytes"] - snapshot["free_bytes"]
    )
    assert (
        summary["required_free_bytes"]
        == cfg["reserve_bytes"] + cfg["launch_allowance_bytes"]
    )
    assert summary["budget_bytes"] == {k: cfg[k] for k in summary["budget_bytes"]}
    assert not (PUBLIC / "native_scores.npz").exists()


def test_all_preoutcome_files_stay_frozen_and_preceding_calibration_stays_qualified():
    summary = json.loads((PUBLIC / "summary.json").read_bytes())
    previous = json.loads(
        (
            ROOT
            / "examples/research_notes/0080_first_step_readout_calibration/summary.json"
        ).read_bytes()
    )
    assert (
        previous["status"] == "VERIFIED"
        and not previous["experimental_readout_panel_released"]
    )
    assert len(previous["code_sha256"]) == 97 and len(summary["code_sha256"]) == 103
    for name, digest in summary["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert all(
        summary["code_sha256"][name] == digest
        for name, digest in previous["code_sha256"].items()
    )
    assert summary["confirmation_cohort"] == "not_generated_not_sealed"
