import hashlib
import json
from pathlib import Path


def test_retained_failure_does_not_admit_any_forced_cache_comparison():
    base = Path(__file__).resolve().parents[1] / "examples/research_notes"
    root = base / "0074_native_suffix_replay_entry_failure"
    summary = json.loads((root / "summary.json").read_text())
    assert (
        hashlib.sha256((root / "arms.jsonl").read_bytes()).hexdigest()
        == summary["partial_arms_sha256"]
    )
    raw = (json.dumps(summary["failure"], indent=2, ensure_ascii=False) + "\n").encode()
    assert hashlib.sha256(raw).hexdigest() == summary["failure_sha256"]
    historical = json.loads(
        (base / "0072_prompt_only_source_calibration/contexts.jsonl")
        .read_text()
        .splitlines()[0]
    )
    arms = [json.loads(line) for line in (root / "arms.jsonl").read_text().splitlines()]
    assert [r["arm"] for r in arms] == ["native_plain", "native_traced"]
    for r in arms:
        assert all(
            r[k] == historical[k] for k in ("ordinal", "cell", "stage", "pair_id")
        )
        assert r["actual_suffix_ids"] == historical["suffix_ids"]
    checkpoints = summary["native_controls_one_row"]
    assert [c["offset"] for c in checkpoints] == [395, 396, 397]
    assert all(c["matched_tensors"] == 48 for c in checkpoints)
    assert (
        checkpoints[-1]["whole_sha256"]
        == arms[0]["terminal_sha256"]
        == arms[1]["terminal_sha256"]
    )
    assert summary["status"] == "NOT_QUALIFIED" and summary["parity_result"] is None
    assert (
        summary["forced_decoder_calls"]
        == summary["completed_native_forced_comparisons"]
        == 0
    )
    assert summary["failure"]["attempted"] == 3 and summary["failure"]["completed"] == 2
    assert summary["failure"]["matched_rows"] == 0 and not summary["retry"]
    assert summary["planned_anchor_rows"] == 96
    assert (
        "thinking_budget_criteria"
        not in summary["actual_outer_stream_step_kwargs_in_synthetic_fixture"]
    )
    assert summary["explicit_none_binds"] and summary["audit_model_calls"] == 0
