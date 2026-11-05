"""Public-only calibration receipt joins; complete score arrays are deliberately local."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "examples/research_notes/0080_first_step_readout_calibration"
HISTORICAL = ROOT / "examples/research_notes/0072_prompt_only_source_calibration"


def read(p):
    return json.loads(p.read_bytes())


def rows(p, sha):
    raw = p.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    return [json.loads(line) for line in raw.splitlines()]


def test_all_three_readout_routes_are_anchored_without_releasing_effect_panel():
    s = read(PUBLIC / "summary.json")
    cfg = read(ROOT / "configs/first_step_readout_calibration_v1.json")
    old = read(HISTORICAL / "summary.json")
    anchors = rows(
        HISTORICAL / "contexts.jsonl", old["ledgers"]["contexts.jsonl"]["sha256"]
    )
    arms = rows(PUBLIC / "arms.jsonl", s["ledgers"]["arms.jsonl"]["sha256"])
    comparisons = rows(
        PUBLIC / "comparisons.jsonl", s["ledgers"]["comparisons.jsonl"]["sha256"]
    )
    assert (
        s["status"] == "VERIFIED" and s["source_contexts"] == 288 and len(arms) == 288
    )
    assert s["rows"] == len(comparisons) == len(anchors) == 96
    assert (
        s["natural_source_contexts"] == 192 and s["independent_prefix_contexts"] == 96
    )
    assert (
        not s["experimental_readout_panel_released"]
        and s["interprompt_effects_computed"] == 0
    )
    assert s["confirmation_cohort"] == "not_generated_not_sealed"
    assert s["verification_model_calls"] == 0
    assert (
        hashlib.sha256(
            (ROOT / "configs/first_step_readout_calibration_v1.json").read_bytes()
        ).hexdigest()
        == s["config_sha256"]
    )
    for name, sha in s["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    for i, a in enumerate(anchors):
        group = arms[3 * i : 3 * i + 3]
        assert [r["arm"] for r in group] == cfg["arms"]
        for r in (*group, comparisons[i]):
            assert all(r[k] == a[k] for k in ("ordinal", "pair_id", "cell", "stage"))
        hashes = {r["vocabulary"]["scores_array_sha256"] for r in group}
        assert hashes == {comparisons[i]["score_array_sha256"]}
        assert (
            group[1]["vocabulary"]["logits_array_sha256"]
            == group[2]["vocabulary"]["logits_array_sha256"]
            == comparisons[i]["logits_array_sha256"]
        )
        for r in group:
            v = r["vocabulary"]
            assert v["source_dtype"] == cfg["native_dtype"]
            assert v["sidecar"]["vocab_size"] == 151936
            assert (
                v["sidecar"]["shape"] == [151936] and v["sidecar"]["dtype"] == "float32"
            )
            assert (
                v["sidecar"]["probability_sum"]
                == comparisons[i]["native_score_probability_sum"]
            )
            assert v["sidecar"]["logsumexp"] == comparisons[i]["native_logsumexp64"]
        assert (
            group[0]["source"]["terminal_sha256"]
            == group[1]["source"]["terminal_sha256"]
        )
        assert group[2]["prefix_word_sha256"] == comparisons[i]["prefix_word_sha256"]
    assert (
        len({(r["pair_id"], r["cell"], tuple(r["prepared_ids"])) for r in anchors})
        == 64
    )


def test_registered_diagnostics_aggregate_without_silent_native_renormalization():
    s = read(PUBLIC / "summary.json")
    c = rows(PUBLIC / "comparisons.jsonl", s["ledgers"]["comparisons.jsonl"]["sha256"])
    assert s["first_step_score_exact_pairs"] == s["raw_logit_exact_pairs"] == 96
    assert s["native_stream_vs_decoder_observer_score_pairs"] == 96
    assert s["first_step_score_coordinate_pairs"] == 96 * 151936
    assert (
        s["prefix_cache_exact_tensor_pairs"]
        == s["native_observer_terminal_exact_tensor_pairs"]
        == 96 * 48
    )
    for r in c:
        assert r["logits_exact"] and r["scores_exact"] and r["vocab_size"] == 151936
        assert (
            r["actual_prefix_decoder_calls"] == 1
            and r["actual_prefix_suffix_calls"] == 0
        )
        assert (
            r["prefix_effective_length"] == 395
            and r["prefix_cache_tensors_exact"] == 48
        )
        assert r["native_observer_terminal_tensors_exact"] == 48
        assert (
            r["native_dtype"] == "mlx.core.bfloat16" and r["saved_dtype"] == "float32"
        )
        assert abs(r["renormalized_probability_sum"] - 1) <= 1e-12
        assert r["natural_first_token_id"] in r["argmax_ids"]
        assert 0 <= r["native_vs_raw_logits_softmax_tv"] <= 1
    assert s["native_score_probability_sum_range"] == [
        min(r["native_score_probability_sum"] for r in c),
        max(r["native_score_probability_sum"] for r in c),
    ]
    assert s["renormalized_probability_sum_range"] == [
        min(r["renormalized_probability_sum"] for r in c),
        max(r["renormalized_probability_sum"] for r in c),
    ]
    assert s["native_vs_raw_logits_softmax_tv_range"] == [
        min(r["native_vs_raw_logits_softmax_tv"] for r in c),
        max(r["native_vs_raw_logits_softmax_tv"] for r in c),
    ]
    assert s["total_negative_infinity_scores"] == sum(
        r["negative_infinity_count"] for r in c
    )


def test_prefill_only_boundary_receipts_and_independent_file_check():
    s = read(PUBLIC / "summary.json")
    audit = read(PUBLIC / "independent_file_check.json")
    assert (
        audit["status"] == "PASS"
        and audit["model_calls"] == audit["processor_calls"] == 0
    )
    assert (
        audit["verified_summary_sha256"]
        == hashlib.sha256((PUBLIC / "summary.json").read_bytes()).hexdigest()
    )
    boundaries = rows(
        PUBLIC / "prefix_boundaries.jsonl", audit["prefix_boundaries_sha256"]
    )
    old = read(HISTORICAL / "summary.json")
    anchors = rows(
        HISTORICAL / "contexts.jsonl", old["ledgers"]["contexts.jsonl"]["sha256"]
    )
    checks = rows(
        PUBLIC / "comparisons.jsonl", s["ledgers"]["comparisons.jsonl"]["sha256"]
    )
    assert len(boundaries) == len(anchors) == 96
    for r, a, c in zip(boundaries, anchors, checks, strict=True):
        assert all(r[k] == a[k] for k in ("ordinal", "pair_id", "cell", "stage"))
        assert r["step"] == 0 and r["input_ids"] == a["prepared_ids"]
        assert r["input_dtype"] == "int64" and r["embedding_shape"] == [1, 395, 896]
        assert r["offset_before"] == 0 and r["offset"] == 395
        assert r["sha256"] == c["prefix_word_sha256"]
        assert set(r["tensor_sha256"]) == {
            f"{i}:{kv}" for i in range(24) for kv in ("keys", "values")
        }
    assert (
        audit["prefix_suffix_calls"] == 0 and audit["actual_prefix_decoder_calls"] == 96
    )
    assert audit["score_coordinate_pairs"] == 2 * 96 * 151936
    assert audit["raw_logit_coordinate_pairs"] == 96 * 151936
    assert audit["prefix_cache_tensor_pairs"] == 96 * 48
    assert audit["interprompt_effects_computed"] == 0
