"""Replay Note 76 receipt joins without MLX, research images or local acquisition files."""

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "examples/research_notes"
PUB = BASE / "0076_native_suffix_replay_parity"


def read(path):
    return json.loads(path.read_bytes())


def ledger(path, sha):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    return [json.loads(line) for line in raw.splitlines()]


def test_all_v2_arms_named_and_distinct_from_retained_v1_failure():
    s = read(PUB / "summary.json")
    pre = read(BASE / "0072_prompt_only_source_calibration/summary.json")
    old = read(BASE / "0074_native_suffix_replay_entry_failure/summary.json")
    cfg = read(REPO / "configs/native_suffix_replay_calibration_v2.json")
    assert s["status"] == "VERIFIED" and old["status"] == "NOT_QUALIFIED"
    assert old["registered_commit"] != s["registered_commit"]
    assert (
        old["completed_native_forced_comparisons"] == 0 and old["parity_result"] is None
    )
    assert cfg["retained_v1_failure_sha256"] == old["failure_sha256"]
    assert (
        s["source_summary_sha256"]
        == cfg["original_summary_sha256"]
        == pre["verified_summary_sha256"]
    )
    for k, v in s["code_sha256"].items():
        assert hashlib.sha256((REPO / k).read_bytes()).hexdigest() == v
    source = ledger(
        BASE / "0072_prompt_only_source_calibration/contexts.jsonl",
        pre["ledgers"]["contexts.jsonl"]["sha256"],
    )
    arms = ledger(PUB / "arms.jsonl", s["ledgers"]["arms.jsonl"]["sha256"])
    assert len(arms) == s["source_contexts"] == 288
    seen = set()
    for i, anchor in enumerate(source):
        rows = arms[3 * i : 3 * i + 3]
        assert [r["arm"] for r in rows] == cfg["arms"]
        assert len({r["terminal_sha256"] for r in rows}) == 1
        for row in rows:
            assert all(
                row[k] == anchor[k] for k in ("ordinal", "stage", "pair_id", "cell")
            )
            assert row["actual_suffix_ids"] == anchor["suffix_ids"]
        seen.add((anchor["frame_sha256"], tuple(anchor["prepared_ids"])))
    assert len(seen) == s["unique_prompt_image_conditions"] == 64
    assert (
        s["native_plain"] == s["native_traced"] == s["forced_own"] == len(source) == 96
    )
    assert s["historical_selected_exact_pairs"] == len(arms) * 3
    assert (
        s["cross_prompt_suffixes"]
        == s["direct_readout_contexts"]
        == s["verification_model_calls"]
        == 0
    )


def test_every_published_checkpoint_reconciles_to_actual_decoder_input_and_all_layers():
    s = read(PUB / "summary.json")
    hist = BASE / "0072_prompt_only_source_calibration"
    anchors = ledger(
        hist / "contexts.jsonl",
        read(hist / "summary.json")["ledgers"]["contexts.jsonl"]["sha256"],
    )
    boundaries = ledger(
        PUB / "boundaries.jsonl", s["ledgers"]["boundaries.jsonl"]["sha256"]
    )
    arms = ledger(PUB / "arms.jsonl", s["ledgers"]["arms.jsonl"]["sha256"])
    words = read(PUB / "independent_word_check.json")
    names = {f"{i}:{kv}" for i in range(24) for kv in ("keys", "values")}
    tensors, scalars, suffixes = Counter(), Counter(), Counter()
    assert (
        len(boundaries) == s["matched_boundary_rows"] == 288
        and set(s["tensor_names"]) == names
    )
    for idx, a in enumerate(anchors):
        suffixes[",".join(map(str, a["suffix_ids"]))] += 1
        for step, n in enumerate((395, 396, 397)):
            r = boundaries[3 * idx + step]
            assert all(
                r[k] == a[k]
                for k in ("ordinal", "stage", "pair_id", "cell", "suffix_ids")
            )
            assert r["step"] == step and r["offset"] == n
            assert r["actual_input_ids"] == (
                a["prepared_ids"] if not step else a["suffix_ids"][step - 1 : step]
            )
            assert r["embedding_shape"] == (None if step else [1, 395, 896])
            assert r["input_dtype"] == (
                "uint32" if step else a["input_fingerprints"]["input_ids"]["dtype"]
            )
            assert (
                r["native_sha256"] == r["forced_sha256"]
                and set(r["tensor_sha256"]) == names
            )
            assert (
                r["bitwise_exact_tensors"] == 48
                and r["scalar_pairs"] == 48 * 2 * n * 64
            )
            if step == 2:
                assert r["native_sha256"] == arms[3 * idx]["terminal_sha256"]
            tensors[str(n)] += 48
            scalars[str(n)] += r["scalar_pairs"]
    assert dict(suffixes) == s["suffix_counts_including_ack_repeats"]
    assert sum(tensors.values()) == s["native_forced_exact_tensor_pairs"] == 13824
    assert sum(scalars.values()) == s["native_forced_exact_scalar_pairs"] == 700710912
    assert words["native_forced_tensor_pairs_by_offset"] == dict(tensors)
    assert words["native_forced_scalar_pairs_by_offset"] == dict(scalars)
    assert words["actual_model_calls"] == words["processor_calls"] == 0
    assert (
        words["verified_summary_sha256"]
        == hashlib.sha256((PUB / "summary.json").read_bytes()).hexdigest()
    )
