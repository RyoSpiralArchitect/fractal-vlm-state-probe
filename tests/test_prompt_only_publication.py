"""Replay the paired prompt receipts without MLX, model weights or local runs."""

import gzip
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "examples/research_notes/0072_prompt_only_source_calibration"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_catalog_adds_only_publication_metadata_to_exact_verifier_output():
    catalog = json.loads((PUBLIC / "summary.json").read_text())
    raw = gzip.decompress((PUBLIC / "verified_summary.json.gz").read_bytes())
    assert catalog.pop("analysis_kind") == "prompt_only_source_calibration"
    assert catalog.pop("verified_summary_sha256") == hashlib.sha256(raw).hexdigest()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "afdc6d94330505f485aaade7f1a3321a9a5bb182085251415571858118d9ed23"
    )
    assert catalog == json.loads(raw)


def test_published_contexts_match_registered_note70_cells_and_only_one_token_changes():
    cfg = json.loads(
        (ROOT / "configs/prompt_only_source_calibration_v1.json").read_text()
    )
    s = json.loads((PUBLIC / "summary.json").read_text())
    assert s["status"] == "VERIFIED" and s["source_contexts"] == 96
    assert s["direct_readout_contexts"] == s["teacher_forced_contexts"] == 0
    assert s["correspondence_scores"] == s["verification_model_calls"] == 0
    assert s["confirmation_cohort"] == "not_generated_not_sealed"
    assert sha(ROOT / cfg["publication"]) == s["source_publication_sha256"]
    source = (ROOT / cfg["publication"]).with_name("cells.jsonl")
    assert (
        sha(source)
        == json.loads((ROOT / cfg["publication"]).read_text())["ledgers"][
            "cells.jsonl"
        ]["sha256"]
    )
    prior = {
        (r["pair_id"], r["cell"]): r
        for r in rows(source)
        if r["phase"] == "reference"
        and r["map"] == "m0_reference"
        and r["pair_id"] in cfg["pair_ids"]
    }
    for name, digest in s["code_sha256"].items():
        assert sha(ROOT / name) == digest
    for name, ledger in s["ledgers"].items():
        assert sha(PUBLIC / name) == ledger["sha256"]
        assert len(rows(PUBLIC / name)) == ledger["rows"]
    contexts = rows(PUBLIC / "contexts.jsonl")
    assert [r["ordinal"] for r in contexts] == list(range(1, 97))
    received = [(r["pair_id"], r["cell"], r["stage"]) for r in contexts]
    assert received == list(
        itertools.product(cfg["pair_ids"], cfg["cells"], cfg["stages"])
    )
    assert len({r["frame_sha256"] for r in contexts}) == len(prior) == 32
    paired = rows(PUBLIC / "pairs.jsonl")
    assert len(paired) == 32
    triples = zip(contexts[::3], contexts[1::3], contexts[2::3], strict=True)
    for (a, b, again), pair in zip(triples, paired, strict=True):
        key = (a["pair_id"], a["cell"])
        assert key == (pair["pair_id"], pair["cell"])
        old = prior[key]
        assert (
            a["cache_array_sha256"]
            == again["cache_array_sha256"]
            == old["cache_array_sha256"]
        )
        assert a["cache_state_ids"] == again["cache_state_ids"]
        assert a["input_fingerprints"] == again["input_fingerprints"]
        assert (
            a["prepared_ids"] == again["prepared_ids"] and len(a["prepared_ids"]) == 140
        )
        assert [
            (n, x, y)
            for n, (x, y) in enumerate(
                zip(a["prepared_ids"], b["prepared_ids"], strict=True)
            )
            if x != y
        ] == [(132, 53763, 14080)]
        assert {
            k
            for k in a["input_fingerprints"]
            if a["input_fingerprints"][k] != b["input_fingerprints"][k]
        } == {"input_ids"}
        for r in (a, b, again):
            assert r["site_array_sha256"] == {
                s: v["array_sha256"] for s, v in old["representation"]["arrays"].items()
            }
            assert (
                r["input_fingerprints"]["pixel_values"]["sha256"]
                == old["pixel_values_sha256"]
            )
            assert r["image_sizes"] == [[320, 320]]
            assert r["cache_state_ids"] == r["prepared_ids"] + r["suffix_ids"]
            assert r["spans"]["suffix"] == [395, 395 + len(r["suffix_ids"])]
            assert r["stream_notification_ids"] in (
                r["suffix_ids"],
                r["suffix_ids"] + r["suffix_ids"][-1:],
            )
        assert (pair["ack_suffix_ids"], pair["edited_suffix_ids"]) == (
            a["suffix_ids"],
            b["suffix_ids"],
        )
        assert pair["suffix_comparable"] == (a["suffix_ids"] == b["suffix_ids"])
        assert (pair["ack_text"], pair["edited_text"]) == (
            a["assistant_text"],
            b["assistant_text"],
        )
    assert dict(Counter(p["ack_text"] for p in paired)) == s["ack_response_counts"]
    assert (
        dict(Counter(p["edited_text"] for p in paired)) == s["edited_response_counts"]
    )
    assert sum(p["suffix_comparable"] for p in paired) == s["aligned_suffix_pairs"]


def test_all_published_region_counts_positions_and_medians_reconcile():
    cfg = json.loads(
        (ROOT / "configs/prompt_only_source_calibration_v1.json").read_text()
    )
    s = json.loads((PUBLIC / "summary.json").read_text())
    regions, pairs = rows(PUBLIC / "regions.jsonl"), rows(PUBLIC / "pairs.jsonl")
    by = {(r["pair_id"], r["cell"], r["target"], r["role"]): r for r in regions}
    spans = {
        "pre_image": [0, 42],
        "image": [42, 298],
        "post_image_before_edit": [298, 387],
        "edit": [387, 388],
        "prompt_tail": [388, 395],
    }
    assert len(by) == len(regions) == 576
    for target in cfg["targets"]:
        firsts = []
        for p in pairs:
            changed = []
            for role in (*spans, "suffix"):
                r = by[(p["pair_id"], p["cell"], target, role)]
                assert r["before_span"] == spans.get(
                    role, [395, 395 + len(p["ack_suffix_ids"])]
                )
                assert r["edited_span"] == spans.get(
                    role, [395, 395 + len(p["edited_suffix_ids"])]
                )
                if role == "suffix" and not p["suffix_comparable"]:
                    assert (
                        not r["available"]
                        and r["reason"] == "realized_suffix_ids_differ"
                    )
                    assert all(
                        r[k] is None
                        for k in (
                            "rms",
                            "max_abs",
                            "changed_scalars",
                            "changed_positions",
                        )
                    )
                    continue
                assert r["available"] and r["reason"] is None
                start, end = r["before_span"]
                positions = r["changed_positions"]
                assert positions == sorted(set(positions)) and all(
                    start <= i < end for i in positions
                )
                assert len(positions) <= r["changed_scalars"] <= 128 * len(positions)
                assert 0 <= r["rms"] <= r["max_abs"]
                changed += positions
            first = min(changed, default=None)
            assert p["first_changed_comparable_position"][target] == first
            firsts.append(str(first))
        assert (
            dict(Counter(firsts))
            == s["cache_regions"][target]["first_changed_position_counts"]
        )
        for role, stats in s["cache_regions"][target]["regions"].items():
            group = [r for r in regions if r["target"] == target and r["role"] == role]
            eligible = [r for r in group if r["available"]]
            assert stats == {
                "planned": 32,
                "available": len(eligible),
                "unavailable": 32 - len(eligible),
                "bitwise_exact": sum(r["changed_scalars"] == 0 for r in eligible),
                "median_cell_rms": median(r["rms"] for r in eligible)
                if eligible
                else None,
                "maximum_abs": max((r["max_abs"] for r in eligible), default=None),
            }
