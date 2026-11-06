"""Weight-free replay of the complete published first-token panel."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.first_token_panel import (
    aggregate,
    check_metrics,
    oracle_expand_scores,
    paired_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "examples/research_notes/0084_first_token_prompt_panel"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledgers(summary):
    result = {}
    for name, meta in summary["ledgers"].items():
        assert sha(PUBLIC / name) == meta["sha256"]
        result[name] = [
            json.loads(x) for x in (PUBLIC / name).read_bytes().splitlines()
        ]
        assert len(result[name]) == meta["rows"]
    return result


def test_complete_panel_provenance_and_claim_boundary():
    s = read(PUBLIC / "summary.json")
    cfg = read(ROOT / "configs/first_token_prompt_panel_v2.json")
    assert s["status"] == "VERIFIED" and s["analysis_kind"] == cfg["study_id"]
    assert s["config_sha256"] == sha(ROOT / "configs/first_token_prompt_panel_v2.json")
    assert len(s["code_sha256"]) == 108
    for name, digest in s["code_sha256"].items():
        assert sha(ROOT / name) == digest
    assert s["fresh_contexts"] == 352 and s["same_model_instance"]
    assert s["calibration_contexts"] == 288 and s["panel_contexts"] == 64
    assert s["paired_images"] == 32 and s["panel_prefix_suffix_calls"] == 0
    assert s["calibration_first_step_score_exact_pairs"] == 96
    assert s["calibration_raw_logit_exact_pairs"] == 96
    assert s["calibration_prefix_cache_exact_tensors"] == 4608
    assert s["panel_prefix_sham_exact_tensors"] == 3072
    assert s["probability_definition"] == cfg["probability_definition"]
    assert s["verification_model_calls"] == 0
    assert s["p_values"] == s["confidence_intervals"] == s["equivalence_tests"] == 0
    assert s["confirmation_cohort"] == "not_generated_not_sealed"
    old = read(
        ROOT
        / "examples/research_notes/0082_first_token_prompt_preflight_stop/summary.json"
    )
    assert old["effect_estimate"] is None and old["new_contexts"] == 0
    assert all(s["code_sha256"][n] == h for n, h in old["code_sha256"].items())
    assert s["qualification_summary_sha256"] == cfg["qualification_summary_sha256"]
    tables = ledgers(s)
    replay = read(PUBLIC / "verification_replay.json")
    assert replay["status"] == "PASS" and replay["verification_model_calls"] == 0
    assert replay["registered_outputs"] == len(replay["byte_identical_files"]) == 6
    assert replay["byte_identical_files"] == {
        name: sha(PUBLIC / name)
        for name in ("summary.json", "native_scores.npz", *s["ledgers"])
    }
    assert {n: len(rows) for n, rows in tables.items()} == {
        "pairs.jsonl": 32,
        "panel_cells.jsonl": 64,
        "calibration_arms.jsonl": 288,
        "calibration_comparisons.jsonl": 96,
    }
    previous = ROOT / "examples/research_notes/0080_first_step_readout_calibration"
    qualification = read(previous / "summary.json")
    assert sha(previous / "summary.json") == s["qualification_summary_sha256"]
    assert (
        sha(previous / "comparisons.jsonl")
        == qualification["ledgers"]["comparisons.jsonl"]["sha256"]
    )
    assert tables["calibration_comparisons.jsonl"] == [
        json.loads(line)
        for line in (previous / "comparisons.jsonl").read_bytes().splitlines()
    ]
    for row in tables["panel_cells.jsonl"]:
        calibration = tables["calibration_arms.jsonl"][3 * (row["ordinal"] - 1) + 2]
        assert row["arm"] == calibration["arm"] == "prefix_only"
        assert row["prefix_word_sha256"] == calibration["prefix_word_sha256"]
        for field in ("scores_array_sha256", "logits_array_sha256"):
            assert row["vocabulary"][field] == calibration["vocabulary"][field]


def test_full_vocabulary_archive_reconstructs_every_pair_and_aggregate():
    s = read(PUBLIC / "summary.json")
    tables = ledgers(s)
    assert sha(PUBLIC / s["archive"]["file"]) == s["archive"]["sha256"]
    with np.load(PUBLIC / s["archive"]["file"], allow_pickle=False) as archive:
        assert set(archive.files) == {"native_score_words", "anchor_ordinals"}
        words = archive["native_score_words"]
        ordinals = archive["anchor_ordinals"]
        assert words.shape == (64, 151936) and words.dtype == np.dtype("<u2")
        assert ordinals.dtype == np.dtype("<i2")
        scores = oracle_expand_scores(words)
    assert ordinals.tolist() == [r["ordinal"] for r in tables["panel_cells.jsonl"]]
    for i, row in enumerate(tables["pairs.jsonl"]):
        a, b = tables["panel_cells.jsonl"][2 * i : 2 * i + 2]
        assert (a["pair_id"], a["cell"]) == (row["pair_id"], row["cell"])
        assert (b["pair_id"], b["cell"]) == (row["pair_id"], row["cell"])
        assert (a["stage"], b["stage"]) == ("ack_before", "yes")
        assert row["ack_anchor_ordinal"] == a["ordinal"]
        assert row["yes_anchor_ordinal"] == b["ordinal"]
        for j, label in ((2 * i, "ack"), (2 * i + 1, "yes")):
            assert (
                hashlib.sha256(scores[j].tobytes()).hexdigest()
                == row[f"{label}_score_array_sha256"]
            )
        metrics = paired_metrics(scores[2 * i], scores[2 * i + 1], 20)
        for key, value in metrics.items():
            if key == "top_delta_coordinates":
                for actual, expected in zip(value, row[key], strict=True):
                    assert actual["token_id"] == expected["token_id"]
                    for field in (
                        "ack_probability",
                        "yes_probability",
                        "yes_minus_ack",
                    ):
                        assert actual[field] == pytest.approx(
                            expected[field], rel=1e-12, abs=1e-14
                        )
            elif isinstance(value, float):
                assert value == pytest.approx(row[key], rel=1e-12, abs=1e-14)
            else:
                assert value == row[key]
        check_metrics(scores[2 * i], scores[2 * i + 1], row, 1e-12, 1e-14)
    assert aggregate(tables["pairs.jsonl"]) == s["result"]
