"""Portable Note 94 replay from original saved scores, without native inputs."""

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0094_margin_decomposition"
OLD = ROOT / "examples/research_notes/0093_stat_matched_source_results"


def read(name):
    return json.loads((PUB / name).read_bytes())


def rows(path):
    return [json.loads(s) for s in path.read_bytes().splitlines()]


def average(values):
    return None if any(v is None for v in values) else math.fsum(values) / len(values)


def assert_number(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected, abs=1e-12, rel=0)


def test_publication_hashes_frozen_inputs_code_and_denominators():
    s, p, r = (
        read("summary.json"),
        read("plan.json"),
        read("verification_receipts.json"),
    )
    assert s["status"] == "VERIFIED_POSTHOC_SCORE_ACCOUNTING"
    for name, digest in s["output_sha256"].items():
        assert hashlib.sha256((PUB / name).read_bytes()).hexdigest() == digest
    for name, digest in p["sources"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    for name, digest in r["provenance"]["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert s["implementation_commit"] == r["provenance"]["implementation_commit"]
    assert (
        s["plan_sha256"] == hashlib.sha256((PUB / "plan.json").read_bytes()).hexdigest()
    )
    assert len(rows(PUB / "decomposition.jsonl")) == s["profiles"] == 18
    assert s["pair_rows"] == 18 * 16 and s["family_rows"] == 18 * 8
    assert s["shift_summary_rows"] == 18 * 4


@pytest.mark.parametrize("index", range(18))
def test_each_decomposition_reconstructs_from_unchanged_source_scores(index):
    original = rows(OLD / "profiles.jsonl")[index]
    current = rows(PUB / "decomposition.jsonl")[index]
    spec = json.loads((ROOT / "configs/stat_matched_source_panel_v1.json").read_bytes())
    metadata = {p["pair_id"]: p for p in spec["sources"]["pairs"]}
    family_class = {p["pairing_family"]: p["broad_class"] for p in metadata.values()}
    assert (current["view"], current["relative_floor"]) == (
        original["view"],
        original["relative_floor"],
    )
    expected_pairs = []
    for i, key in enumerate(original["test_pair_ids"]):
        pair = metadata[key]
        own = original["families"].index(pair["pairing_family"])
        others = [
            j
            for j, f in enumerate(original["families"])
            if j != own and family_class[f] == pair["broad_class"]
        ]
        assert len(others) == 3
        values = [original["scores"][s][i][own] for s in range(4)]
        competitors = [
            average([original["scores"][s][i][j] for j in others]) for s in range(4)
        ]
        do = None if any(v is None for v in values) else average(values[1:]) - values[0]
        dc = (
            None
            if any(v is None for v in competitors)
            else average(competitors[1:]) - competitors[0]
        )
        d = None if do is None or dc is None else do - dc
        row = current["pairs"][i]
        assert row["pair_id"] == key
        for name, value in (
            ("delta_own", do),
            ("delta_other", dc),
            ("other_contribution", None if dc is None else -dc),
            ("margin_change", d),
        ):
            assert_number(row[name], value)
        expected_pairs.append(
            {"family": pair["pairing_family"], "own": do, "other": dc, "D": d}
        )
    for i, family in enumerate(original["families"]):
        group = [p for p in expected_pairs if p["family"] == family]
        assert len(group) == 2
        r = current["families"][i]
        assert r["family"] == family
        assert_number(r["delta_own"], average([p["own"] for p in group]))
        assert_number(r["delta_other"], average([p["other"] for p in group]))
        assert_number(r["margin_change"], original["family_deltas"][i])
    assert_number(current["overall"]["margin_change"], original["paired_margin_change"])
    assert (
        current["overall"]["source_margin_change"] == original["paired_margin_change"]
    )
    assert current["available_test_pairs"] == original["available_test_pairs"]
    assert current["available"] == original["available"]


def test_two_rational_replays_agree_and_boundary_is_posthoc_score_only():
    s, r = read("summary.json"), read("verification_receipts.json")
    assert r["first_summary"] == r["second_summary"]
    assert r["first_tree_sha256"] == r["second_tree_sha256"]
    assert r["first_summary"]["status"] == "VERIFIED_SCORE_DECOMPOSITION"
    assert s["source_run_summary_sha256"] == r["first_summary"]["run_summary_sha256"]
    checks = rows(PUB / "oracle_checks.jsonl")
    assert len(checks) == 18 and all(x["status"] == "PASS" for x in checks)
    assert (
        max(c["maximum_absolute_difference"] for c in checks)
        == s["maximum_absolute_oracle_difference"]
    )
    assert s["maximum_absolute_source_reconstruction_difference"] <= 1e-12
    assert s["maximum_absolute_closure_residual"] <= 1e-12
    assert s["posthoc"] and s["not_preregistered_before_note0093"]
    assert (
        s["source_generation"]
        == s["processor_calls"]
        == s["model_contexts"]
        == s["native_array_reads"]
        == 0
    )
    assert (
        not s["ratio_attribution"]
        and not s["reference_reselection"]
        and not s["new_reference_fits"]
    )
    assert (
        s["p_values"] is None
        and s["confidence_intervals"] is None
        and not s["equivalence_tested"]
    )


def test_signed_overall_terms_and_unavailable_profile_are_retained():
    s = read("summary.json")
    assert [p["view"] for p in s["primary_profiles"]] == ["encoder", "projector"]
    for p in s["primary_profiles"]:
        assert p["available_test_pairs"] == 16
        assert p["delta_own"] < 0 and p["delta_other"] < 0
        assert p["other_contribution"] > 0 and p["margin_change"] < 0
        assert_number(
            p["delta_own"] + p["other_contribution"], p["source_margin_change"]
        )
    assert s["unavailable_profiles"] == [
        {
            "view": "raw_rgb/global_moments",
            "relative_floor": None,
            "available_test_pairs": 0,
        }
    ]
