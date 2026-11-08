"""Portable aggregate replay; no native arrays, processor or model required."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "examples/research_notes/0093_stat_matched_source_results"


def read(name):
    return json.loads((PUBLIC / name).read_bytes())


def profiles():
    return [
        json.loads(line)
        for line in (PUBLIC / "profiles.jsonl").read_bytes().splitlines()
    ]


def test_portable_outputs_match_qualified_hashes_and_counts():
    s = read("summary.json")
    assert s["status"] == "DESCRIPTIVE_PROFILES_VERIFIED"
    for name, digest in s["output_sha256"].items():
        assert hashlib.sha256((PUBLIC / name).read_bytes()).hexdigest() == digest
    assert len(profiles()) == s["profiles"] == 18
    assert len((PUBLIC / "spectra.jsonl").read_bytes().splitlines()) == 512
    assert s["counts"] == {
        "source_frames": 64,
        "processor_only_calls": 665,
        "model_source_contexts": 920,
        "decoder_calls": 2760,
        "historical_model_contexts": 384,
        "sham_model_contexts": 24,
        "experimental_model_contexts": 512,
        "candidate_cells": 512,
        "primary_test_pairs": 16,
        "known_pairing_families": 8,
    }
    assert s["reference_seal"]["last_context"] == 664
    assert s["reference_seal"]["first_test_context"] == 665
    assert s["reference_seal"]["counts"] == {
        k: {"attempted": n, "returned": n}
        for k, n in (("source", 32), ("processor", 409), ("model", 664))
    }
    assert s["reference_seal"]["artifact_bytes_independently_verified"]


@pytest.mark.parametrize("index", range(18))
def test_each_score_margin_matrix_and_paired_change_reconstructs(index):
    p = profiles()[index]
    spec = json.loads((ROOT / "configs/stat_matched_source_panel_v1.json").read_bytes())
    pairs = {r["pair_id"]: r for r in spec["sources"]["pairs"]}
    family_class = {r["pairing_family"]: r["broad_class"] for r in pairs.values()}
    scores = np.asarray(p["scores"], dtype=float)
    assert scores.shape == (4, 16, 8)
    margins = np.full((4, 16), np.nan)
    for i, pair_id in enumerate(p["test_pair_ids"]):
        pair = pairs[pair_id]
        own = p["families"].index(pair["pairing_family"])
        other = [
            j
            for j, f in enumerate(p["families"])
            if j != own and family_class[f] == pair["broad_class"]
        ]
        assert len(other) == 3
        for shift in range(4):
            if np.isfinite(scores[shift, i, [own, *other]]).all():
                margins[shift, i] = (
                    scores[shift, i, own]
                    - math.fsum(scores[shift, i, j] for j in other) / 3
                )
    assert np.allclose(
        margins,
        np.asarray(p["margins"], dtype=float),
        atol=1e-12,
        rtol=0,
        equal_nan=True,
    )
    changes = margins[1:].mean(axis=0) - margins[0]
    assert int(np.isfinite(changes).sum()) == p["available_test_pairs"]
    deltas = []
    for fi, family in enumerate(p["families"]):
        indices = [
            i
            for i, key in enumerate(p["test_pair_ids"])
            if pairs[key]["pairing_family"] == family
        ]
        assert len(indices) == 2
        deltas.append(float(changes[indices].mean()))
        for shift in range(4):
            assert np.allclose(
                scores[shift, indices].mean(axis=0),
                np.asarray(p["score_matrices"][shift][fi], dtype=float),
                atol=1e-12,
                rtol=0,
                equal_nan=True,
            )
    assert np.allclose(
        deltas,
        np.asarray(p["family_deltas"], dtype=float),
        atol=1e-12,
        rtol=0,
        equal_nan=True,
    )
    if np.isfinite(deltas).all():
        assert p["available"]
        assert p["paired_margin_change"] == pytest.approx(
            math.fsum(deltas) / 8, abs=1e-12, rel=0
        )
    else:
        assert not p["available"] and p["paired_margin_change"] is None
    assert p["p_values"] is None and p["confidence_intervals"] is None
    assert not p["equivalence_tested"]


def test_primary_effects_sensitivity_heterogeneity_and_missingness_are_retained():
    a, s = profiles(), read("summary.json")
    primary = [p for p in a if p["primary_learned_profile"]]
    assert [p["view"] for p in primary] == ["encoder", "projector"]
    for p, endpoint in zip(primary, s["primary_endpoints"], strict=True):
        assert p["available_test_pairs"] == 16 and p["available"]
        assert p["paired_margin_change"] == endpoint["paired_margin_change"]
        assert sum(x < 0 for x in p["family_deltas"]) == 6
        assert sum(x > 0 for x in p["family_deltas"]) == 2
        assert len(p["diagnostics"]) == 48
        variants = [q for q in a if q["view"] == p["view"]]
        assert [q["relative_floor"] for q in variants] == [1e-8, 1e-6, 1e-4]
        assert all(
            q["scores"] == p["scores"]
            and q["paired_margin_change"] == p["paired_margin_change"]
            for q in variants
        )
        assert math.fsum(p["margins"][0]) / 16 == pytest.approx(
            endpoint["mean_identity_margin"], abs=1e-14, rel=0
        )
        assert math.fsum(
            v for row in p["margins"][1:] for v in row
        ) / 48 == pytest.approx(endpoint["mean_nonidentity_margin"], abs=1e-14, rel=0)
        assert math.fsum(
            r["direction_cosine"] for r in p["diagnostics"]
        ) / 48 == pytest.approx(endpoint["mean_direction_cosine"], abs=1e-14, rel=0)
        assert (
            min(r["norm_ratio"] for r in p["diagnostics"]) == endpoint["norm_ratio_min"]
        )
        assert (
            max(r["norm_ratio"] for r in p["diagnostics"]) == endpoint["norm_ratio_max"]
        )
    baseline = [p for p in a if p["relative_floor"] is None]
    assert len(baseline) == 12 and sum(p["available"] for p in baseline) == 11
    missing = [p for p in baseline if not p["available"]]
    assert len(missing) == 1 and missing[0]["view"] == "raw_rgb/global_moments"
    assert (
        missing[0]["available_test_pairs"] == 0
        and missing[0]["paired_margin_change"] is None
    )
    assert (
        max(abs(p["paired_margin_change"]) for p in baseline if p["available"])
        == s["maximum_absolute_baseline_D"]
    )


def test_both_file_verifications_agree_without_promoting_native_reexecution():
    r, s = read("verification_receipts.json"), read("summary.json")
    assert r["first_tree_sha256"] == r["second_tree_sha256"] and r["byte_identical"]
    assert r["first_summary"] == r["second_summary"]
    v = r["first_summary"]
    assert v["status"] == "VERIFIED" and v["capture_sha256"] == s["capture_sha256"]
    assert v["proof"]["saved_binary_artifacts_checked"] == 24794
    assert v["new_model_contexts"] == v["new_processor_calls"] == 0
    assert s["verification"]["receipted_files"] == 27838
    assert r["preflight"]["status"] == "PASS_FOR_APPROVED_RELEASE"
    assert not r["preflight"]["model_weights_loaded"]
    assert (
        r["preflight"]["new_model_contexts"]
        == r["preflight"]["new_processor_calls"]
        == 0
    )
    assert not s["full_native_payloads_published"]
    assert s["confirmation_cohort"] == "not_generated_not_sealed"
    assert (
        s["p_values"] is None
        and s["confidence_intervals"] is None
        and not s["equivalence_tested"]
    )
