import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.analyze_controlled_overlap import pair_scores
from fractal_vlm_state_probe.cli.summarize_controlled_overlap import (
    close,
    oracle_bootstrap,
    oracle_score,
    verify_summary,
)
from fractal_vlm_state_probe.controlled_overlap import paired_bootstrap


@pytest.fixture
def scored(tmp_path):
    metadata = {
        f"{b}{f}_r{r}": {"pairing_family": f"{b}{f}", "broad_class": b, "replicate": r}
        for b in ("a", "b")
        for f in range(4)
        for r in range(1, 5)
    }
    generator = np.random.default_rng(73)
    paths = {}
    for label in metadata:
        paths[label] = tmp_path / f"{label}.npy"
        np.save(paths[label], generator.normal(size=(4, 4, 3)))
    categories = np.tile(np.arange(4), (4, 1))
    result = pair_scores(paths, paths, metadata, categories=categories)
    return metadata, paths, categories, result


def test_independent_scores_components_and_summary(scored):
    metadata, paths, categories, result = scored
    scores, parts = oracle_score(
        paths, paths, result["labels"], result["families"], metadata, categories
    )
    close(scores, result["scores"], "full scores")
    for key in parts:
        close(parts[key], result["component_scores"][key], key)
    close(sum(parts.values()), scores, "additive reconstruction")
    verify_summary(
        result["summary"], scores, result["labels"], result["families"], metadata
    )


@pytest.mark.parametrize(
    "field",
    ["margin", "own_family_cosine_contribution", "other_family_cosine_contribution"],
)
def test_corrupted_seed_contribution_rejected(scored, field):
    metadata, _, _, result = scored
    broken = copy.deepcopy(result["summary"])
    broken["test_seed_margins"][0][field] += 0.01
    with pytest.raises(ValueError, match="independent verification differs"):
        verify_summary(
            broken,
            np.asarray(result["scores"]),
            result["labels"],
            result["families"],
            metadata,
        )


@pytest.mark.parametrize(
    "field",
    [
        "test_seed_count",
        "retrieval_correct_count",
        "retrieval_tied_count",
        "positive_family_margin_count",
    ],
)
def test_corrupted_denominator_or_counts_rejected(scored, field):
    metadata, _, _, result = scored
    broken = copy.deepcopy(result["summary"])
    broken[field] += 1
    with pytest.raises(ValueError):
        verify_summary(
            broken,
            np.asarray(result["scores"]),
            result["labels"],
            result["families"],
            metadata,
        )


def test_independent_bootstrap_with_crossed_random_effects():
    config = json.loads(
        (
            Path(__file__).parents[1] / "configs/controlled_overlap_phase1_v1.json"
        ).read_text()
    )
    config["bootstrap"]["draws"] = 100
    margins = np.random.default_rng(82).normal(size=(2, 4, 2, 3, 8, 2))
    close(
        oracle_bootstrap(margins, config["bootstrap"]),
        paired_bootstrap(margins, config)["ranges"],
        "bootstrap",
    )
    margins[:, :, :, 2] = margins[:, :, :, 0]
    close(
        oracle_bootstrap(margins, config["bootstrap"])[:, 1],
        np.zeros((2, 2)),
        "paired zero difference",
    )


def test_independent_zero_and_nonfinite_vectors(scored):
    metadata, paths, categories, result = scored
    np.save(paths["a0_r1"], np.zeros((4, 4, 3)))
    assert (
        oracle_score(
            paths, paths, result["labels"], result["families"], metadata, categories
        )
        is None
    )
    np.save(paths["a0_r1"], np.full((4, 4, 3), np.nan))
    with pytest.raises(ValueError, match="nonfinite"):
        oracle_score(
            paths, paths, result["labels"], result["families"], metadata, categories
        )


@pytest.mark.parametrize(
    "actual,expected",
    [(1, [1]), ([np.nan], [np.nan]), ([np.inf], [np.inf]), ([1], [1.01])],
)
def test_independent_comparator_fails_closed(actual, expected):
    with pytest.raises(ValueError):
        close(actual, expected, "corrupted")


@pytest.fixture(scope="module")
def publication():
    root = (
        Path(__file__).parents[1]
        / "examples/research_notes/0056_controlled_overlap_phase1"
    )
    return root, json.loads((root / "summary.json").read_text())


def test_published_ledgers_and_figures_are_hash_locked(publication):
    root, summary = publication
    for filename, receipt in summary["ledgers"].items():
        payload = (root / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == receipt["sha256"]
        assert len(payload.splitlines()) == receipt["rows"]
    for filename, digest in summary["figures"].items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
    assert {k: v["rows"] for k, v in summary["ledgers"].items()} == {
        "cells.jsonl": 3584,
        "vectors.jsonl": 1024,
        "scores.jsonl": 56,
        "between_level_spectra.jsonl": 512,
    }


def test_published_registration_counts_and_claim_limits(publication):
    _, summary = publication
    frozen = summary["frozen_specification"]
    assert (
        frozen["registered_design_commit"] == "1e19014a54a999f14b091f9ce0467b4aa034afdf"
    )
    assert len(frozen["code_sha256"]) == 30
    assert summary["cache_panel_eligible"]
    assert len(summary["input_gate_counts"]) == 28
    assert all(
        v == {"accepted_blocks": 32, "total_blocks": 32}
        for v in summary["input_gate_counts"].values()
    )
    assert summary["new_model_forwards"] == summary["new_p_values"] == 0
    assert summary["confirmation_cohort"] == "not_generated_not_sealed"
    verification = summary["independent_verification"]
    assert verification["bootstrap_verified"]
    assert verification["verification_processor_calls"] == 3584
    assert summary["measurement_processor_calls"] == 3712
    assert verification["maximum_vector_oracle_abs_error"] < 1e-12
    assert verification["maximum_score_oracle_abs_error"] < 1e-12
    for row in summary["primary_endpoints"]:
        assert (
            row["available"]
            and row["p_value"] is None
            and not row["equivalence_tested"]
        )
    for row in summary["map_construction"]["maps"].values():
        counts, recipe = row["counts"], row["recipe"]
        assert counts["left_fixed"] == counts["right_fixed"] == 19200
        if recipe["q"] is not None:
            assert counts["shared_fixed"] == 76800 * recipe["q"] // 16
            assert counts["shared_moved"] == 0
            assert 1 <= len(recipe["attempt_shared_moved"]) <= 128


def test_published_scores_reconstruct_every_aggregate(publication):
    root, summary = publication
    rows = [
        json.loads(line) for line in (root / "scores.jsonl").read_text().splitlines()
    ]
    metadata = summary["frozen_specification"]["hierarchy_metadata"]
    for row in rows:
        assert row["available"]
        verify_summary(
            row["summary"],
            np.asarray(row["scores"]),
            row["labels"],
            row["families"],
            metadata,
        )
    for row in summary["aggregates"]:
        selected = [
            r
            for r in rows
            if r["view"] == row["view"] and r["q"] == row["q_sixteenths"]
        ]
        assert len(selected) == 8
        scores = np.mean([r["scores"] for r in selected], axis=0)
        verify_summary(
            row, scores, selected[0]["labels"], selected[0]["families"], metadata
        )
    for view in summary["frozen_specification"]["config"]["views"]:
        endpoints = {
            r["endpoint"]: r for r in summary["primary_endpoints"] if r["view"] == view
        }
        aggregates = {
            r["q_sixteenths"]: r for r in summary["aggregates"] if r["view"] == view
        }
        close(
            endpoints["zero_overlap_margin"]["mean"],
            aggregates[0]["mean_family_margin"],
            "zero mean",
        )
        close(
            endpoints["high_minus_zero_margin"]["mean"],
            aggregates[4]["mean_family_margin"] - aggregates[0]["mean_family_margin"],
            "paired difference",
        )
        for f in endpoints["high_minus_zero_margin"]["family_values"]:
            close(
                endpoints["high_minus_zero_margin"]["family_values"][f],
                aggregates[4]["family_margins"][f] - aggregates[0]["family_margins"][f],
                "family difference",
            )
