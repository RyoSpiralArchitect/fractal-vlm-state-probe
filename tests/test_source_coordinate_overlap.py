import copy
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_direction_holdout import evaluate_pairing_holdout
from fractal_vlm_state_probe.graded_cache import endpoint_reference_vectors
from fractal_vlm_state_probe.map_seed_cache import aggregate_map_scores
from fractal_vlm_state_probe.permutation_maps import MAPS, NEW_MAPS
from fractal_vlm_state_probe.source_coordinate_overlap import (
    COMPONENTS,
    analyze,
    decompose_pair,
    linear_summary,
    partition_maps,
    partition_receipt,
    signed_ratio,
    validate_config,
)


def metadata():
    return {
        f"{b}{f}_r{r}": {"broad_class": b, "pairing_family": f"{b}{f}", "replicate": r}
        for b in ("a", "b")
        for f in range(4)
        for r in range(1, 5)
    }


def test_registered_config():
    config = json.loads(
        (
            Path(__file__).parents[1] / "configs/source_coordinate_overlap_v1.json"
        ).read_text()
    )
    validate_config(config)
    config["normalization"] = "masked_norms"
    with pytest.raises(ValueError, match="registration"):
        validate_config(config)


def test_partition_distinguishes_fixed_moved_noncoincident_and_padding():
    left = np.array([0, 2, 1, 3, 5, 4])
    right = np.array([0, 2, 3, 1, 4, 5])
    labels = partition_maps(left, right, shape=(2, 3), padding_rows=1)
    np.testing.assert_array_equal(labels[1:3].ravel(), [0, 1, 2, 2, 2, 2])
    receipt = partition_receipt(labels)
    assert receipt["pixel_counts"] == dict(zip(COMPONENTS, (1, 1, 4, 6)))
    assert receipt["coincident_content_fraction"] == 2 / 6
    assert receipt["coincident_content_fraction"] != 8 / 12
    assert sum(receipt["pixel_counts"].values()) == labels.size


def test_diagonal_has_no_noncoincident_content():
    mapping = np.array([0, 2, 1, 3, 5, 4])
    receipt = partition_receipt(
        partition_maps(mapping, mapping, shape=(2, 3), padding_rows=1)
    )
    assert receipt["pixel_counts"] == dict(zip(COMPONENTS, (2, 4, 0, 6)))


@pytest.mark.parametrize(
    "broken",
    [
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([0, 0, 2, 3, 4, 5]),
        np.arange(5),
    ],
)
def test_nonbijective_or_noninteger_maps_rejected(broken):
    with pytest.raises(ValueError, match="integer bijection"):
        partition_maps(broken, np.arange(6), shape=(2, 3), padding_rows=1)


def test_decomposition_keeps_full_norms_not_masked_cosines():
    meta = metadata()
    values = {k: np.ones((4, 3, 3)) for k in meta}
    labels = partition_maps(
        np.arange(6), np.array([0, 2, 1, 3, 5, 4]), shape=(2, 3), padding_rows=1
    )
    result = decompose_pair(values, values, meta, labels)
    assert result["available"]
    counts = partition_receipt(labels)["pixel_counts"]
    for row in result["dots"]:
        assert row["full_cosine"] == pytest.approx(1)
        assert sum(row["components"].values()) == pytest.approx(1)
        for c in COMPONENTS:
            assert row["components"][c] == pytest.approx(counts[c] / labels.size)
    assert (
        result["components"]["shared_moved_content"]["mean_margin_fraction_of_full"]
        is None
    )
    assert "retrieval_correct_count" not in result["components"]["shared_fixed_content"]


def test_zero_and_cancelling_full_vectors_stay_unavailable():
    meta = metadata()
    values = {k: np.ones((4, 3, 3)) for k in meta}
    labels = partition_maps(np.arange(6), np.arange(6), shape=(2, 3), padding_rows=1)
    values["a0_r1"] = np.zeros((4, 3, 3))
    assert not decompose_pair(values, values, meta, labels)["available"]
    values["a0_r1"] = -np.ones((4, 3, 3))
    result = decompose_pair(values, values, meta, labels)
    assert not result["available"]
    assert "cancelling" in result["reason"]


@pytest.mark.parametrize(
    "value,total,expected",
    [(1.1, 1, 1.1), (-0.1, 1, -0.1), (1, 0, None), (1, 1e-13, None)],
)
def test_signed_ratios_are_not_clamped(value, total, expected):
    assert signed_ratio(value, total) == expected


def test_nonfinite_or_missing_scores_rejected():
    meta = metadata()
    families = sorted({m["pairing_family"] for m in meta.values()})
    scores = {
        k: {f: 1.0 for f in families}
        for k, m in meta.items()
        if m["replicate"] in (3, 4)
    }
    scores["a0_r3"]["a0"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        linear_summary(scores, meta)
    scores.pop("a0_r3")
    with pytest.raises(ValueError, match="coverage"):
        linear_summary(scores, meta)


@pytest.fixture(scope="module")
def synthetic_panel():
    meta, rng = metadata(), np.random.default_rng(9)
    maps = {k: rng.permutation(6) for k in MAPS}
    originals = {k: rng.normal(size=(4, 3, 3)) for k in meta}
    vectors = {k: {} for k in MAPS}
    for key in MAPS:
        for label, original in originals.items():
            value = original.copy()
            value[1:3] = original[1:3].reshape(6, 3)[maps[key]].reshape(2, 3, 3)
            vectors[key][label] = value
    rows = []
    for left, right in product(MAPS, repeat=2):
        result = evaluate_pairing_holdout(
            endpoint_reference_vectors(vectors[left], vectors[right], meta), meta
        )
        result.pop("exact_block_test")
        rows.append(
            {"reference_map": left, "test_map": right, "view": "raw_rgb", **result}
        )
    primary = []
    for regime in ("same_map", "cross_map"):
        selected = [
            r
            for r in rows
            if r["reference_map"] in NEW_MAPS
            and r["test_map"] in NEW_MAPS
            and (r["reference_map"] == r["test_map"]) == (regime == "same_map")
        ]
        primary.append(
            {
                "view": "raw_rgb",
                "regime": regime,
                **aggregate_map_scores(selected, meta, regime),
            }
        )
    baseline = {"map_matrix": rows, "primary_tests": primary}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "fractal_vlm_state_probe.source_coordinate_overlap.partition_maps",
            lambda a, b: partition_maps(a, b, shape=(2, 3), padding_rows=1),
        )
        result = analyze(vectors, maps, meta, baseline)
    return vectors, maps, meta, baseline, result


def test_complete_panel_and_aggregates_reconstruct_baseline(synthetic_panel):
    _, _, _, _, result = synthetic_panel
    assert len(result["matrix"]) == 16
    assert len(result["dot_products"]) == result["available_dot_products"] == 2048
    assert result["new_statistical_tests"] == 0
    for row, pair_count in zip(result["aggregates"], (3, 6)):
        assert len(row["ordered_map_pairs"]) == pair_count
        assert row["full"]["test_seed_count"] == 16
        assert sum(
            c["mean_family_margin"] for c in row["components"].values()
        ) == pytest.approx(row["full"]["mean_family_margin"])
        assert row["coincident_content"]["mean_family_margin"] == pytest.approx(
            sum(row["components"][c]["mean_family_margin"] for c in COMPONENTS[:2])
        )


def test_wrong_baseline_score_fails_closed(synthetic_panel, monkeypatch):
    vectors, maps, meta, source, _ = synthetic_panel
    baseline = copy.deepcopy(source)
    baseline["map_matrix"][0]["test_seed_scores"][0]["reference_cosines"]["a0"] += 0.01
    monkeypatch.setattr(
        "fractal_vlm_state_probe.source_coordinate_overlap.partition_maps",
        lambda a, b: partition_maps(a, b, shape=(2, 3), padding_rows=1),
    )
    with pytest.raises(ValueError, match="frozen baseline"):
        analyze(vectors, maps, meta, baseline)
