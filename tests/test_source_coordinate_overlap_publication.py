import copy
from itertools import product

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_direction_holdout import evaluate_pairing_holdout
from fractal_vlm_state_probe.cli.summarize_source_coordinate_overlap import (
    check_ratio,
    oracle_partition,
    verify_decomposition,
)
from fractal_vlm_state_probe.graded_cache import endpoint_reference_vectors
from fractal_vlm_state_probe.map_seed_cache import aggregate_map_scores
from fractal_vlm_state_probe.permutation_maps import MAPS, NEW_MAPS
from fractal_vlm_state_probe.source_coordinate_overlap import analyze, partition_maps


def test_independent_coordinate_categories():
    a, b = np.array([0, 2, 1, 3, 5, 4]), np.array([0, 2, 3, 1, 4, 5])
    expected = np.array([[3, 3, 3], [0, 1, 2], [2, 2, 2], [3, 3, 3]])
    np.testing.assert_array_equal(
        oracle_partition(a, b, shape=(2, 3), padding_rows=1), expected
    )


@pytest.mark.parametrize(
    "saved,value,total", [(1.2, 1.2, 1), (-0.2, -0.2, 1), (None, 1, 0)]
)
def test_independent_ratios_preserve_signed_and_missing_values(saved, value, total):
    check_ratio(saved, value, total)


@pytest.mark.parametrize(
    "saved,value,total", [(1, 1.2, 1), (0, -0.2, 1), (0, 1, 0), (float("nan"), 1, 1)]
)
def test_independent_ratios_reject_clipping_or_invention(saved, value, total):
    with pytest.raises(ValueError, match="signed component ratio"):
        check_ratio(saved, value, total)


@pytest.fixture(scope="module")
def panel():
    metadata = {
        f"{b}{f}_r{r}": {"broad_class": b, "pairing_family": f"{b}{f}", "replicate": r}
        for b in ("a", "b")
        for f in range(4)
        for r in range(1, 5)
    }
    rng = np.random.default_rng(21)
    maps = {k: rng.permutation(6) for k in MAPS}
    original = {k: rng.normal(size=(4, 3, 3)) for k in metadata}
    vectors = {k: {} for k in MAPS}
    for key in MAPS:
        for label, array in original.items():
            value = array.copy()
            value[1:3] = array[1:3].reshape(6, 3)[maps[key]].reshape(2, 3, 3)
            vectors[key][label] = value
    matrix = []
    for a, b in product(MAPS, repeat=2):
        row = evaluate_pairing_holdout(
            endpoint_reference_vectors(vectors[a], vectors[b], metadata), metadata
        )
        row.pop("exact_block_test")
        matrix.append({"reference_map": a, "test_map": b, "view": "raw_rgb", **row})
    primary = []
    for regime in ("same_map", "cross_map"):
        selected = [
            r
            for r in matrix
            if r["reference_map"] in NEW_MAPS
            and r["test_map"] in NEW_MAPS
            and (r["reference_map"] == r["test_map"]) == (regime == "same_map")
        ]
        primary.append(
            {
                "view": "raw_rgb",
                "regime": regime,
                **aggregate_map_scores(selected, metadata, regime),
            }
        )
    baseline = {"map_matrix": matrix, "primary_tests": primary}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "fractal_vlm_state_probe.source_coordinate_overlap.partition_maps",
            lambda a, b: partition_maps(a, b, shape=(2, 3), padding_rows=1),
        )
        result = analyze(vectors, maps, metadata, baseline)
    dots = result.pop("dot_products")
    return result, dots, vectors, maps, metadata, baseline


@pytest.fixture(autouse=True)
def small_oracle(monkeypatch):
    monkeypatch.setattr(
        "fractal_vlm_state_probe.cli.summarize_source_coordinate_overlap.oracle_partition",
        lambda a, b: oracle_partition(a, b, shape=(2, 3), padding_rows=1),
    )


def test_independent_score_accounting_matches(panel):
    checked = verify_decomposition(*panel)
    assert checked["ordered_map_partitions"] == 16
    assert checked["dot_products_recomputed"] == 2048
    assert checked["additive_terms_recomputed"] == 8192
    assert checked["aggregate_regimes_recomputed"] == 2
    assert checked["maximum_component_oracle_abs_error"] < 1e-12


@pytest.mark.parametrize(
    "corruption",
    [
        "term",
        "full",
        "nonfinite_full",
        "duplicate",
        "component",
        "normalization",
        "partition",
        "weights",
        "family",
        "ratio",
        "retrieval",
        "seed",
        "test_count",
    ],
)
def test_corrupted_publication_rejected(panel, corruption):
    source, original_dots, vectors, maps, metadata, baseline = panel
    result, dots = copy.deepcopy(source), copy.deepcopy(original_dots)
    if corruption == "term":
        dots[0]["components"]["shared_fixed_content"] += 0.01
    elif corruption == "full":
        dots[0]["full_cosine"] += 0.01
    elif corruption == "nonfinite_full":
        dots[0]["full_cosine"] = float("nan")
    elif corruption == "duplicate":
        dots[-1] = dots[0]
    elif corruption == "component":
        dots[0]["components"].pop("padding")
    elif corruption == "normalization":
        result["matrix"][0]["normalization"]["tests"]["a0_r3"] *= 2
    elif corruption == "partition":
        result["matrix"][0]["partition"]["pixel_counts"]["padding"] += 1
    elif corruption == "weights":
        result["aggregates"][1]["ordered_map_pairs"].pop()
    elif corruption == "family":
        result["matrix"][0]["components"]["padding"]["family_margins"]["a0"] += 0.01
    elif corruption == "ratio":
        result["matrix"][0]["coincident_content"]["mean_margin_fraction_of_full"] += 1
    elif corruption == "retrieval":
        result["aggregates"][1]["components"]["padding"]["retrieval_correct_count"] = 16
    elif corruption == "seed":
        result["matrix"][0]["full"]["test_seed_margins"].pop()
    else:
        result["new_statistical_tests"] = 1
    with pytest.raises(ValueError):
        verify_decomposition(result, dots, vectors, maps, metadata, baseline)
