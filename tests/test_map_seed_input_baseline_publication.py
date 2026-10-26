import copy

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.analyze_pairing_input_holdout import input_interaction
from fractal_vlm_state_probe.cli.summarize_map_seed_input_baseline import (
    verify_equivariance,
    verify_interaction,
    verify_scores,
)
from fractal_vlm_state_probe.map_seed_input_baseline import VIEWS, analyze_vectors
from fractal_vlm_state_probe.permutation_maps import ANCHOR, MAPS, NEW_MAPS


@pytest.mark.parametrize("view", VIEWS)
def test_independent_interaction_oracle(view):
    rng = np.random.default_rng(3)
    cells = {
        c: rng.integers(0, 256, (4, 5, 3), dtype=np.uint8)
        for c in ("mm", "jj", "mj", "jm")
    }
    if view != "raw_rgb":
        cells = {c: a.astype(np.float32) / 80 - 1 for c, a in cells.items()}
    converted = {
        c: a.astype(np.float64) / (255 if view == "raw_rgb" else 1)
        for c, a in cells.items()
    }
    saved = input_interaction(converted)
    assert verify_interaction(saved, cells, view) < 1e-12
    saved[0, 0, 0] += 0.001
    with pytest.raises(ValueError, match="independent cell arithmetic"):
        verify_interaction(saved, cells, view)


@pytest.fixture(scope="module")
def scored(tmp_path_factory):
    root = tmp_path_factory.mktemp("input-score-publication")
    metadata = {
        f"{b}{f}_r{r}": {"broad_class": b, "pairing_family": f"{b}{f}", "replicate": r}
        for b in ("a", "b")
        for f in range(4)
        for r in range(1, 5)
    }
    rng = np.random.default_rng(5)
    base = {label: rng.normal(size=20) for label in metadata}
    paths = {v: {k: {} for k in MAPS} for v in VIEWS}
    for label, vector in base.items():
        path = root / f"{label}.npy"
        np.save(path, vector)
        for view in VIEWS:
            for key in MAPS:
                paths[view][key][label] = path
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "fractal_vlm_state_probe.map_seed_input_baseline.raw_equivariance",
            lambda *a: {},
        )
        result = analyze_vectors(paths, metadata, {})
    vectors = {v: {k: base for k in MAPS} for v in VIEWS}
    return result, vectors, metadata


def test_independent_scores_reproduce_full_denominators(scored):
    assert verify_scores(*scored) == {
        "matrix_cosines": 4096,
        "aggregate_cosines": 512,
        "synchronized_exact_tests": 4,
        "directional_records": 384,
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_matrix",
        "cosine",
        "primary_count",
        "holm",
        "retrieval",
        "direction",
        "direction_phase",
        "gap",
        "missing_family_gap",
    ],
)
def test_independent_scores_reject_corruption(scored, corruption):
    source, vectors, metadata = scored
    result = copy.deepcopy(source)
    if corruption == "duplicate_matrix":
        result["map_matrix"][-1] = result["map_matrix"][0]
    elif corruption == "cosine":
        result["map_matrix"][0]["test_seed_scores"][0]["reference_cosines"]["a0"] += 0.1
    elif corruption == "primary_count":
        result["primary_tests"].pop()
    elif corruption == "holm":
        result["primary_tests"][0]["holm_family_size"] = 12
    elif corruption == "retrieval":
        result["primary_tests"][0]["retrieval_correct_count"] += 1
    elif corruption == "direction":
        result["directional_decompositions"][0]["cosine"] = -1
    elif corruption == "direction_phase":
        result["directional_decompositions"][0]["phase"] = "test"
    elif corruption == "gap":
        result["same_cross_gaps"][0]["same_minus_cross_mean_margin"] += 0.1
    else:
        result["same_cross_gaps"][0]["family_margin_differences"].pop("a0")
    with pytest.raises(ValueError):
        verify_scores(result, vectors, metadata)


def test_independent_inverse_scatter_and_negative_control():
    rng = np.random.default_rng(12)
    original = rng.normal(size=(320, 320, 3))
    maps = {k: rng.permutation(76800) for k in MAPS}
    vectors = {}
    for key, mapping in maps.items():
        value = original.copy()
        value[40:280] = original[40:280].reshape(-1, 3)[mapping].reshape(240, 320, 3)
        vectors[key] = {str(i): value for i in range(32)}
    result = {
        "raw_equivariance": {
            "inverse_checks": [
                {"pair_id": label, "map": key, "exact_inverse_equal": True}
                for label in vectors[ANCHOR]
                for key in NEW_MAPS
            ],
            "diagonal_cosine_max_abs_error": 0,
        },
        "map_matrix": [
            {
                "view": "raw_rgb",
                "reference_map": k,
                "test_map": k,
                "available": True,
                "test_seed_scores": [
                    {"label": "x", "reference_cosines": {"family": 0.25}}
                ],
            }
            for k in MAPS
        ],
    }
    assert verify_equivariance(result, vectors, maps) == 96
    vectors[MAPS[-1]]["0"] = vectors[MAPS[-1]]["0"].copy()
    vectors[MAPS[-1]]["0"][40, 0, 0] += 1
    with pytest.raises(ValueError, match="independent raw inverse"):
        verify_equivariance(result, vectors, maps)
