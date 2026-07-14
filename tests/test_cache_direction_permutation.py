from __future__ import annotations

import json
from itertools import combinations

import pytest

from fractal_vlm_state_probe.cache_direction_permutation import (
    analyze_cache_direction_class_permutation,
    format_cache_direction_class_permutation_markdown,
)


def test_source_pair_permutation_detects_class_organized_direction() -> None:
    labels = ("f1", "f2", "g1", "g2")
    classes = {
        "f1": "fractal",
        "f2": "fractal",
        "g1": "geometry",
        "g2": "geometry",
    }
    cosine = {
        ("f1", "f2"): 1.0,
        ("g1", "g2"): 1.0,
        ("f1", "g1"): -1.0,
        ("f1", "g2"): -1.0,
        ("f2", "g1"): -1.0,
        ("f2", "g2"): -1.0,
    }

    result = analyze_cache_direction_class_permutation(
        _replication(labels, cosine), class_by_label=classes
    )

    region = result["groups"][0]["regions"][0]
    assert region["observed"]["within_class"]["mean"] == 1.0
    assert region["observed"]["between_class"]["mean"] == -1.0
    assert region["observed"]["mean_difference"] == 2.0
    assert region["global_test"]["mode"] == "exact"
    assert region["global_test"]["assignment_count"] == 6
    assert region["global_test"]["p_greater"] == pytest.approx(2.0 / 6.0)
    assert all(record["assignment_count"] == 6 for record in region["class_tests"])
    json.dumps(result)

    markdown = format_cache_direction_class_permutation_markdown(result)
    assert markdown.startswith("# Source-Level Cache Direction Permutation")
    assert "exact / 6" in markdown
    assert "`fractal` vs `geometry`" in markdown


def test_source_pair_permutation_uses_monte_carlo_above_exact_limit() -> None:
    labels = tuple(f"p{index}" for index in range(6))
    classes = {label: "a" if index < 3 else "b" for index, label in enumerate(labels)}
    cosine = {
        pair: 0.5 if classes[pair[0]] == classes[pair[1]] else 0.0
        for pair in combinations(labels, 2)
    }

    result = analyze_cache_direction_class_permutation(
        _replication(labels, cosine),
        class_by_label=classes,
        exact_limit=1,
        monte_carlo_permutations=25,
        seed=7,
    )

    test = result["groups"][0]["regions"][0]["global_test"]
    assert test["mode"] == "monte_carlo"
    assert test["assignment_count"] == 25
    assert 0.0 < test["p_greater"] <= 1.0


def test_source_pair_permutation_requires_complete_class_labels() -> None:
    labels = ("a1", "a2", "b1", "b2")
    cosine = {pair: 0.0 for pair in combinations(labels, 2)}

    with pytest.raises(ValueError, match="missing class labels"):
        analyze_cache_direction_class_permutation(
            _replication(labels, cosine),
            class_by_label={"a1": "a", "a2": "a", "b1": "b"},
        )


def _replication(labels: tuple[str, ...], cosine: dict[tuple[str, str], float]) -> dict:
    pairwise = [
        {"left": left, "right": right, "cosine": value, "available": True}
        for (left, right), value in cosine.items()
    ]
    region = {
        "region": "image_tokens",
        "pairwise_direction_cosines": pairwise,
        "effect_direction_replication": {
            "interaction": {"pairwise": pairwise},
        },
    }
    return {
        "analysis_kind": "source_cache_tensor_cross_pair_replication",
        "groups": [
            {
                "model_id": "example/model",
                "layer_index": 3,
                "tensor": "values",
                "points": [{"label": label} for label in labels],
                "regions": [region],
            }
        ],
    }
