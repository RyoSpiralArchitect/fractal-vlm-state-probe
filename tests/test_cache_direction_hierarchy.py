from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

from fractal_vlm_state_probe.cache_direction_hierarchy import (
    analyze_cache_direction_hierarchy,
    format_cache_direction_hierarchy_markdown,
)
from fractal_vlm_state_probe.cli.analyze_cache_direction_hierarchy import (
    _load_panel_hierarchy,
)


def test_hierarchy_separates_seed_replication_and_pairing_transfer() -> None:
    labels = ("g11", "g12", "g21", "g22", "s11", "s12", "s21", "s22")
    broad = {
        label: "geometry" if label.startswith("g") else "stochastic" for label in labels
    }
    pairing = {label: label[:2] for label in labels}
    cosine = {}
    for left, right in combinations(labels, 2):
        if pairing[left] == pairing[right]:
            value = 0.9
        elif broad[left] == broad[right]:
            value = 0.6
        else:
            value = -0.2
        cosine[(left, right)] = value

    result = analyze_cache_direction_hierarchy(
        _replication(labels, cosine),
        broad_class_by_label=broad,
        pairing_family_by_label=pairing,
    )

    region = result["groups"][0]["regions"][0]
    categories = region["direction_categories"]
    assert categories["within_pairing_family"]["count"] == 4
    assert categories["cross_pairing_same_broad"]["count"] == 8
    assert categories["between_broad"]["count"] == 16
    assert region["seed_replication_test"]["statistic"] == pytest.approx(0.3)
    assert region["seed_replication_test"]["assignment_count"] == 9
    assert region["broad_class_transfer_test"]["statistic"] == pytest.approx(0.8)
    assert region["broad_class_transfer_test"]["assignment_count"] == 6
    assert len(region["broad_class_specific_tests"]) == 2
    assert all(
        record["statistic"] == pytest.approx(0.8)
        for record in region["broad_class_specific_tests"]
    )
    assert all(
        record["assignment_count"] == 6
        for record in region["broad_class_specific_tests"]
    )

    markdown = format_cache_direction_hierarchy_markdown(result)
    assert markdown.startswith("# Source-Cache Direction Pairing Hierarchy")
    assert "exact / 9" in markdown
    assert "exact / 6" in markdown
    assert "Broad-Class-Specific Transfer" in markdown


def test_hierarchy_rejects_pairing_family_that_crosses_broad_classes() -> None:
    labels = ("g1", "g2", "g3", "g4", "s1", "s2", "s3", "s4")
    broad = {
        label: "geometry" if label.startswith("g") else "stochastic" for label in labels
    }
    pairing = {
        "g1": "crossed_1",
        "g2": "geometry_1",
        "g3": "geometry_1",
        "g4": "crossed_2",
        "s1": "crossed_1",
        "s2": "stochastic_1",
        "s3": "stochastic_1",
        "s4": "crossed_2",
    }
    cosine = {pair: 0.0 for pair in combinations(labels, 2)}

    with pytest.raises(ValueError, match="pairing families cross broad classes"):
        analyze_cache_direction_hierarchy(
            _replication(labels, cosine),
            broad_class_by_label=broad,
            pairing_family_by_label=pairing,
        )


def test_hierarchy_enumerates_the_full_sixteen_pair_design() -> None:
    labels = tuple(
        f"{broad_prefix}{family}_r{replicate}"
        for broad_prefix in ("g", "s")
        for family in range(1, 5)
        for replicate in (1, 2)
    )
    broad = {
        label: "geometry" if label.startswith("g") else "stochastic" for label in labels
    }
    pairing = {label: label.rsplit("_r", 1)[0] for label in labels}
    cosine = {}
    for left, right in combinations(labels, 2):
        if pairing[left] == pairing[right]:
            value = 0.9
        elif broad[left] == broad[right]:
            value = 0.4
        else:
            value = -0.1
        cosine[(left, right)] = value

    result = analyze_cache_direction_hierarchy(
        _replication(labels, cosine),
        broad_class_by_label=broad,
        pairing_family_by_label=pairing,
    )

    region = result["groups"][0]["regions"][0]
    categories = region["direction_categories"]
    assert categories["within_pairing_family"]["count"] == 8
    assert categories["cross_pairing_same_broad"]["count"] == 48
    assert categories["between_broad"]["count"] == 64
    assert region["seed_replication_test"]["assignment_count"] == 11_025
    assert region["seed_replication_test"]["p_greater"] == pytest.approx(1 / 11_025)
    assert region["broad_class_transfer_test"]["assignment_count"] == 70
    assert region["broad_class_transfer_test"]["p_greater"] == pytest.approx(2 / 70)


def test_load_panel_hierarchy_uses_generated_summary_maps(tmp_path: Path) -> None:
    path = tmp_path / "panel_summary.json"
    path.write_text(
        json.dumps(
            {
                "analysis_kind": "generator_pairing_factorial_panel",
                "broad_class_by_pair": {"geometry_vq_r1": "geometry"},
                "pairing_family_by_pair": {
                    "geometry_vq_r1": "geometry_voronoi_quasicrystal"
                },
            }
        ),
        encoding="utf-8",
    )

    broad, pairing = _load_panel_hierarchy(path)

    assert broad == {"geometry_vq_r1": "geometry"}
    assert pairing == {"geometry_vq_r1": "geometry_voronoi_quasicrystal"}


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
