import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_direction_holdout import evaluate_pairing_holdout
from fractal_vlm_state_probe.cli.run_map_seed_cache_study import validate_config
from fractal_vlm_state_probe.cli.summarize_padding_cache_study import verify_exact_test
from fractal_vlm_state_probe.graded_cache import endpoint_reference_vectors
from fractal_vlm_state_probe.map_seed_cache import aggregate_map_scores
from fractal_vlm_state_probe.permutation_maps import NEW_MAPS


def toy_matrix(rotate=False):
    metadata = {
        f"f{f}_r{r}": {
            "pairing_family": f"f{f}",
            "replicate": r,
            "broad_class": "a" if f < 4 else "b",
        }
        for f in range(8)
        for r in range(1, 5)
    }
    vectors = {
        key: {
            label: np.eye(8)[(int(m["pairing_family"][1:]) + (i if rotate else 0)) % 8]
            for label, m in metadata.items()
        }
        for i, key in enumerate(NEW_MAPS)
    }
    rows = []
    for a in NEW_MAPS:
        for b in NEW_MAPS:
            result = evaluate_pairing_holdout(
                endpoint_reference_vectors(vectors[a], vectors[b], metadata), metadata
            )
            result.pop("exact_block_test")
            rows.append({"reference_map": a, "test_map": b, **result})
    return rows, metadata


def select(rows, same):
    return [r for r in rows if (r["reference_map"] == r["test_map"]) == same]


def test_identical_maps_have_same_and_cross_correspondence_without_sample_inflation():
    rows, metadata = toy_matrix()
    for regime, same, count in (("same_map", True, 3), ("cross_map", False, 6)):
        result = aggregate_map_scores(select(rows, same), metadata, regime)
        verify_exact_test(result, metadata)
        assert result["test_seed_count"] == 16
        assert result["map_pair_count"] == count
        assert result["mean_family_margin"] == 1
        assert result["retrieval_correct_count"] == 16
        assert result["exact_block_test"]["assignment_count"] == 576
        assert result["exact_block_test"]["extreme_count"] == 1


def test_map_specific_coordinates_separate_within_and_cross_scores():
    rows, metadata = toy_matrix(rotate=True)
    same = aggregate_map_scores(select(rows, True), metadata, "same_map")
    cross = aggregate_map_scores(select(rows, False), metadata, "cross_map")
    verify_exact_test(same, metadata)
    verify_exact_test(cross, metadata)
    assert same["mean_family_margin"] == 1
    assert cross["mean_family_margin"] < 0
    assert cross["retrieval_tied_count"] == 16


@pytest.mark.parametrize("regime,same", [("same_map", True), ("cross_map", False)])
def test_missing_map_pair_is_not_silently_renormalized(regime, same):
    rows, metadata = toy_matrix()
    with pytest.raises(ValueError):
        aggregate_map_scores(select(rows, same)[:-1], metadata, regime)


def test_unavailable_map_pair_keeps_whole_primary_unavailable():
    rows, metadata = toy_matrix()
    selected = select(rows, False)
    selected[0] = {**selected[0], "available": False}
    result = aggregate_map_scores(selected, metadata, "cross_map")
    assert not result["available"]
    assert "mean_family_margin" not in result


@pytest.mark.parametrize(
    "corruption", ["duplicate_map", "missing_seed", "missing_family", "nonfinite"]
)
def test_score_coverage_and_finiteness_fail_closed(corruption):
    rows, metadata = toy_matrix()
    selected = copy.deepcopy(select(rows, True))
    if corruption == "duplicate_map":
        selected[-1] = copy.deepcopy(selected[0])
    elif corruption == "missing_seed":
        selected[0]["test_seed_scores"].pop()
    elif corruption == "missing_family":
        selected[0]["test_seed_scores"][0]["reference_cosines"].pop("f0")
    else:
        selected[0]["test_seed_scores"][0]["reference_cosines"]["f0"] = float("nan")
    with pytest.raises(ValueError):
        aggregate_map_scores(selected, metadata, "same_map")


def test_cache_config_rejects_new_hypotheses_or_source_changes():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/map_seed_cache_fastvlm_v1.json").read_text())
    validate_config(config)
    for key, value in (
        ("primary_test_family_size", 6),
        ("primary_map_scope", "all_maps"),
        ("expected_panel_cells", 384),
        ("expected_source_response", "ACK"),
    ):
        with pytest.raises(ValueError):
            validate_config({**config, key: value})
