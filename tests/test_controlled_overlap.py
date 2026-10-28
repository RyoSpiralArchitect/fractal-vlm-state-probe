import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.analyze_controlled_overlap import pair_scores
from fractal_vlm_state_probe.controlled_overlap import (
    construct_maps,
    coordinate_counts,
    margin_cube,
    paired_bootstrap,
    verify_coordinate_recipe,
)


def config():
    return json.loads(
        (
            Path(__file__).parents[1] / "configs/controlled_overlap_phase1_v1.json"
        ).read_text()
    )


@pytest.mark.parametrize("size", [16, 64, 320])
def test_coordinate_recipe_and_exact_coincidences(size):
    cfg = config()
    maps, recipes = construct_maps(cfg, size=size)
    assert verify_coordinate_recipe(cfg, maps, recipes)["maps_verified"] == 28
    for name, row in recipes.items():
        counts = coordinate_counts(maps[row["reference"]], maps[name])
        assert counts["left_fixed"] == counts["right_fixed"] == size // 4
        if row["q"] is not None:
            assert counts["shared_fixed"] == size * row["q"] // 16
            assert counts["shared_moved"] == 0
            assert counts["noncoincident"] == size - counts["shared_fixed"]
            same_mask = maps[f"m{row['mask']}_q{row['q']}_d{1 - row['destination']}"]
            np.testing.assert_array_equal(
                maps[name] == np.arange(size), same_mask == np.arange(size)
            )
            assert 1 <= len(row["attempt_shared_moved"]) <= 128


def test_recipe_corruption_and_retry_limit_rejected():
    cfg = config()
    maps, recipes = construct_maps(cfg, size=64)
    broken = copy.deepcopy(recipes)
    broken["m0_q0_d0"]["attempt_shared_moved"].append(0)
    with pytest.raises(ValueError, match="retry receipt"):
        verify_coordinate_recipe(cfg, maps, broken)
    broken = copy.deepcopy(recipes)
    broken["m0_q0_d0"]["q"] = 4
    with pytest.raises(ValueError, match="recipe metadata"):
        verify_coordinate_recipe(cfg, maps, broken)
    changed = {k: v.copy() for k, v in maps.items()}
    changed["m0_q0_d0"][[0, 1]] = changed["m0_q0_d0"][[1, 0]]
    with pytest.raises(ValueError, match="reconstruction"):
        verify_coordinate_recipe(cfg, changed, recipes)
    cfg["maximum_attempts_per_test_map"] = 0
    with pytest.raises(ValueError, match="exhausted"):
        construct_maps(cfg, size=64)


@pytest.mark.parametrize("size", [0, 15, 17])
def test_invalid_size_rejected(size):
    with pytest.raises(ValueError):
        construct_maps(config(), size=size)


def test_malformed_maps_rejected():
    with pytest.raises(ValueError):
        coordinate_counts(np.array([0, 0]), np.array([0, 1]))
    with pytest.raises(ValueError):
        coordinate_counts(np.arange(2), np.arange(3))


def test_bootstrap_preserves_pairing_and_constant_endpoints():
    cfg = config()
    cfg["bootstrap"]["draws"] = 100
    a = np.ones((2, 4, 2, 3, 8, 2))
    a[:, :, :, 2] = 3
    checked = paired_bootstrap(a, cfg)
    np.testing.assert_allclose(checked["ranges"], [[[1, 1], [2, 2]]] * 2)
    generator = np.random.default_rng(51)
    values = generator.normal(size=(2, 4, 2, 1, 8, 2))
    paired = np.repeat(values, 3, axis=3)
    checked = paired_bootstrap(paired, cfg)
    np.testing.assert_array_equal(np.asarray(checked["ranges"])[:, 1], 0)
    assert checked == paired_bootstrap(paired, cfg)


def test_bootstrap_rejects_partial_or_nonfinite_panel():
    with pytest.raises(ValueError):
        paired_bootstrap(np.ones((2, 3, 2, 3, 8, 2)), config())
    with pytest.raises(ValueError):
        paired_bootstrap(np.full((2, 4, 2, 3, 8, 2), np.nan), config())


@pytest.fixture
def inputs(tmp_path):
    metadata = {
        f"{b}{f}_r{r}": {"pairing_family": f"{b}{f}", "broad_class": b, "replicate": r}
        for b in ("a", "b")
        for f in range(4)
        for r in range(1, 5)
    }
    generator = np.random.default_rng(7)
    paths = {}
    for label in metadata:
        paths[label] = tmp_path / f"{label}.npy"
        np.save(paths[label], generator.normal(size=(4, 4, 3)))
    return metadata, paths


def test_full_normalization_partition_and_margin_oracle(inputs):
    metadata, paths = inputs
    categories = np.tile(np.arange(4), (4, 1))
    result = pair_scores(paths, paths, metadata, categories=categories)
    assert result["available"]
    scores = np.asarray(result["scores"])
    np.testing.assert_allclose(
        sum(np.asarray(v) for v in result["component_scores"].values()),
        scores,
        atol=1e-12,
    )
    cube = np.broadcast_to(scores, (2, 4, 2, 3, 16, 8))
    margins = margin_cube(cube, result["labels"], result["families"], metadata)
    assert margins.mean() == pytest.approx(result["summary"]["mean_family_margin"])
    for fi, family in enumerate(result["families"]):
        assert margins[..., fi, :].mean() == pytest.approx(
            result["summary"]["family_margins"][family]
        )
    for row, label in zip(scores, result["labels"]):
        test = np.load(paths[label]).ravel()
        for score, family in zip(row, result["families"]):
            ref = [np.load(paths[f"{family}_r{i}"]).ravel() for i in (1, 2)]
            direction = sum(v / np.sqrt(np.sum(v * v)) for v in ref)
            expected = np.sum(test * direction) / np.sqrt(
                np.sum(test * test) * np.sum(direction * direction)
            )
            assert score == pytest.approx(expected, abs=1e-12)


def test_zero_cancelled_and_nonfinite_vectors(inputs):
    metadata, paths = inputs
    np.save(paths["a0_r1"], np.zeros((4, 4, 3)))
    assert not pair_scores(paths, paths, metadata)["available"]
    np.save(paths["a0_r1"], -np.load(paths["a0_r2"]))
    assert not pair_scores(paths, paths, metadata)["available"]
    np.save(paths["a0_r1"], np.full((4, 4, 3), np.nan))
    with pytest.raises(ValueError, match="nonfinite"):
        pair_scores(paths, paths, metadata)
