import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.run_map_seed_input_baseline import (
    array_hash,
    checked_pixels,
)
from fractal_vlm_state_probe.map_seed_input_baseline import (
    VIEWS,
    analyze_vectors,
    canonical_raw,
    raw_equivariance,
    validate_config,
)
from fractal_vlm_state_probe.permutation_maps import ANCHOR, MAPS


def test_config_matches_registration():
    config = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/map_seed_input_baseline_fastvlm_v1.json"
        ).read_text()
    )
    validate_config(config)
    config["primary_test_family_size"] = 12
    with pytest.raises(ValueError, match="registration"):
        validate_config(config)


def test_inverse_raw_map_preserves_padding_and_channels():
    original = np.arange(320 * 320 * 3).reshape(320, 320, 3)
    mapping = np.random.default_rng(5).permutation(76800)
    permuted = original.copy()
    permuted[40:280] = original[40:280].reshape(-1, 3)[mapping].reshape(240, 320, 3)
    np.testing.assert_array_equal(canonical_raw(permuted, mapping), original)


@pytest.mark.parametrize("broken", ["shape", "duplicate"])
def test_inverse_rejects_invalid_map(broken):
    mapping = np.arange(76800)
    if broken == "shape":
        mapping = mapping[:-1]
    else:
        mapping[0] = 1
    with pytest.raises(ValueError):
        canonical_raw(np.zeros((320, 320, 3)), mapping)


def test_raw_equivariance_detects_nonpermutation_effect():
    maps = {k: np.arange(76800) for k in MAPS}
    vectors = {k: {"a": np.ones((320, 320, 3))} for k in MAPS}
    vectors[MAPS[-1]]["a"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="common coordinate"):
        raw_equivariance(vectors, maps, [])


@pytest.fixture
def pixel_contract():
    pixels = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    payload = {"pixel_values": pixels, "image_sizes": np.array([[320, 320]])}
    expected = {
        "pixel_values_shape": [3, 1024, 1024],
        "pixel_values_dtype": "float32",
        "pixel_values_sha256": array_hash(pixels),
        "processor_non_pixel_payload": {"image_sizes": [[320, 320]]},
    }
    audit = {
        "native": {"pixel_values": {"sha256": array_hash(pixels)}},
        "override_image_sizes": None,
    }
    audit["effective"] = copy.deepcopy(audit["native"])
    return payload, expected, audit


def test_processor_checks_exact_native_values(pixel_contract):
    payload, expected, audit = pixel_contract
    pixels, digest = checked_pixels(
        np.zeros((320, 320, 3), dtype=np.uint8), lambda **kw: payload, expected, audit
    )
    assert pixels.dtype == np.float32
    assert digest == expected["pixel_values_sha256"]


@pytest.mark.parametrize(
    "field", ["pixels", "metadata", "cache_hash", "override", "dtype"]
)
def test_processor_rejects_contract_drift(pixel_contract, field):
    payload, expected, audit = pixel_contract
    if field == "pixels":
        payload["pixel_values"][0, 0, 0, 0] = 1
    elif field == "metadata":
        payload["image_sizes"][0, 0] = 240
    elif field == "cache_hash":
        audit["effective"]["pixel_values"]["sha256"] = "different"
    elif field == "override":
        audit["override_image_sizes"] = [240, 320]
    else:
        payload["pixel_values"] = payload["pixel_values"].astype(np.float64)
    with pytest.raises(ValueError, match="qualified inputs"):
        checked_pixels(
            np.zeros((320, 320, 3), dtype=np.uint8),
            lambda **kw: payload,
            expected,
            audit,
        )


@pytest.fixture
def tiny_paths(tmp_path, monkeypatch):
    metadata = {
        f"{b}{f}_r{r}": {"broad_class": b, "pairing_family": f"{b}{f}", "replicate": r}
        for b in ("a", "b")
        for f in range(4)
        for r in range(1, 5)
    }
    paths = {v: {k: {} for k in MAPS} for v in VIEWS}
    rng = np.random.default_rng(8)
    vectors = {label: rng.normal(size=20) for label in metadata}
    for view in VIEWS:
        for key in MAPS:
            for label, vector in vectors.items():
                p = tmp_path / f"{view}_{key}_{label}.npy"
                np.save(p, vector)
                paths[view][key][label] = p
    monkeypatch.setattr(
        "fractal_vlm_state_probe.map_seed_input_baseline.raw_equivariance",
        lambda *a: {"tested_separately": True},
    )
    return paths, metadata


def test_analysis_retains_full_map_matrix_and_joint_four_tests(tiny_paths):
    paths, metadata = tiny_paths
    result = analyze_vectors(paths, metadata, {})
    assert len(result["map_matrix"]) == 32
    assert len(result["directional_decompositions"]) == 384
    assert len(result["primary_tests"]) == 4
    for row in result["primary_tests"]:
        assert row["test_seed_count"] == 16
        assert row["holm_family_size"] == 4
        assert row["exact_block_test"]["assignment_count"] == 576
        assert all(ANCHOR not in pair for pair in row["ordered_map_pairs"])
    assert all("exact_block_test" not in r for r in result["map_matrix"])
    assert all(
        r["same_minus_cross_mean_margin"] == 0 for r in result["same_cross_gaps"]
    )


def test_zero_input_retained_unavailable(tiny_paths):
    paths, metadata = tiny_paths
    label = next(iter(metadata))
    np.save(paths["raw_rgb"][MAPS[1]][label], np.zeros(20))
    result = analyze_vectors(paths, metadata, {})
    for row in result["primary_tests"]:
        if row["view"] == "raw_rgb":
            assert not row["available"]
            assert row["holm_p_greater"] == 1
    assert len(result["directional_decompositions"]) == 384


def test_missing_map_rejected(tiny_paths):
    paths, metadata = tiny_paths
    del paths["raw_rgb"][ANCHOR]
    with pytest.raises(ValueError, match="coverage"):
        analyze_vectors(paths, metadata, {})
