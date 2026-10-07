import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.run_graded_permutation_study import validate_config
from fractal_vlm_state_probe.frequency_control import CELLS, rgb_multiset_sha256
from fractal_vlm_state_probe.graded_permutation import (
    audit_level,
    mapping_metrics,
    restricted_permutation,
)
from fractal_vlm_state_probe.padding_policy import fill_color, pad_to_square


def test_cycle_restriction_has_exact_endpoints_and_nested_support():
    full = np.array([2, 3, 4, 1, 0, 5])
    order = np.array([4, 1, 0, 3, 5, 2])
    previous = set()
    for count in range(7):
        actual = restricted_permutation(full, order, count)
        selected = set(order[:count])
        moved = set(np.flatnonzero(actual != np.arange(6)))
        assert previous <= moved <= selected
        assert np.array_equal(np.sort(actual), np.arange(6))
        assert np.all(
            actual[list(set(range(6)) - selected)] == list(set(range(6)) - selected)
        )
        previous = moved
        # Independent definition: follow the full map until the next selected site.
        for position in selected:
            successor = full[position]
            while successor not in selected:
                successor = full[successor]
            assert actual[position] == successor
    assert np.array_equal(restricted_permutation(full, order, 0), np.arange(6))
    assert np.array_equal(restricted_permutation(full, order, 6), full)


@pytest.mark.parametrize("size", [2, 17, 96])
def test_random_restrictions_roundtrip_and_preserve_rgb_multiset(size):
    rng = np.random.default_rng(85)
    full, order = rng.permutation(size), rng.permutation(size)
    pixels = rng.integers(0, 256, size=(size, 3), dtype=np.uint8)
    for count in range(size + 1):
        mapping = restricted_permutation(full, order, count)
        assert np.array_equal(pixels[mapping][np.argsort(mapping)], pixels)
        assert rgb_multiset_sha256(pixels[mapping]) == rgb_multiset_sha256(pixels)


@pytest.mark.parametrize(
    "full,order,count",
    [
        ([0, 0], [0, 1], 1),
        ([0, 1], [1, 1], 1),
        ([0.0, 1.0], [0, 1], 1),
        ([0, 1], [0, 1], -1),
        ([0, 1], [0, 1], 3),
        ([0, 1], [0], 1),
        ([], [], 0),
    ],
)
def test_invalid_permutations_fail_closed(full, order, count):
    with pytest.raises(ValueError):
        restricted_permutation(np.array(full), np.array(order), count)


def test_mapping_metrics_identity_and_shape_guard():
    result = mapping_metrics(np.arange(48), (6, 8))
    assert result["moved_index_fraction"] == 0
    assert result["mean_displacement_pixels"] == 0
    assert result["retained_undirected_grid_edge_fraction"] == 1
    with pytest.raises(ValueError):
        mapping_metrics(np.arange(48), (5, 8))


class ToyProcessor:
    def __call__(self, *, images, text, return_tensors):
        rgb = np.asarray(images).astype(np.float32) / 255
        return {
            "pixel_values": rgb.transpose(2, 0, 1)[None],
            "image_sizes": [[images.width, images.height]],
            "input_ids": [[1, 2]],
        }


@pytest.fixture
def toy_inputs():
    rgb = np.arange(144, dtype=np.uint8).reshape(6, 8, 3)
    other = np.flip(rgb, axis=2)
    originals = {"mm": rgb, "jj": other, "mj": other[::-1], "jm": rgb[::-1]}
    fills = {c: fill_color("palette_mean", originals[c]) for c in ("mm", "jj")}
    hashes = {
        c: rgb_multiset_sha256(
            pad_to_square(x, fills["mm" if c in ("mm", "jm") else "jj"])[0]
        )
        for c, x in originals.items()
    }
    payloads = {c: {"image_sizes": [[8, 8]], "input_ids": [[1, 2]]} for c in CELLS}
    return originals, fills, hashes, payloads


def test_audit_level_keeps_padding_marginal_metadata_and_inverse(toy_inputs):
    originals, fills, hashes, payloads = toy_inputs
    permutation = np.random.default_rng(3).permutation(48)
    result, canvases = audit_level(
        originals,
        permutation,
        fills,
        hashes,
        payloads,
        processor=ToyProcessor(),
        normalization={"image_mean": [0, 0, 0], "image_std": [1, 1, 1]},
    )
    for cell in CELLS:
        assert result["cells"][cell]["inverse_roundtrip_exact"]
        assert result["cells"][cell]["non_pixel_payload_unchanged"]
        assert rgb_multiset_sha256(canvases[cell]) == hashes[cell]
        assert np.array_equal(
            canvases[cell][1:7].reshape(-1, 3),
            originals[cell].reshape(-1, 3)[permutation],
        )


@pytest.mark.parametrize("corruption", ["marginal", "metadata", "missing_cell"])
def test_audit_rejects_changed_controls(toy_inputs, corruption):
    originals, fills, hashes, payloads = toy_inputs
    if corruption == "marginal":
        hashes["mm"] = "wrong"
    elif corruption == "metadata":
        payloads["mm"]["image_sizes"] = [[8, 6]]
    else:
        originals.pop("jm")
    with pytest.raises(ValueError):
        audit_level(
            originals,
            np.arange(48),
            fills,
            hashes,
            payloads,
            processor=ToyProcessor(),
            normalization={"image_mean": [0, 0, 0], "image_std": [1, 1, 1]},
        )


def test_registered_configuration_rejects_retuning():
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/graded_permutation_fastvlm_v1.json").read_text()
    )
    validate_config(config)
    for key, value in (
        ("centroid_relative_tolerance", 0.06),
        ("selection_seed", 1),
        ("level_numerators", [0, 8]),
        ("new_cache_forwards", 1),
    ):
        with pytest.raises(ValueError):
            validate_config({**config, key: value})
