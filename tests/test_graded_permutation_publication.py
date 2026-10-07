import copy

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.summarize_graded_permutation_study import (
    cycle_oracle,
    verify_gate,
    verify_pixels,
)
from fractal_vlm_state_probe.frequency_control import CELLS, rgb_multiset_sha256
from fractal_vlm_state_probe.graded_permutation import (
    array_sha256,
    restricted_permutation,
)
from fractal_vlm_state_probe.padding_policy import (
    marginal_audit,
    pad_to_square,
    padding_gate,
)


def test_independent_cycle_oracle_matches_all_registered_levels():
    rng = np.random.default_rng(2)
    full, order = rng.permutation(256), rng.permutation(256)
    for n in (0, 1, 2, 4, 6, 7, 8):
        assert np.array_equal(
            cycle_oracle(full, order, 32 * n),
            restricted_permutation(full, order, 32 * n),
        )


@pytest.fixture
def gate_cells():
    return {
        cell: {
            "processor": {
                "spectral_centroid": 0.2,
                "high_frequency_energy_ratio": 0.1,
                "luminance_std": 0.2,
            },
            "pixel_values_shape": [3, 1024, 1024],
            "registered_expanded_palette_verified": True,
            "marginal_audit": {
                "content_pixels_unchanged": True,
                "padding_pixels_equal_registered_fill": True,
                "mean_max_abs_error": 0.0,
                "variance_max_abs_error": 0.0,
            },
        }
        for cell in CELLS
    }


@pytest.mark.parametrize(
    "field,value", [(None, None), ("spectral_centroid", 0.5), ("luminance_std", 0.0)]
)
def test_independent_gate_keeps_acceptance_and_rejections(gate_cells, field, value):
    if field:
        gate_cells["mm"]["processor"][field] = value
    gate = padding_gate(gate_cells)
    verify_gate(gate_cells, gate)
    corrupt = {**gate, "accepted": not gate["accepted"]}
    with pytest.raises(ValueError):
        verify_gate(gate_cells, corrupt)


def test_moment_gate_failure_cannot_be_hidden_by_exact_histograms(gate_cells):
    gate_cells["mm"]["marginal_audit"]["mean_max_abs_error"] = 1.1e-12
    gate = padding_gate(gate_cells)
    assert not gate["accepted"]
    verify_gate(gate_cells, gate)
    with pytest.raises(ValueError):
        verify_gate(gate_cells, {**gate, "content_and_fill_verified": True})


@pytest.fixture
def pixel_case():
    rng = np.random.default_rng(8)
    source = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    color = np.array([128, 127, 129], dtype=np.uint8)
    baseline, _ = pad_to_square(source, color)
    mapping = rng.permutation(76800)
    content = source.reshape(-1, 3)[mapping].reshape(source.shape)
    canvas, bounds = pad_to_square(content, color)
    row = {
        "canvas_rgb_multiset_sha256": rgb_multiset_sha256(canvas),
        "canvas_rgb_bytes_sha256": array_sha256(canvas),
        "content_rgb_bytes_sha256": array_sha256(content),
        "changed_rgb_site_fraction": float(np.any(content != source, axis=2).mean()),
        "marginal_audit": marginal_audit(content, canvas, color, bounds),
        "registered_expanded_palette_verified": True,
        "inverse_roundtrip_exact": True,
    }
    return canvas, baseline, mapping, color, row


def test_publication_recomputes_exact_pixel_controls(pixel_case):
    result = verify_pixels(*pixel_case)
    assert set(result) == {"mean", "variance"}


@pytest.mark.parametrize("field", ["pixel", "fill", "fraction", "hash", "moment"])
def test_publication_rejects_pixel_receipt_corruption(pixel_case, field):
    canvas, baseline, mapping, color, row = copy.deepcopy(pixel_case)
    if field == "pixel":
        canvas[100, 100, 0] ^= 1
    elif field == "fill":
        canvas[0, 0, 0] ^= 1
    elif field == "fraction":
        row["changed_rgb_site_fraction"] = 0
    elif field == "hash":
        row["canvas_rgb_multiset_sha256"] = "wrong"
    else:
        row["marginal_audit"]["mean_max_abs_error"] = 0.1
    with pytest.raises(ValueError):
        verify_pixels(canvas, baseline, mapping, color, row)
