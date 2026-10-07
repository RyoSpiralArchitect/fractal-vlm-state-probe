import numpy as np
import pytest

from fractal_vlm_state_probe.padding_policy import (
    fill_color,
    marginal_audit,
    measure_processor,
    pad_to_square,
    padding_gate,
)


@pytest.mark.parametrize(
    "shape,bounds", [((2, 5, 3), (1, 3, 0, 5)), ((5, 2, 3), (0, 5, 1, 3))]
)
def test_padding_keeps_exact_content_and_centers_odd_difference(shape, bounds):
    content = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    fill = np.array([12, 45, 67], dtype=np.uint8)
    canvas, actual = pad_to_square(content, fill)
    assert actual == bounds
    audit = marginal_audit(content, canvas, fill, bounds)
    assert audit["content_pixels_unchanged"]
    assert audit["padding_pixels_equal_registered_fill"]
    assert audit["mean_max_abs_error"] < 1e-14
    assert audit["variance_max_abs_error"] < 1e-14


def test_joint_histogram_change_matches_declared_mixture():
    content = np.array(
        [[[0, 0, 0], [255, 0, 0], [0, 0, 0], [255, 0, 0]]], dtype=np.uint8
    )
    fill = np.array([0, 0, 0], dtype=np.uint8)
    canvas, bounds = pad_to_square(content, fill)
    audit = marginal_audit(content, canvas, fill, bounds)
    original_red_mass = np.mean(np.all(content == [255, 0, 0], axis=2))
    canvas_red_mass = np.mean(np.all(canvas == [255, 0, 0], axis=2))
    assert (
        audit["joint_rgb_total_variation_from_content"]
        == original_red_mass - canvas_red_mass
    )
    assert audit["content_fraction"] == 0.25


def test_palette_mean_fill_is_order_invariant_and_uses_even_tie_rounding():
    palette = np.array([[[0, 1, 2], [1, 2, 3]]], dtype=np.uint8)
    np.testing.assert_array_equal(fill_color("palette_mean", palette), [0, 2, 2])
    np.testing.assert_array_equal(
        fill_color("palette_mean", palette[:, ::-1]), [0, 2, 2]
    )
    with pytest.raises(ValueError, match="unknown"):
        fill_color("adaptive_search", palette)


def test_pixel_sham_does_not_imply_full_processor_payload_equality():
    def processor(*, images, text, return_tensors):
        rgb = np.asarray(images)
        canvas, _ = pad_to_square(rgb, np.zeros(3, dtype=np.uint8))
        return {
            "pixel_values": np.moveaxis(canvas.astype(np.float32) / 255, -1, 0)[None],
            "image_sizes": [[rgb.shape[1], rgb.shape[0]]],
            "input_ids": [[-200]],
            "attention_mask": [[1]],
        }

    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    canvas, _ = pad_to_square(rgb, fill_color("black", rgb))
    norm = {"image_mean": [0, 0, 0], "image_std": [1, 1, 1]}
    before, pixels_before = measure_processor(rgb, processor, norm)
    after, pixels_after = measure_processor(canvas, processor, norm)
    assert pixels_before.tobytes() == pixels_after.tobytes()
    assert before["pixel_values_sha256"] == after["pixel_values_sha256"]
    assert before["processor_non_pixel_payload"]["image_sizes"] == [[4, 2]]
    assert after["processor_non_pixel_payload"]["image_sizes"] == [[4, 4]]


def test_padding_gate_rejects_unregistered_marginal_and_changed_content():
    def cells():
        return {
            c: {
                "processor": {
                    "spectral_centroid": 0.1,
                    "high_frequency_energy_ratio": 0.02,
                    "luminance_std": 0.2,
                },
                "pixel_values_shape": [3, 4, 4],
                "registered_expanded_palette_verified": True,
                "marginal_audit": {
                    "content_pixels_unchanged": True,
                    "padding_pixels_equal_registered_fill": True,
                    "mean_max_abs_error": 0,
                    "variance_max_abs_error": 0,
                },
            }
            for c in ("mm", "jj", "mj", "jm")
        }

    assert padding_gate(cells())["accepted"]
    changed = cells()
    changed["mj"]["registered_expanded_palette_verified"] = False
    assert not padding_gate(changed)["accepted"]
    changed = cells()
    changed["jm"]["marginal_audit"]["content_pixels_unchanged"] = False
    assert not padding_gate(changed)["accepted"]
