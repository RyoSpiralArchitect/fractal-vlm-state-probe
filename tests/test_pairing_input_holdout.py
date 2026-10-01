from __future__ import annotations

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.analyze_pairing_input_holdout import (
    _ModelProcessorPixelView,
    input_interaction,
)


def test_input_interaction_preserves_signed_coordinate_contrast() -> None:
    cells = {
        "mm": np.array([1.0, 2.0]),
        "jj": np.array([4.0, 8.0]),
        "mj": np.array([2.0, 3.0]),
        "jm": np.array([3.0, 4.0]),
    }
    np.testing.assert_array_equal(input_interaction(cells), [0.0, 3.0])
    cells.pop("jm")
    with pytest.raises(ValueError, match="all four cells"):
        input_interaction(cells)
    cells["jm"] = np.array([1.0])
    with pytest.raises(ValueError, match="shapes differ"):
        input_interaction(cells)


def test_pixel_view_uses_the_full_model_processor_image_path() -> None:
    calls = []

    def processor(**kwargs):
        calls.append(kwargs)
        return {"pixel_values": np.ones((1, 3, 4, 4))}

    image = object()
    result = _ModelProcessorPixelView(processor)(images=image, return_tensors="np")
    assert result["pixel_values"].shape == (1, 3, 4, 4)
    assert calls == [{"images": image, "text": ["<image>"], "return_tensors": "np"}]
