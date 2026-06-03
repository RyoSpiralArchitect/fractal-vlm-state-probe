from __future__ import annotations

import pytest

from fractal_vlm_state_probe.trace import select_layer_indices
from fractal_vlm_state_probe.trace import select_sequence_positions


def test_select_layer_indices_spreads_samples() -> None:
    assert select_layer_indices(0, 4) == []
    assert select_layer_indices(1, 4) == [0]
    assert select_layer_indices(12, 4) == [0, 4, 7, 11]
    assert select_layer_indices(12, 1) == [11]
    assert select_layer_indices(3, None) == [0, 1, 2]


def test_select_layer_indices_rejects_negative() -> None:
    with pytest.raises(ValueError, match="max_layers"):
        select_layer_indices(12, -1)


def test_select_sequence_positions() -> None:
    assert select_sequence_positions(0) == []
    assert select_sequence_positions(1) == [0]
    assert select_sequence_positions(4) == [0, 2, 3]
    assert select_sequence_positions(8) == [0, 4, 6, 7]
