import numpy as np
import pytest

from fractal_vlm_state_probe.cli.analyze_frequency_padding_audit import (
    padding_variance_decomposition,
    support_bounds,
)


def test_support_requires_nonempty_rectangular_probe():
    probe = np.zeros((3, 8, 8))
    probe[:, 2:6, :] = 1
    assert support_bounds(probe) == (2, 6, 0, 8)
    probe[:, 3, 3] = 0
    with pytest.raises(ValueError, match="rectangular"):
        support_bounds(probe)
    with pytest.raises(ValueError, match="no active"):
        support_bounds(np.zeros_like(probe))


def test_black_padding_variance_decomposition_is_exact():
    rgb = np.zeros((8, 8, 3))
    rgb[2:6] = 0.5
    result = padding_variance_decomposition(rgb, (2, 6, 0, 8))
    assert result["black_padding_fraction"] == 0.5
    assert result["between_region_variance_fraction"] == 1
    assert result["variance_decomposition_abs_error"] < 1e-15
    rgb[0, 0] = 0.1
    with pytest.raises(ValueError, match="not exactly black"):
        padding_variance_decomposition(rgb, (2, 6, 0, 8))
