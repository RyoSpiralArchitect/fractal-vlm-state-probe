from __future__ import annotations

import hashlib
import pytest

from fractal_vlm_state_probe.fractals import FractalSpec, generate_frame


def test_generate_frame_is_deterministic() -> None:
    spec = FractalSpec(
        kind="mandelbrot",
        width=64,
        height=48,
        total_frames=3,
        fps=1.0,
        center_start=(-0.75, 0.1),
        center_end=(-0.74, 0.12),
        scale_start=2.0,
        scale_end=0.5,
        max_iter=32,
        color_seed=3,
    )
    frame_a = generate_frame(spec, 1)
    frame_b = generate_frame(spec, 1)
    assert frame_a.shape == (48, 64, 3)
    assert hashlib.sha256(frame_a.tobytes()).hexdigest() == hashlib.sha256(
        frame_b.tobytes()
    ).hexdigest()


def test_different_frame_changes_pixels() -> None:
    spec = FractalSpec(
        kind="julia",
        width=64,
        height=48,
        total_frames=3,
        fps=1.0,
        center_start=(0.0, 0.0),
        center_end=(0.0, 0.0),
        scale_start=2.5,
        scale_end=1.0,
        max_iter=32,
        color_seed=7,
    )
    assert generate_frame(spec, 0).tobytes() != generate_frame(spec, 2).tobytes()


def test_validate_rejects_direct_invalid_kind() -> None:
    spec = FractalSpec(
        kind="burning_ship",  # type: ignore[arg-type]
        width=64,
        height=48,
        total_frames=3,
        fps=1.0,
        center_start=(0.0, 0.0),
        center_end=(0.0, 0.0),
        scale_start=2.5,
        scale_end=1.0,
        max_iter=32,
    )
    with pytest.raises(ValueError, match="unsupported fractal kind"):
        spec.validate()
