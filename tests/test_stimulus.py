from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.fractals import FractalSpec
from fractal_vlm_state_probe.stimulus import render_stimulus, validate_manifest


def test_render_stimulus_writes_valid_manifest(tmp_path: Path) -> None:
    spec = FractalSpec(
        kind="mandelbrot",
        width=48,
        height=36,
        total_frames=2,
        fps=2.0,
        center_start=(-0.75, 0.1),
        center_end=(-0.74, 0.12),
        scale_start=2.0,
        scale_end=1.0,
        max_iter=24,
        color_seed=11,
    )
    manifest = render_stimulus(spec, tmp_path)
    assert len(manifest["frames"]) == 2
    assert manifest["frames"][1]["t_seconds"] == 0.5
    assert manifest["stimulus_condition"]["condition_family"] == "fractal"
    assert validate_manifest(tmp_path / "manifest.json") == []

