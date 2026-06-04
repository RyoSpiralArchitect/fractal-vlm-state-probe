from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.control_stimulus import render_blank_stimulus
from fractal_vlm_state_probe.stimulus import validate_manifest


def test_render_blank_stimulus_writes_control_manifest(tmp_path: Path) -> None:
    manifest = render_blank_stimulus(
        tmp_path,
        width=32,
        height=24,
        total_frames=3,
        fps=1.0,
        rgb=(1, 2, 3),
    )
    assert len(manifest["frames"]) == 3
    assert manifest["stimulus_condition"]["condition_family"] == "control"
    assert manifest["stimulus_condition"]["condition_id"] == "blank_visual_null"
    assert manifest["stimulus_config"]["rgb"] == [1, 2, 3]
    assert validate_manifest(tmp_path / "manifest.json") == []
