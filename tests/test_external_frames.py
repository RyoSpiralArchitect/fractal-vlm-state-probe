from __future__ import annotations

from pathlib import Path

from PIL import Image

from fractal_vlm_state_probe.conditions import StimulusCondition
from fractal_vlm_state_probe.external_frames import build_external_frame_manifest


def test_build_external_frame_manifest(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (12, 10), color=(20, 30, 40)).save(frames_dir / "000.png")
    Image.new("RGB", (12, 10), color=(50, 60, 70)).save(frames_dir / "001.png")

    condition = StimulusCondition(
        condition_id="natural_city_ordered",
        condition_family="natural",
        temporal_policy="ordered",
        semantic_load="high",
        deterministic=False,
        source_kind="external_frames",
    )
    manifest = build_external_frame_manifest(
        frames_dir=frames_dir,
        output_path=tmp_path / "manifest.json",
        condition=condition,
        fps=2.0,
    )
    assert manifest["stimulus_condition"]["condition_id"] == "natural_city_ordered"
    assert manifest["frames"][1]["t_seconds"] == 0.5
    assert len(manifest["frames"][0]["sha256"]) == 64
