from __future__ import annotations

from pathlib import Path

from PIL import Image

from fractal_vlm_state_probe.frame_artifacts import prepare_frame_artifacts


def test_prepare_frame_artifacts_copies_selected_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_path = frames_dir / "frame_000000.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(frame_path)

    records = prepare_frame_artifacts(
        output_path=tmp_path / "run.json",
        manifest_base=tmp_path,
        frames=[
            {
                "index": 0,
                "path": "frames/frame_000000.png",
                "sha256": "abc",
                "width": 8,
                "height": 8,
            }
        ],
        enabled=True,
    )

    copied = tmp_path / records[0]["path"]
    assert copied.exists()
    assert records[0]["source_path"] == "frames/frame_000000.png"
