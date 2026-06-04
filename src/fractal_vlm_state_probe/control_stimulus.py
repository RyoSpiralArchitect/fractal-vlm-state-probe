from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .conditions import StimulusCondition
from .stimulus import sha256_file, sha256_json, write_json


def render_blank_stimulus(
    output_dir: Path,
    *,
    width: int,
    height: int,
    total_frames: int,
    fps: float,
    rgb: tuple[int, int, int] = (0, 0, 0),
    condition_id: str = "blank_visual_null",
    overwrite: bool = False,
) -> dict[str, Any]:
    _validate_blank_spec(width=width, height=height, total_frames=total_frames, fps=fps, rgb=rgb)
    condition = StimulusCondition(
        condition_id=condition_id,
        condition_family="control",
        temporal_policy="static_repeat",
        semantic_load="none",
        deterministic=True,
        source_kind="generated",
        comparison_role="blank_visual_null",
        description="Generated blank visual stream used as a null image-token control.",
    )
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for frame_index in range(total_frames):
        filename = f"frame_{frame_index:06d}.png"
        frame_path = frames_dir / filename
        if frame_path.exists() and not overwrite:
            raise FileExistsError(f"{frame_path} exists; pass overwrite=True to replace")
        Image.new("RGB", (width, height), color=rgb).save(frame_path)
        frame_records.append(
            {
                "index": frame_index,
                "t_seconds": frame_index / fps,
                "path": str(Path("frames") / filename),
                "sha256": sha256_file(frame_path),
                "width": width,
                "height": height,
            }
        )

    stimulus_config = {
        "schema_version": 1,
        "kind": "blank_visual",
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "fps": fps,
        "rgb": list(rgb),
        "stimulus_condition": condition.to_dict(),
    }
    manifest = {
        "schema_version": 1,
        "generator": "fractal_vlm_state_probe",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "stimulus_condition": condition.to_dict(),
        "stimulus_config": stimulus_config,
        "stimulus_config_sha256": sha256_json(stimulus_config),
        "frames": frame_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def _validate_blank_spec(
    *,
    width: int,
    height: int,
    total_frames: int,
    fps: float,
    rgb: tuple[int, int, int],
) -> None:
    if width < 8 or height < 8:
        raise ValueError("width and height must be at least 8")
    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(rgb) != 3 or any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("rgb must be three integers in 0..255")
