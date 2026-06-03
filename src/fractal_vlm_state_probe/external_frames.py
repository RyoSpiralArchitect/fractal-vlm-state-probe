from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .conditions import StimulusCondition
from .stimulus import sha256_file, sha256_json, write_json

SUPPORTED_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def build_external_frame_manifest(
    *,
    frames_dir: Path,
    output_path: Path,
    condition: StimulusCondition,
    fps: float,
    frame_glob: str = "*",
    relative_to_output: bool = True,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames_dir does not exist: {frames_dir}")

    frame_paths = [
        path
        for path in sorted(frames_dir.glob(frame_glob))
        if path.is_file() and path.suffix.lower() in SUPPORTED_FRAME_SUFFIXES
    ]
    if not frame_paths:
        raise ValueError(f"no supported frames found in {frames_dir} with glob {frame_glob!r}")

    condition.validate()
    frame_records = []
    base = output_path.parent if relative_to_output else Path.cwd()
    for index, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as image:
            width, height = image.size
        try:
            path_value = str(frame_path.resolve().relative_to(base.resolve()))
        except ValueError:
            path_value = str(frame_path.resolve())
        frame_records.append(
            {
                "index": index,
                "t_seconds": index / fps,
                "path": path_value,
                "sha256": sha256_file(frame_path),
                "width": width,
                "height": height,
            }
        )

    manifest = {
        "schema_version": 1,
        "generator": "fractal_vlm_state_probe.external_frames",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "stimulus_condition": condition.to_dict(),
        "stimulus_config": {
            "source_kind": condition.source_kind,
            "frames_dir": str(frames_dir),
            "frame_glob": frame_glob,
            "fps": fps,
        },
        "stimulus_config_sha256": sha256_json(
            {
                "condition": condition.to_dict(),
                "frames": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for path in frame_paths
                ],
                "fps": fps,
            }
        ),
        "frames": frame_records,
    }
    write_json(output_path, manifest)
    return manifest


def load_condition(path: Path) -> StimulusCondition:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return StimulusCondition.from_dict(data)

