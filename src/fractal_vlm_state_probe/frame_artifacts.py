from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def prepare_frame_artifacts(
    *,
    output_path: Path,
    manifest_base: Path,
    frames: list[dict[str, Any]],
    enabled: bool,
) -> dict[int, dict[str, Any]]:
    if not enabled:
        return {}

    artifact_dir = output_path.with_suffix(output_path.suffix + ".frames")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    records: dict[int, dict[str, Any]] = {}
    for frame in frames:
        source_path = manifest_base / frame["path"]
        dest_name = f"frame_{int(frame['index']):06d}{source_path.suffix.lower() or '.png'}"
        dest_path = artifact_dir / dest_name
        shutil.copy2(source_path, dest_path)
        records[int(frame["index"])] = {
            "path": str(dest_path.relative_to(output_path.parent)),
            "source_path": frame["path"],
            "sha256": frame.get("sha256"),
            "width": frame.get("width"),
            "height": frame.get("height"),
        }
    return records

