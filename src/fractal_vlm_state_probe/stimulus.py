from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .conditions import condition_from_config
from .fractals import FractalSpec, generate_frame


def load_spec(path: Path) -> FractalSpec:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return FractalSpec.from_dict(data)


def render_stimulus(
    spec: FractalSpec,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render frames and write a manifest with relative paths and hashes."""
    spec.validate()
    condition = condition_from_config(spec.to_dict(), fallback_kind=spec.kind)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for frame_index in range(spec.total_frames):
        filename = f"frame_{frame_index:06d}.png"
        frame_path = frames_dir / filename
        if frame_path.exists() and not overwrite:
            raise FileExistsError(f"{frame_path} exists; pass overwrite=True to replace")

        array = generate_frame(spec, frame_index)
        Image.fromarray(array).save(frame_path)
        frame_records.append(
            {
                "index": frame_index,
                "t_seconds": frame_index / spec.fps,
                "path": str(Path("frames") / filename),
                "sha256": sha256_file(frame_path),
                "width": spec.width,
                "height": spec.height,
            }
        )

    manifest = {
        "schema_version": 1,
        "generator": "fractal_vlm_state_probe",
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "stimulus_condition": condition.to_dict(),
        "stimulus_config": spec.to_dict(),
        "stimulus_config_sha256": sha256_json(spec.to_dict()),
        "frames": frame_records,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_manifest(manifest_path: Path) -> list[str]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    issues: list[str] = []
    base = manifest_path.parent
    for frame in manifest.get("frames", []):
        frame_path = base / frame["path"]
        if not frame_path.exists():
            issues.append(f"missing frame: {frame['path']}")
            continue
        actual = sha256_file(frame_path)
        if actual != frame.get("sha256"):
            issues.append(f"hash mismatch: {frame['path']}")
    return issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
