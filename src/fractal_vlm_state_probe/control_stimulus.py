from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import cos, pi, sin, sqrt
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

from .conditions import StimulusCondition
from .stimulus import sha256_file, sha256_json, write_json

GeneratedControlKind = Literal[
    "blank",
    "white_noise",
    "blue_noise",
    "random_dots",
    "checkerboard",
    "square_tiling",
    "triangle_tiling",
    "hex_tiling",
    "voronoi",
    "quasicrystal",
]
TransformControlKind = Literal[
    "phase_scrambled",
    "static_repeat",
    "shuffled",
    "reversed",
]

GENERATED_CONTROL_KINDS: tuple[str, ...] = (
    "blank",
    "white_noise",
    "blue_noise",
    "random_dots",
    "checkerboard",
    "square_tiling",
    "triangle_tiling",
    "hex_tiling",
    "voronoi",
    "quasicrystal",
)
TRANSFORM_CONTROL_KINDS: tuple[str, ...] = (
    "phase_scrambled",
    "static_repeat",
    "shuffled",
    "reversed",
)


@dataclass(frozen=True)
class GeneratedControlSpec:
    kind: GeneratedControlKind
    width: int
    height: int
    total_frames: int
    fps: float
    seed: int = 0
    rgb: tuple[int, int, int] = (0, 0, 0)
    cell_size: int = 24
    sites: int = 32
    dot_density: float = 0.035
    motion_speed: float = 1.0
    condition_id: str | None = None
    schema_version: int = 1

    def validate(self) -> None:
        if self.kind not in GENERATED_CONTROL_KINDS:
            raise ValueError(f"unsupported generated control kind: {self.kind}")
        if self.width < 8 or self.height < 8:
            raise ValueError("width and height must be at least 8")
        if self.total_frames < 1:
            raise ValueError("total_frames must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if len(self.rgb) != 3 or any(channel < 0 or channel > 255 for channel in self.rgb):
            raise ValueError("rgb must be three integers in 0..255")
        if self.cell_size < 2:
            raise ValueError("cell_size must be at least 2")
        if self.sites < 2:
            raise ValueError("sites must be at least 2")
        if not 0.0 < self.dot_density <= 0.5:
            raise ValueError("dot_density must be in (0, 0.5]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "seed": self.seed,
            "rgb": list(self.rgb),
            "cell_size": self.cell_size,
            "sites": self.sites,
            "dot_density": self.dot_density,
            "motion_speed": self.motion_speed,
            "condition_id": self.condition_id,
        }


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
    spec = GeneratedControlSpec(
        kind="blank",
        width=width,
        height=height,
        total_frames=total_frames,
        fps=fps,
        rgb=rgb,
        condition_id=condition_id,
    )
    return render_generated_control_stimulus(spec, output_dir, overwrite=overwrite)


def render_generated_control_stimulus(
    spec: GeneratedControlSpec,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    spec.validate()
    condition = _generated_condition(spec)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for frame_index in range(spec.total_frames):
        filename = f"frame_{frame_index:06d}.png"
        frame_path = frames_dir / filename
        if frame_path.exists() and not overwrite:
            raise FileExistsError(f"{frame_path} exists; pass overwrite=True to replace")

        array = generate_control_frame(spec, frame_index)
        Image.fromarray(array).save(frame_path)
        frame_records.append(_frame_record(frame_index, spec.fps, frame_path, output_dir, spec.width, spec.height))

    stimulus_config = spec.to_dict()
    stimulus_config["stimulus_condition"] = condition.to_dict()
    manifest = _manifest(
        generator="fractal_vlm_state_probe.control_stimulus",
        condition=condition,
        stimulus_config=stimulus_config,
        frames=frame_records,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def render_manifest_transform(
    output_dir: Path,
    *,
    source_manifest_path: Path,
    transform_kind: TransformControlKind,
    seed: int = 0,
    source_frame_index: int = 0,
    max_frames: int | None = None,
    condition_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if transform_kind not in TRANSFORM_CONTROL_KINDS:
        raise ValueError(f"unsupported transform control kind: {transform_kind}")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")

    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    source_frames = list(source_manifest.get("frames") or [])
    if not source_frames:
        raise ValueError(f"source manifest contains no frames: {source_manifest_path}")

    selected = source_frames[:max_frames] if max_frames is not None else source_frames
    order = _transform_order(
        transform_kind,
        selected,
        seed=seed,
        source_frame_index=source_frame_index,
    )
    fps = _manifest_fps(source_manifest)
    condition = _transform_condition(
        source_manifest=source_manifest,
        transform_kind=transform_kind,
        condition_id=condition_id,
    )
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records = []
    for output_index, source_record in enumerate(order):
        filename = f"frame_{output_index:06d}.png"
        output_frame_path = frames_dir / filename
        if output_frame_path.exists() and not overwrite:
            raise FileExistsError(f"{output_frame_path} exists; pass overwrite=True to replace")

        source_path = _resolve_source_frame_path(source_manifest_path, source_record)
        with Image.open(source_path) as image:
            array = np.asarray(image.convert("RGB"))

        if transform_kind == "phase_scrambled":
            array = phase_scramble_rgb(array, seed=seed + output_index)

        Image.fromarray(array).save(output_frame_path)
        frame_record = _frame_record(
            output_index,
            fps,
            output_frame_path,
            output_dir,
            int(array.shape[1]),
            int(array.shape[0]),
        )
        frame_record["source_index"] = source_record.get("index")
        frame_record["source_path"] = source_record.get("path")
        frame_record["source_sha256"] = source_record.get("sha256")
        frame_records.append(frame_record)

    stimulus_config = {
        "schema_version": 1,
        "kind": transform_kind,
        "seed": seed,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_condition": source_manifest.get("stimulus_condition"),
        "source_frame_index": source_frame_index,
        "max_frames": max_frames,
        "stimulus_condition": condition.to_dict(),
    }
    manifest = _manifest(
        generator="fractal_vlm_state_probe.control_stimulus",
        condition=condition,
        stimulus_config=stimulus_config,
        frames=frame_records,
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def generate_control_frame(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    spec.validate()
    if not 0 <= frame_index < spec.total_frames:
        raise IndexError(f"frame_index {frame_index} outside 0..{spec.total_frames - 1}")

    if spec.kind == "blank":
        array = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
        array[:, :] = np.array(spec.rgb, dtype=np.uint8)
        return array
    if spec.kind == "white_noise":
        return _white_noise(spec, frame_index)
    if spec.kind == "blue_noise":
        return _blue_noise(spec, frame_index)
    if spec.kind == "random_dots":
        return _random_dots(spec, frame_index)
    if spec.kind == "checkerboard":
        return _checkerboard(spec, frame_index)
    if spec.kind == "square_tiling":
        return _square_tiling(spec, frame_index)
    if spec.kind == "triangle_tiling":
        return _triangle_tiling(spec, frame_index)
    if spec.kind == "hex_tiling":
        return _hex_tiling(spec, frame_index)
    if spec.kind == "voronoi":
        return _voronoi(spec, frame_index)
    if spec.kind == "quasicrystal":
        return _quasicrystal(spec, frame_index)
    raise ValueError(f"unsupported generated control kind: {spec.kind}")


def phase_scramble_rgb(array: np.ndarray, *, seed: int) -> np.ndarray:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("phase_scramble_rgb expects an RGB array")
    rng = np.random.default_rng(seed)
    output = np.empty_like(array, dtype=np.float64)
    for channel in range(3):
        source = array[:, :, channel].astype(np.float64)
        spectrum = np.fft.rfft2(source)
        amplitude = np.abs(spectrum)
        phase = rng.uniform(-pi, pi, size=amplitude.shape)
        scrambled = np.fft.irfft2(amplitude * np.exp(1j * phase), s=source.shape)
        output[:, :, channel] = _match_mean_std(scrambled, source)
    return np.clip(output, 0, 255).astype(np.uint8)


def _generated_condition(spec: GeneratedControlSpec) -> StimulusCondition:
    if spec.kind == "blank":
        return StimulusCondition(
            condition_id=spec.condition_id or "blank_visual_null",
            condition_family="control",
            temporal_policy="static_repeat",
            semantic_load="none",
            deterministic=True,
            source_kind="generated",
            comparison_role="blank_visual_null",
            description="Generated blank visual stream used as a null image-token control.",
        )
    if spec.kind in ("white_noise", "blue_noise", "random_dots"):
        family = "control"
        semantic_load = "none"
        role = "low_level_visual_control"
    else:
        family = "geometric"
        semantic_load = "low"
        role = "non_fractal_geometry_control"
    return StimulusCondition(
        condition_id=spec.condition_id or f"{spec.kind}_seed_{spec.seed}",
        condition_family=family,
        temporal_policy="ordered",
        semantic_load=semantic_load,
        deterministic=True,
        source_kind="generated",
        comparison_role=role,
        description=f"Deterministic {spec.kind} control stream.",
    )


def _transform_condition(
    *,
    source_manifest: dict[str, Any],
    transform_kind: str,
    condition_id: str | None,
) -> StimulusCondition:
    raw_source = source_manifest.get("stimulus_condition") or {}
    source = StimulusCondition.from_dict(raw_source)
    source_id = source.condition_id
    if transform_kind == "phase_scrambled":
        return StimulusCondition(
            condition_id=condition_id or f"{source_id}_phase_scrambled",
            condition_family="control",
            temporal_policy="ordered",
            semantic_load="none",
            deterministic=True,
            source_kind="generated",
            comparison_role="phase_scrambled_visual_control",
            description=f"Phase-scrambled transform of {source_id}; approximate spectrum-preserving low-level control.",
        )
    temporal_policy = {
        "static_repeat": "static_repeat",
        "shuffled": "shuffled",
        "reversed": "reversed",
    }[transform_kind]
    return StimulusCondition(
        condition_id=condition_id or f"{source_id}_{transform_kind}",
        condition_family=source.condition_family,
        temporal_policy=temporal_policy,
        semantic_load=source.semantic_load,
        deterministic=True,
        source_kind="generated",
        comparison_role=f"{transform_kind}_temporal_control",
        description=f"{transform_kind} temporal transform of {source_id}.",
    )


def _transform_order(
    transform_kind: str,
    selected: list[dict[str, Any]],
    *,
    seed: int,
    source_frame_index: int,
) -> list[dict[str, Any]]:
    if transform_kind == "static_repeat":
        if not 0 <= source_frame_index < len(selected):
            raise IndexError(f"source_frame_index {source_frame_index} outside 0..{len(selected) - 1}")
        return [selected[source_frame_index] for _ in selected]
    if transform_kind == "shuffled":
        rng = np.random.default_rng(seed)
        return [selected[int(index)] for index in rng.permutation(len(selected))]
    if transform_kind == "reversed":
        return list(reversed(selected))
    return selected


def _white_noise(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    rng = np.random.default_rng(_frame_seed(spec.seed, frame_index, 101))
    return rng.integers(0, 256, size=(spec.height, spec.width, 3), dtype=np.uint8)


def _blue_noise(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    rng = np.random.default_rng(_frame_seed(spec.seed, frame_index, 211))
    noise = rng.random((spec.height, spec.width, 3), dtype=np.float64)
    high_pass = noise - _box_blur(noise)
    return _normalize_to_uint8(high_pass)


def _random_dots(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    rng = np.random.default_rng(_frame_seed(spec.seed, frame_index, 307))
    array = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
    array[:, :] = np.array([8, 8, 12], dtype=np.uint8)
    count = max(1, int(spec.width * spec.height * spec.dot_density))
    radius = max(1, spec.cell_size // 12)
    colors = _palette(spec.seed + frame_index, count)
    xs = rng.integers(0, spec.width, size=count)
    ys = rng.integers(0, spec.height, size=count)
    for index, (x, y) in enumerate(zip(xs, ys)):
        x0 = max(0, int(x) - radius)
        x1 = min(spec.width, int(x) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(spec.height, int(y) + radius + 1)
        array[y0:y1, x0:x1] = colors[index]
    return array


def _checkerboard(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    yy, xx = np.mgrid[0 : spec.height, 0 : spec.width]
    cell = max(2, spec.cell_size)
    phase = int(frame_index * spec.motion_speed) % (2 * cell)
    mask = (((xx + phase) // cell) + ((yy + phase) // cell)) % 2
    colors = _palette(spec.seed, 2)
    return colors[mask].astype(np.uint8)


def _square_tiling(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    image = _line_image(spec)
    draw = ImageDraw.Draw(image)
    cell = max(4, spec.cell_size)
    offset = int(frame_index * spec.motion_speed) % cell
    line_color = tuple(int(value) for value in _palette(spec.seed, 1)[0])
    for x in range(-cell, spec.width + cell, cell):
        draw.line((x + offset, 0, x + offset, spec.height), fill=line_color, width=2)
    for y in range(-cell, spec.height + cell, cell):
        draw.line((0, y + offset, spec.width, y + offset), fill=line_color, width=2)
    return np.asarray(image, dtype=np.uint8)


def _triangle_tiling(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    image = _line_image(spec)
    draw = ImageDraw.Draw(image)
    cell = max(6, spec.cell_size)
    offset = int(frame_index * spec.motion_speed) % cell
    height = max(2, int(cell * sqrt(3.0) / 2.0))
    line_color = tuple(int(value) for value in _palette(spec.seed + 1, 1)[0])
    for y in range(-height, spec.height + height, height):
        draw.line((0, y + offset, spec.width, y + offset), fill=line_color, width=2)
    diagonal_span = int(spec.height / sqrt(3.0)) + cell
    for x in range(-spec.width, spec.width * 2, cell):
        draw.line((x + offset, 0, x + diagonal_span + offset, spec.height), fill=line_color, width=2)
        draw.line((x + offset, spec.height, x + diagonal_span + offset, 0), fill=line_color, width=2)
    return np.asarray(image, dtype=np.uint8)


def _hex_tiling(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    image = _line_image(spec)
    draw = ImageDraw.Draw(image)
    radius = max(4, spec.cell_size // 2)
    dx = sqrt(3.0) * radius
    dy = 1.5 * radius
    offset = frame_index * spec.motion_speed
    line_color = tuple(int(value) for value in _palette(spec.seed + 2, 1)[0])
    row = -2
    y = -2 * dy + offset % dy
    while y < spec.height + 2 * dy:
        x_shift = (row % 2) * dx / 2.0
        x = -2 * dx + x_shift + offset % dx
        while x < spec.width + 2 * dx:
            points = [
                (
                    x + radius * cos(pi / 6.0 + side * pi / 3.0),
                    y + radius * sin(pi / 6.0 + side * pi / 3.0),
                )
                for side in range(6)
            ]
            draw.line(points + [points[0]], fill=line_color, width=2)
            x += dx
        y += dy
        row += 1
    return np.asarray(image, dtype=np.uint8)


def _voronoi(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    base_points = rng.random((spec.sites, 2))
    angles = rng.uniform(0.0, 2.0 * pi, size=spec.sites)
    alpha = frame_index / max(1, spec.total_frames - 1)
    drift = 0.08 * np.stack(
        [
            np.cos(2.0 * pi * alpha + angles),
            np.sin(2.0 * pi * alpha + angles),
        ],
        axis=1,
    )
    points = np.mod(base_points + drift, 1.0)
    points[:, 0] *= spec.width
    points[:, 1] *= spec.height
    yy, xx = np.mgrid[0 : spec.height, 0 : spec.width]
    best = np.full((spec.height, spec.width), np.inf, dtype=np.float64)
    labels = np.zeros((spec.height, spec.width), dtype=np.int32)
    for index, (px, py) in enumerate(points):
        distance = (xx - px) ** 2 + (yy - py) ** 2
        mask = distance < best
        labels[mask] = index
        best[mask] = distance[mask]
    colors = _palette(spec.seed + 3, spec.sites)
    return colors[labels].astype(np.uint8)


def _quasicrystal(spec: GeneratedControlSpec, frame_index: int) -> np.ndarray:
    yy, xx = np.mgrid[0 : spec.height, 0 : spec.width]
    scale = max(4.0, float(spec.cell_size))
    x = (xx - spec.width / 2.0) / scale
    y = (yy - spec.height / 2.0) / scale
    phase = 2.0 * pi * frame_index / max(1, spec.total_frames)
    field = np.zeros((spec.height, spec.width), dtype=np.float64)
    for index in range(5):
        theta = index * pi / 5.0
        field += np.cos(x * cos(theta) + y * sin(theta) + phase * (1.0 + index * 0.07))
    values = (field - field.min()) / max(1e-12, field.max() - field.min())
    return _scalar_palette(values, spec.seed + 5)


def _line_image(spec: GeneratedControlSpec) -> Image.Image:
    base = tuple(int(value) for value in _palette(spec.seed + 17, 1)[0] // 5)
    return Image.new("RGB", (spec.width, spec.height), color=base)


def _box_blur(array: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(array, dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += np.roll(np.roll(array, dy, axis=0), dx, axis=1)
    return acc / 9.0


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    low = array.min(axis=(0, 1), keepdims=True)
    high = array.max(axis=(0, 1), keepdims=True)
    normalized = (array - low) / np.maximum(1e-12, high - low)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _match_mean_std(array: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref_std = float(reference.std())
    if ref_std < 1e-12:
        return np.full_like(array, float(reference.mean()))
    std = float(array.std())
    if std < 1e-12:
        return np.full_like(array, float(reference.mean()))
    return (array - float(array.mean())) / std * ref_std + float(reference.mean())


def _scalar_palette(values: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 1.0, size=3)
    contrast = rng.uniform(0.35, 0.5, size=3)
    offset = rng.uniform(0.42, 0.58, size=3)
    channels = [
        np.clip(offset[channel] + contrast[channel] * np.cos(2.0 * pi * (values + phases[channel])), 0.0, 1.0)
        for channel in range(3)
    ]
    return (np.stack(channels, axis=-1) * 255.0).astype(np.uint8)


def _palette(seed: int, count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1.0, size=(count, 3))
    return (0.18 * 255.0 + base * 0.72 * 255.0).astype(np.uint8)


def _frame_seed(seed: int, frame_index: int, salt: int) -> int:
    return int((seed * 1_000_003 + frame_index * 9_176 + salt) % (2**32 - 1))


def _frame_record(
    frame_index: int,
    fps: float,
    frame_path: Path,
    output_dir: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "index": frame_index,
        "t_seconds": frame_index / fps,
        "path": str(frame_path.relative_to(output_dir)),
        "sha256": sha256_file(frame_path),
        "width": width,
        "height": height,
    }


def _manifest(
    *,
    generator: str,
    condition: StimulusCondition,
    stimulus_config: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generator": generator,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "stimulus_condition": condition.to_dict(),
        "stimulus_config": stimulus_config,
        "stimulus_config_sha256": sha256_json(stimulus_config),
        "frames": frames,
    }


def _resolve_source_frame_path(source_manifest_path: Path, frame_record: dict[str, Any]) -> Path:
    raw = Path(str(frame_record["path"]))
    if raw.is_absolute():
        return raw
    return source_manifest_path.parent / raw


def _manifest_fps(manifest: dict[str, Any]) -> float:
    raw = (manifest.get("stimulus_config") or {}).get("fps")
    if raw is not None:
        return float(raw)
    frames = manifest.get("frames") or []
    if len(frames) >= 2:
        delta = float(frames[1].get("t_seconds", 1.0)) - float(frames[0].get("t_seconds", 0.0))
        if delta > 0:
            return 1.0 / delta
    return 1.0
