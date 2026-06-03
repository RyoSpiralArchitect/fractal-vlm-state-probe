from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

FractalKind = Literal["mandelbrot", "julia"]


@dataclass(frozen=True)
class FractalSpec:
    kind: FractalKind
    width: int
    height: int
    total_frames: int
    fps: float
    center_start: tuple[float, float]
    center_end: tuple[float, float]
    scale_start: float
    scale_end: float
    max_iter: int
    color_seed: int = 0
    julia_c: tuple[float, float] = (-0.7269, 0.1889)
    schema_version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FractalSpec":
        required = {
            "kind",
            "width",
            "height",
            "total_frames",
            "fps",
            "center_start",
            "center_end",
            "scale_start",
            "scale_end",
            "max_iter",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"missing fractal config keys: {', '.join(missing)}")

        known = required | {"schema_version", "color_seed", "julia_c"}
        extra = {key: value for key, value in data.items() if key not in known}
        kind = data["kind"]
        if kind not in ("mandelbrot", "julia"):
            raise ValueError(f"unsupported fractal kind: {kind}")

        return cls(
            schema_version=int(data.get("schema_version", 1)),
            kind=kind,
            width=int(data["width"]),
            height=int(data["height"]),
            total_frames=int(data["total_frames"]),
            fps=float(data["fps"]),
            center_start=_pair(data["center_start"], "center_start"),
            center_end=_pair(data["center_end"], "center_end"),
            scale_start=float(data["scale_start"]),
            scale_end=float(data["scale_end"]),
            max_iter=int(data["max_iter"]),
            color_seed=int(data.get("color_seed", 0)),
            julia_c=_pair(data.get("julia_c", (-0.7269, 0.1889)), "julia_c"),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "center_start": list(self.center_start),
            "center_end": list(self.center_end),
            "scale_start": self.scale_start,
            "scale_end": self.scale_end,
            "max_iter": self.max_iter,
            "color_seed": self.color_seed,
        }
        if self.kind == "julia":
            data["julia_c"] = list(self.julia_c)
        data.update(self.extra)
        return data

    def validate(self) -> None:
        if self.kind not in ("mandelbrot", "julia"):
            raise ValueError(f"unsupported fractal kind: {self.kind}")
        if self.width < 8 or self.height < 8:
            raise ValueError("width and height must be at least 8")
        if self.total_frames < 1:
            raise ValueError("total_frames must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.scale_start <= 0 or self.scale_end <= 0:
            raise ValueError("scale values must be positive")
        if self.max_iter < 8:
            raise ValueError("max_iter must be at least 8")


def generate_frame(spec: FractalSpec, frame_index: int) -> np.ndarray:
    """Generate one deterministic RGB fractal frame."""
    spec.validate()
    if not 0 <= frame_index < spec.total_frames:
        raise IndexError(f"frame_index {frame_index} outside 0..{spec.total_frames - 1}")

    center, scale = frame_view(spec, frame_index)
    aspect = spec.width / spec.height
    real = np.linspace(
        center[0] - scale * aspect / 2.0,
        center[0] + scale * aspect / 2.0,
        spec.width,
        dtype=np.float64,
    )
    imag = np.linspace(
        center[1] - scale / 2.0,
        center[1] + scale / 2.0,
        spec.height,
        dtype=np.float64,
    )
    grid = real[None, :] + 1j * imag[:, None]

    if spec.kind == "mandelbrot":
        c = grid
        z = np.zeros_like(c)
    else:
        c = complex(*spec.julia_c)
        z = grid.copy()

    counts = np.full(grid.shape, spec.max_iter, dtype=np.float32)
    mask = np.ones(grid.shape, dtype=bool)
    smooth = np.zeros(grid.shape, dtype=np.float32)

    for iteration in range(spec.max_iter):
        z[mask] = z[mask] * z[mask] + c if np.isscalar(c) else z[mask] * z[mask] + c[mask]
        escaped = mask & (np.abs(z) > 2.0)
        if escaped.any():
            counts[escaped] = iteration
            # Smooth escape count for less banding.
            abs_z = np.abs(z[escaped])
            smooth[escaped] = iteration + 1.0 - np.log2(np.log2(np.maximum(abs_z, 2.000001)))
        mask[escaped] = False
        if not mask.any():
            break

    normalized = np.clip(smooth / max(1, spec.max_iter), 0.0, 1.0)
    normalized[counts >= spec.max_iter] = 0.0
    rgb = cosine_palette(normalized, counts < spec.max_iter, spec.color_seed)
    return rgb


def frame_view(spec: FractalSpec, frame_index: int) -> tuple[tuple[float, float], float]:
    if spec.total_frames == 1:
        alpha = 0.0
    else:
        alpha = frame_index / (spec.total_frames - 1)

    cx = _lerp(spec.center_start[0], spec.center_end[0], alpha)
    cy = _lerp(spec.center_start[1], spec.center_end[1], alpha)
    log_scale = _lerp(np.log(spec.scale_start), np.log(spec.scale_end), alpha)
    return (float(cx), float(cy)), float(np.exp(log_scale))


def cosine_palette(values: np.ndarray, escaped: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 1.0, size=3)
    contrast = rng.uniform(0.35, 0.5, size=3)
    offset = rng.uniform(0.42, 0.58, size=3)
    channels = []
    for channel in range(3):
        wave = offset[channel] + contrast[channel] * np.cos(
            2.0 * np.pi * (values + phases[channel])
        )
        channels.append(np.clip(wave, 0.0, 1.0))
    rgb = np.stack(channels, axis=-1)
    rgb[~escaped] = np.array([0.015, 0.012, 0.025])
    return (rgb * 255.0).astype(np.uint8)


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-item list or tuple")
    return float(value[0]), float(value[1])


def _lerp(start: float, end: float, alpha: float) -> float:
    return start * (1.0 - alpha) + end * alpha
