from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
from PIL import Image

from .stimulus import sha256_file, write_json

SCALAR_FRAME_FIELDS = (
    "luminance_mean",
    "luminance_std",
    "luminance_entropy_bits",
    "edge_density",
    "edge_strength_mean",
    "spectral_centroid",
    "high_frequency_energy_ratio",
    "colorfulness",
    "mean_abs_luminance_delta_from_previous",
)


def analyze_manifest_image_stats(
    manifest_path: Path,
    *,
    max_frames: int | None = None,
    include_frame_stats: bool = True,
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    frames = list(manifest.get("frames") or [])
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"manifest contains no frames: {manifest_path}")

    frame_stats = []
    previous_luminance: np.ndarray | None = None
    for frame in frames:
        frame_path = _resolve_frame_path(manifest_path, frame)
        with Image.open(frame_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        stats = _frame_image_stats(rgb)
        stats["index"] = int(frame["index"])
        stats["t_seconds"] = float(frame["t_seconds"])
        stats["path"] = str(frame.get("path"))
        stats["sha256"] = str(frame.get("sha256") or sha256_file(frame_path))

        luminance = _luminance(rgb)
        if previous_luminance is None:
            stats["mean_abs_luminance_delta_from_previous"] = None
        else:
            stats["mean_abs_luminance_delta_from_previous"] = float(np.mean(np.abs(luminance - previous_luminance)))
        previous_luminance = luminance
        frame_stats.append(stats)

    aggregate = _aggregate_frame_stats(frame_stats)
    record = {
        "schema_version": 1,
        "analysis_kind": "manifest_image_statistics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "condition": manifest.get("stimulus_condition"),
        "stimulus_config_sha256": manifest.get("stimulus_config_sha256"),
        "frame_count_available": len(manifest.get("frames") or []),
        "frame_count_analyzed": len(frame_stats),
        "aggregate": aggregate,
    }
    if include_frame_stats:
        record["frame_stats"] = frame_stats
    return record


def analyze_manifest_batch(
    manifest_paths: list[Path],
    *,
    max_frames: int | None = None,
    include_frame_stats: bool = False,
) -> dict[str, Any]:
    records = [
        analyze_manifest_image_stats(
            path,
            max_frames=max_frames,
            include_frame_stats=include_frame_stats,
        )
        for path in manifest_paths
    ]
    return {
        "schema_version": 1,
        "analysis_kind": "image_statistics_batch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "manifest_count": len(records),
        "records": records,
        "pairwise_deltas": _pairwise_deltas(records),
    }


def write_image_stats_json(analysis: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, analysis)


def write_image_stats_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_image_stats_markdown(analysis), encoding="utf-8")


def format_image_stats_markdown(analysis: dict[str, Any]) -> str:
    records = analysis["records"] if analysis.get("analysis_kind") == "image_statistics_batch" else [analysis]
    lines = [
        "# Image Statistics",
        "",
        f"- Manifests: `{len(records)}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Condition | Family | Frames | Lum mean | Lum std | Entropy | Edge density | Edge strength | HF ratio | Spectral centroid | Temporal delta |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        condition = record.get("condition") or {}
        aggregate = record["aggregate"]
        lines.append(
            "| "
            f"`{condition.get('condition_id')}` | "
            f"`{condition.get('condition_family')}` | "
            f"{record['frame_count_analyzed']} | "
            f"{_fmt(aggregate['luminance_mean']['mean'])} | "
            f"{_fmt(aggregate['luminance_std']['mean'])} | "
            f"{_fmt(aggregate['luminance_entropy_bits']['mean'])} | "
            f"{_fmt(aggregate['edge_density']['mean'])} | "
            f"{_fmt(aggregate['edge_strength_mean']['mean'])} | "
            f"{_fmt(aggregate['high_frequency_energy_ratio']['mean'])} | "
            f"{_fmt(aggregate['spectral_centroid']['mean'])} | "
            f"{_fmt(aggregate['mean_abs_luminance_delta_from_previous']['mean'])} |"
        )

    if analysis.get("analysis_kind") == "image_statistics_batch":
        lines.extend(
            [
                "",
                "## Pairwise Deltas",
                "",
                "| Left | Right | Lum mean | Lum std | Entropy | Edge density | HF ratio | Spectral centroid |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in analysis["pairwise_deltas"]:
            deltas = item["aggregate_mean_abs_deltas"]
            lines.append(
                "| "
                f"`{item['left_condition_id']}` | "
                f"`{item['right_condition_id']}` | "
                f"{_fmt(deltas['luminance_mean'])} | "
                f"{_fmt(deltas['luminance_std'])} | "
                f"{_fmt(deltas['luminance_entropy_bits'])} | "
                f"{_fmt(deltas['edge_density'])} | "
                f"{_fmt(deltas['high_frequency_energy_ratio'])} | "
                f"{_fmt(deltas['spectral_centroid'])} |"
            )

    lines.extend(
        [
            "",
            "## Reading Note",
            "",
            "These are low-level image statistics for generated or imported frame manifests. "
            "They are useful controls, not claims about what the model perceives.",
            "",
        ]
    )
    return "\n".join(lines)


def _frame_image_stats(rgb: np.ndarray) -> dict[str, Any]:
    luminance = _luminance(rgb)
    edge_strength = _edge_strength(luminance)
    spectral = _spectral_stats(luminance)
    return {
        "rgb_mean": [float(value) for value in rgb.mean(axis=(0, 1))],
        "rgb_std": [float(value) for value in rgb.std(axis=(0, 1))],
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "luminance_entropy_bits": _entropy_bits(luminance),
        "edge_density": float(np.mean(edge_strength > 0.12)),
        "edge_strength_mean": float(edge_strength.mean()),
        "spectral_centroid": spectral["spectral_centroid"],
        "high_frequency_energy_ratio": spectral["high_frequency_energy_ratio"],
        "colorfulness": _colorfulness(rgb),
    }


def _aggregate_frame_stats(frame_stats: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in SCALAR_FRAME_FIELDS:
        values = [float(frame[field]) for frame in frame_stats if frame.get(field) is not None]
        output[field] = _summary(values)
    output["rgb_mean"] = _channel_summary(frame_stats, "rgb_mean")
    output["rgb_std"] = _channel_summary(frame_stats, "rgb_std")
    return output


def _pairwise_deltas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            left_condition = left.get("condition") or {}
            right_condition = right.get("condition") or {}
            output.append(
                {
                    "left_condition_id": left_condition.get("condition_id"),
                    "right_condition_id": right_condition.get("condition_id"),
                    "aggregate_mean_abs_deltas": {
                        field: _abs_delta(
                            left["aggregate"][field]["mean"],
                            right["aggregate"][field]["mean"],
                        )
                        for field in SCALAR_FRAME_FIELDS
                    },
                }
            )
    return output


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722


def _edge_strength(luminance: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(luminance)
    return np.sqrt(gx * gx + gy * gy)


def _spectral_stats(luminance: np.ndarray) -> dict[str, float]:
    centered = luminance - float(luminance.mean())
    spectrum = np.fft.rfft2(centered)
    power = np.abs(spectrum) ** 2
    power[0, 0] = 0.0
    total = float(power.sum())
    if total <= 1e-18:
        return {"spectral_centroid": 0.0, "high_frequency_energy_ratio": 0.0}

    fy = np.fft.fftfreq(luminance.shape[0])[:, None]
    fx = np.fft.rfftfreq(luminance.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    max_radius = float(radius.max()) or 1.0
    normalized_radius = radius / max_radius
    return {
        "spectral_centroid": float((normalized_radius * power).sum() / total),
        "high_frequency_energy_ratio": float(power[normalized_radius > 0.35].sum() / total),
    }


def _entropy_bits(luminance: np.ndarray) -> float:
    bins = np.clip((luminance * 255.0).astype(np.int32), 0, 255)
    counts = np.bincount(bins.ravel(), minlength=256).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _colorfulness(rgb: np.ndarray) -> float:
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    rg = red - green
    yb = 0.5 * (red + green) - blue
    std_root = np.sqrt(float(rg.std()) ** 2 + float(yb.std()) ** 2)
    mean_root = np.sqrt(float(rg.mean()) ** 2 + float(yb.mean()) ** 2)
    return float(std_root + 0.3 * mean_root)


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": fmean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _channel_summary(frame_stats: list[dict[str, Any]], field: str) -> dict[str, Any]:
    channels = []
    for channel in range(3):
        values = [float(frame[field][channel]) for frame in frame_stats]
        channels.append(_summary(values))
    return {"channels": channels}


def _abs_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(right) - float(left))


def _resolve_frame_path(manifest_path: Path, frame: dict[str, Any]) -> Path:
    path = Path(str(frame["path"]))
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
