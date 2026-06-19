from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np
from PIL import Image

from .stimulus import sha256_file, write_json

SCALAR_PROCESSOR_FRAME_FIELDS = (
    "tensor_mean",
    "tensor_std",
    "tensor_min",
    "tensor_max",
    "spectral_centroid",
    "high_frequency_energy_ratio",
    "spectral_centroid_cycles_per_patch",
    "energy_ratio_above_half_cycle_per_patch",
    "energy_ratio_above_one_cycle_per_patch",
    "mean_abs_tensor_delta_from_previous",
)


def analyze_manifest_processor_image_stats(
    manifest_path: Path,
    *,
    processor: Any,
    patch_size: int | None = 14,
    max_frames: int | None = None,
    include_frame_stats: bool = True,
) -> dict[str, Any]:
    if patch_size is not None and patch_size < 1:
        raise ValueError("patch_size must be positive")
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
    previous_tensor_view: np.ndarray | None = None
    for frame in frames:
        frame_path = _resolve_frame_path(manifest_path, frame)
        with Image.open(frame_path) as image:
            tensor = extract_processor_pixel_tensor(processor, image.convert("RGB"))
        stats = _processor_tensor_stats(tensor, patch_size=patch_size)
        stats["index"] = int(frame["index"])
        stats["t_seconds"] = float(frame["t_seconds"])
        stats["path"] = str(frame.get("path"))
        stats["sha256"] = str(frame.get("sha256") or sha256_file(frame_path))

        tensor_view = _tensor_view(tensor)
        if previous_tensor_view is None:
            stats["mean_abs_tensor_delta_from_previous"] = None
        else:
            stats["mean_abs_tensor_delta_from_previous"] = float(
                np.mean(np.abs(tensor_view - previous_tensor_view))
            )
        previous_tensor_view = tensor_view
        frame_stats.append(stats)

    record = {
        "schema_version": 1,
        "analysis_kind": "manifest_processor_image_statistics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "condition": manifest.get("stimulus_condition"),
        "stimulus_config_sha256": manifest.get("stimulus_config_sha256"),
        "processor_patch_size": patch_size,
        "frame_count_available": len(manifest.get("frames") or []),
        "frame_count_analyzed": len(frame_stats),
        "aggregate": _aggregate_processor_frame_stats(frame_stats),
    }
    if include_frame_stats:
        record["frame_stats"] = frame_stats
    return record


def analyze_processor_manifest_batch(
    manifest_paths: list[Path],
    *,
    processor: Any,
    patch_size: int | None = 14,
    max_frames: int | None = None,
    include_frame_stats: bool = False,
) -> dict[str, Any]:
    records = [
        analyze_manifest_processor_image_stats(
            path,
            processor=processor,
            patch_size=patch_size,
            max_frames=max_frames,
            include_frame_stats=include_frame_stats,
        )
        for path in manifest_paths
    ]
    return {
        "schema_version": 1,
        "analysis_kind": "processor_image_statistics_batch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "manifest_count": len(records),
        "records": records,
        "pairwise_deltas": _pairwise_deltas(records),
    }


def extract_processor_pixel_tensor(processor: Any, image: Image.Image) -> np.ndarray:
    payload = _call_processor(processor, image)
    pixel_values = _get_pixel_values(payload)
    array = _to_numpy(pixel_values).astype(np.float64, copy=False)
    return _normalize_pixel_tensor(array)


def load_hf_processor(model_id_or_path: str, *, trust_remote_code: bool = False) -> Any:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("install the hf extra to load processors: pip install -e '.[hf]'") from exc
    return AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)


def write_processor_image_stats_json(analysis: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, analysis)


def write_processor_image_stats_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_processor_image_stats_markdown(analysis), encoding="utf-8")


def format_processor_image_stats_markdown(analysis: dict[str, Any]) -> str:
    records = analysis["records"] if analysis.get("analysis_kind") == "processor_image_statistics_batch" else [analysis]
    lines = [
        "# Processor Image Statistics",
        "",
        f"- Manifests: `{len(records)}`",
        f"- Patch size: `{records[0].get('processor_patch_size')}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Condition | Family | Frames | Tensor mean | Tensor std | HF ratio | Centroid | Centroid cpp | >0.5 cpp | >1.0 cpp | Temporal delta |",
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
            f"{_fmt(aggregate['tensor_mean']['mean'])} | "
            f"{_fmt(aggregate['tensor_std']['mean'])} | "
            f"{_fmt(aggregate['high_frequency_energy_ratio']['mean'])} | "
            f"{_fmt(aggregate['spectral_centroid']['mean'])} | "
            f"{_fmt(aggregate['spectral_centroid_cycles_per_patch']['mean'])} | "
            f"{_fmt(aggregate['energy_ratio_above_half_cycle_per_patch']['mean'])} | "
            f"{_fmt(aggregate['energy_ratio_above_one_cycle_per_patch']['mean'])} | "
            f"{_fmt(aggregate['mean_abs_tensor_delta_from_previous']['mean'])} |"
        )

    if analysis.get("analysis_kind") == "processor_image_statistics_batch":
        lines.extend(
            [
                "",
                "## Pairwise Deltas",
                "",
                "| Left | Right | Tensor std | HF ratio | Centroid cpp | >0.5 cpp | >1.0 cpp | Temporal delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in analysis["pairwise_deltas"]:
            deltas = item["aggregate_mean_abs_deltas"]
            lines.append(
                "| "
                f"`{item['left_condition_id']}` | "
                f"`{item['right_condition_id']}` | "
                f"{_fmt(deltas['tensor_std'])} | "
                f"{_fmt(deltas['high_frequency_energy_ratio'])} | "
                f"{_fmt(deltas['spectral_centroid_cycles_per_patch'])} | "
                f"{_fmt(deltas['energy_ratio_above_half_cycle_per_patch'])} | "
                f"{_fmt(deltas['energy_ratio_above_one_cycle_per_patch'])} | "
                f"{_fmt(deltas['mean_abs_tensor_delta_from_previous'])} |"
            )

    lines.extend(
        [
            "",
            "## Reading Note",
            "",
            "These statistics are computed after the model processor has produced `pixel_values`. "
            "They audit resize, normalization, tiling, and aliasing effects; they do not identify "
            "which visual features the model actually uses.",
            "",
        ]
    )
    return "\n".join(lines)


def _call_processor(processor: Any, image: Image.Image) -> Any:
    try:
        return processor(images=image, return_tensors="np")
    except TypeError:
        pass
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise TypeError("processor must accept images=... or expose image_processor")
    try:
        return image_processor(images=image, return_tensors="np")
    except TypeError:
        return image_processor(image, return_tensors="np")


def _get_pixel_values(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "pixel_values" not in payload:
            raise KeyError("processor output did not include pixel_values")
        return payload["pixel_values"]
    if hasattr(payload, "pixel_values"):
        return payload.pixel_values
    raise TypeError("processor output must be a mapping or expose pixel_values")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _normalize_pixel_tensor(array: np.ndarray) -> np.ndarray:
    while array.ndim >= 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4:
        tiles = [_channel_first_3d(tile) for tile in array]
        return np.concatenate(tiles, axis=2)
    if array.ndim == 3:
        return _channel_first_3d(array)
    if array.ndim == 2:
        return array[None, :, :]
    raise ValueError(f"unsupported pixel_values shape: {array.shape}")


def _channel_first_3d(array: np.ndarray) -> np.ndarray:
    if array.shape[0] in (1, 3, 4):
        return array[:3]
    if array.shape[-1] in (1, 3, 4):
        return np.moveaxis(array[:, :, :3], -1, 0)
    raise ValueError(f"cannot infer channel axis for pixel_values shape: {array.shape}")


def _processor_tensor_stats(tensor: np.ndarray, *, patch_size: int | None) -> dict[str, Any]:
    view = _tensor_view(tensor)
    spectral = _spectral_stats(view, patch_size=patch_size)
    return {
        "pixel_values_shape": [int(value) for value in tensor.shape],
        "tensor_mean": float(tensor.mean()),
        "tensor_std": float(tensor.std()),
        "tensor_min": float(tensor.min()),
        "tensor_max": float(tensor.max()),
        "spectral_centroid": spectral["spectral_centroid"],
        "high_frequency_energy_ratio": spectral["high_frequency_energy_ratio"],
        "spectral_centroid_cycles_per_patch": spectral["spectral_centroid_cycles_per_patch"],
        "energy_ratio_above_half_cycle_per_patch": spectral["energy_ratio_above_half_cycle_per_patch"],
        "energy_ratio_above_one_cycle_per_patch": spectral["energy_ratio_above_one_cycle_per_patch"],
    }


def _tensor_view(tensor: np.ndarray) -> np.ndarray:
    if tensor.shape[0] == 1:
        return tensor[0]
    return tensor.mean(axis=0)


def _spectral_stats(signal: np.ndarray, *, patch_size: int | None) -> dict[str, float | None]:
    centered = signal - float(signal.mean())
    spectrum = np.fft.rfft2(centered)
    power = np.abs(spectrum) ** 2
    power[0, 0] = 0.0
    total = float(power.sum())
    if total <= 1e-18:
        return {
            "spectral_centroid": 0.0,
            "high_frequency_energy_ratio": 0.0,
            "spectral_centroid_cycles_per_patch": 0.0 if patch_size is not None else None,
            "energy_ratio_above_half_cycle_per_patch": 0.0 if patch_size is not None else None,
            "energy_ratio_above_one_cycle_per_patch": 0.0 if patch_size is not None else None,
        }

    fy = np.fft.fftfreq(signal.shape[0])[:, None]
    fx = np.fft.rfftfreq(signal.shape[1])[None, :]
    radius_cycles_per_pixel = np.sqrt(fx * fx + fy * fy)
    max_radius = float(radius_cycles_per_pixel.max()) or 1.0
    normalized_radius = radius_cycles_per_pixel / max_radius

    output: dict[str, float | None] = {
        "spectral_centroid": float((normalized_radius * power).sum() / total),
        "high_frequency_energy_ratio": float(power[normalized_radius > 0.35].sum() / total),
        "spectral_centroid_cycles_per_patch": None,
        "energy_ratio_above_half_cycle_per_patch": None,
        "energy_ratio_above_one_cycle_per_patch": None,
    }
    if patch_size is not None:
        cycles_per_patch = radius_cycles_per_pixel * float(patch_size)
        output["spectral_centroid_cycles_per_patch"] = float((cycles_per_patch * power).sum() / total)
        output["energy_ratio_above_half_cycle_per_patch"] = float(power[cycles_per_patch > 0.5].sum() / total)
        output["energy_ratio_above_one_cycle_per_patch"] = float(power[cycles_per_patch > 1.0].sum() / total)
    return output


def _aggregate_processor_frame_stats(frame_stats: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in SCALAR_PROCESSOR_FRAME_FIELDS:
        values = [float(frame[field]) for frame in frame_stats if frame.get(field) is not None]
        output[field] = _summary(values)
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
                        for field in SCALAR_PROCESSOR_FRAME_FIELDS
                    },
                }
            )
    return output


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": fmean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


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
