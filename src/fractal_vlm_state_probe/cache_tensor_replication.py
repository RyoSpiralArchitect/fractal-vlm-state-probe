from __future__ import annotations

from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .cache_tensor_artifact import load_cache_tensor_artifact
from .cache_tensor_factorial import (
    REGION_ORDER,
    cache_tensor_regions,
    factorial_effect_vectors,
)
from .probe_readout import CELL_KEYS
from .stimulus import write_json


def analyze_cache_tensor_replication(
    analyses: dict[str, dict[str, Any]],
    *,
    analysis_paths: dict[str, Path],
) -> dict[str, Any]:
    if len(analyses) < 2:
        raise ValueError("at least two cache tensor factorial analyses are required")
    if set(analyses) != set(analysis_paths):
        raise ValueError("analysis_paths must align with analyses")

    points = [
        _point_record(label, analyses[label], analysis_paths[label])
        for label in sorted(analyses)
    ]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for point in points:
        key = (point["model_id"], point["layer_index"], point["tensor"])
        groups.setdefault(key, []).append(point)

    group_records = []
    for (model_id, layer_index, tensor), records in sorted(groups.items()):
        if len(records) < 2:
            raise ValueError(
                f"replication group needs at least two source pairs: {model_id} layer {layer_index} {tensor}"
            )
        group_records.append(
            _group_record(
                model_id=model_id,
                layer_index=layer_index,
                tensor=tensor,
                points=records,
            )
        )

    return {
        "schema_version": 1,
        "analysis_kind": "source_cache_tensor_cross_pair_replication",
        "analysis_count": len(points),
        "group_count": len(group_records),
        "model_count": len({point["model_id"] for point in points}),
        "source_pair_count": len({point["source_pair_id"] for point in points}),
        "points": [
            {key: value for key, value in point.items() if not key.startswith("_")}
            for point in points
        ],
        "groups": group_records,
        "interpretation_notes": [
            "Direction cosines compare the same layer, tensor, model, and token region across independent source pairs.",
            "No raw vector cosine is computed across models or layers because their cache coordinates are not assumed to align.",
            "The image-token region carries total interaction energy, while the fixed post-image suffix can carry a more repeatable low-energy direction.",
            "Balanced factorial contrast shares compare spatial, palette, and interaction axes on the same +/-1 coefficient scale; 1/3 is the exchangeable isotropic reference.",
            "Group-level spatial, palette, and interaction shares are componentwise medians and therefore need not sum exactly to one.",
            "The pre-image prefix is an autoregressive alignment control: it should not depend on later image tokens.",
            "These are deterministic descriptive replications over four source pairs, not null-hypothesis tests.",
        ],
    }


def write_cache_tensor_replication_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_cache_tensor_replication_markdown(
    analysis: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_cache_tensor_replication_markdown(analysis), encoding="utf-8"
    )


def format_cache_tensor_replication_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Source-Cache Tensor Cross-Pair Replication",
        "",
        f"- Models: `{analysis['model_count']}`",
        f"- Source pairs: `{analysis['source_pair_count']}`",
        f"- Layer groups: `{analysis['group_count']}`",
        "",
        "## Group Summary",
        "",
        "| Model | Layer | Tensor | Pairs | Image energy median | Image argmax | Pre-image zero | Image direction median | Post-image direction median |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in analysis["groups"]:
        image_direction = _region(group, "image_tokens")["direction_cosine_summary"]
        post_direction = _region(group, "post_image")["direction_cosine_summary"]
        lines.append(
            f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
            f"`{group['tensor']}` | {group['source_pair_count']} | "
            f"{_fmt(group['image_energy_fraction']['median'])} | "
            f"{group['interaction_argmax_in_image_count']}/{group['source_pair_count']} | "
            f"{group['pre_image_all_effects_zero_count']}/{group['source_pair_count']} | "
            f"{_fmt(image_direction['median'])} | "
            f"{_fmt(post_direction['median'])} |"
        )

    lines.extend(
        [
            "",
            "## Region Direction Replication",
            "",
            "| Model | Layer | Region | Interaction RMS median (range) | Relative to pairwise median | Balanced S / P / I energy shares | Cross-pair cosine median (range) | Positive cosine share |",
            "| --- | ---: | --- | --- | ---: | --- | --- | ---: |",
        ]
    )
    for group in analysis["groups"]:
        for region in group["regions"]:
            direction = region["direction_cosine_summary"]
            shares = region["balanced_contrast_energy_shares"]
            lines.append(
                f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                f"`{region['region']}` | {_range_text(region['interaction_rms'])} | "
                f"{_fmt(region['relative_rms_to_mean_pairwise_rms']['median'])} | "
                f"{_fmt(shares['spatial_contrast']['median'])} / "
                f"{_fmt(shares['palette_contrast']['median'])} / "
                f"{_fmt(shares['interaction_contrast']['median'])} | "
                f"{_range_text(direction)} | {_fmt(direction['positive_share'])} |"
            )

    lines.extend(
        [
            "",
            "## Effect Direction Replication",
            "",
            "| Model | Layer | Region | Effect | Cross-pair cosine median (range) | Positive cosine share |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for group in analysis["groups"]:
        for region in group["regions"]:
            if region["region"] not in {"image_tokens", "post_image"}:
                continue
            for effect, record in region["effect_direction_replication"].items():
                summary = record["summary"]
                lines.append(
                    f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                    f"`{region['region']}` | `{effect}` | {_range_text(summary)} | "
                    f"{_fmt(summary['positive_share'])} |"
                )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _point_record(
    label: str,
    analysis: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    if analysis.get("analysis_kind") != "source_cache_tensor_factorial_contrast":
        raise ValueError(f"unsupported analysis kind for {label}")
    regions = {record["region"]: record for record in analysis.get("regions") or []}
    image_positions = _image_position_set(analysis)
    all_interaction = regions["all_effective"]["effects"]["interaction"]
    pre_image = regions.get("pre_image")
    tensor_shape = [int(value) for value in analysis["tensor_shape"]]
    region_positions = cache_tensor_regions(
        analysis["cells"]["mm"]["cache_token_layout"],
        sequence_length=tensor_shape[-2],
    )
    region_metrics = {}
    for name, record in regions.items():
        balanced_shares = (record.get("balanced_contrast_energy") or {}).get(
            "energy_shares"
        ) or {}
        region_metrics[name] = {
            "interaction_rms": float(record["effects"]["interaction"]["rms"]),
            "relative_rms_to_mean_pairwise_rms": record["effects"]["interaction"][
                "relative_rms_to_mean_pairwise_rms"
            ],
            "balanced_contrast_energy_shares": {
                contrast: balanced_shares.get(contrast)
                for contrast in (
                    "spatial_contrast",
                    "palette_contrast",
                    "interaction_contrast",
                )
            },
            "balanced_interaction_energy_share": balanced_shares.get(
                "interaction_contrast"
            ),
        }
    return {
        "label": label,
        "analysis_path": str(path),
        "model_id": str(analysis["model_id"]),
        "source_pair_id": str(analysis["source_pair_id"]),
        "layer_index": int(analysis["layer_index"]),
        "tensor": str(analysis["tensor"]),
        "tensor_shape": tensor_shape,
        "interaction_argmax_sequence_position": int(
            all_interaction["argmax_sequence_position"]
        ),
        "interaction_argmax_in_image": int(all_interaction["argmax_sequence_position"])
        in image_positions,
        "image_energy_fraction": float(
            analysis["interaction_partition"]["image_energy_fraction"]
        ),
        "pre_image_all_effects_zero": bool(
            pre_image
            and all(
                float(effect["l2_norm"]) == 0.0
                for effect in pre_image["effects"].values()
            )
        ),
        "region_metrics": region_metrics,
        "_analysis": analysis,
        "_region_positions": region_positions,
    }


def _group_record(
    *,
    model_id: str,
    layer_index: int,
    tensor: str,
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    source_pairs = [point["source_pair_id"] for point in points]
    if len(set(source_pairs)) != len(source_pairs):
        raise ValueError(
            f"duplicate source-pair coverage for {model_id} layer {layer_index}"
        )
    tensor_shapes = {tuple(point["tensor_shape"]) for point in points}
    if len(tensor_shapes) != 1:
        raise ValueError(
            f"replication group tensor shapes do not align: {sorted(tensor_shapes)}"
        )
    reference_positions = points[0]["_region_positions"]
    for point in points[1:]:
        if point["_region_positions"] != reference_positions:
            raise ValueError(
                "replication group token-region positions do not align: "
                f"{points[0]['label']} != {point['label']}"
            )

    vectors = {
        point["label"]: _factorial_vectors(point["_analysis"]) for point in points
    }
    region_records = []
    for region in REGION_ORDER:
        available = {
            point["label"]: vectors[point["label"]][region]
            for point in points
            if region in vectors[point["label"]]
        }
        if len(available) != len(points):
            continue
        effect_replication = {
            effect: _direction_replication(
                {label: records[effect] for label, records in available.items()}
            )
            for effect in (
                "spatial_main_effect",
                "palette_main_effect",
                "interaction",
            )
        }
        interaction_replication = effect_replication["interaction"]
        region_records.append(
            {
                "region": region,
                "interaction_rms": _summary(
                    point["region_metrics"][region]["interaction_rms"]
                    for point in points
                ),
                "relative_rms_to_mean_pairwise_rms": _summary(
                    point["region_metrics"][region]["relative_rms_to_mean_pairwise_rms"]
                    for point in points
                ),
                "balanced_interaction_energy_share": _summary(
                    point["region_metrics"][region]["balanced_interaction_energy_share"]
                    for point in points
                ),
                "balanced_contrast_energy_shares": {
                    contrast: _summary(
                        point["region_metrics"][region][
                            "balanced_contrast_energy_shares"
                        ][contrast]
                        for point in points
                    )
                    for contrast in (
                        "spatial_contrast",
                        "palette_contrast",
                        "interaction_contrast",
                    )
                },
                "pairwise_direction_cosines": interaction_replication["pairwise"],
                "direction_cosine_summary": interaction_replication["summary"],
                "effect_direction_replication": effect_replication,
            }
        )

    public_points = [
        {key: value for key, value in point.items() if not key.startswith("_")}
        for point in points
    ]
    return {
        "model_id": model_id,
        "layer_index": layer_index,
        "tensor": tensor,
        "source_pair_count": len(points),
        "source_pair_ids": sorted(source_pairs),
        "points": public_points,
        "interaction_argmax_in_image_count": sum(
            int(point["interaction_argmax_in_image"]) for point in points
        ),
        "pre_image_all_effects_zero_count": sum(
            int(point["pre_image_all_effects_zero"]) for point in points
        ),
        "image_energy_fraction": _summary(
            point["image_energy_fraction"] for point in points
        ),
        "regions": region_records,
    }


def _factorial_vectors(
    analysis: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    arrays = {}
    layouts = {}
    for cell in CELL_KEYS:
        record = analysis["cells"][cell]
        run_path = _resolve_source_path(str(record["source_path"]))
        arrays[cell] = load_cache_tensor_artifact(
            run_path, record["cache_tensor_artifact"]
        ).astype(np.float64)
        layouts[cell] = record["cache_token_layout"]

    shape = arrays[CELL_KEYS[0]].shape
    regions = cache_tensor_regions(layouts[CELL_KEYS[0]], sequence_length=shape[-2])
    output = {}
    for region, positions in regions.items():
        if not positions:
            continue
        cells = {cell: np.take(arrays[cell], positions, axis=-2) for cell in CELL_KEYS}
        output[region] = factorial_effect_vectors(cells)
    return output


def _direction_replication(vectors: dict[str, np.ndarray]) -> dict[str, Any]:
    pairwise = []
    for left, right in combinations(sorted(vectors), 2):
        cosine = _cosine(vectors[left], vectors[right])
        pairwise.append(
            {
                "left": left,
                "right": right,
                "cosine": cosine,
                "available": cosine is not None,
            }
        )
    return {
        "pairwise": pairwise,
        "summary": _summary(record["cosine"] for record in pairwise),
    }


def _image_position_set(analysis: dict[str, Any]) -> set[int]:
    layout = analysis["cells"]["mm"]["cache_token_layout"]
    positions = set()
    for run in layout.get("image_token_runs") or []:
        positions.update(range(int(run["start"]), int(run["end"]) + 1))
    return positions


def _resolve_source_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    raise FileNotFoundError(f"source run path does not exist: {path}")


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator == 0.0:
        return None
    return float(np.dot(left_flat, right_flat) / denominator)


def _summary(values: Any) -> dict[str, Any]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "positive_share": None,
        }
    return {
        "count": len(cleaned),
        "min": min(cleaned),
        "median": median(cleaned),
        "mean": sum(cleaned) / len(cleaned),
        "max": max(cleaned),
        "positive_share": sum(value > 0 for value in cleaned) / len(cleaned),
    }


def _region(group: dict[str, Any], name: str) -> dict[str, Any]:
    return next(record for record in group["regions"] if record["region"] == name)


def _range_text(record: dict[str, Any]) -> str:
    return (
        f"{_fmt(record.get('median'))} "
        f"({_fmt(record.get('min'))}-{_fmt(record.get('max'))})"
    )


def _short_model(model_id: str) -> str:
    if "InternVL3" in model_id:
        return "InternVL3-2B"
    if "Qwen2.5" in model_id:
        return "Qwen2.5-VL-3B"
    return model_id


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
