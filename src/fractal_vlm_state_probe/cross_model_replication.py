from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from .stimulus import write_json


CELL_KEYS = ("mm", "jj", "mj", "jm")


def analyze_cross_model_replication(
    family_trajectory: dict[str, Any],
    frequency_trajectory: dict[str, Any],
    *,
    family_path: Path,
    frequency_path: Path,
    frame_count: int = 1,
) -> dict[str, Any]:
    family_points = _points_by_label(family_trajectory, frame_count=frame_count)
    frequency_points = _points_by_label(frequency_trajectory, frame_count=frame_count)
    if set(family_points) != set(frequency_points):
        raise ValueError("family and frequency trajectory labels do not align")
    if not family_points:
        raise ValueError(f"no trajectory points found for frame_count={frame_count}")

    points = [
        _combined_point(label, family_points[label], frequency_points[label])
        for label in sorted(family_points)
    ]
    model_groups = _group(points, "model_id")
    pair_groups = _group(points, "source_pair_id")
    return {
        "schema_version": 1,
        "analysis_kind": "cross_model_source_pair_replication",
        "family_trajectory": str(family_path),
        "frequency_trajectory": str(frequency_path),
        "frame_count": frame_count,
        "point_count": len(points),
        "model_count": len(model_groups),
        "source_pair_count": len(pair_groups),
        "points": sorted(
            points, key=lambda item: (item["model_id"], item["source_pair_id"])
        ),
        "model_summaries": [
            _model_summary(model_id, records)
            for model_id, records in sorted(model_groups.items())
        ],
        "source_pair_summaries": [
            _source_pair_summary(pair_id, records)
            for pair_id, records in sorted(pair_groups.items())
        ],
        "interpretation_notes": [
            "Independent source pairs support replication claims; frame-count points from one pair do not.",
            "Cache argmax stability refers to scalar summary-stat factorial interactions, not full hidden-state vectors.",
            "Exact layer stability, K/V component stability, and interaction-sign stability are reported separately.",
            "Forced-choice readouts are prompt-conditioned measurements and should be interpreted with the prompt robustness audit.",
        ],
    }


def write_cross_model_replication_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_cross_model_replication_markdown(
    analysis: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_cross_model_replication_markdown(analysis), encoding="utf-8")


def format_cross_model_replication_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Cross-Model Source-Pair Replication",
        "",
        f"- Frames: `{analysis['frame_count']}`",
        f"- Models: `{analysis['model_count']}`",
        f"- Independent source pairs: `{analysis['source_pair_count']}`",
        "",
        "## Model Summary",
        "",
        "| Model | Pairs | Exact cache mode | Exact share | Component mode | Component share | Sign share | Depth range | Family L1 median (range) | Frequency L1 median (range) |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for record in analysis["model_summaries"]:
        lines.append(
            f"| `{_short_model(record['model_id'])}` | {record['source_pair_count']} | "
            f"{_location_text(record['dominant_cache_location'])} | "
            f"{_fmt(record['dominant_cache_location_share'])} | "
            f"`{record['dominant_cache_component']}` | "
            f"{_fmt(record['dominant_cache_component_share'])} | "
            f"{_fmt(record['dominant_interaction_sign_share'])} | "
            f"{_range_text(record['normalized_depth_range'])} | "
            f"{_metric_range_text(record['family_full_vocab_interaction_l1'])} | "
            f"{_metric_range_text(record['frequency_full_vocab_interaction_l1'])} |"
        )

    lines.extend(
        [
            "",
            "## Pair-Level Table",
            "",
            "| Model | Pair | Cache argmax | Interaction | Relative | Family generated (MM/JJ/MJ/JM) | Family max JS | Family interaction L1 | Frequency generated (MM/JJ/MJ/JM) | Frequency max JS | Frequency interaction L1 |",
            "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for point in analysis["points"]:
        cache = point["cache_scalar_argmax"]
        lines.append(
            f"| `{_short_model(point['model_id'])}` | `{point['source_pair_id']}` | "
            f"{_location_text(cache)} | {_fmt(cache.get('interaction_effect'))} | "
            f"{_fmt(cache.get('relative_abs_interaction'))} | "
            f"{_pattern(point['family']['generated_tokens'])} | "
            f"{_fmt(point['family']['max_pair_jensen_shannon'])} | "
            f"{_fmt(point['family']['interaction_l1_norm'])} | "
            f"{_pattern(point['frequency']['generated_tokens'])} | "
            f"{_fmt(point['frequency']['max_pair_jensen_shannon'])} | "
            f"{_fmt(point['frequency']['interaction_l1_norm'])} |"
        )

    lines.extend(
        [
            "",
            "## Source-Pair Summary",
            "",
            "| Pair | Models | Cache components | Family patterns agree | Frequency patterns agree | Cache interaction abs median |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for record in analysis["source_pair_summaries"]:
        components = ", ".join(
            f"{key}:{value}" for key, value in record["cache_component_counts"].items()
        )
        lines.append(
            f"| `{record['source_pair_id']}` | {record['model_count']} | {components} | "
            f"`{record['family_generated_pattern_consistent']}` | "
            f"`{record['frequency_generated_pattern_consistent']}` | "
            f"{_fmt(record['cache_abs_interaction_median'])} |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _points_by_label(
    trajectory: dict[str, Any],
    *,
    frame_count: int,
) -> dict[str, dict[str, Any]]:
    return {
        str(point["label"]): point
        for point in trajectory.get("points") or []
        if int(point.get("frame_count") or 0) == frame_count
    }


def _combined_point(
    label: str,
    family: dict[str, Any],
    frequency: dict[str, Any],
) -> dict[str, Any]:
    if family.get("series_id") != frequency.get("series_id"):
        raise ValueError(f"trajectory series mismatch for {label}")
    family_location = _location_key(family.get("scalar_argmax"))
    frequency_location = _location_key(frequency.get("scalar_argmax"))
    if family_location != frequency_location:
        raise ValueError(f"cache argmax mismatch across probe reports for {label}")
    condition_ids = str(family["series_id"]).rsplit("__", 2)[-2:]
    source_pair_id = _source_pair_id(*condition_ids)
    return {
        "label": label,
        "model_id": family["model_id"],
        "source_pair_id": source_pair_id,
        "condition_ids": {"mm": condition_ids[0], "jj": condition_ids[1]},
        "frame_count": family["frame_count"],
        "cache_scalar_argmax": family.get("scalar_argmax") or {},
        "family": _readout_summary(family),
        "frequency": _readout_summary(frequency),
    }


def _readout_summary(point: dict[str, Any]) -> dict[str, Any]:
    full_vocab = point.get("full_vocab_readout") or {}
    return {
        "generated_tokens": {
            cell: str(value).strip() if value is not None else None
            for cell, value in (point.get("family_labels") or {}).items()
        },
        "generated_cell_invariant": point.get("readout_cell_invariant"),
        "max_pair_jensen_shannon": full_vocab.get("max_pair_jensen_shannon"),
        "interaction_l1_norm": full_vocab.get("interaction_l1_norm"),
        "interaction_max_abs": full_vocab.get("interaction_max_abs"),
        "interaction_argmax_token": full_vocab.get("interaction_argmax_token"),
    }


def _model_summary(model_id: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    locations = [_location_key(point["cache_scalar_argmax"]) for point in points]
    components = [point["cache_scalar_argmax"].get("tensor") for point in points]
    signs = [
        _sign(point["cache_scalar_argmax"].get("interaction_effect"))
        for point in points
    ]
    dominant_location, location_count = _mode(locations)
    dominant_component, component_count = _mode(components)
    dominant_sign, sign_count = _mode(signs)
    depths = [
        float(point["cache_scalar_argmax"]["normalized_depth"])
        for point in points
        if point["cache_scalar_argmax"].get("normalized_depth") is not None
    ]
    return {
        "model_id": model_id,
        "source_pair_count": len(points),
        "source_pair_ids": sorted(point["source_pair_id"] for point in points),
        "dominant_cache_location": {
            "layer_index": dominant_location[0],
            "tensor": dominant_location[1],
        },
        "dominant_cache_location_count": location_count,
        "dominant_cache_location_share": location_count / len(points),
        "exact_cache_location_consistent": location_count == len(points),
        "dominant_cache_component": dominant_component,
        "dominant_cache_component_count": component_count,
        "dominant_cache_component_share": component_count / len(points),
        "cache_component_consistent": component_count == len(points),
        "dominant_interaction_sign": dominant_sign,
        "dominant_interaction_sign_count": sign_count,
        "dominant_interaction_sign_share": sign_count / len(points),
        "cache_interaction_sign_consistent": sign_count == len(points),
        "normalized_depth_range": _range(depths),
        "family_full_vocab_interaction_l1": _metric_summary(
            point["family"]["interaction_l1_norm"] for point in points
        ),
        "frequency_full_vocab_interaction_l1": _metric_summary(
            point["frequency"]["interaction_l1_norm"] for point in points
        ),
        "family_generated_pattern_counts": dict(
            Counter(
                _pattern_key(point["family"]["generated_tokens"]) for point in points
            )
        ),
        "frequency_generated_pattern_counts": dict(
            Counter(
                _pattern_key(point["frequency"]["generated_tokens"]) for point in points
            )
        ),
    }


def _source_pair_summary(
    source_pair_id: str,
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    family_patterns = {
        _pattern_key(point["family"]["generated_tokens"]) for point in points
    }
    frequency_patterns = {
        _pattern_key(point["frequency"]["generated_tokens"]) for point in points
    }
    return {
        "source_pair_id": source_pair_id,
        "model_count": len(points),
        "model_ids": sorted(point["model_id"] for point in points),
        "cache_component_counts": dict(
            Counter(point["cache_scalar_argmax"].get("tensor") for point in points)
        ),
        "family_generated_pattern_consistent": len(family_patterns) == 1,
        "frequency_generated_pattern_consistent": len(frequency_patterns) == 1,
        "cache_abs_interaction_median": median(
            abs(float(point["cache_scalar_argmax"]["interaction_effect"]))
            for point in points
        ),
    }


def _metric_summary(values: Any) -> dict[str, float | None]:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return {"median": None, "min": None, "max": None}
    return {"median": median(cleaned), "min": min(cleaned), "max": max(cleaned)}


def _group(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    return grouped


def _source_pair_id(mm_condition: str, jj_condition: str) -> str:
    mm = mm_condition.removeprefix("mandelbrot_zoom_").removesuffix("_50f")
    jj = jj_condition.removeprefix("julia_zoom_").removesuffix("_50f")
    return f"{mm}_{jj}"


def _location_key(record: dict[str, Any] | None) -> tuple[Any, Any]:
    record = record or {}
    return record.get("layer_index"), record.get("tensor")


def _mode(values: list[Any]) -> tuple[Any, int]:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], str(value))), max(
        counts.values()
    )


def _sign(value: Any) -> str:
    number = float(value)
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def _range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": min(values), "max": max(values)}


def _pattern_key(labels: dict[str, Any]) -> str:
    return "/".join(str(labels.get(cell)) for cell in CELL_KEYS)


def _pattern(labels: dict[str, Any]) -> str:
    return " / ".join(str(labels.get(cell)) for cell in CELL_KEYS)


def _location_text(record: dict[str, Any]) -> str:
    return f"layer {record.get('layer_index')} `{record.get('tensor')}`"


def _range_text(record: dict[str, Any]) -> str:
    return f"{_fmt(record.get('min'))}-{_fmt(record.get('max'))}"


def _metric_range_text(record: dict[str, Any]) -> str:
    return (
        f"{_fmt(record.get('median'))} "
        f"({_fmt(record.get('min'))}-{_fmt(record.get('max'))})"
    )


def _short_model(model_id: str) -> str:
    if "InternVL3" in model_id:
        return "InternVL3-2B"
    if "Qwen2.5" in model_id:
        return "Qwen2.5-VL-3B"
    if "SmolVLM2" in model_id:
        return "SmolVLM2-2.2B"
    if "gemma-3" in model_id:
        return "Gemma-3-4B"
    return model_id


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
