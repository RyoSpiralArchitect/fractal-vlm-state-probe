from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .stimulus import write_json


def analyze_factorial_cache_trajectory(
    analyses: dict[str, dict[str, Any]],
    *,
    analysis_paths: dict[str, Path],
    phase: str = "after",
    probe_id: str = "forced_family_choice",
) -> dict[str, Any]:
    if len(analyses) < 2:
        raise ValueError("at least two factorial analyses are required")
    if set(analyses) != set(analysis_paths):
        raise ValueError("analysis_paths must align with analyses")
    points = [
        _trajectory_point(
            label=label,
            analysis=analysis,
            analysis_path=analysis_paths[label],
            phase=phase,
            probe_id=probe_id,
        )
        for label, analysis in analyses.items()
    ]
    series: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        series.setdefault(point["series_id"], []).append(point)
    series_records = [
        _series_summary(series_id, records)
        for series_id, records in sorted(series.items())
    ]
    return {
        "schema_version": 1,
        "analysis_kind": "factorial_cache_trajectory",
        "phase": phase,
        "probe_id": probe_id,
        "point_count": len(points),
        "series_count": len(series_records),
        "points": sorted(points, key=lambda item: (item["series_id"], item["frame_count"])),
        "series": series_records,
        "interpretation_notes": [
            "Frame-count trajectories are descriptive and are not independent source-pair replications.",
            "Scalar cache effects are summary-stat contrasts rather than full hidden-state vector contrasts.",
            "Sequence-position argmax is restricted to positions sampled into each run artifact.",
            "Cumulative replay is distinct from incremental multi-turn cache persistence.",
        ],
    }


def write_factorial_cache_trajectory_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_factorial_cache_trajectory_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_factorial_cache_trajectory_markdown(analysis), encoding="utf-8")


def format_factorial_cache_trajectory_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Factorial Cache Trajectory",
        "",
        f"- Phase: `{analysis['phase']}`",
        f"- Probe: `{analysis['probe_id']}`",
        "",
        "## Points",
        "",
        "| Series | Label | Frames | Cache tokens | Image tokens | Scalar argmax | Interaction | Relative | Abs / frame | Position argmax | Position roles | Family labels | Top-k Jaccard |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for point in analysis["points"]:
        scalar = point.get("scalar_argmax") or {}
        position = point.get("position_argmax") or {}
        labels = ", ".join(
            f"{cell}={token}" for cell, token in (point.get("family_labels") or {}).items()
        )
        lines.append(
            f"| `{point['series_id']}` | `{point['label']}` | {point['frame_count']} | "
            f"{_fmt(point.get('cache_token_count'), integer=True)} | "
            f"{_fmt(point.get('image_token_count'), integer=True)} | "
            f"{_location(scalar)} | {_fmt(scalar.get('interaction_effect'))} | "
            f"{_fmt(scalar.get('relative_abs_interaction'))} | "
            f"{_fmt(point.get('scalar_abs_interaction_per_frame'))} | "
            f"{_location(position)} | "
            f"{', '.join(position.get('roles') or []) or 'n/a'} | "
            f"{labels or 'n/a'} | {_fmt(point.get('family_top_k_jaccard'))} |"
        )

    lines.extend(
        [
            "",
            "## Series Summary",
            "",
            "| Series | Points | Frames | Scalar location stable | Position location stable | Readout cell-invariant | Frame/effect Pearson |",
            "| --- | ---: | --- | --- | --- | --- | ---: |",
        ]
    )
    for record in analysis["series"]:
        lines.append(
            f"| `{record['series_id']}` | {record['point_count']} | "
            f"{', '.join(str(value) for value in record['frame_counts'])} | "
            f"`{record['scalar_argmax_location_consistent']}` | "
            f"`{record['position_argmax_location_consistent']}` | "
            f"`{record['readout_cell_invariant_at_all_points']}` | "
            f"{_fmt(record.get('frame_count_vs_abs_scalar_interaction_pearson'))} |"
        )
    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _trajectory_point(
    *,
    label: str,
    analysis: dict[str, Any],
    analysis_path: Path,
    phase: str,
    probe_id: str,
) -> dict[str, Any]:
    cells = analysis.get("cells") or {}
    mm_header = cells.get("mm") or {}
    jj_header = cells.get("jj") or {}
    run_path = Path(str(mm_header.get("source_path") or ""))
    run = _load_json(run_path) if run_path.is_file() else {}
    stimulus = run.get("stimulus") or {}
    context = run.get("context_policy") or {}
    event = ((run.get("stream_events") or [{}])[0])
    layout = event.get("cache_token_layout") or {}
    role_map = {
        int(record["position"]): list(record.get("roles") or [])
        for record in layout.get("sequence_position_plan") or []
    }
    scalar = _argmax(analysis, "scalar", phase=phase, probe_id=probe_id)
    position = _argmax(analysis, "sequence_position", phase=phase, probe_id=probe_id)
    if scalar:
        scalar["relative_abs_interaction"] = _relative_effect(
            analysis.get("scalar_records") or [],
            scalar,
        )
    if position:
        position["relative_abs_interaction"] = _relative_effect(
            analysis.get("sequence_position_records") or [],
            position,
        )
        position["roles"] = _position_roles(
            int(position["position"]),
            role_map=role_map,
            image_runs=layout.get("image_token_runs") or [],
        )
    frame_count = int(stimulus.get("frame_count_selected") or 0)
    readout = _readout_record(analysis_path, phase=phase, probe_id=probe_id)
    series_id = f"{mm_header.get('condition_id')}__{jj_header.get('condition_id')}"
    scalar_abs = (scalar or {}).get("abs_interaction_effect")
    return {
        "label": label,
        "analysis_path": str(analysis_path),
        "series_id": series_id,
        "model_id": mm_header.get("model"),
        "context_protocol": context.get("visual_context_protocol")
        or context.get("frame_delivery"),
        "frame_count": frame_count,
        "cache_token_count": layout.get("token_count"),
        "image_token_count": layout.get("image_token_count"),
        "image_token_runs": layout.get("image_token_runs") or [],
        "scalar_argmax": scalar,
        "position_argmax": position,
        "scalar_abs_interaction_per_frame": (
            float(scalar_abs) / frame_count if scalar_abs is not None and frame_count else None
        ),
        "family_labels": readout.get("labels"),
        "family_top_k_jaccard": readout.get("mean_top_k_jaccard"),
        "family_top_interaction_effect": readout.get("top_interaction_effect"),
        "readout_cell_invariant": readout.get("cell_invariant"),
    }


def _series_summary(series_id: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(points, key=lambda item: item["frame_count"])
    scalar_locations = {_location_key(point.get("scalar_argmax")) for point in ordered}
    position_locations = {_location_key(point.get("position_argmax")) for point in ordered}
    frame_counts = [point["frame_count"] for point in ordered]
    effects = [
        float(point["scalar_argmax"]["abs_interaction_effect"])
        for point in ordered
        if point.get("scalar_argmax")
    ]
    return {
        "series_id": series_id,
        "model_id": ordered[0].get("model_id"),
        "context_protocols": sorted({str(point.get("context_protocol")) for point in ordered}),
        "point_count": len(ordered),
        "frame_counts": frame_counts,
        "scalar_argmax_location_consistent": len(scalar_locations) == 1,
        "position_argmax_location_consistent": len(position_locations) == 1,
        "readout_cell_invariant_at_all_points": all(
            point.get("readout_cell_invariant") is True for point in ordered
        ),
        "frame_count_vs_abs_scalar_interaction_pearson": _pearson(frame_counts, effects),
    }


def _argmax(
    analysis: dict[str, Any],
    record_type: str,
    *,
    phase: str,
    probe_id: str,
) -> dict[str, Any] | None:
    for record in analysis.get("interaction_argmax_summary") or []:
        if record.get("record_type") == record_type and record.get("probe_id") == probe_id:
            value = record.get(phase)
            return dict(value) if value else None
    return None


def _relative_effect(records: list[dict[str, Any]], argmax: dict[str, Any]) -> float | None:
    for record in records:
        keys = ("phase", "probe_id", "layer_index", "tensor", "position")
        if all(record.get(key) == argmax.get(key) for key in keys if key in argmax):
            if record.get("field") == "l2_norm":
                value = record.get("relative_abs_interaction_to_grand_mean")
                return float(value) if isinstance(value, (int, float)) else None
    return None


def _readout_record(analysis_path: Path, *, phase: str, probe_id: str) -> dict[str, Any]:
    path = analysis_path.parent / "probe_readout_contrast.json"
    if not path.is_file():
        return {}
    analysis = _load_json(path)
    for record in analysis.get("records") or []:
        if record.get("phase") != phase or record.get("probe_id") != probe_id:
            continue
        labels = {
            cell: value.get("token")
            for cell, value in (record.get("generated_tokens") or {}).items()
        }
        top = (record.get("top_common_token_effects") or [{}])[0]
        return {
            "labels": labels,
            "mean_top_k_jaccard": record.get("mean_top_k_jaccard"),
            "top_interaction_effect": top.get("interaction_effect"),
            "cell_invariant": len(set(labels.values())) <= 1 if labels else None,
        }
    return {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _location_key(record: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if not record:
        return None
    return record.get("layer_index"), record.get("tensor"), record.get("position")


def _position_roles(
    position: int,
    *,
    role_map: dict[int, list[str]],
    image_runs: list[dict[str, Any]],
) -> list[str]:
    roles = set(role_map.get(position, []))
    for run_index, run in enumerate(image_runs):
        start = int(run["start"])
        end = int(run["end"])
        if start < position < end:
            roles.add(f"image_run_{run_index}_interior")
    return sorted(roles)


def _location(record: dict[str, Any]) -> str:
    if not record:
        return "n/a"
    value = f"layer {record.get('layer_index')} `{record.get('tensor')}`"
    if record.get("position") is not None:
        value += f" pos {record['position']}"
    return value


def _pearson(left: list[float | int], right: list[float | int]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_mean = mean(left_values)
    right_mean = mean(right_values)
    left_centered = [value - left_mean for value in left_values]
    right_centered = [value - right_mean for value in right_values]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def _fmt(value: Any, *, integer: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return str(int(value)) if integer else f"{float(value):.3f}"
