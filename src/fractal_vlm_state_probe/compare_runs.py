from __future__ import annotations

import json
from statistics import fmean
from pathlib import Path
from typing import Any

from .stimulus import write_json

_CACHE_STAT_FIELDS = ("mean", "variance", "std", "abs_mean", "l2_norm")


def load_run(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_source_path"] = str(path)
    return data


def compare_runs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "comparison_kind": "paired_run_comparison",
        "left": _run_header(left),
        "right": _run_header(right),
        "matched_context": _matched_context(left, right),
        "probe_comparison": _compare_probes(left, right),
        "stream_cache_comparison": _compare_stream_cache(left, right),
        "probe_source_cache_comparison": _compare_probe_source_cache(left, right),
    }


def write_comparison_markdown(comparison: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_comparison_markdown(comparison), encoding="utf-8")


def write_comparison_json(comparison: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, comparison)


def format_comparison_markdown(comparison: dict[str, Any]) -> str:
    left = comparison["left"]
    right = comparison["right"]
    lines = [
        "# Paired Run Comparison",
        "",
        f"- Left: `{left['condition_id']}` ({left['source_path']})",
        f"- Right: `{right['condition_id']}` ({right['source_path']})",
        f"- Model: `{left['model']}` / `{right['model']}`",
        f"- Seed: `{left.get('seed')}` / `{right.get('seed')}`",
        f"- Delivery: `{left.get('delivery_mode')}` / `{right.get('delivery_mode')}`",
        "",
        "## Matched Context",
        "",
    ]
    for key, value in comparison["matched_context"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Probe Comparison", ""])
    for record in comparison["probe_comparison"]:
        lines.append(f"### {record['phase']} / {record['probe_id']}")
        lines.append("")
        lines.append(f"- Left tokens: `{record['left_generation_tokens']}`")
        lines.append(f"- Right tokens: `{record['right_generation_tokens']}`")
        lines.append(f"- Same text: `{record['same_text']}`")
        lines.append("")
        lines.append("Left:")
        lines.append("")
        lines.append(f"> {record['left_text']}")
        lines.append("")
        lines.append("Right:")
        lines.append("")
        lines.append(f"> {record['right_text']}")
        lines.append("")

    lines.extend(["## Stream Cache Comparison", ""])
    for record in comparison["stream_cache_comparison"]:
        delta = record.get("cache_stat_delta")
        delta_text = _format_cache_delta_short(delta)
        lines.append(
            "- "
            f"frame {record['frame_index']}: "
            f"left_tokens={record.get('left_token_count')}, "
            f"right_tokens={record.get('right_token_count')}, "
            f"cache_delta={delta_text}, "
            f"left_artifact={record.get('left_frame_artifact')}, "
            f"right_artifact={record.get('right_frame_artifact')}"
        )
        if delta and delta.get("top_l2_deltas"):
            top = delta["top_l2_deltas"][0]
            lines.append(
                "  "
                f"top_l2_delta: layer {top['layer_index']} {top['tensor']} "
                f"left={top['left']:.6g} right={top['right']:.6g} "
                f"abs_delta={top['abs_delta']:.6g}"
            )

    lines.extend(["", "## Probe Source Cache Comparison", ""])
    for record in comparison["probe_source_cache_comparison"]:
        delta = record.get("source_cache_delta")
        lines.append(
            "- "
            f"{record['phase']} / {record['probe_id']}: "
            f"left_tokens={record.get('left_token_count')}, "
            f"right_tokens={record.get('right_token_count')}, "
            f"source_cache_delta={_format_cache_delta_short(delta)}"
        )
        if delta and delta.get("top_l2_deltas"):
            top = delta["top_l2_deltas"][0]
            lines.append(
                "  "
                f"top_l2_delta: layer {top['layer_index']} {top['tensor']} "
                f"left={top['left']:.6g} right={top['right']:.6g} "
                f"abs_delta={top['abs_delta']:.6g}"
            )

    lines.extend(
        [
            "",
            "## Reading Note",
            "",
            "This paired comparison is descriptive. It can show wording drift or trace-shape differences, "
            "but it does not establish a stable condition effect without repeated seeds, longer streams, "
            "and additional controls.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_header(run: dict[str, Any]) -> dict[str, Any]:
    stimulus = run.get("stimulus") or {}
    condition = stimulus.get("condition") or {}
    reproducibility = run.get("reproducibility") or {}
    delivery = run.get("stimulus_delivery") or {}
    return {
        "source_path": run.get("_source_path"),
        "run_kind": run.get("run_kind"),
        "model": run.get("model_id") or run.get("model_ref"),
        "adapter_id": (run.get("adapter_capabilities") or {}).get("adapter_id"),
        "condition_id": condition.get("condition_id"),
        "condition_family": condition.get("condition_family"),
        "frame_count_selected": stimulus.get("frame_count_selected"),
        "source_frame_count_selected": stimulus.get("source_frame_count_selected"),
        "seed": reproducibility.get("seed"),
        "delivery_mode": delivery.get("mode") or (run.get("context_policy") or {}).get("frame_delivery"),
    }


def _matched_context(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    return {
        "same_model": _run_header(left)["model"] == _run_header(right)["model"],
        "same_adapter": _run_header(left)["adapter_id"] == _run_header(right)["adapter_id"],
        "same_frame_count": _run_header(left)["frame_count_selected"] == _run_header(right)["frame_count_selected"],
        "same_source_frame_count": _run_header(left).get("source_frame_count_selected")
        == _run_header(right).get("source_frame_count_selected"),
        "same_seed": _run_header(left).get("seed") == _run_header(right).get("seed"),
        "same_delivery_mode": _run_header(left).get("delivery_mode") == _run_header(right).get("delivery_mode"),
        "same_probe_policy": (left.get("context_policy") or {}).get("probe_cache_policy")
        == (right.get("context_policy") or {}).get("probe_cache_policy"),
    }


def _compare_probes(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    left_probes = left.get("probes") or {}
    right_probes = right.get("probes") or {}
    for phase in ("before", "mid", "after"):
        left_by_id = {record.get("probe_id"): record for record in left_probes.get(phase, [])}
        right_by_id = {record.get("probe_id"): record for record in right_probes.get(phase, [])}
        for probe_id in sorted(set(left_by_id) & set(right_by_id)):
            left_record = left_by_id[probe_id]
            right_record = right_by_id[probe_id]
            records.append(
                {
                    "phase": phase,
                    "probe_id": probe_id,
                    "left_text": _one_line(left_record.get("assistant_text")),
                    "right_text": _one_line(right_record.get("assistant_text")),
                    "same_text": _one_line(left_record.get("assistant_text"))
                    == _one_line(right_record.get("assistant_text")),
                    "left_generation_tokens": (left_record.get("generation") or {}).get("generation_tokens"),
                    "right_generation_tokens": (right_record.get("generation") or {}).get("generation_tokens"),
                }
            )
    return records


def _compare_stream_cache(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_events = {event.get("frame_index"): event for event in left.get("stream_events", [])}
    right_events = {event.get("frame_index"): event for event in right.get("stream_events", [])}
    records = []
    for frame_index in sorted(set(left_events) & set(right_events)):
        left_event = left_events[frame_index]
        right_event = right_events[frame_index]
        records.append(
            {
                "frame_index": frame_index,
                "left_token_count": ((left_event.get("cache_summary") or {}).get("token_count")),
                "right_token_count": ((right_event.get("cache_summary") or {}).get("token_count")),
                "left_frame_artifact": ((left_event.get("frame_artifact") or {}).get("path")),
                "right_frame_artifact": ((right_event.get("frame_artifact") or {}).get("path")),
                "cache_stat_delta": _compare_cache_summary(
                    left_event.get("cache_summary"),
                    right_event.get("cache_summary"),
                ),
            }
        )
    return records


def _compare_probe_source_cache(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    left_probes = left.get("probes") or {}
    right_probes = right.get("probes") or {}
    for phase in ("before", "mid", "after"):
        left_by_id = {record.get("probe_id"): record for record in left_probes.get(phase, [])}
        right_by_id = {record.get("probe_id"): record for record in right_probes.get(phase, [])}
        for probe_id in sorted(set(left_by_id) & set(right_by_id)):
            left_summary = left_by_id[probe_id].get("source_cache_summary_before_probe")
            right_summary = right_by_id[probe_id].get("source_cache_summary_before_probe")
            records.append(
                {
                    "phase": phase,
                    "probe_id": probe_id,
                    "left_token_count": (left_summary or {}).get("token_count"),
                    "right_token_count": (right_summary or {}).get("token_count"),
                    "source_cache_delta": _compare_cache_summary(left_summary, right_summary),
                }
            )
    return records


def _compare_cache_summary(
    left_summary: dict[str, Any] | None,
    right_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not left_summary or not right_summary:
        return None
    left_layers = {layer.get("layer_index"): layer for layer in left_summary.get("layers", [])}
    right_layers = {layer.get("layer_index"): layer for layer in right_summary.get("layers", [])}
    deltas = []
    for layer_index in sorted(set(left_layers) & set(right_layers)):
        left_layer = left_layers[layer_index]
        right_layer = right_layers[layer_index]
        for tensor_name in ("keys", "values"):
            left_tensor = left_layer.get(tensor_name)
            right_tensor = right_layer.get(tensor_name)
            if not left_tensor or not right_tensor:
                continue
            for field in _CACHE_STAT_FIELDS:
                delta = _numeric_delta(left_tensor.get(field), right_tensor.get(field))
                if delta is None:
                    continue
                deltas.append(
                    {
                        "layer_index": layer_index,
                        "tensor": tensor_name,
                        "field": field,
                        **delta,
                    }
                )

    l2_deltas = [record for record in deltas if record["field"] == "l2_norm"]
    variance_deltas = [record for record in deltas if record["field"] == "variance"]
    return {
        "available": bool(deltas),
        "left_available": left_summary.get("available"),
        "right_available": right_summary.get("available"),
        "left_token_count": left_summary.get("token_count"),
        "right_token_count": right_summary.get("token_count"),
        "token_count_delta": _simple_number_delta(
            left_summary.get("token_count"),
            right_summary.get("token_count"),
        ),
        "comparable_layers": sorted(set(left_layers) & set(right_layers)),
        "stat_fields": list(_CACHE_STAT_FIELDS),
        "max_abs_l2_delta": _max_abs_delta(l2_deltas),
        "mean_abs_l2_delta": _mean_abs_delta(l2_deltas),
        "max_abs_variance_delta": _max_abs_delta(variance_deltas),
        "top_l2_deltas": _top_abs_deltas(l2_deltas),
        "top_stat_deltas": _top_abs_deltas(deltas),
    }


def _numeric_delta(left_value: Any, right_value: Any) -> dict[str, float] | None:
    try:
        left_float = float(left_value)
        right_float = float(right_value)
    except (TypeError, ValueError):
        return None
    delta = right_float - left_float
    return {
        "left": left_float,
        "right": right_float,
        "delta": delta,
        "abs_delta": abs(delta),
    }


def _simple_number_delta(left_value: Any, right_value: Any) -> float | None:
    delta = _numeric_delta(left_value, right_value)
    if delta is None:
        return None
    return delta["delta"]


def _max_abs_delta(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return max(record["abs_delta"] for record in records)


def _mean_abs_delta(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return fmean(record["abs_delta"] for record in records)


def _top_abs_deltas(records: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: record["abs_delta"], reverse=True)[:limit]


def _format_cache_delta_short(delta: dict[str, Any] | None) -> str:
    if not delta:
        return "not captured"
    if not delta.get("available"):
        return "no comparable stats"
    max_l2 = delta.get("max_abs_l2_delta")
    mean_l2 = delta.get("mean_abs_l2_delta")
    max_var = delta.get("max_abs_variance_delta")
    return (
        f"max_abs_l2={_format_optional_number(max_l2)}, "
        f"mean_abs_l2={_format_optional_number(mean_l2)}, "
        f"max_abs_var={_format_optional_number(max_var)}"
    )


def _format_optional_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _one_line(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())
