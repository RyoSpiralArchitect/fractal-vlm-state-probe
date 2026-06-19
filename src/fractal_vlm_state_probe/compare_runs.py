from __future__ import annotations

import json
from statistics import fmean
from pathlib import Path
from typing import Any

from .stimulus import write_json

_CACHE_STAT_FIELDS = ("mean", "variance", "std", "abs_mean", "l2_norm")
_SEQUENCE_POSITION_STAT_FIELDS = ("mean", "variance", "abs_mean", "l2_norm")


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
        readout_delta = record.get("generation_readout_delta")
        if readout_delta and readout_delta.get("available"):
            lines.append(
                "- First-step top-k readout: "
                f"jaccard_distance=`{_format_optional_number(readout_delta.get('top_k_jaccard_distance'))}`, "
                f"max_abs_shared_logprob_delta=`{_format_optional_number(readout_delta.get('max_abs_shared_logprob_delta'))}`"
            )
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
        if delta and delta.get("top_sequence_position_l2_deltas"):
            top = delta["top_sequence_position_l2_deltas"][0]
            lines.append(
                "  "
                f"top_sequence_l2_delta: layer {top['layer_index']} {top['tensor']} "
                f"position {top['position']} left={top['left']:.6g} "
                f"right={top['right']:.6g} abs_delta={top['abs_delta']:.6g}"
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
        if delta and delta.get("top_sequence_position_l2_deltas"):
            top = delta["top_sequence_position_l2_deltas"][0]
            lines.append(
                "  "
                f"top_sequence_l2_delta: layer {top['layer_index']} {top['tensor']} "
                f"position {top['position']} left={top['left']:.6g} "
                f"right={top['right']:.6g} abs_delta={top['abs_delta']:.6g}"
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
                    "generation_readout_delta": _compare_generation_readout(
                        left_record.get("generation"),
                        right_record.get("generation"),
                    ),
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
    sequence_position_deltas = []
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
            sequence_position_deltas.extend(
                _compare_sequence_position_stats(
                    layer_index=layer_index,
                    tensor_name=tensor_name,
                    left_tensor=left_tensor,
                    right_tensor=right_tensor,
                )
            )

    l2_deltas = [record for record in deltas if record["field"] == "l2_norm"]
    variance_deltas = [record for record in deltas if record["field"] == "variance"]
    sequence_l2_deltas = [
        record for record in sequence_position_deltas if record["field"] == "l2_norm"
    ]
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
        "sequence_position_stat_fields": list(_SEQUENCE_POSITION_STAT_FIELDS),
        "max_abs_sequence_position_l2_delta": _max_abs_delta(sequence_l2_deltas),
        "mean_abs_sequence_position_l2_delta": _mean_abs_delta(sequence_l2_deltas),
        "top_l2_deltas": _top_abs_deltas(l2_deltas),
        "top_stat_deltas": _top_abs_deltas(deltas),
        "top_sequence_position_l2_deltas": _top_abs_deltas(sequence_l2_deltas),
        "top_sequence_position_stat_deltas": _top_abs_deltas(sequence_position_deltas),
    }


def _compare_sequence_position_stats(
    *,
    layer_index: int,
    tensor_name: str,
    left_tensor: dict[str, Any],
    right_tensor: dict[str, Any],
) -> list[dict[str, Any]]:
    left_positions = {
        record.get("position"): record
        for record in left_tensor.get("sequence_position_stats", [])
    }
    right_positions = {
        record.get("position"): record
        for record in right_tensor.get("sequence_position_stats", [])
    }
    records = []
    for position in sorted(set(left_positions) & set(right_positions)):
        if position is None:
            continue
        left_record = left_positions[position]
        right_record = right_positions[position]
        for field in _SEQUENCE_POSITION_STAT_FIELDS:
            delta = _numeric_delta(left_record.get(field), right_record.get(field))
            if delta is None:
                continue
            records.append(
                {
                    "layer_index": layer_index,
                    "tensor": tensor_name,
                    "position": int(position),
                    "field": field,
                    **delta,
                }
            )
    return records


def _compare_generation_readout(
    left_generation: dict[str, Any] | None,
    right_generation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    left_step = _generation_step(left_generation, 0)
    right_step = _generation_step(right_generation, 0)
    if not left_step or not right_step:
        return None

    left_top = _top_logprob_map(left_step)
    right_top = _top_logprob_map(right_step)
    if not left_top or not right_top:
        return None

    left_ids = set(left_top)
    right_ids = set(right_top)
    union = left_ids | right_ids
    intersection = left_ids & right_ids
    jaccard = len(intersection) / len(union) if union else None
    shared_deltas = []
    for token_id in sorted(intersection):
        delta = _numeric_delta(left_top[token_id]["logprob"], right_top[token_id]["logprob"])
        if delta is None:
            continue
        shared_deltas.append(
            {
                "token_id": token_id,
                "left_token": left_top[token_id].get("token"),
                "right_token": right_top[token_id].get("token"),
                **delta,
            }
        )

    left_top1 = _first_top_logprob(left_step)
    right_top1 = _first_top_logprob(right_step)
    return {
        "available": True,
        "metric_scope": "first_generation_step_top_k_logprobs",
        "left_top_k": len(left_ids),
        "right_top_k": len(right_ids),
        "shared_top_k": len(intersection),
        "top_k_jaccard": jaccard,
        "top_k_jaccard_distance": None if jaccard is None else 1.0 - jaccard,
        "left_top1": left_top1,
        "right_top1": right_top1,
        "top1_same_token_id": (
            left_top1 is not None
            and right_top1 is not None
            and left_top1.get("token_id") == right_top1.get("token_id")
        ),
        "mean_abs_shared_logprob_delta": _mean_abs_delta(shared_deltas),
        "max_abs_shared_logprob_delta": _max_abs_delta(shared_deltas),
        "top_shared_logprob_deltas": _top_abs_deltas(shared_deltas),
    }


def _generation_step(generation: dict[str, Any] | None, step_index: int) -> dict[str, Any] | None:
    if not generation:
        return None
    for step in generation.get("steps") or []:
        if int(step.get("step_index", -1)) == step_index:
            return step
    return None


def _top_logprob_map(step: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output = {}
    for item in step.get("top_logprobs") or []:
        try:
            token_id = int(item["token_id"])
            logprob = float(item["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        output[token_id] = {
            "token_id": token_id,
            "token": item.get("token"),
            "logprob": logprob,
        }
    return output


def _first_top_logprob(step: dict[str, Any]) -> dict[str, Any] | None:
    values = step.get("top_logprobs") or []
    if not values:
        return None
    item = values[0]
    try:
        return {
            "token_id": int(item["token_id"]),
            "token": item.get("token"),
            "logprob": float(item["logprob"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


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
