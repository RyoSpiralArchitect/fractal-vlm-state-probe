from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_run(data: dict[str, Any]) -> str:
    lines: list[str] = []
    stimulus = data.get("stimulus") or {}
    condition = stimulus.get("condition") or {}
    capabilities = data.get("adapter_capabilities") or {}
    context = data.get("context_policy") or {}
    schedule = data.get("probe_schedule") or {}
    reproducibility = data.get("reproducibility") or {}
    delivery = data.get("stimulus_delivery") or {}
    stream_events = data.get("stream_events") or []
    probes = data.get("probes") or {}

    lines.append(f"Run kind: {data.get('run_kind', '<unknown>')}")
    lines.append(f"Model: {data.get('model_id') or data.get('model_ref') or '<unknown>'}")
    lines.append(f"Adapter: {capabilities.get('adapter_id', '<unknown>')} ({capabilities.get('tier', '<unknown>')})")
    lines.append(
        "Condition: "
        f"{condition.get('condition_id', '<unknown>')} "
        f"[{condition.get('condition_family', '?')}, {condition.get('temporal_policy', '?')}, "
        f"semantic={condition.get('semantic_load', '?')}]"
    )
    lines.append(
        "Frames: "
        f"{stimulus.get('frame_count_selected', len(stream_events))}/"
        f"{stimulus.get('frame_count_available', '?')} selected"
    )
    lines.append(f"Delivery mode: {delivery.get('mode', context.get('frame_delivery', '<unknown>'))}")
    lines.append(
        "Probe policy: "
        f"{context.get('probe_cache_policy', '<unknown>')} | "
        f"mid after position {schedule.get('mid_probe_after_position', '<none>')}"
    )
    lines.append(f"Seed: {reproducibility.get('seed')}")
    lines.append("")

    lines.append("Stream events:")
    for event in stream_events:
        generation = event.get("generation") or {}
        cache = event.get("cache_summary")
        event_delivery = event.get("delivery") or {}
        lines.append(
            f"- frame {event.get('frame_index')} @ {event.get('timecode')}: "
            f"{_one_line(event.get('assistant_text'))!r}, "
            f"gen_tokens={generation.get('generation_tokens')}, "
            f"cache_captured={bool(cache)}, "
            f"delivery={event_delivery.get('input_kind', '<unknown>')}"
        )
        artifact = event.get("frame_artifact")
        if artifact:
            lines.append(f"  frame artifact: {artifact.get('path')}")
        if cache:
            lines.append(
                "  "
                f"cache tokens={cache.get('token_count')} "
                f"layers={cache.get('reported_layers')}/{cache.get('total_layers')} "
                f"truncated={cache.get('truncated')}"
            )
    lines.append("")

    lines.append("Probe outputs:")
    for phase in ("before", "mid", "after"):
        records = probes.get(phase) or []
        if not records:
            continue
        for record in records:
            generation = record.get("generation") or {}
            before_cache = record.get("source_cache_summary_before_probe")
            branch_cache = record.get("cache_summary")
            lines.append(
                f"- {phase}/{record.get('probe_id')}: "
                f"branch={record.get('cache_branch_status', record.get('history_mutated', '<n/a>'))}, "
                f"gen_tokens={generation.get('generation_tokens')}, "
                f"text={_one_line(record.get('assistant_text'))!r}"
            )
            if before_cache:
                lines.append(
                    "  "
                    f"stream cache before probe: tokens={before_cache.get('token_count')} "
                    f"layers={before_cache.get('reported_layers')}/{before_cache.get('total_layers')}"
                )
            if branch_cache:
                lines.append(
                    "  "
                    f"probe branch after probe: tokens={branch_cache.get('token_count')} "
                    f"layers={branch_cache.get('reported_layers')}/{branch_cache.get('total_layers')}"
                )

    return "\n".join(lines)


def _one_line(value: Any, max_len: int = 180) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
