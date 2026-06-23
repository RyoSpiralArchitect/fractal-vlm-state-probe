from __future__ import annotations

from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from .stimulus import write_json

CELL_KEYS = ("mm", "jj", "mj", "jm")


def analyze_first_token_readout_contrast(
    *,
    mm: dict[str, Any],
    jj: dict[str, Any],
    mj: dict[str, Any],
    jm: dict[str, Any],
    max_token_effects: int = 20,
) -> dict[str, Any]:
    runs = {"mm": mm, "jj": jj, "mj": mj, "jm": jm}
    records = []
    top_token_effects = []
    for phase, probe_id in _phase_probe_keys(runs.values()):
        record = _analyze_phase_probe(runs, phase=phase, probe_id=probe_id)
        records.append(record)
        for token_effect in record.get("top_common_token_effects", [])[:max_token_effects]:
            top_token_effects.append(
                {
                    "phase": phase,
                    "probe_id": probe_id,
                    **token_effect,
                }
            )

    top_token_effects.sort(key=lambda item: item["abs_interaction_effect"], reverse=True)
    return {
        "schema_version": 1,
        "analysis_kind": "first_token_probe_readout_contrast",
        "cells": {
            key: _run_condition(runs[key])
            for key in CELL_KEYS
        },
        "cell_semantics": {
            "mm": "Mandelbrot spatial rank x Mandelbrot palette",
            "jj": "Julia spatial rank x Julia palette",
            "mj": "Mandelbrot spatial rank x Julia palette",
            "jm": "Julia spatial rank x Mandelbrot palette",
        },
        "contrast_formulas": {
            "spatial_main_effect": "((jm - mm) + (jj - mj)) / 2",
            "palette_main_effect": "((mj - mm) + (jj - jm)) / 2",
            "interaction_effect": "jj - jm - mj + mm",
        },
        "records": records,
        "top_common_token_effects": top_token_effects[:max_token_effects],
        "interpretation_notes": [
            "This analysis uses saved first generated-token top-k logprobs, not full-vocabulary distributions.",
            "KL/JS and full teacher-forced scoring require a dedicated scoring forward pass or full-vocabulary logits.",
            "Missing records usually mean the run was produced before generation step readouts were saved.",
        ],
    }


def write_first_token_readout_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_first_token_readout_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_first_token_readout_markdown(analysis), encoding="utf-8")


def format_first_token_readout_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# First-Token Probe Readout Contrast",
        "",
        "## Cells",
        "",
        "| Cell | Condition | Meaning |",
        "| --- | --- | --- |",
    ]
    for key in CELL_KEYS:
        lines.append(
            f"| `{key}` | `{analysis['cells'].get(key)}` | {analysis['cell_semantics'][key]} |"
        )
    lines.extend(
        [
            "",
            "## Phase / Probe Records",
            "",
            "| Phase | Probe | Available | Generated tokens | Mean top-k Jaccard | Common top-k tokens | Top interaction token | Interaction |",
            "| --- | --- | --- | --- | ---: | ---: | --- | ---: |",
        ]
    )
    for record in analysis["records"]:
        generated = ", ".join(
            f"{cell}=`{value.get('token', '')}`"
            for cell, value in record.get("generated_tokens", {}).items()
        )
        top_effect = (record.get("top_common_token_effects") or [{}])[0]
        token_label = (
            f"`{top_effect.get('token', '')}` ({top_effect.get('token_id', '')})"
            if top_effect
            else "n/a"
        )
        interaction = top_effect.get("interaction_effect")
        interaction_text = f"{interaction:.3f}" if isinstance(interaction, (int, float)) else "n/a"
        jaccard = record.get("mean_top_k_jaccard")
        jaccard_text = f"{jaccard:.3f}" if isinstance(jaccard, (int, float)) else "n/a"
        lines.append(
            f"| `{record['phase']}` | `{record['probe_id']}` | `{record['available']}` | "
            f"{generated or 'n/a'} | {jaccard_text} | {record.get('common_top_k_token_count', 0)} | "
            f"{token_label} | {interaction_text} |"
        )

    lines.extend(["", "## Top Common-Token Interaction Effects", ""])
    if analysis["top_common_token_effects"]:
        lines.extend(
            [
                "| Phase | Probe | Token | MM | JJ | MJ | JM | Interaction |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for effect in analysis["top_common_token_effects"]:
            lines.append(
                f"| `{effect['phase']}` | `{effect['probe_id']}` | "
                f"`{effect.get('token', '')}` ({effect['token_id']}) | "
                f"{effect['mm']:.3f} | {effect['jj']:.3f} | {effect['mj']:.3f} | {effect['jm']:.3f} | "
                f"{effect['interaction_effect']:.3f} |"
            )
    else:
        lines.append("No common top-k token effects were available.")

    lines.extend(["", "## Interpretation Notes", ""])
    for note in analysis.get("interpretation_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _analyze_phase_probe(
    runs: dict[str, dict[str, Any]],
    *,
    phase: str,
    probe_id: str,
) -> dict[str, Any]:
    steps = {
        cell: _first_generation_step(run, phase=phase, probe_id=probe_id)
        for cell, run in runs.items()
    }
    missing = [cell for cell, step in steps.items() if step is None]
    generated_tokens = {
        cell: {
            "token_id": step.get("token_id"),
            "token": step.get("token"),
            "token_logprob": step.get("token_logprob"),
        }
        for cell, step in steps.items()
        if step is not None
    }
    top_maps = {
        cell: _top_logprob_map(step)
        for cell, step in steps.items()
        if step is not None
    }
    common_ids = set.intersection(*(set(mapping) for mapping in top_maps.values())) if len(top_maps) == 4 else set()
    token_effects = [
        _token_effect(token_id, top_maps)
        for token_id in sorted(common_ids)
    ]
    token_effects.sort(key=lambda item: item["abs_interaction_effect"], reverse=True)
    return {
        "phase": phase,
        "probe_id": probe_id,
        "available": not missing,
        "missing_cells": missing,
        "generated_tokens": generated_tokens,
        "top1_tokens": {
            cell: _top1_record(step)
            for cell, step in steps.items()
            if step is not None
        },
        "mean_top_k_jaccard": _mean_jaccard(top_maps),
        "common_top_k_token_count": len(common_ids),
        "top_common_token_effects": token_effects[:20],
    }


def _token_effect(token_id: int, top_maps: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    values = {cell: top_maps[cell][token_id]["logprob"] for cell in CELL_KEYS}
    spatial = ((values["jm"] - values["mm"]) + (values["jj"] - values["mj"])) / 2.0
    palette = ((values["mj"] - values["mm"]) + (values["jj"] - values["jm"])) / 2.0
    interaction = values["jj"] - values["jm"] - values["mj"] + values["mm"]
    first = top_maps["mm"][token_id]
    return {
        "token_id": token_id,
        "token": first.get("token", ""),
        "mm": values["mm"],
        "jj": values["jj"],
        "mj": values["mj"],
        "jm": values["jm"],
        "spatial_main_effect": spatial,
        "palette_main_effect": palette,
        "interaction_effect": interaction,
        "abs_interaction_effect": abs(interaction),
    }


def _phase_probe_keys(runs: Any) -> list[tuple[str, str]]:
    keys = set()
    for run in runs:
        for phase, records in (run.get("probes") or {}).items():
            for record in records or []:
                probe_id = record.get("probe_id")
                if probe_id:
                    keys.add((phase, probe_id))
    phase_order = {"before": 0, "mid": 1, "after": 2}
    return sorted(keys, key=lambda item: (phase_order.get(item[0], 99), item[1]))


def _first_generation_step(
    run: dict[str, Any],
    *,
    phase: str,
    probe_id: str,
) -> dict[str, Any] | None:
    for record in (run.get("probes") or {}).get(phase, []) or []:
        if record.get("probe_id") != probe_id:
            continue
        steps = (record.get("generation") or {}).get("steps") or []
        return steps[0] if steps else None
    return None


def _top_logprob_map(step: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not step:
        return {}
    records = {}
    for item in step.get("top_logprobs") or []:
        token_id = item.get("token_id")
        if token_id is not None and item.get("logprob") is not None:
            records[int(token_id)] = item
    return records


def _top1_record(step: dict[str, Any] | None) -> dict[str, Any] | None:
    if not step:
        return None
    top = step.get("top_logprobs") or []
    return top[0] if top else None


def _mean_jaccard(top_maps: dict[str, dict[int, dict[str, Any]]]) -> float | None:
    if len(top_maps) < 2:
        return None
    values = []
    for left, right in combinations(top_maps, 2):
        left_ids = set(top_maps[left])
        right_ids = set(top_maps[right])
        union = left_ids | right_ids
        values.append(len(left_ids & right_ids) / len(union) if union else 1.0)
    return mean(values) if values else None


def _run_condition(run: dict[str, Any]) -> str | None:
    return ((run.get("stimulus") or {}).get("condition") or {}).get("condition_id")
