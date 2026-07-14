from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .image_stats import SCALAR_FRAME_FIELDS
from .processor_image_stats import SCALAR_PROCESSOR_FRAME_FIELDS
from .probe_readout import CELL_SEMANTICS
from .stimulus import write_json

EFFECT_FIELDS = ("spatial_main_effect", "palette_main_effect", "interaction_effect")


def load_stats(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_factorial_image_contrast(
    *,
    stats: dict[str, Any],
    mm_condition_id: str,
    jj_condition_id: str,
    mj_condition_id: str,
    jm_condition_id: str,
) -> dict[str, Any]:
    records_by_condition = _records_by_condition(stats)
    cells = {
        "mm": _require_record(records_by_condition, mm_condition_id),
        "jj": _require_record(records_by_condition, jj_condition_id),
        "mj": _require_record(records_by_condition, mj_condition_id),
        "jm": _require_record(records_by_condition, jm_condition_id),
    }
    metric_fields = _metric_fields(stats, cells)
    records = []
    for field in metric_fields:
        values = _cell_values(cells, field)
        if values is None:
            continue
        records.append(
            {
                "field": field,
                **values,
                **_effects(values),
            }
        )

    return {
        "schema_version": 1,
        "analysis_kind": "factorial_image_contrast_2x2",
        "source_analysis_kind": stats.get("analysis_kind"),
        "cells": {cell: _cell_header(record) for cell, record in cells.items()},
        "cell_semantics": dict(CELL_SEMANTICS),
        "contrast_formulas": {
            "spatial_main_effect": "((jm - mm) + (jj - mj)) / 2",
            "palette_main_effect": "((mj - mm) + (jj - jm)) / 2",
            "interaction_effect": "jj - jm - mj + mm",
        },
        "metric_fields": metric_fields,
        "records": records,
        "top_effects": _top_effects(records),
        "top_interaction_effects": _top_effects(records, effects=("interaction_effect",)),
        "interpretation_notes": [
            "This is a 2x2 contrast over aggregate image-statistic means, not model-state evidence.",
            "Use it beside cache-factorial contrasts to audit whether cache interactions track raw or processor-space image interactions.",
            "Processor-space statistics are usually the safer comparison surface for VLM inputs because they include resize, normalization, and tiling effects.",
        ],
    }


def write_factorial_image_contrast_json(analysis: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, analysis)


def write_factorial_image_contrast_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_factorial_image_contrast_markdown(analysis), encoding="utf-8")


def format_factorial_image_contrast_markdown(analysis: dict[str, Any]) -> str:
    title = _title_for_source_kind(analysis.get("source_analysis_kind"))
    lines = [
        f"# 2x2 Factorial {title} Contrast",
        "",
        "## Cells",
        "",
        "| Cell | Condition | Frames | Meaning |",
        "| --- | --- | ---: | --- |",
    ]
    for cell in ("mm", "jj", "mj", "jm"):
        header = analysis["cells"][cell]
        lines.append(
            "| "
            f"`{cell}` | "
            f"`{header.get('condition_id')}` | "
            f"{header.get('frame_count_analyzed')} | "
            f"{analysis['cell_semantics'][cell]} |"
        )

    lines.extend(["", "## Formulas", ""])
    for effect, formula in analysis["contrast_formulas"].items():
        lines.append(f"- `{effect}`: `{formula}`")

    lines.extend(
        [
            "",
            "## Top Effects",
            "",
            "| Effect | Field | Value | Relative | MM | JJ | MJ | JM |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["top_effects"][:16]:
        lines.append(
            "| "
            f"`{row['effect'].replace('_effect', '')}` | "
            f"`{row['field']}` | "
            f"{_fmt(row['effect_value'])} | "
            f"{_fmt(row.get('relative_abs_interaction_to_grand_mean'))} | "
            f"{_fmt(row['mm'])} | "
            f"{_fmt(row['jj'])} | "
            f"{_fmt(row['mj'])} | "
            f"{_fmt(row['jm'])} |"
        )

    lines.extend(
        [
            "",
            "## All Metric Effects",
            "",
            "| Field | Spatial | Palette | Interaction | Relative interaction |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in analysis["records"]:
        lines.append(
            "| "
            f"`{row['field']}` | "
            f"{_fmt(row['spatial_main_effect'])} | "
            f"{_fmt(row['palette_main_effect'])} | "
            f"{_fmt(row['interaction_effect'])} | "
            f"{_fmt(row.get('relative_abs_interaction_to_grand_mean'))} |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    for note in analysis["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _records_by_condition(stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = stats.get("records") if stats.get("records") is not None else [stats]
    output = {}
    for record in records or []:
        condition = record.get("condition") or {}
        condition_id = condition.get("condition_id")
        if condition_id:
            output[str(condition_id)] = record
    return output


def _require_record(records_by_condition: dict[str, dict[str, Any]], condition_id: str) -> dict[str, Any]:
    try:
        return records_by_condition[condition_id]
    except KeyError as exc:
        available = ", ".join(sorted(records_by_condition))
        raise ValueError(f"condition_id not found in stats: {condition_id}; available: {available}") from exc


def _metric_fields(stats: dict[str, Any], cells: dict[str, dict[str, Any]]) -> list[str]:
    if stats.get("analysis_kind") == "processor_image_statistics_batch":
        candidates = SCALAR_PROCESSOR_FRAME_FIELDS
    elif stats.get("analysis_kind") == "manifest_processor_image_statistics":
        candidates = SCALAR_PROCESSOR_FRAME_FIELDS
    else:
        candidates = SCALAR_FRAME_FIELDS
    fields = []
    for field in candidates:
        if all(field in (record.get("aggregate") or {}) for record in cells.values()):
            fields.append(field)
    return fields


def _cell_values(cells: dict[str, dict[str, Any]], field: str) -> dict[str, float] | None:
    output = {}
    for cell in ("mm", "jj", "mj", "jm"):
        summary = (cells[cell].get("aggregate") or {}).get(field) or {}
        try:
            output[cell] = float(summary["mean"])
        except (KeyError, TypeError, ValueError):
            return None
    return output


def _effects(values: dict[str, float]) -> dict[str, float | None]:
    mm = values["mm"]
    jj = values["jj"]
    mj = values["mj"]
    jm = values["jm"]
    grand_mean = (mm + jj + mj + jm) / 4.0
    spatial = ((jm - mm) + (jj - mj)) / 2.0
    palette = ((mj - mm) + (jj - jm)) / 2.0
    interaction = jj - jm - mj + mm
    relative = abs(interaction) / abs(grand_mean) if grand_mean != 0.0 else None
    return {
        "grand_mean": grand_mean,
        "spatial_main_effect": spatial,
        "palette_main_effect": palette,
        "interaction_effect": interaction,
        "abs_spatial_main_effect": abs(spatial),
        "abs_palette_main_effect": abs(palette),
        "abs_interaction_effect": abs(interaction),
        "relative_abs_interaction_to_grand_mean": relative,
    }


def _top_effects(
    records: list[dict[str, Any]],
    *,
    effects: tuple[str, ...] = EFFECT_FIELDS,
    limit: int = 12,
) -> list[dict[str, Any]]:
    output = []
    for effect in effects:
        ranked = sorted(records, key=lambda item: abs(float(item.get(effect) or 0.0)), reverse=True)
        output.extend(
            {
                **record,
                "effect": effect,
                "effect_value": record[effect],
                "abs_effect_value": abs(record[effect]),
            }
            for record in ranked[:limit]
            if abs(record[effect]) > 0.0
        )
    return sorted(output, key=lambda item: item["abs_effect_value"], reverse=True)[: limit * len(effects)]


def _cell_header(record: dict[str, Any]) -> dict[str, Any]:
    condition = record.get("condition") or {}
    return {
        "condition_id": condition.get("condition_id"),
        "condition_family": condition.get("condition_family"),
        "manifest_path": record.get("manifest_path"),
        "frame_count_analyzed": record.get("frame_count_analyzed"),
    }


def _title_for_source_kind(source_kind: Any) -> str:
    if source_kind in ("processor_image_statistics_batch", "manifest_processor_image_statistics"):
        return "Processor-Image"
    return "Image-Statistic"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
