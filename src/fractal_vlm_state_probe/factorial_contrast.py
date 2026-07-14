from __future__ import annotations

from pathlib import Path
from typing import Any

from .probe_readout import CELL_SEMANTICS
from .stimulus import write_json

CACHE_STAT_FIELDS = ("mean", "variance", "std", "abs_mean", "l2_norm")
SEQUENCE_POSITION_STAT_FIELDS = ("mean", "variance", "abs_mean", "l2_norm")
EFFECT_FIELDS = ("spatial_main_effect", "palette_main_effect", "interaction_effect")


def analyze_factorial_cache_contrast(
    *,
    mm: dict[str, Any],
    jj: dict[str, Any],
    mj: dict[str, Any],
    jm: dict[str, Any],
) -> dict[str, Any]:
    cells = {"mm": mm, "jj": jj, "mj": mj, "jm": jm}
    summary_sets = {cell: _source_cache_summaries(run) for cell, run in cells.items()}
    common_probe_keys = sorted(set.intersection(*(set(items) for items in summary_sets.values())))

    scalar_records: list[dict[str, Any]] = []
    sequence_position_records: list[dict[str, Any]] = []
    for phase, probe_id in common_probe_keys:
        summaries = {cell: items[(phase, probe_id)] for cell, items in summary_sets.items()}
        scalar_records.extend(_scalar_contrast_records(phase=phase, probe_id=probe_id, summaries=summaries))
        sequence_position_records.extend(
            _sequence_position_contrast_records(phase=phase, probe_id=probe_id, summaries=summaries)
        )

    return {
        "schema_version": 1,
        "analysis_kind": "factorial_cache_contrast_2x2",
        "cells": {cell: _run_header(run) for cell, run in cells.items()},
        "cell_semantics": dict(CELL_SEMANTICS),
        "contrast_formulas": {
            "spatial_main_effect": "((jm - mm) + (jj - mj)) / 2",
            "palette_main_effect": "((mj - mm) + (jj - jm)) / 2",
            "interaction_effect": "jj - jm - mj + mm",
        },
        "probe_keys": [{"phase": phase, "probe_id": probe_id} for phase, probe_id in common_probe_keys],
        "scalar_records": scalar_records,
        "sequence_position_records": sequence_position_records,
        "top_scalar_l2_effects": _top_l2_effects(scalar_records),
        "top_sequence_position_l2_effects": _top_l2_effects(sequence_position_records),
        "interaction_argmax_summary": _interaction_argmax_summary(
            scalar_records=scalar_records,
            sequence_position_records=sequence_position_records,
        ),
        "interpretation_notes": [
            "This is a factorial contrast over saved cache summary statistics, not a full hidden-state vector contrast.",
            "Signed effects preserve direction for scalar summaries but cannot recover vector direction lost before saving.",
            "The interaction term estimates whether palette effect changes with spatial-rank donor in the saved summary field.",
            "Sequence-position argmax is only over positions sampled into the run artifact, not a full token-position search.",
        ],
    }


def write_factorial_contrast_json(analysis: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, analysis)


def write_factorial_contrast_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_factorial_contrast_markdown(analysis), encoding="utf-8")


def format_factorial_contrast_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# 2x2 Factorial Cache Contrast",
        "",
        "## Cells",
        "",
        "| Cell | Condition | Meaning |",
        "| --- | --- | --- |",
    ]
    for cell in ("mm", "jj", "mj", "jm"):
        header = analysis["cells"][cell]
        lines.append(
            f"| `{cell}` | `{header.get('condition_id')}` | {analysis['cell_semantics'][cell]} |"
        )

    lines.extend(
        [
            "",
            "## Formulas",
            "",
        ]
    )
    for effect, formula in analysis["contrast_formulas"].items():
        lines.append(f"- `{effect}`: `{formula}`")

    lines.extend(
        [
            "",
            "## Top Scalar L2 Effects",
            "",
            "| Effect | Phase | Probe | Layer | Tensor | Value | Relative |",
            "| --- | --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in analysis["top_scalar_l2_effects"][:12]:
        lines.append(_format_effect_row(row, include_position=False))

    lines.extend(
        [
            "",
            "## Top Sequence-Position L2 Effects",
            "",
            "| Effect | Phase | Probe | Layer | Tensor | Position | Value | Relative |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    if analysis["top_sequence_position_l2_effects"]:
        for row in analysis["top_sequence_position_l2_effects"][:12]:
            lines.append(_format_effect_row(row, include_position=True))
    else:
        lines.append("| `none` | `n/a` | `n/a` |  | `n/a` |  |  |  |")

    lines.extend(
        [
            "",
            "## Interaction Argmax Persistence",
            "",
            "| Record type | Probe | Mid location | After location | Same location |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in analysis["interaction_argmax_summary"]:
        lines.append(
            "| "
            f"`{row['record_type']}` | "
            f"`{row['probe_id']}` | "
            f"{_location_text(row.get('mid'))} | "
            f"{_location_text(row.get('after'))} | "
            f"`{row['same_location']}` |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    for note in analysis["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _source_cache_summaries(run: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for phase, probes in (run.get("probes") or {}).items():
        for probe in probes:
            summary = probe.get("source_cache_summary_before_probe")
            if summary and summary.get("available"):
                output[(str(phase), str(probe.get("probe_id")))] = summary
    return output


def _scalar_contrast_records(
    *,
    phase: str,
    probe_id: str,
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    layer_maps = {cell: _layers_by_index(summary) for cell, summary in summaries.items()}
    common_layers = sorted(set.intersection(*(set(layers) for layers in layer_maps.values())))
    records = []
    for layer_index in common_layers:
        layers = {cell: layer_maps[cell][layer_index] for cell in summaries}
        for tensor in ("keys", "values"):
            tensors = {cell: layers[cell].get(tensor) for cell in summaries}
            if not all(tensors.values()):
                continue
            for field in CACHE_STAT_FIELDS:
                values = _cell_values(tensors, field)
                if values is None:
                    continue
                records.append(
                    {
                        "record_type": "scalar",
                        "phase": phase,
                        "probe_id": probe_id,
                        "layer_index": layer_index,
                        "tensor": tensor,
                        "field": field,
                        **values,
                        **_effects(values),
                    }
                )
    return records


def _sequence_position_contrast_records(
    *,
    phase: str,
    probe_id: str,
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    layer_maps = {cell: _layers_by_index(summary) for cell, summary in summaries.items()}
    common_layers = sorted(set.intersection(*(set(layers) for layers in layer_maps.values())))
    records = []
    for layer_index in common_layers:
        layers = {cell: layer_maps[cell][layer_index] for cell in summaries}
        for tensor in ("keys", "values"):
            tensors = {cell: layers[cell].get(tensor) for cell in summaries}
            if not all(tensors.values()):
                continue
            position_maps = {cell: _positions_by_index(tensors[cell]) for cell in tensors}
            common_positions = sorted(set.intersection(*(set(positions) for positions in position_maps.values())))
            for position in common_positions:
                position_records = {cell: position_maps[cell][position] for cell in position_maps}
                for field in SEQUENCE_POSITION_STAT_FIELDS:
                    values = _cell_values(position_records, field)
                    if values is None:
                        continue
                    records.append(
                        {
                            "record_type": "sequence_position",
                            "phase": phase,
                            "probe_id": probe_id,
                            "layer_index": layer_index,
                            "tensor": tensor,
                            "position": position,
                            "field": field,
                            **values,
                            **_effects(values),
                        }
                    )
    return records


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


def _cell_values(records: dict[str, dict[str, Any]], field: str) -> dict[str, float] | None:
    output = {}
    for cell in ("mm", "jj", "mj", "jm"):
        try:
            output[cell] = float(records[cell][field])
        except (KeyError, TypeError, ValueError):
            return None
    return output


def _layers_by_index(summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(layer["layer_index"]): layer
        for layer in summary.get("layers", [])
        if layer.get("layer_index") is not None
    }


def _positions_by_index(tensor: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(record["position"]): record
        for record in tensor.get("sequence_position_stats", [])
        if record.get("position") is not None
    }


def _top_l2_effects(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    l2_records = [record for record in records if record.get("field") == "l2_norm"]
    output = []
    for effect in EFFECT_FIELDS:
        ranked = [
            record
            for record in sorted(l2_records, key=lambda item: abs(item[effect]), reverse=True)
            if abs(record[effect]) > 0.0
        ]
        output.extend(
            {
                **record,
                "effect": effect,
                "effect_value": record[effect],
                "abs_effect_value": abs(record[effect]),
            }
            for record in ranked[:limit]
        )
    return sorted(output, key=lambda item: item["abs_effect_value"], reverse=True)[: limit * len(EFFECT_FIELDS)]


def _interaction_argmax_summary(
    *,
    scalar_records: list[dict[str, Any]],
    sequence_position_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for record_type, records in (
        ("scalar", scalar_records),
        ("sequence_position", sequence_position_records),
    ):
        probe_ids = sorted({record["probe_id"] for record in records})
        for probe_id in probe_ids:
            mid = _top_interaction_l2(records, phase="mid", probe_id=probe_id)
            after = _top_interaction_l2(records, phase="after", probe_id=probe_id)
            output.append(
                {
                    "record_type": record_type,
                    "probe_id": probe_id,
                    "mid": mid,
                    "after": after,
                    "same_location": _same_location(mid, after),
                }
            )
    return output


def _top_interaction_l2(records: list[dict[str, Any]], *, phase: str, probe_id: str) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if record.get("phase") == phase and record.get("probe_id") == probe_id and record.get("field") == "l2_norm"
    ]
    if not candidates:
        return None
    top = max(candidates, key=lambda item: item.get("abs_interaction_effect") or 0.0)
    if not top.get("abs_interaction_effect"):
        return None
    keys = ("phase", "probe_id", "layer_index", "tensor", "position", "interaction_effect", "abs_interaction_effect")
    return {key: top.get(key) for key in keys if key in top}


def _same_location(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool | None:
    if not left or not right:
        return None
    return (
        left.get("layer_index"),
        left.get("tensor"),
        left.get("position"),
    ) == (
        right.get("layer_index"),
        right.get("tensor"),
        right.get("position"),
    )


def _format_effect_row(row: dict[str, Any], *, include_position: bool) -> str:
    effect_name = str(row["effect"]).replace("_effect", "")
    relative = row.get("relative_abs_interaction_to_grand_mean")
    cells = [
        f"`{effect_name}`",
        f"`{row['phase']}`",
        f"`{row['probe_id']}`",
        str(row["layer_index"]),
        f"`{row['tensor']}`",
    ]
    if include_position:
        cells.append(str(row.get("position")))
    cells.extend(
        [
            _fmt(row["effect_value"]),
            _fmt(relative),
        ]
    )
    return "| " + " | ".join(cells) + " |"


def _location_text(record: dict[str, Any] | None) -> str:
    if not record:
        return "`n/a`"
    location = f"layer {record['layer_index']} `{record['tensor']}`"
    if "position" in record:
        location += f" pos {record['position']}"
    return f"{location}, interaction={_fmt(record.get('interaction_effect'))}"


def _run_header(run: dict[str, Any]) -> dict[str, Any]:
    stimulus = run.get("stimulus") or {}
    condition = stimulus.get("condition") or {}
    reproducibility = run.get("reproducibility") or {}
    return {
        "source_path": run.get("_source_path"),
        "condition_id": condition.get("condition_id"),
        "model": run.get("model_id") or run.get("model_ref"),
        "seed": reproducibility.get("seed"),
        "probe_seed": reproducibility.get("probe_seed"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
