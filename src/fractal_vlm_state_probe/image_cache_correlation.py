from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from .image_stats import SCALAR_FRAME_FIELDS
from .stimulus import write_json

CACHE_METRICS = ("max_abs_l2_delta", "mean_abs_l2_delta")


@dataclass(frozen=True)
class CacheDistanceRow:
    left: str
    right: str
    phase: str
    max_abs_l2_delta: float | None
    mean_abs_l2_delta: float | None
    source_path: str
    comparison_kind: str | None = None


def analyze_image_cache_correlations(
    *,
    image_stats_paths: list[Path],
    cache_analysis_paths: list[Path],
    batch_summary_paths: list[Path] | None = None,
    condition_aliases: dict[str, str] | None = None,
    min_samples: int = 3,
) -> dict[str, Any]:
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    image_index = _load_image_index(image_stats_paths)
    aliases = _build_aliases(
        image_index=image_index,
        batch_summary_paths=batch_summary_paths or [],
        explicit_aliases=condition_aliases or {},
    )
    cache_rows = _load_cache_rows(cache_analysis_paths)
    joined_rows = _join_rows(cache_rows=cache_rows, image_index=image_index, aliases=aliases)
    unmatched_rows = [
        _cache_row_to_dict(cache_row)
        for cache_row in cache_rows
        if not _can_join_cache_row(cache_row, image_index=image_index, aliases=aliases)
    ]
    return {
        "schema_version": 1,
        "analysis_kind": "image_cache_correlation",
        "image_stats_paths": [str(path) for path in image_stats_paths],
        "cache_analysis_paths": [str(path) for path in cache_analysis_paths],
        "batch_summary_paths": [str(path) for path in batch_summary_paths or []],
        "min_samples": min_samples,
        "image_delta_fields": list(SCALAR_FRAME_FIELDS),
        "cache_metrics": list(CACHE_METRICS),
        "condition_aliases": dict(sorted(aliases.items())),
        "joined_row_count": len(joined_rows),
        "unmatched_cache_rows": unmatched_rows,
        "joined_rows": joined_rows,
        "correlations": _correlations(joined_rows, min_samples=min_samples),
        "top_joined_by_cache_distance": _top_joined_rows(joined_rows),
        "interpretation_notes": [
            "Correlation here is descriptive over saved pilot artifacts; it is not a causal estimate.",
            "Rows join pairwise image-stat deltas to sampled source-cache summary distances.",
            "Probe-seed repeats do not create independent source-cache replications when streams are deterministic and probes are isolated branch reads.",
        ],
    }


def write_image_cache_correlation_json(analysis: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, analysis)


def write_image_cache_correlation_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_image_cache_correlation_markdown(analysis), encoding="utf-8")


def format_image_cache_correlation_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Image-Stat / Cache-Distance Correlation",
        "",
        f"- Joined comparisons: `{analysis['joined_row_count']}`",
        f"- Unmatched cache rows: `{len(analysis['unmatched_cache_rows'])}`",
        f"- Minimum samples per correlation: `{analysis['min_samples']}`",
        "",
        "## Strongest Correlations",
        "",
        "| Phase | Cache metric | Image-stat delta | N | Pearson | Spearman |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    strongest = sorted(
        analysis["correlations"],
        key=lambda item: abs(item["pearson"]) if item.get("pearson") is not None else -1,
        reverse=True,
    )[:12]
    for item in strongest:
        lines.append(
            "| "
            f"`{item['phase']}` | "
            f"`{item['cache_metric']}` | "
            f"`{item['image_delta_field']}` | "
            f"{item['sample_count']} | "
            f"{_fmt(item.get('pearson'))} | "
            f"{_fmt(item.get('spearman'))} |"
        )

    lines.extend(
        [
            "",
            "## Largest Joined Cache Distances",
            "",
            "| Phase | Left | Right | Cache max L2 | Cache mean L2 | Largest image-stat deltas |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in analysis["top_joined_by_cache_distance"][:16]:
        top_deltas = ", ".join(
            f"{item['field']}={_fmt(item['value'])}" for item in row["top_image_deltas"][:3]
        )
        lines.append(
            "| "
            f"`{row['phase']}` | "
            f"`{row['left']}` | "
            f"`{row['right']}` | "
            f"{_fmt(row.get('max_abs_l2_delta'))} | "
            f"{_fmt(row.get('mean_abs_l2_delta'))} | "
            f"{top_deltas} |"
        )

    if analysis["unmatched_cache_rows"]:
        lines.extend(["", "## Unmatched Cache Rows", ""])
        for row in analysis["unmatched_cache_rows"][:20]:
            lines.append(f"- `{row['left']}` vs `{row['right']}` / `{row['phase']}` from `{row['source_path']}`")

    lines.extend(["", "## Interpretation Notes", ""])
    for note in analysis["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _load_image_index(paths: list[Path]) -> dict[str, Any]:
    condition_records: dict[str, dict[str, Any]] = {}
    manifest_to_condition: dict[str, str] = {}
    pairwise_deltas: dict[tuple[str, str], dict[str, float]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        records = data.get("records") if data.get("analysis_kind") == "image_statistics_batch" else [data]
        for record in records or []:
            condition = record.get("condition") or {}
            condition_id = condition.get("condition_id")
            if not condition_id:
                continue
            condition_records[str(condition_id)] = record
            manifest_path = record.get("manifest_path")
            if manifest_path:
                manifest_to_condition[_normalize_path(str(manifest_path))] = str(condition_id)
        for item in data.get("pairwise_deltas") or []:
            left = item.get("left_condition_id")
            right = item.get("right_condition_id")
            if left and right:
                pairwise_deltas[_pair_key(str(left), str(right))] = {
                    field: float(value)
                    for field, value in (item.get("aggregate_mean_abs_deltas") or {}).items()
                    if value is not None
                }
    return {
        "condition_records": condition_records,
        "manifest_to_condition": manifest_to_condition,
        "pairwise_deltas": pairwise_deltas,
    }


def _build_aliases(
    *,
    image_index: dict[str, Any],
    batch_summary_paths: list[Path],
    explicit_aliases: dict[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for condition_id in image_index["condition_records"]:
        aliases[condition_id] = condition_id
        for alias in _inferred_aliases(condition_id):
            aliases.setdefault(alias, condition_id)

    manifest_to_condition = image_index["manifest_to_condition"]
    for path in batch_summary_paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for key, manifest_path in (data.get("conditions") or {}).items():
            condition_id = manifest_to_condition.get(_normalize_path(str(manifest_path)))
            if condition_id:
                aliases[str(key)] = condition_id

    for alias, condition_id in explicit_aliases.items():
        if condition_id not in image_index["condition_records"]:
            raise ValueError(f"explicit alias {alias!r} points to unknown condition_id {condition_id!r}")
        aliases[alias] = condition_id
    return aliases


def _inferred_aliases(condition_id: str) -> set[str]:
    aliases = {condition_id}
    if condition_id == "blank_visual_null":
        aliases.update({"blank", "null_blank"})
    for prefix in ("mandelbrot", "julia", "checkerboard", "voronoi", "quasicrystal", "white_noise", "blue_noise"):
        if condition_id.startswith(prefix):
            aliases.add(prefix)
    for family in ("mandelbrot", "julia"):
        if condition_id.startswith(f"{family}_low_lq_cutoff_"):
            aliases.add(condition_id.replace(f"{family}_low_lq_cutoff_", f"{family}_low_"))
        if condition_id.startswith(f"{family}_high_lq_cutoff_"):
            aliases.add(condition_id.replace(f"{family}_high_lq_cutoff_", f"{family}_high_"))
        if condition_id.startswith(f"{family}_zoom") and condition_id.endswith("_phase_scrambled"):
            aliases.add(f"{family}_phase")
        if condition_id.startswith(f"{family}_zoom") and condition_id.endswith("_phase_scrambled_quantile_matched"):
            aliases.add(f"{family}_rgb_quantile")
        if condition_id.startswith(f"{family}_zoom") and condition_id.endswith("_phase_scrambled_luminance_quantile_matched"):
            aliases.add(f"{family}_luminance_quantile")
            aliases.add(f"{family}_phase_lq")
        if condition_id.startswith(f"{family}_") and condition_id.endswith("_low_pass_luminance_quantile_matched"):
            aliases.add(f"{family}_low_lq")
        if condition_id.startswith(f"{family}_") and condition_id.endswith("_high_pass_luminance_quantile_matched"):
            aliases.add(f"{family}_high_lq")
    return aliases


def _load_cache_rows(paths: list[Path]) -> list[CacheDistanceRow]:
    rows: list[CacheDistanceRow] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        source_path = str(path)
        if data.get("analysis_kind") == "paired_stochastic_probe_batch":
            for item in data.get("pairwise_cache_distances") or []:
                left, right = _split_comparison_id(str(item.get("comparison_id")))
                rows.append(
                    CacheDistanceRow(
                        left=left,
                        right=right,
                        phase=str(item.get("phase")),
                        max_abs_l2_delta=_optional_float(item.get("mean_max_abs_l2_delta")),
                        mean_abs_l2_delta=_optional_float(item.get("mean_mean_abs_l2_delta")),
                        source_path=source_path,
                    )
                )
        elif data.get("comparison_kind") == "paired_run_comparison":
            left = ((data.get("left") or {}).get("condition_id")) or ((data.get("left") or {}).get("source_path"))
            right = ((data.get("right") or {}).get("condition_id")) or ((data.get("right") or {}).get("source_path"))
            for item in data.get("probe_source_cache_comparison") or []:
                delta = item.get("source_cache_delta") or {}
                if not delta.get("available"):
                    continue
                rows.append(
                    CacheDistanceRow(
                        left=str(left),
                        right=str(right),
                        phase=str(item.get("phase")),
                        max_abs_l2_delta=_optional_float(delta.get("max_abs_l2_delta")),
                        mean_abs_l2_delta=_optional_float(delta.get("mean_abs_l2_delta")),
                        source_path=source_path,
                    )
                )
        elif data.get("records"):
            for item in data.get("records") or []:
                for phase in ("mid", "after"):
                    rows.append(
                        CacheDistanceRow(
                            left=str(item.get("left")),
                            right=str(item.get("right")),
                            phase=phase,
                            max_abs_l2_delta=_optional_float(item.get(f"{phase}_max_l2")),
                            mean_abs_l2_delta=_optional_float(item.get(f"{phase}_mean_l2")),
                            source_path=source_path,
                            comparison_kind=item.get("comparison_kind"),
                        )
                    )
        else:
            raise ValueError(f"unsupported cache analysis shape: {path}")
    return [row for row in rows if row.phase in ("mid", "after")]


def _join_rows(
    *,
    cache_rows: list[CacheDistanceRow],
    image_index: dict[str, Any],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    output = []
    pairwise_deltas = image_index["pairwise_deltas"]
    for row in cache_rows:
        left_condition = aliases.get(row.left, row.left)
        right_condition = aliases.get(row.right, row.right)
        deltas = pairwise_deltas.get(_pair_key(left_condition, right_condition))
        if not deltas:
            continue
        output.append(
            {
                "left": row.left,
                "right": row.right,
                "left_condition_id": left_condition,
                "right_condition_id": right_condition,
                "phase": row.phase,
                "comparison_kind": row.comparison_kind,
                "source_path": row.source_path,
                "max_abs_l2_delta": row.max_abs_l2_delta,
                "mean_abs_l2_delta": row.mean_abs_l2_delta,
                "image_stat_deltas": deltas,
            }
        )
    return output


def _can_join_cache_row(
    row: CacheDistanceRow,
    *,
    image_index: dict[str, Any],
    aliases: dict[str, str],
) -> bool:
    left_condition = aliases.get(row.left, row.left)
    right_condition = aliases.get(row.right, row.right)
    return _pair_key(left_condition, right_condition) in image_index["pairwise_deltas"]


def _correlations(joined_rows: list[dict[str, Any]], *, min_samples: int) -> list[dict[str, Any]]:
    output = []
    phases = sorted({row["phase"] for row in joined_rows})
    for phase in ["all", *phases]:
        phase_rows = joined_rows if phase == "all" else [row for row in joined_rows if row["phase"] == phase]
        for cache_metric in CACHE_METRICS:
            for image_field in SCALAR_FRAME_FIELDS:
                pairs = [
                    (row.get(cache_metric), (row.get("image_stat_deltas") or {}).get(image_field))
                    for row in phase_rows
                ]
                pairs = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
                if len(pairs) < min_samples:
                    continue
                xs = [x for x, _ in pairs]
                ys = [y for _, y in pairs]
                output.append(
                    {
                        "phase": phase,
                        "cache_metric": cache_metric,
                        "image_delta_field": image_field,
                        "sample_count": len(pairs),
                        "pearson": _pearson(xs, ys),
                        "spearman": _spearman(xs, ys),
                    }
                )
    return output


def _top_joined_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        image_deltas = row.get("image_stat_deltas") or {}
        enriched.append(
            {
                **{key: value for key, value in row.items() if key != "image_stat_deltas"},
                "top_image_deltas": [
                    {"field": field, "value": value}
                    for field, value in sorted(image_deltas.items(), key=lambda item: abs(item[1]), reverse=True)
                ],
            }
        )
    return sorted(enriched, key=lambda item: item.get("max_abs_l2_delta") or -1, reverse=True)


def _split_comparison_id(value: str) -> tuple[str, str]:
    if "_vs_" not in value:
        raise ValueError(f"comparison_id does not contain _vs_: {value}")
    left, right = value.split("_vs_", 1)
    return left, right


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _normalize_path(path: str) -> str:
    return str(Path(path))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _cache_row_to_dict(row: CacheDistanceRow) -> dict[str, Any]:
    return {
        "left": row.left,
        "right": row.right,
        "phase": row.phase,
        "max_abs_l2_delta": row.max_abs_l2_delta,
        "mean_abs_l2_delta": row.mean_abs_l2_delta,
        "source_path": row.source_path,
        "comparison_kind": row.comparison_kind,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0.0 or y_den == 0.0:
        return None
    return numerator / (x_den * y_den)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        rank = (position + 1 + end) / 2
        for index, _ in indexed[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
