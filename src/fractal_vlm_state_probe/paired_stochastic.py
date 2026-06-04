from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from .stimulus import write_json

PHASES = ("before", "mid", "after")


def analyze_paired_stochastic_batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _probe_rows(records)
    condition_keys = sorted({row["condition_key"] for row in rows})
    probe_seeds = sorted({row["probe_seed"] for row in rows})
    return {
        "schema_version": 1,
        "analysis_kind": "paired_stochastic_probe_batch",
        "probe_seed_count": len(probe_seeds),
        "condition_keys": condition_keys,
        "before_alignment": _before_alignment(rows),
        "condition_phase_drift": _condition_phase_drift(rows),
        "pairwise_phase_distances": _pairwise_phase_distances(rows),
        "pairwise_cache_distances": _pairwise_cache_distances(records),
        "interpretation_notes": [
            "This summary uses surface lexical distances only; it is not an embedding-level semantic analysis.",
            "Phase distances are paired by probe seed and report mid/after distances after subtracting before-phase distance.",
            "The stream can remain deterministic while probe sampling is varied through matched probe seeds.",
        ],
    }


def write_paired_stochastic_json(analysis: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, analysis)


def write_paired_stochastic_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_paired_stochastic_markdown(analysis), encoding="utf-8")


def format_paired_stochastic_markdown(analysis: dict[str, Any]) -> str:
    alignment = analysis["before_alignment"]
    lines = [
        "# Paired Stochastic Probe Batch",
        "",
        f"- Probe seeds: `{analysis['probe_seed_count']}`",
        f"- Conditions: `{', '.join(analysis['condition_keys'])}`",
        f"- Before probes identical across conditions: `{alignment['all_conditions_same_count']}/{alignment['seed_count']}` seeds",
        "",
        "## Condition Drift From Before",
        "",
        "| Condition | Phase | Mean lexical distance | Std | Samples |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in analysis["condition_phase_drift"]:
        lines.append(
            f"| `{item['condition_key']}` | `{item['phase']}` | "
            f"{_fmt(item['mean_distance_from_before'])} | {_fmt(item['std_distance_from_before'])} | "
            f"{item['sample_count']} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Condition Distances",
            "",
            "| Left | Right | Phase | Mean distance | Mean before-adjusted distance | Samples |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in analysis["pairwise_phase_distances"]:
        lines.append(
            f"| `{item['left']}` | `{item['right']}` | `{item['phase']}` | "
            f"{_fmt(item['mean_distance'])} | {_fmt(item.get('mean_before_adjusted_distance'))} | "
            f"{item['sample_count']} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Source-Cache Summary Distances",
            "",
            "| Comparison | Phase | Mean max L2 delta | Std | Mean mean-L2 delta | Samples |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in analysis["pairwise_cache_distances"]:
        lines.append(
            f"| `{item['comparison_id']}` | `{item['phase']}` | "
            f"{_fmt(item['mean_max_abs_l2_delta'])} | {_fmt(item['std_max_abs_l2_delta'])} | "
            f"{_fmt(item['mean_mean_abs_l2_delta'])} | {item['sample_count']} |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    for note in analysis["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def lexical_distance(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(right))
    if not left_tokens and not right_tokens:
        return 0.0
    if not left_tokens or not right_tokens:
        return 1.0
    return 1.0 - len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _probe_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for batch_record in records:
        probe_seed = int(batch_record["probe_seed"])
        for condition_key, run_path in batch_record["runs"].items():
            with Path(run_path).open("r", encoding="utf-8") as handle:
                run = json.load(handle)
            condition = ((run.get("stimulus") or {}).get("condition") or {}).get("condition_id")
            for phase in PHASES:
                for probe in (run.get("probes") or {}).get(phase, []) or []:
                    rows.append(
                        {
                            "probe_seed": probe_seed,
                            "condition_key": condition_key,
                            "condition_id": condition,
                            "phase": phase,
                            "probe_id": probe.get("probe_id"),
                            "text": " ".join(str(probe.get("assistant_text", "")).split()),
                        }
                    )
    return rows


def _before_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed_probe: dict[tuple[int, str], list[str]] = {}
    for row in rows:
        if row["phase"] != "before":
            continue
        by_seed_probe.setdefault((row["probe_seed"], row["probe_id"]), []).append(row["text"])
    records = []
    same_count = 0
    for (probe_seed, probe_id), texts in sorted(by_seed_probe.items()):
        unique_texts = sorted(set(texts))
        same = len(unique_texts) == 1
        same_count += int(same)
        records.append(
            {
                "probe_seed": probe_seed,
                "probe_id": probe_id,
                "same_across_conditions": same,
                "unique_text_count": len(unique_texts),
            }
        )
    return {
        "seed_count": len(records),
        "all_conditions_same_count": same_count,
        "records": records,
    }


def _condition_phase_drift(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = _index_rows(rows)
    records = []
    condition_keys = sorted({key[1] for key in indexed})
    probe_ids = sorted({key[2] for key in indexed})
    for condition_key in condition_keys:
        for phase in ("mid", "after"):
            distances = []
            for probe_seed in sorted({key[0] for key in indexed}):
                for probe_id in probe_ids:
                    before = indexed.get((probe_seed, condition_key, probe_id, "before"))
                    current = indexed.get((probe_seed, condition_key, probe_id, phase))
                    if before is None or current is None:
                        continue
                    distances.append(lexical_distance(before, current))
            records.append(
                {
                    "condition_key": condition_key,
                    "phase": phase,
                    "sample_count": len(distances),
                    **_mean_std("distance_from_before", distances),
                }
            )
    return records


def _pairwise_phase_distances(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = _index_rows(rows)
    condition_keys = sorted({key[1] for key in indexed})
    probe_seeds = sorted({key[0] for key in indexed})
    probe_ids = sorted({key[2] for key in indexed})
    records = []
    for left, right in combinations(condition_keys, 2):
        for phase in PHASES:
            distances = []
            adjusted = []
            for probe_seed in probe_seeds:
                for probe_id in probe_ids:
                    left_text = indexed.get((probe_seed, left, probe_id, phase))
                    right_text = indexed.get((probe_seed, right, probe_id, phase))
                    if left_text is None or right_text is None:
                        continue
                    distance = lexical_distance(left_text, right_text)
                    distances.append(distance)
                    if phase != "before":
                        left_before = indexed.get((probe_seed, left, probe_id, "before"))
                        right_before = indexed.get((probe_seed, right, probe_id, "before"))
                        if left_before is not None and right_before is not None:
                            adjusted.append(distance - lexical_distance(left_before, right_before))
            record = {
                "left": left,
                "right": right,
                "phase": phase,
                "sample_count": len(distances),
                **_mean_std("distance", distances),
            }
            if phase != "before":
                record.update(_mean_std("before_adjusted_distance", adjusted))
            else:
                record["mean_before_adjusted_distance"] = None
                record["std_before_adjusted_distance"] = None
            records.append(record)
    return records


def _pairwise_cache_distances(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for batch_record in records:
        for comparison_id, paths in (batch_record.get("comparisons") or {}).items():
            json_path = paths.get("json")
            if not json_path:
                continue
            with Path(json_path).open("r", encoding="utf-8") as handle:
                comparison = json.load(handle)
            for item in comparison.get("probe_source_cache_comparison") or []:
                phase = item.get("phase")
                if phase not in ("mid", "after"):
                    continue
                delta = item.get("source_cache_delta") or {}
                if not delta.get("available"):
                    continue
                bucket = values.setdefault(
                    (comparison_id, phase),
                    {"max_abs_l2_delta": [], "mean_abs_l2_delta": []},
                )
                for field in bucket:
                    raw = delta.get(field)
                    if raw is not None:
                        bucket[field].append(float(raw))

    output = []
    for (comparison_id, phase), bucket in sorted(values.items()):
        max_l2 = bucket["max_abs_l2_delta"]
        mean_l2 = bucket["mean_abs_l2_delta"]
        output.append(
            {
                "comparison_id": comparison_id,
                "phase": phase,
                "sample_count": max(len(max_l2), len(mean_l2)),
                **_mean_std("max_abs_l2_delta", max_l2),
                **_mean_std("mean_abs_l2_delta", mean_l2),
            }
        )
    return output


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, str, str, str], str]:
    return {
        (row["probe_seed"], row["condition_key"], row["probe_id"], row["phase"]): row["text"]
        for row in rows
    }


def _mean_std(name: str, values: list[float]) -> dict[str, Any]:
    if not values:
        return {f"mean_{name}": None, f"std_{name}": None}
    return {
        f"mean_{name}": fmean(values),
        f"std_{name}": pstdev(values) if len(values) > 1 else 0.0,
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
