from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from .stimulus import write_json


def compare_cache_intervention_profiles(
    analyses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(analyses) < 2:
        raise ValueError("at least two intervention analyses are required")

    profiles = {
        label: _extract_profile(label, analysis)
        for label, analysis in analyses.items()
    }
    comparisons = [
        _compare_profiles(profiles[left], profiles[right])
        for left, right in combinations(profiles, 2)
    ]
    return {
        "schema_version": 1,
        "analysis_kind": "cache_intervention_layer_profile_comparison",
        "profile_count": len(profiles),
        "profiles": profiles,
        "comparisons": comparisons,
        "interpretation_notes": [
            "Effect ratio is the mean per-trial intervention-to-origin top-k RMSE divided by source-donor baseline RMSE.",
            "Profile correlations are descriptive; cache layers are not independent statistical samples.",
            "A screening profile without reciprocal or sham branches requires controlled follow-up at selected layers.",
            "Directional comparisons align source-with-donor and reciprocal donor-with-source effects within one controlled profile.",
        ],
    }


def write_profile_comparison_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_profile_comparison_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_profile_comparison_markdown(analysis), encoding="utf-8")


def format_profile_comparison_markdown(analysis: dict[str, Any]) -> str:
    profiles = analysis["profiles"]
    labels = list(profiles)
    all_layers = sorted(
        {record["layer_index"] for profile in profiles.values() for record in profile["layers"]}
    )
    by_label = {
        label: {record["layer_index"]: record for record in profile["layers"]}
        for label, profile in profiles.items()
    }
    lines = [
        "# Cache Intervention Layer Profiles",
        "",
        "## Profiles",
        "",
        "| Layer | " + " | ".join(f"{label} effect / baseline" for label in labels) + " |",
        "| ---: | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for layer in all_layers:
        values = [
            _format_number((by_label[label].get(layer) or {}).get("effect_to_baseline_ratio"))
            for label in labels
        ]
        lines.append(f"| {layer} | " + " | ".join(values) + " |")

    directional = [
        (label, profile.get("directional_comparison"))
        for label, profile in profiles.items()
        if (profile.get("directional_comparison") or {}).get("common_layer_count")
    ]
    if directional:
        lines.extend(
            [
                "",
                "## Reciprocal Direction Checks",
                "",
                "| Profile | Common layers | Pearson | Spearman | Mean abs difference | Argmax layers | Same argmax |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for label, record in directional:
            lines.append(
                f"| `{label}` | {record['common_layer_count']} | "
                f"{_format_number(record['pearson'])} | {_format_number(record['spearman'])} | "
                f"{_format_number(record['mean_abs_difference'])} | "
                f"{record['source_argmax_layer']} / {record['reciprocal_argmax_layer']} | "
                f"`{record['argmax_same']}` |"
            )

    lines.extend(["", "## Pairwise Comparisons", ""])
    lines.extend(
        [
            "| Left | Right | Common layers | Pearson | Spearman | Mean abs difference | Argmax layers | Top-3 overlap |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for record in analysis["comparisons"]:
        lines.append(
            f"| `{record['left']}` | `{record['right']}` | {record['common_layer_count']} | "
            f"{_format_number(record['pearson'])} | {_format_number(record['spearman'])} | "
            f"{_format_number(record['mean_abs_difference'])} | "
            f"{record['left_argmax_layer']} / {record['right_argmax_layer']} | "
            f"{record['top_3_overlap_count']} |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _extract_profile(label: str, analysis: dict[str, Any]) -> dict[str, Any]:
    groups = analysis.get("group_summaries") or []
    if not groups:
        raise ValueError(f"analysis {label!r} has no group summaries")
    identities = {
        (
            group.get("model_id"),
            group.get("source_condition"),
            group.get("donor_condition"),
            group.get("probe_phase"),
            group.get("tensor"),
        )
        for group in groups
    }
    if len(identities) != 1:
        raise ValueError(f"analysis {label!r} contains multiple model/pair/phase/tensor profiles")
    model_id, source, donor, phase, tensor = identities.pop()
    layers = []
    for group in sorted(groups, key=lambda value: int(value["layer_index"])):
        metrics = group.get("metrics") or {}
        layers.append(
            {
                "layer_index": int(group["layer_index"]),
                "effect_to_baseline_ratio": _mean_metric(
                    metrics, "top_k_effect_to_baseline_ratio"
                ),
                "effect_rmse": _mean_metric(
                    metrics, "source_intervention_top_k_effect"
                ),
                "donor_pull": _mean_metric(metrics, "top_k_donor_pull_index"),
                "reciprocal_effect_to_baseline_ratio": _mean_metric(
                    metrics, "reciprocal_top_k_effect_to_baseline_ratio"
                ),
                "reciprocal_source_pull": _mean_metric(
                    metrics, "top_k_source_pull_index"
                ),
                "self_sham_top_k_effect": _mean_metric(
                    metrics, "self_sham_top_k_effect"
                ),
                "trial_count": group.get("trial_count"),
                "probe_seeds": group.get("probe_seeds") or [],
            }
        )
    ranked = sorted(
        (record for record in layers if record["effect_to_baseline_ratio"] is not None),
        key=lambda record: record["effect_to_baseline_ratio"],
        reverse=True,
    )
    reciprocal_ranked = sorted(
        (
            record
            for record in layers
            if record["reciprocal_effect_to_baseline_ratio"] is not None
        ),
        key=lambda record: record["reciprocal_effect_to_baseline_ratio"],
        reverse=True,
    )
    return {
        "label": label,
        "model_id": model_id,
        "source_condition": source,
        "donor_condition": donor,
        "probe_phase": phase,
        "tensor": tensor,
        "layer_count": len(layers),
        "argmax_layer": ranked[0]["layer_index"] if ranked else None,
        "argmax_effect_to_baseline_ratio": (
            ranked[0]["effect_to_baseline_ratio"] if ranked else None
        ),
        "top_3_layers": [record["layer_index"] for record in ranked[:3]],
        "reciprocal_argmax_layer": (
            reciprocal_ranked[0]["layer_index"] if reciprocal_ranked else None
        ),
        "reciprocal_top_3_layers": [
            record["layer_index"] for record in reciprocal_ranked[:3]
        ],
        "directional_comparison": _directional_comparison(layers),
        "layers": layers,
    }


def _directional_comparison(layers: list[dict[str, Any]]) -> dict[str, Any]:
    common = [
        record
        for record in layers
        if record["effect_to_baseline_ratio"] is not None
        and record["reciprocal_effect_to_baseline_ratio"] is not None
    ]
    source_values = [record["effect_to_baseline_ratio"] for record in common]
    reciprocal_values = [
        record["reciprocal_effect_to_baseline_ratio"] for record in common
    ]
    source_ranked = sorted(
        common,
        key=lambda record: record["effect_to_baseline_ratio"],
        reverse=True,
    )
    reciprocal_ranked = sorted(
        common,
        key=lambda record: record["reciprocal_effect_to_baseline_ratio"],
        reverse=True,
    )
    source_argmax = source_ranked[0]["layer_index"] if source_ranked else None
    reciprocal_argmax = reciprocal_ranked[0]["layer_index"] if reciprocal_ranked else None
    return {
        "common_layers": [record["layer_index"] for record in common],
        "common_layer_count": len(common),
        "pearson": _pearson(source_values, reciprocal_values),
        "spearman": _pearson(
            _average_ranks(source_values),
            _average_ranks(reciprocal_values),
        ),
        "mean_abs_difference": (
            mean(abs(a - b) for a, b in zip(source_values, reciprocal_values))
            if common
            else None
        ),
        "source_argmax_layer": source_argmax,
        "reciprocal_argmax_layer": reciprocal_argmax,
        "argmax_same": source_argmax == reciprocal_argmax if common else None,
    }


def _compare_profiles(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_by_layer = {
        record["layer_index"]: record["effect_to_baseline_ratio"]
        for record in left["layers"]
        if record["effect_to_baseline_ratio"] is not None
    }
    right_by_layer = {
        record["layer_index"]: record["effect_to_baseline_ratio"]
        for record in right["layers"]
        if record["effect_to_baseline_ratio"] is not None
    }
    common_layers = sorted(set(left_by_layer) & set(right_by_layer))
    left_values = [left_by_layer[layer] for layer in common_layers]
    right_values = [right_by_layer[layer] for layer in common_layers]
    return {
        "left": left["label"],
        "right": right["label"],
        "common_layers": common_layers,
        "common_layer_count": len(common_layers),
        "pearson": _pearson(left_values, right_values),
        "spearman": _pearson(_average_ranks(left_values), _average_ranks(right_values)),
        "mean_abs_difference": (
            mean(abs(a - b) for a, b in zip(left_values, right_values))
            if common_layers
            else None
        ),
        "left_argmax_layer": left["argmax_layer"],
        "right_argmax_layer": right["argmax_layer"],
        "argmax_same": left["argmax_layer"] == right["argmax_layer"],
        "top_3_overlap_count": len(set(left["top_3_layers"]) & set(right["top_3_layers"])),
    }


def _mean_metric(metrics: dict[str, Any], name: str) -> float | None:
    value = (metrics.get(name) or {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def _average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2
        for index, _ in ordered[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def _format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.3f}"
