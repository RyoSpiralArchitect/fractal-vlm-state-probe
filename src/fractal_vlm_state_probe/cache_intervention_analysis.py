from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .stimulus import write_json


def analyze_values_swap_interventions(
    runs: list[dict[str, Any]],
    *,
    run_paths: list[str] | None = None,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one values-swap run is required")
    if run_paths is not None and len(run_paths) != len(runs):
        raise ValueError("run_paths must align with runs")

    paths = run_paths or [None] * len(runs)
    expanded = _expand_runs(runs, paths)
    trials = [_analyze_run(run, run_path=path) for run, path in expanded]
    return {
        "schema_version": 1,
        "analysis_kind": "mlx_cache_values_swap_intervention",
        "input_run_count": len(runs),
        "trial_count": len(trials),
        "available_trial_count": sum(bool(trial["available"]) for trial in trials),
        "trials": trials,
        "group_summaries": _group_summaries(trials),
        "metric_semantics": {
            "effect_to_baseline_ratio": (
                "distance(intervention, origin) / distance(source, donor); "
                "0 is no measured effect and 1 matches baseline separation magnitude"
            ),
            "donor_pull_index": (
                "(distance(intervention, source) - distance(intervention, donor)) "
                "/ distance(source, donor); -1 is source-like and +1 is donor-like"
            ),
            "source_pull_index": (
                "(distance(intervention, donor) - distance(intervention, source)) "
                "/ distance(source, donor); -1 is donor-like and +1 is source-like"
            ),
        },
        "interpretation_notes": [
            "Token distance is normalized Levenshtein distance over saved generated token ids.",
            "Logprob distance is RMSE over tokens common to each aligned pair of saved top-k sets.",
            "Top-k distances are partial readout measurements, not full-vocabulary KL or JS divergence.",
            "Pull indices are unavailable when source and donor baselines are not separated on that metric.",
            "A causal claim requires matched sham or off-locus swaps and replication across probe seeds.",
        ],
    }


def write_values_swap_analysis_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_values_swap_analysis_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_values_swap_analysis_markdown(analysis), encoding="utf-8")


def format_values_swap_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Cache Values-Swap Intervention Analysis",
        "",
        "## Trials",
        "",
        "| Model | Pair | Phase | Layer | Seed | Baseline token distance | Intervention token effect | Token donor pull | Baseline top-k RMSE | Intervention top-k effect | Effect / baseline | Top-k donor pull |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in analysis.get("trials", []):
        metadata = trial["metadata"]
        source = metadata.get("source_condition") or metadata.get("source_label")
        donor = metadata.get("donor_condition") or metadata.get("donor_label")
        baseline = trial.get("baseline_distance") or {}
        intervention = trial.get("source_intervention") or {}
        lines.append(
            f"| `{metadata.get('model_id')}` | `{source}` -> `{donor}` | "
            f"`{metadata.get('probe_phase')}` | {metadata.get('layer_index')} | "
            f"{metadata.get('probe_seed')} | "
            f"{_format_number(_metric(baseline, 'token', 'normalized_edit_distance'))} | "
            f"{_format_number(_metric(intervention, 'distance_to_source', 'token', 'normalized_edit_distance'))} | "
            f"{_format_number(_metric(intervention, 'donor_pull_index', 'token'))} | "
            f"{_format_number(_metric(baseline, 'top_k', 'rmse'))} | "
            f"{_format_number(_metric(intervention, 'distance_to_source', 'top_k', 'rmse'))} | "
            f"{_format_number(_metric(intervention, 'effect_to_baseline_ratio', 'top_k'))} | "
            f"{_format_number(_metric(intervention, 'donor_pull_index', 'top_k'))} |"
        )

    lines.extend(
        [
            "",
            "## Group Summaries",
            "",
            "| Model | Pair | Phase | Layer | Trials | Token donor pull mean | Top-k effect / baseline mean | Top-k donor pull mean | Exact source/intervention | Exact self-sham |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in analysis.get("group_summaries", []):
        sham_available = group["source_self_sham_available_count"]
        sham_exact = group["source_self_sham_exact_match_count"]
        sham_display = f"{sham_exact}/{sham_available}" if sham_available else "n/a"
        lines.append(
            f"| `{group['model_id']}` | `{group['source_condition']}` -> `{group['donor_condition']}` | "
            f"`{group['probe_phase']}` | {group['layer_index']} | {group['trial_count']} | "
            f"{_format_number(_metric(group, 'metrics', 'token_donor_pull_index', 'mean'))} | "
            f"{_format_number(_metric(group, 'metrics', 'top_k_effect_to_baseline_ratio', 'mean'))} | "
            f"{_format_number(_metric(group, 'metrics', 'top_k_donor_pull_index', 'mean'))} | "
            f"{group['source_intervention_exact_match_count']}/{group['trial_count']} | "
            f"{sham_display} |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    for note in analysis.get("interpretation_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _analyze_run(run: dict[str, Any], *, run_path: str | None) -> dict[str, Any]:
    probes = run.get("probes") or {}
    required = ("source_baseline", "donor_baseline", "source_with_donor_values")
    missing = [label for label in required if not probes.get(label)]
    metadata = _run_metadata(run, run_path=run_path)
    if missing:
        return {
            "available": False,
            "missing_probe_labels": missing,
            "metadata": metadata,
        }

    source = probes["source_baseline"]
    donor = probes["donor_baseline"]
    source_intervention = probes["source_with_donor_values"]
    baseline_distance = _pair_distance(source, donor)
    source_result = _intervention_result(
        source=source,
        donor=donor,
        intervention=source_intervention,
        baseline_distance=baseline_distance,
        pull_toward="donor",
    )
    reciprocal = None
    if probes.get("donor_with_source_values"):
        reciprocal = _intervention_result(
            source=source,
            donor=donor,
            intervention=probes["donor_with_source_values"],
            baseline_distance=baseline_distance,
            pull_toward="source",
        )
    self_sham_distance = None
    if probes.get("source_with_source_values"):
        self_sham_distance = _pair_distance(probes["source_with_source_values"], source)
    return {
        "available": True,
        "missing_probe_labels": [],
        "metadata": metadata,
        "baseline_distance": baseline_distance,
        "source_intervention": source_result,
        "reciprocal_intervention": reciprocal,
        "source_self_sham_distance": self_sham_distance,
    }


def _expand_runs(
    runs: list[dict[str, Any]],
    paths: list[str | None],
) -> list[tuple[dict[str, Any], str | None]]:
    expanded = []
    for run, path in zip(runs, paths):
        if run.get("run_kind") != "mlx_cache_values_swap_probe_sweep":
            expanded.append((run, path))
            continue
        for trial in run.get("trials") or []:
            pseudo_run = {
                **run,
                "run_kind": "mlx_cache_values_swap_probe",
                "reproducibility": {
                    **(run.get("reproducibility") or {}),
                    "probe_seed": trial.get("probe_seed"),
                },
                "intervention_policy": {
                    **(run.get("intervention_policy") or {}),
                    "layer_index": trial.get("layer_index"),
                    "tensor": trial.get("tensor", "values"),
                },
                "probes": trial.get("probes") or {},
            }
            trial_path = (
                f"{path}#trial={trial.get('trial_index')}" if path is not None else None
            )
            expanded.append((pseudo_run, trial_path))
    return expanded


def _run_metadata(run: dict[str, Any], *, run_path: str | None) -> dict[str, Any]:
    stimulus = run.get("stimulus") or {}
    policy = run.get("intervention_policy") or {}
    manifests = run.get("manifests") or {}
    source_condition = (stimulus.get("source_condition") or {}).get("condition_id")
    donor_condition = (stimulus.get("donor_condition") or {}).get("condition_id")
    labels = list(manifests)
    return {
        "run_path": run_path,
        "model_id": run.get("model_id"),
        "source_label": labels[0] if labels else "source",
        "donor_label": labels[1] if len(labels) > 1 else "donor",
        "source_condition": source_condition,
        "donor_condition": donor_condition,
        "probe_phase": stimulus.get("probe_phase"),
        "frame_count_consumed": stimulus.get("frame_count_consumed"),
        "layer_index": policy.get("layer_index"),
        "tensor": policy.get("tensor"),
        "probe_seed": (run.get("reproducibility") or {}).get("probe_seed"),
        "probe_temperature": (run.get("context_policy") or {}).get("probe_temperature"),
    }


def _intervention_result(
    *,
    source: dict[str, Any],
    donor: dict[str, Any],
    intervention: dict[str, Any],
    baseline_distance: dict[str, Any],
    pull_toward: str,
) -> dict[str, Any]:
    distance_to_source = _pair_distance(intervention, source)
    distance_to_donor = _pair_distance(intervention, donor)
    token_source = _metric(distance_to_source, "token", "normalized_edit_distance")
    token_donor = _metric(distance_to_donor, "token", "normalized_edit_distance")
    token_baseline = _metric(baseline_distance, "token", "normalized_edit_distance")
    topk_source = _metric(distance_to_source, "top_k", "rmse")
    topk_donor = _metric(distance_to_donor, "top_k", "rmse")
    topk_baseline = _metric(baseline_distance, "top_k", "rmse")

    if pull_toward == "donor":
        token_origin = token_source
        topk_origin = topk_source
        token_pull = _normalized_pull(token_source, token_donor, token_baseline)
        topk_pull = _normalized_pull(topk_source, topk_donor, topk_baseline)
        pull_key = "donor_pull_index"
    else:
        token_origin = token_donor
        topk_origin = topk_donor
        token_pull = _normalized_pull(token_donor, token_source, token_baseline)
        topk_pull = _normalized_pull(topk_donor, topk_source, topk_baseline)
        pull_key = "source_pull_index"
    return {
        "assistant_text": intervention.get("assistant_text"),
        "distance_to_source": distance_to_source,
        "distance_to_donor": distance_to_donor,
        "effect_to_baseline_ratio": {
            "token": _safe_ratio(token_origin, token_baseline),
            "top_k": _safe_ratio(topk_origin, topk_baseline),
        },
        pull_key: {"token": token_pull, "top_k": topk_pull},
    }


def _pair_distance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_steps = _generation_steps(left)
    right_steps = _generation_steps(right)
    left_ids = [int(step["token_id"]) for step in left_steps if step.get("token_id") is not None]
    right_ids = [int(step["token_id"]) for step in right_steps if step.get("token_id") is not None]
    edit_distance = _levenshtein(left_ids, right_ids)
    denominator = max(len(left_ids), len(right_ids))
    return {
        "token": {
            "left_count": len(left_ids),
            "right_count": len(right_ids),
            "exact_match": left_ids == right_ids,
            "first_divergent_step": _first_divergent_step(left_ids, right_ids),
            "edit_distance": edit_distance,
            "normalized_edit_distance": edit_distance / denominator if denominator else 0.0,
        },
        "top_k": _top_k_distance(left_steps, right_steps),
    }


def _generation_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list((record.get("generation") or {}).get("steps") or [])


def _top_k_distance(
    left_steps: list[dict[str, Any]],
    right_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    deltas: list[float] = []
    jaccards: list[float] = []
    common_counts: list[int] = []
    for left, right in zip(left_steps, right_steps):
        left_map = _top_logprob_map(left)
        right_map = _top_logprob_map(right)
        left_ids = set(left_map)
        right_ids = set(right_map)
        union = left_ids | right_ids
        common = left_ids & right_ids
        jaccards.append(len(common) / len(union) if union else 1.0)
        common_counts.append(len(common))
        deltas.extend(left_map[token_id] - right_map[token_id] for token_id in common)
    return {
        "aligned_steps": min(len(left_steps), len(right_steps)),
        "mean_jaccard": mean(jaccards) if jaccards else None,
        "mean_common_token_count": mean(common_counts) if common_counts else None,
        "compared_logprob_count": len(deltas),
        "mean_abs_delta": mean(abs(value) for value in deltas) if deltas else None,
        "rmse": math.sqrt(mean(value * value for value in deltas)) if deltas else None,
        "max_abs_delta": max((abs(value) for value in deltas), default=None),
    }


def _top_logprob_map(step: dict[str, Any]) -> dict[int, float]:
    result = {}
    for item in step.get("top_logprobs") or []:
        token_id = item.get("token_id")
        logprob = item.get("logprob")
        if token_id is not None and logprob is not None:
            result[int(token_id)] = float(logprob)
    return result


def _levenshtein(left: list[int], right: list[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _first_divergent_step(left: list[int], right: list[int]) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _normalized_pull(
    distance_to_origin: float | None,
    distance_to_target: float | None,
    baseline_distance: float | None,
) -> float | None:
    if (
        distance_to_origin is None
        or distance_to_target is None
        or baseline_distance is None
        or baseline_distance <= 0
    ):
        return None
    return (distance_to_origin - distance_to_target) / baseline_distance


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _group_summaries(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for trial in trials:
        if not trial.get("available"):
            continue
        metadata = trial["metadata"]
        key = (
            metadata.get("model_id"),
            metadata.get("source_condition") or metadata.get("source_label"),
            metadata.get("donor_condition") or metadata.get("donor_label"),
            metadata.get("probe_phase"),
            metadata.get("layer_index"),
            metadata.get("tensor"),
        )
        groups.setdefault(key, []).append(trial)

    summaries = []
    for key, records in sorted(groups.items(), key=_group_sort_key):
        model_id, source, donor, phase, layer_index, tensor = key
        summaries.append(
            {
                "model_id": model_id,
                "source_condition": source,
                "donor_condition": donor,
                "probe_phase": phase,
                "layer_index": layer_index,
                "tensor": tensor,
                "trial_count": len(records),
                "probe_seeds": [record["metadata"].get("probe_seed") for record in records],
                "source_intervention_exact_match_count": sum(
                    bool(_metric(record, "source_intervention", "distance_to_source", "token", "exact_match"))
                    for record in records
                ),
                "source_self_sham_exact_match_count": sum(
                    bool(_metric(record, "source_self_sham_distance", "token", "exact_match"))
                    for record in records
                ),
                "source_self_sham_available_count": sum(
                    record.get("source_self_sham_distance") is not None for record in records
                ),
                "metrics": {
                    name: _summarize(values)
                    for name, values in _group_metric_values(records).items()
                },
            }
        )
    return summaries


def _group_sort_key(
    item: tuple[tuple[Any, ...], list[dict[str, Any]]],
) -> tuple[str, str, str, str, int, str]:
    model_id, source, donor, phase, layer_index, tensor = item[0]
    numeric_layer = int(layer_index) if isinstance(layer_index, int) else -1
    return (
        str(model_id),
        str(source),
        str(donor),
        str(phase),
        numeric_layer,
        str(tensor),
    )


def _group_metric_values(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    paths = {
        "baseline_token_distance": ("baseline_distance", "token", "normalized_edit_distance"),
        "baseline_top_k_rmse": ("baseline_distance", "top_k", "rmse"),
        "source_intervention_token_effect": (
            "source_intervention",
            "distance_to_source",
            "token",
            "normalized_edit_distance",
        ),
        "source_intervention_top_k_effect": (
            "source_intervention",
            "distance_to_source",
            "top_k",
            "rmse",
        ),
        "token_donor_pull_index": ("source_intervention", "donor_pull_index", "token"),
        "top_k_donor_pull_index": ("source_intervention", "donor_pull_index", "top_k"),
        "top_k_effect_to_baseline_ratio": (
            "source_intervention",
            "effect_to_baseline_ratio",
            "top_k",
        ),
        "token_source_pull_index": ("reciprocal_intervention", "source_pull_index", "token"),
        "top_k_source_pull_index": ("reciprocal_intervention", "source_pull_index", "top_k"),
        "self_sham_token_effect": (
            "source_self_sham_distance",
            "token",
            "normalized_edit_distance",
        ),
        "self_sham_top_k_effect": ("source_self_sham_distance", "top_k", "rmse"),
    }
    return {
        name: [float(value) for record in records if (value := _metric(record, *path)) is not None]
        for name, path in paths.items()
    }


def _summarize(values: Iterable[float]) -> dict[str, Any]:
    materialized = list(values)
    if not materialized:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(materialized),
        "mean": mean(materialized),
        "min": min(materialized),
        "max": max(materialized),
    }


def _metric(record: Any, *path: str) -> Any:
    value = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if value != 0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"
