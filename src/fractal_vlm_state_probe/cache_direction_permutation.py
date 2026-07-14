from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Iterator

import numpy as np

from .stimulus import write_json


DEFAULT_REGIONS = ("image_tokens", "post_image")


def analyze_cache_direction_class_permutation(
    replication: dict[str, Any],
    *,
    class_by_label: dict[str, str],
    regions: Iterable[str] = DEFAULT_REGIONS,
    effect: str = "interaction",
    exact_limit: int = 100_000,
    monte_carlo_permutations: int = 100_000,
    seed: int = 20260714,
) -> dict[str, Any]:
    """Calibrate cross-pair direction by permuting atomic source-pair labels."""
    if replication.get("analysis_kind") != "source_cache_tensor_cross_pair_replication":
        raise ValueError("expected a source-cache tensor replication analysis")
    if effect not in {"spatial_main_effect", "palette_main_effect", "interaction"}:
        raise ValueError(f"unsupported factorial effect: {effect}")
    if exact_limit < 1:
        raise ValueError("exact_limit must be positive")
    if monte_carlo_permutations < 1:
        raise ValueError("monte_carlo_permutations must be positive")

    selected_regions = tuple(dict.fromkeys(str(region) for region in regions))
    if not selected_regions:
        raise ValueError("at least one token region is required")

    groups = []
    all_labels: set[str] = set()
    for group in replication.get("groups") or []:
        labels = tuple(sorted(str(point["label"]) for point in group["points"]))
        all_labels.update(labels)
        groups.append(
            _analyze_group(
                group,
                labels=labels,
                class_by_label=class_by_label,
                regions=selected_regions,
                effect=effect,
                exact_limit=exact_limit,
                monte_carlo_permutations=monte_carlo_permutations,
                seed=seed,
            )
        )

    extra_labels = sorted(set(class_by_label) - all_labels)
    if extra_labels:
        raise ValueError(f"class labels do not occur in replication: {extra_labels}")
    return {
        "schema_version": 1,
        "analysis_kind": "source_cache_direction_class_permutation",
        "source_analysis_kind": replication["analysis_kind"],
        "source_pair_is_permutation_unit": True,
        "cell_labels_are_permuted": False,
        "effect": effect,
        "regions": list(selected_regions),
        "group_count": len(groups),
        "source_pair_count": len(all_labels),
        "class_count": len(set(class_by_label.values())),
        "class_by_label": dict(sorted(class_by_label.items())),
        "exact_limit": exact_limit,
        "monte_carlo_permutations": monte_carlo_permutations,
        "seed": seed,
        "groups": groups,
        "interpretation_notes": [
            "Each complete source-pair factorial is one atomic permutation unit; MM/JJ/MJ/JM cells and effect-vector orientation remain fixed.",
            "The global one-sided statistic is mean within-class cosine minus mean between-class cosine.",
            "Class-specific one-sided tests compare within-class cosine with cosine across that class boundary.",
            "Pairwise class tests use the absolute difference between class-specific within-class mean cosines and are two-sided.",
            "Exact tests enumerate every class-label assignment that preserves observed class sizes when the assignment count is within exact_limit.",
            "Permutation p-values are unadjusted across regions, targets, and class contrasts.",
            "The test conditions on the observed source generators and does not establish exchangeability with unobserved visual populations.",
        ],
    }


def write_cache_direction_class_permutation_json(
    analysis: dict[str, Any], path: Path
) -> None:
    write_json(path, analysis)


def write_cache_direction_class_permutation_markdown(
    analysis: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_cache_direction_class_permutation_markdown(analysis),
        encoding="utf-8",
    )


def format_cache_direction_class_permutation_markdown(
    analysis: dict[str, Any],
) -> str:
    lines = [
        "# Source-Level Cache Direction Permutation",
        "",
        f"- Source-pair units: `{analysis['source_pair_count']}`",
        f"- Classes: `{analysis['class_count']}`",
        f"- Effect: `{analysis['effect']}`",
        "- Permutation unit: one complete source-pair factorial",
        "",
        "## Global Class Organization",
        "",
        "| Model | Layer | Tensor | Region | Units | Mean within | Mean between | Difference | Null 95% interval | p (greater) | Mode / assignments |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for group in analysis["groups"]:
        for region in group["regions"]:
            if not region["available"]:
                continue
            observed = region["observed"]
            test = region["global_test"]
            lines.append(
                f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                f"`{group['tensor']}` | `{region['region']}` | "
                f"{group['source_pair_count']} | "
                f"{_fmt(observed['within_class']['mean'])} | "
                f"{_fmt(observed['between_class']['mean'])} | "
                f"{_fmt(observed['mean_difference'])} | "
                f"{_interval(test['null'])} | {_fmt(test['p_greater'])} | "
                f"{test['mode']} / {test['assignment_count']} |"
            )

    lines.extend(
        [
            "",
            "## Class-Specific Coherence",
            "",
            "| Model | Layer | Region | Class | Units | Within mean | Boundary mean | Difference | p (greater) |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in analysis["groups"]:
        for region in group["regions"]:
            if not region["available"]:
                continue
            for record in region["class_tests"]:
                lines.append(
                    f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                    f"`{region['region']}` | `{record['class_label']}` | "
                    f"{record['source_pair_count']} | {_fmt(record['within_mean'])} | "
                    f"{_fmt(record['boundary_mean'])} | "
                    f"{_fmt(record['statistic'])} | {_fmt(record['p_greater'])} |"
                )

    lines.extend(
        [
            "",
            "## Pairwise Class Contrasts",
            "",
            "| Model | Layer | Region | Classes | Within means | Difference | p (two-sided) |",
            "| --- | ---: | --- | --- | --- | ---: | ---: |",
        ]
    )
    for group in analysis["groups"]:
        for region in group["regions"]:
            if not region["available"]:
                continue
            for record in region["pairwise_class_tests"]:
                lines.append(
                    f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                    f"`{region['region']}` | `{record['left_class']}` vs "
                    f"`{record['right_class']}` | "
                    f"{_fmt(record['left_within_mean'])} / "
                    f"{_fmt(record['right_within_mean'])} | "
                    f"{_fmt(record['difference'])} | "
                    f"{_fmt(record['p_two_sided'])} |"
                )

    unavailable = [
        (group, region)
        for group in analysis["groups"]
        for region in group["regions"]
        if not region["available"]
    ]
    if unavailable:
        lines.extend(["", "## Unavailable Regions", ""])
        lines.extend(
            f"- `{_short_model(group['model_id'])}` layer {group['layer_index']} "
            f"`{region['region']}`: {region['reason']}"
            for group, region in unavailable
        )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _analyze_group(
    group: dict[str, Any],
    *,
    labels: tuple[str, ...],
    class_by_label: dict[str, str],
    regions: tuple[str, ...],
    effect: str,
    exact_limit: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, Any]:
    missing = [label for label in labels if label not in class_by_label]
    if missing:
        raise ValueError(f"missing class labels for source pairs: {missing}")
    classes = tuple(class_by_label[label] for label in labels)
    counts = {name: classes.count(name) for name in sorted(set(classes))}
    if len(counts) < 2:
        raise ValueError("at least two source classes are required")
    if any(count < 2 for count in counts.values()):
        raise ValueError(f"every source class needs at least two pairs: {counts}")

    by_region = {record["region"]: record for record in group["regions"]}
    region_records = []
    for region_name in regions:
        record = by_region.get(region_name)
        if record is None:
            region_records.append(
                {
                    "region": region_name,
                    "available": False,
                    "reason": "region is absent from the replication analysis",
                }
            )
            continue
        pairwise = _effect_pairwise(record, effect)
        if any(not item.get("available") for item in pairwise):
            region_records.append(
                {
                    "region": region_name,
                    "available": False,
                    "reason": "one or more direction cosines are undefined",
                }
            )
            continue
        cosine_by_pair = _complete_cosine_map(labels, pairwise)
        region_records.append(
            _analyze_region(
                region_name,
                labels=labels,
                classes=classes,
                cosine_by_pair=cosine_by_pair,
                exact_limit=exact_limit,
                monte_carlo_permutations=monte_carlo_permutations,
                seed=seed,
            )
        )

    return {
        "model_id": str(group["model_id"]),
        "layer_index": int(group["layer_index"]),
        "tensor": str(group["tensor"]),
        "source_pair_count": len(labels),
        "source_pair_labels": list(labels),
        "class_counts": counts,
        "regions": region_records,
    }


def _analyze_region(
    region_name: str,
    *,
    labels: tuple[str, ...],
    classes: tuple[str, ...],
    cosine_by_pair: dict[tuple[int, int], float],
    exact_limit: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, Any]:
    observed = _class_organization(classes, cosine_by_pair)
    assignments = _assignment_stream(
        classes,
        exact_limit=exact_limit,
        monte_carlo_permutations=monte_carlo_permutations,
        seed=seed,
    )
    null_values = [
        _class_organization(assignment, cosine_by_pair)["mean_difference"]
        for assignment in assignments["assignments"]
    ]
    global_test = _permutation_test(
        observed["mean_difference"],
        null_values,
        mode=assignments["mode"],
        alternative="greater",
    )

    class_tests = [
        _class_specific_test(
            class_label,
            labels=labels,
            classes=classes,
            cosine_by_pair=cosine_by_pair,
        )
        for class_label in sorted(set(classes))
    ]
    pairwise_class_tests = [
        _pairwise_class_test(
            left,
            right,
            labels=labels,
            classes=classes,
            cosine_by_pair=cosine_by_pair,
        )
        for left, right in combinations(sorted(set(classes)), 2)
    ]
    return {
        "region": region_name,
        "available": True,
        "pairwise_cosine_count": len(cosine_by_pair),
        "observed": observed,
        "global_test": global_test,
        "class_tests": class_tests,
        "pairwise_class_tests": pairwise_class_tests,
    }


def _class_organization(
    classes: tuple[str, ...],
    cosine_by_pair: dict[tuple[int, int], float],
) -> dict[str, Any]:
    within = [
        cosine
        for (left, right), cosine in cosine_by_pair.items()
        if classes[left] == classes[right]
    ]
    between = [
        cosine
        for (left, right), cosine in cosine_by_pair.items()
        if classes[left] != classes[right]
    ]
    if not within or not between:
        raise ValueError(
            "class assignment must contain within- and between-class pairs"
        )
    return {
        "within_class": _summary(within),
        "between_class": _summary(between),
        "mean_difference": fmean(within) - fmean(between),
        "median_difference": median(within) - median(between),
    }


def _class_specific_test(
    class_label: str,
    *,
    labels: tuple[str, ...],
    classes: tuple[str, ...],
    cosine_by_pair: dict[tuple[int, int], float],
) -> dict[str, Any]:
    member_indices = tuple(
        index for index, value in enumerate(classes) if value == class_label
    )
    observed = _membership_statistic(set(member_indices), cosine_by_pair)
    null_values = [
        _membership_statistic(set(selected), cosine_by_pair)["statistic"]
        for selected in combinations(range(len(labels)), len(member_indices))
    ]
    test = _permutation_test(
        observed["statistic"],
        null_values,
        mode="exact",
        alternative="greater",
    )
    return {
        "class_label": class_label,
        "source_pair_count": len(member_indices),
        "source_pair_labels": [labels[index] for index in member_indices],
        **observed,
        **test,
    }


def _pairwise_class_test(
    left_class: str,
    right_class: str,
    *,
    labels: tuple[str, ...],
    classes: tuple[str, ...],
    cosine_by_pair: dict[tuple[int, int], float],
) -> dict[str, Any]:
    left_indices = tuple(
        index for index, value in enumerate(classes) if value == left_class
    )
    right_indices = tuple(
        index for index, value in enumerate(classes) if value == right_class
    )
    union = tuple(sorted(left_indices + right_indices))
    observed_left = _within_mean(set(left_indices), cosine_by_pair)
    observed_right = _within_mean(set(right_indices), cosine_by_pair)
    observed_difference = observed_left - observed_right
    null_values = []
    for selected in combinations(union, len(left_indices)):
        selected_set = set(selected)
        complement = set(union) - selected_set
        null_values.append(
            _within_mean(selected_set, cosine_by_pair)
            - _within_mean(complement, cosine_by_pair)
        )
    test = _permutation_test(
        observed_difference,
        null_values,
        mode="exact",
        alternative="two_sided",
    )
    return {
        "left_class": left_class,
        "right_class": right_class,
        "left_source_pair_count": len(left_indices),
        "right_source_pair_count": len(right_indices),
        "left_within_mean": observed_left,
        "right_within_mean": observed_right,
        "difference": observed_difference,
        **test,
    }


def _membership_statistic(
    selected: set[int], cosine_by_pair: dict[tuple[int, int], float]
) -> dict[str, float]:
    within = [
        value
        for (left, right), value in cosine_by_pair.items()
        if left in selected and right in selected
    ]
    boundary = [
        value
        for (left, right), value in cosine_by_pair.items()
        if (left in selected) != (right in selected)
    ]
    return {
        "within_mean": fmean(within),
        "boundary_mean": fmean(boundary),
        "statistic": fmean(within) - fmean(boundary),
    }


def _within_mean(
    selected: set[int], cosine_by_pair: dict[tuple[int, int], float]
) -> float:
    values = [
        value
        for (left, right), value in cosine_by_pair.items()
        if left in selected and right in selected
    ]
    return fmean(values)


def _assignment_stream(
    observed: tuple[str, ...],
    *,
    exact_limit: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, Any]:
    class_counts = {name: observed.count(name) for name in sorted(set(observed))}
    assignment_count = math.factorial(len(observed))
    for count in class_counts.values():
        assignment_count //= math.factorial(count)
    if assignment_count <= exact_limit:
        return {
            "mode": "exact",
            "assignment_count": assignment_count,
            "assignments": _exact_assignments(len(observed), class_counts),
        }

    rng = np.random.default_rng(seed)
    base = np.asarray(observed, dtype=object)

    def sampled() -> Iterator[tuple[str, ...]]:
        for _ in range(monte_carlo_permutations):
            yield tuple(str(value) for value in rng.permutation(base))

    return {
        "mode": "monte_carlo",
        "assignment_count": monte_carlo_permutations,
        "assignments": sampled(),
    }


def _exact_assignments(
    item_count: int, class_counts: dict[str, int]
) -> Iterator[tuple[str, ...]]:
    classes = tuple(sorted(class_counts))
    assignment = [""] * item_count

    def visit(
        class_index: int, remaining: tuple[int, ...]
    ) -> Iterator[tuple[str, ...]]:
        class_label = classes[class_index]
        if class_index == len(classes) - 1:
            for index in remaining:
                assignment[index] = class_label
            yield tuple(assignment)
            return
        for selected in combinations(remaining, class_counts[class_label]):
            selected_set = set(selected)
            for index in selected:
                assignment[index] = class_label
            next_remaining = tuple(
                index for index in remaining if index not in selected_set
            )
            yield from visit(class_index + 1, next_remaining)

    yield from visit(0, tuple(range(item_count)))


def _permutation_test(
    observed: float,
    null_values: list[float],
    *,
    mode: str,
    alternative: str,
) -> dict[str, Any]:
    if alternative == "greater":
        extreme = sum(value >= observed - 1e-15 for value in null_values)
    elif alternative == "two_sided":
        extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in null_values)
    else:  # pragma: no cover - internal contract.
        raise ValueError(f"unsupported alternative: {alternative}")
    if mode == "exact":
        p_value = extreme / len(null_values)
    else:
        p_value = (extreme + 1) / (len(null_values) + 1)
    key = "p_greater" if alternative == "greater" else "p_two_sided"
    return {
        "mode": mode,
        "assignment_count": len(null_values),
        "alternative": alternative,
        key: p_value,
        "null": _null_summary(null_values),
    }


def _effect_pairwise(region: dict[str, Any], effect: str) -> list[dict[str, Any]]:
    if effect == "interaction":
        return list(region.get("pairwise_direction_cosines") or [])
    effects = region.get("effect_direction_replication") or {}
    return list((effects.get(effect) or {}).get("pairwise") or [])


def _complete_cosine_map(
    labels: tuple[str, ...], pairwise: list[dict[str, Any]]
) -> dict[tuple[int, int], float]:
    index = {label: position for position, label in enumerate(labels)}
    output = {}
    for record in pairwise:
        left = str(record["left"])
        right = str(record["right"])
        if left not in index or right not in index:
            raise ValueError(
                f"pairwise cosine references unknown labels: {left}, {right}"
            )
        key = tuple(sorted((index[left], index[right])))
        if key in output:
            raise ValueError(f"duplicate pairwise cosine: {left}, {right}")
        output[key] = float(record["cosine"])
    expected = len(labels) * (len(labels) - 1) // 2
    if len(output) != expected:
        raise ValueError(
            f"incomplete pairwise cosine matrix: {len(output)} != {expected}"
        )
    return output


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": fmean(values),
        "max": max(values),
        "positive_share": sum(value > 0 for value in values) / len(values),
    }


def _null_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.median(array)),
        "q975": float(np.quantile(array, 0.975)),
        "max": float(np.max(array)),
    }


def _interval(record: dict[str, Any]) -> str:
    return f"{_fmt(record.get('q025'))}-{_fmt(record.get('q975'))}"


def _short_model(model_id: str) -> str:
    if "Ministral-3" in model_id:
        return "Ministral-3-3B"
    if "Qwen2.5" in model_id:
        return "Qwen2.5-VL-3B"
    return model_id


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
