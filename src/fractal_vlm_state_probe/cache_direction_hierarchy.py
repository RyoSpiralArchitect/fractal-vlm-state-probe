from __future__ import annotations

import math
from itertools import combinations, product
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Iterator

import numpy as np

from .stimulus import write_json


DEFAULT_REGIONS = ("image_tokens", "post_image")


def analyze_cache_direction_hierarchy(
    replication: dict[str, Any],
    *,
    broad_class_by_label: dict[str, str],
    pairing_family_by_label: dict[str, str],
    regions: Iterable[str] = DEFAULT_REGIONS,
    effect: str = "interaction",
    exact_limit: int = 100_000,
    monte_carlo_permutations: int = 100_000,
    seed: int = 20260714,
) -> dict[str, Any]:
    """Separate seed replication from transfer across generator pairings."""
    if replication.get("analysis_kind") != "source_cache_tensor_cross_pair_replication":
        raise ValueError("expected a source-cache tensor replication analysis")
    if effect not in {"spatial_main_effect", "palette_main_effect", "interaction"}:
        raise ValueError(f"unsupported factorial effect: {effect}")
    if exact_limit < 1 or monte_carlo_permutations < 1:
        raise ValueError("permutation limits must be positive")
    selected_regions = tuple(dict.fromkeys(str(region) for region in regions))
    if not selected_regions:
        raise ValueError("at least one token region is required")
    if set(broad_class_by_label) != set(pairing_family_by_label):
        raise ValueError("broad-class and pairing-family labels must align")

    groups = []
    observed_labels: set[str] = set()
    for group_index, group in enumerate(replication.get("groups") or []):
        labels = tuple(sorted(str(point["label"]) for point in group["points"]))
        observed_labels.update(labels)
        groups.append(
            _analyze_group(
                group,
                labels=labels,
                broad_class_by_label=broad_class_by_label,
                pairing_family_by_label=pairing_family_by_label,
                regions=selected_regions,
                effect=effect,
                exact_limit=exact_limit,
                monte_carlo_permutations=monte_carlo_permutations,
                seed=seed + group_index,
            )
        )
    extra = sorted(set(broad_class_by_label) - observed_labels)
    if extra:
        raise ValueError(f"hierarchy labels do not occur in replication: {extra}")

    return {
        "schema_version": 1,
        "analysis_kind": "source_cache_direction_pairing_hierarchy",
        "source_analysis_kind": replication["analysis_kind"],
        "effect": effect,
        "regions": list(selected_regions),
        "source_pair_is_atomic": True,
        "pairing_family_is_broad_class_permutation_unit": True,
        "broad_class_by_label": dict(sorted(broad_class_by_label.items())),
        "pairing_family_by_label": dict(sorted(pairing_family_by_label.items())),
        "source_pair_count": len(observed_labels),
        "pairing_family_count": len(set(pairing_family_by_label.values())),
        "broad_class_count": len(set(broad_class_by_label.values())),
        "group_count": len(groups),
        "exact_limit": exact_limit,
        "monte_carlo_permutations": monte_carlo_permutations,
        "seed": seed,
        "groups": groups,
        "interpretation_notes": [
            "Within-pairing cosines compare independent seeds of the same ordered generator pairing.",
            "Cross-pairing same-class cosines exclude within-pairing replicate pairs and measure transfer across generator pairings.",
            "The seed-replication test randomizes perfect replicate matchings inside each fixed broad class.",
            "The broad-class transfer test moves complete pairing families between broad classes while preserving family size and class counts.",
            "Simultaneously exchanging source A and source B maps MM to JJ and MJ to JM, leaving JJ - JM - MJ + MM unchanged; role reversal is an invariant check, not an independent interaction experiment.",
            "Permutation p-values are unadjusted across regions and tensor targets.",
            "The tests condition on the observed generator families and do not establish exchangeability with unobserved visual populations.",
        ],
    }


def write_cache_direction_hierarchy_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_cache_direction_hierarchy_markdown(
    analysis: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_cache_direction_hierarchy_markdown(analysis), encoding="utf-8"
    )


def format_cache_direction_hierarchy_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Source-Cache Direction Pairing Hierarchy",
        "",
        f"- Source-pair units: `{analysis['source_pair_count']}`",
        f"- Pairing families: `{analysis['pairing_family_count']}`",
        f"- Broad classes: `{analysis['broad_class_count']}`",
        f"- Effect: `{analysis['effect']}`",
        "",
        "## Direction Categories",
        "",
        "| Model | Layer | Tensor | Region | Within pairing | Cross pairing, same broad | Between broad |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for group in analysis["groups"]:
        for region in group["regions"]:
            if not region["available"]:
                continue
            categories = region["direction_categories"]
            lines.append(
                f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                f"`{group['tensor']}` | `{region['region']}` | "
                f"{_summary_text(categories['within_pairing_family'])} | "
                f"{_summary_text(categories['cross_pairing_same_broad'])} | "
                f"{_summary_text(categories['between_broad'])} |"
            )

    lines.extend(
        [
            "",
            "## Exact Hierarchy Tests",
            "",
            "| Model | Layer | Region | Test | Difference | Null 95% interval | p (greater) | Mode / assignments |",
            "| --- | ---: | --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    for group in analysis["groups"]:
        for region in group["regions"]:
            if not region["available"]:
                continue
            for key, label in (
                ("seed_replication_test", "within pairing - cross pairing same broad"),
                (
                    "broad_class_transfer_test",
                    "cross pairing same broad - between broad",
                ),
            ):
                test = region[key]
                lines.append(
                    f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                    f"`{region['region']}` | {label} | {_fmt(test['statistic'])} | "
                    f"{_interval(test['null'])} | {_fmt(test['p_greater'])} | "
                    f"{test['mode']} / {test['assignment_count']} |"
                )

    lines.extend(
        [
            "",
            "## Broad-Class-Specific Transfer",
            "",
            "| Model | Layer | Region | Broad class | Within pairing | Cross pairing | Boundary | Cross pairing - boundary | p (greater) |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in analysis["groups"]:
        for region in group["regions"]:
            if not region["available"]:
                continue
            for record in region["broad_class_specific_tests"]:
                categories = record["direction_categories"]
                lines.append(
                    f"| `{_short_model(group['model_id'])}` | {group['layer_index']} | "
                    f"`{region['region']}` | `{record['broad_class']}` | "
                    f"{_fmt(categories['within_pairing_family']['mean'])} | "
                    f"{_fmt(categories['cross_pairing_same_broad']['mean'])} | "
                    f"{_fmt(categories['boundary']['mean'])} | "
                    f"{_fmt(record['statistic'])} | {_fmt(record['p_greater'])} |"
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
    lines.extend(f"- {note}" for note in analysis["interpretation_notes"])
    lines.append("")
    return "\n".join(lines)


def _analyze_group(
    group: dict[str, Any],
    *,
    labels: tuple[str, ...],
    broad_class_by_label: dict[str, str],
    pairing_family_by_label: dict[str, str],
    regions: tuple[str, ...],
    effect: str,
    exact_limit: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, Any]:
    missing = [label for label in labels if label not in broad_class_by_label]
    if missing:
        raise ValueError(f"missing hierarchy labels for source pairs: {missing}")
    _validate_hierarchy(
        labels,
        broad_class_by_label=broad_class_by_label,
        pairing_family_by_label=pairing_family_by_label,
    )

    by_region = {record["region"]: record for record in group["regions"]}
    region_records = []
    for region_name in regions:
        source = by_region.get(region_name)
        if source is None:
            region_records.append(
                {
                    "region": region_name,
                    "available": False,
                    "reason": "region is absent from the replication analysis",
                }
            )
            continue
        pairwise = _effect_pairwise(source, effect)
        if any(not record.get("available") for record in pairwise):
            region_records.append(
                {
                    "region": region_name,
                    "available": False,
                    "reason": "one or more direction cosines are undefined",
                }
            )
            continue
        cosine = _complete_cosine_map(labels, pairwise)
        region_records.append(
            _analyze_region(
                region_name,
                labels=labels,
                broad_class_by_label=broad_class_by_label,
                pairing_family_by_label=pairing_family_by_label,
                cosine=cosine,
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
        "pairing_family_count": len(
            {pairing_family_by_label[label] for label in labels}
        ),
        "broad_class_count": len({broad_class_by_label[label] for label in labels}),
        "source_pair_labels": list(labels),
        "regions": region_records,
    }


def _validate_hierarchy(
    labels: tuple[str, ...],
    *,
    broad_class_by_label: dict[str, str],
    pairing_family_by_label: dict[str, str],
) -> None:
    broad_classes = {broad_class_by_label[label] for label in labels}
    if len(broad_classes) != 2:
        raise ValueError(
            "hierarchy analysis currently requires exactly two broad classes"
        )
    family_members: dict[str, list[str]] = {}
    for label in labels:
        family_members.setdefault(pairing_family_by_label[label], []).append(label)
    bad_counts = {
        family: len(members)
        for family, members in family_members.items()
        if len(members) != 2
    }
    if bad_counts:
        raise ValueError(
            f"every pairing family must have exactly two replicates: {bad_counts}"
        )
    family_broad = {
        family: {broad_class_by_label[label] for label in members}
        for family, members in family_members.items()
    }
    crossed = {
        family: values for family, values in family_broad.items() if len(values) != 1
    }
    if crossed:
        raise ValueError(f"pairing families cross broad classes: {crossed}")
    counts = {
        broad: sum(next(iter(values)) == broad for values in family_broad.values())
        for broad in broad_classes
    }
    if any(count < 2 for count in counts.values()):
        raise ValueError(
            f"every broad class needs at least two pairing families: {counts}"
        )


def _analyze_region(
    region_name: str,
    *,
    labels: tuple[str, ...],
    broad_class_by_label: dict[str, str],
    pairing_family_by_label: dict[str, str],
    cosine: dict[tuple[str, str], float],
    exact_limit: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, Any]:
    categories = _direction_categories(
        labels,
        broad_class_by_label=broad_class_by_label,
        pairing_family_by_label=pairing_family_by_label,
        cosine=cosine,
    )
    observed_seed = (
        categories["within_pairing_family"]["mean"]
        - categories["cross_pairing_same_broad"]["mean"]
    )
    observed_transfer = (
        categories["cross_pairing_same_broad"]["mean"]
        - categories["between_broad"]["mean"]
    )

    seed_assignments = _replicate_matching_stream(
        labels,
        broad_class_by_label=broad_class_by_label,
        exact_limit=exact_limit,
        monte_carlo_permutations=monte_carlo_permutations,
        seed=seed,
    )
    seed_null = [
        _matching_statistic(
            labels,
            matched_pairs=matched,
            broad_class_by_label=broad_class_by_label,
            cosine=cosine,
        )
        for matched in seed_assignments["assignments"]
    ]
    seed_test = _permutation_test(
        observed_seed,
        seed_null,
        mode=seed_assignments["mode"],
    )

    broad_assignments = _family_class_assignment_stream(
        labels,
        broad_class_by_label=broad_class_by_label,
        pairing_family_by_label=pairing_family_by_label,
        exact_limit=exact_limit,
        monte_carlo_permutations=monte_carlo_permutations,
        seed=seed + 10_000,
    )
    broad_assignment_values = list(broad_assignments["assignments"])
    transfer_null = [
        _transfer_statistic(
            labels,
            broad_class_by_family=assignment,
            pairing_family_by_label=pairing_family_by_label,
            cosine=cosine,
        )
        for assignment in broad_assignment_values
    ]
    transfer_test = _permutation_test(
        observed_transfer,
        transfer_null,
        mode=broad_assignments["mode"],
    )
    categories_by_broad = _direction_categories_by_broad(
        labels,
        broad_class_by_label=broad_class_by_label,
        pairing_family_by_label=pairing_family_by_label,
        cosine=cosine,
    )
    broad_class_specific_tests = []
    for broad_class, class_categories in sorted(categories_by_broad.items()):
        statistic = (
            class_categories["cross_pairing_same_broad"]["mean"]
            - class_categories["boundary"]["mean"]
        )
        null_values = [
            _class_transfer_statistic(
                labels,
                selected_broad_class=broad_class,
                broad_class_by_family=assignment,
                pairing_family_by_label=pairing_family_by_label,
                cosine=cosine,
            )
            for assignment in broad_assignment_values
        ]
        broad_class_specific_tests.append(
            {
                "broad_class": broad_class,
                "direction_categories": class_categories,
                **_permutation_test(
                    statistic,
                    null_values,
                    mode=broad_assignments["mode"],
                ),
            }
        )
    return {
        "region": region_name,
        "available": True,
        "pairwise_cosine_count": len(cosine),
        "direction_categories": categories,
        "seed_replication_test": seed_test,
        "broad_class_transfer_test": transfer_test,
        "broad_class_specific_tests": broad_class_specific_tests,
    }


def _direction_categories(
    labels: tuple[str, ...],
    *,
    broad_class_by_label: dict[str, str],
    pairing_family_by_label: dict[str, str],
    cosine: dict[tuple[str, str], float],
) -> dict[str, dict[str, Any]]:
    values = {
        "within_pairing_family": [],
        "cross_pairing_same_broad": [],
        "between_broad": [],
    }
    for left, right in combinations(labels, 2):
        pair_cosine = cosine[tuple(sorted((left, right)))]
        if pairing_family_by_label[left] == pairing_family_by_label[right]:
            values["within_pairing_family"].append(pair_cosine)
        elif broad_class_by_label[left] == broad_class_by_label[right]:
            values["cross_pairing_same_broad"].append(pair_cosine)
        else:
            values["between_broad"].append(pair_cosine)
    if any(not records for records in values.values()):
        raise ValueError("hierarchy must produce all three direction categories")
    return {name: _summary(records) for name, records in values.items()}


def _direction_categories_by_broad(
    labels: tuple[str, ...],
    *,
    broad_class_by_label: dict[str, str],
    pairing_family_by_label: dict[str, str],
    cosine: dict[tuple[str, str], float],
) -> dict[str, dict[str, dict[str, Any]]]:
    output = {}
    for broad_class in sorted({broad_class_by_label[label] for label in labels}):
        within_pairing = []
        cross_pairing = []
        boundary = []
        for left, right in combinations(labels, 2):
            key = tuple(sorted((left, right)))
            left_in = broad_class_by_label[left] == broad_class
            right_in = broad_class_by_label[right] == broad_class
            if left_in and right_in:
                destination = (
                    within_pairing
                    if pairing_family_by_label[left] == pairing_family_by_label[right]
                    else cross_pairing
                )
                destination.append(cosine[key])
            elif left_in != right_in:
                boundary.append(cosine[key])
        output[broad_class] = {
            "within_pairing_family": _summary(within_pairing),
            "cross_pairing_same_broad": _summary(cross_pairing),
            "boundary": _summary(boundary),
        }
    return output


def _replicate_matching_stream(
    labels: tuple[str, ...],
    *,
    broad_class_by_label: dict[str, str],
    exact_limit: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, Any]:
    broad_classes = sorted({broad_class_by_label[label] for label in labels})
    members = {
        broad: tuple(label for label in labels if broad_class_by_label[label] == broad)
        for broad in broad_classes
    }
    counts = {
        broad: _perfect_matching_count(len(class_members))
        for broad, class_members in members.items()
    }
    assignment_count = math.prod(counts.values())
    if assignment_count <= exact_limit:
        matching_lists = [
            tuple(_perfect_matchings(members[broad])) for broad in broad_classes
        ]

        def exact() -> Iterator[frozenset[tuple[str, str]]]:
            for selected in product(*matching_lists):
                yield frozenset(pair for matching in selected for pair in matching)

        return {
            "mode": "exact",
            "assignment_count": assignment_count,
            "assignments": exact(),
        }

    rng = np.random.default_rng(seed)

    def sampled() -> Iterator[frozenset[tuple[str, str]]]:
        for _ in range(monte_carlo_permutations):
            pairs = []
            for broad in broad_classes:
                shuffled = list(rng.permutation(members[broad]))
                pairs.extend(
                    tuple(sorted((str(shuffled[index]), str(shuffled[index + 1]))))
                    for index in range(0, len(shuffled), 2)
                )
            yield frozenset(pairs)

    return {
        "mode": "monte_carlo",
        "assignment_count": monte_carlo_permutations,
        "assignments": sampled(),
    }


def _family_class_assignment_stream(
    labels: tuple[str, ...],
    *,
    broad_class_by_label: dict[str, str],
    pairing_family_by_label: dict[str, str],
    exact_limit: int,
    monte_carlo_permutations: int,
    seed: int,
) -> dict[str, Any]:
    families = tuple(sorted({pairing_family_by_label[label] for label in labels}))
    broad_classes = tuple(sorted({broad_class_by_label[label] for label in labels}))
    family_broad = {
        family: broad_class_by_label[
            next(label for label in labels if pairing_family_by_label[label] == family)
        ]
        for family in families
    }
    first_class = broad_classes[0]
    first_count = sum(value == first_class for value in family_broad.values())
    assignment_count = math.comb(len(families), first_count)
    if assignment_count <= exact_limit:

        def exact() -> Iterator[dict[str, str]]:
            for selected in combinations(families, first_count):
                selected_set = set(selected)
                yield {
                    family: first_class if family in selected_set else broad_classes[1]
                    for family in families
                }

        return {
            "mode": "exact",
            "assignment_count": assignment_count,
            "assignments": exact(),
        }

    rng = np.random.default_rng(seed)
    base = np.asarray([family_broad[family] for family in families], dtype=object)

    def sampled() -> Iterator[dict[str, str]]:
        for _ in range(monte_carlo_permutations):
            values = rng.permutation(base)
            yield {family: str(value) for family, value in zip(families, values)}

    return {
        "mode": "monte_carlo",
        "assignment_count": monte_carlo_permutations,
        "assignments": sampled(),
    }


def _matching_statistic(
    labels: tuple[str, ...],
    *,
    matched_pairs: frozenset[tuple[str, str]],
    broad_class_by_label: dict[str, str],
    cosine: dict[tuple[str, str], float],
) -> float:
    matched = []
    unmatched_same_broad = []
    for left, right in combinations(labels, 2):
        if broad_class_by_label[left] != broad_class_by_label[right]:
            continue
        key = tuple(sorted((left, right)))
        if key in matched_pairs:
            matched.append(cosine[key])
        else:
            unmatched_same_broad.append(cosine[key])
    return fmean(matched) - fmean(unmatched_same_broad)


def _transfer_statistic(
    labels: tuple[str, ...],
    *,
    broad_class_by_family: dict[str, str],
    pairing_family_by_label: dict[str, str],
    cosine: dict[tuple[str, str], float],
) -> float:
    same_broad = []
    between_broad = []
    for left, right in combinations(labels, 2):
        left_family = pairing_family_by_label[left]
        right_family = pairing_family_by_label[right]
        if left_family == right_family:
            continue
        value = cosine[tuple(sorted((left, right)))]
        if broad_class_by_family[left_family] == broad_class_by_family[right_family]:
            same_broad.append(value)
        else:
            between_broad.append(value)
    return fmean(same_broad) - fmean(between_broad)


def _class_transfer_statistic(
    labels: tuple[str, ...],
    *,
    selected_broad_class: str,
    broad_class_by_family: dict[str, str],
    pairing_family_by_label: dict[str, str],
    cosine: dict[tuple[str, str], float],
) -> float:
    within = []
    boundary = []
    for left, right in combinations(labels, 2):
        left_family = pairing_family_by_label[left]
        right_family = pairing_family_by_label[right]
        left_in = broad_class_by_family[left_family] == selected_broad_class
        right_in = broad_class_by_family[right_family] == selected_broad_class
        if left_in and right_in and left_family != right_family:
            within.append(cosine[tuple(sorted((left, right)))])
        elif left_in != right_in:
            boundary.append(cosine[tuple(sorted((left, right)))])
    return fmean(within) - fmean(boundary)


def _perfect_matchings(items: tuple[str, ...]) -> Iterator[tuple[tuple[str, str], ...]]:
    if not items:
        yield ()
        return
    first = items[0]
    for index in range(1, len(items)):
        second = items[index]
        remaining = items[1:index] + items[index + 1 :]
        pair = tuple(sorted((first, second)))
        for rest in _perfect_matchings(remaining):
            yield (pair, *rest)


def _perfect_matching_count(item_count: int) -> int:
    if item_count < 2 or item_count % 2:
        raise ValueError("each broad class needs a positive even source-pair count")
    output = 1
    for value in range(item_count - 1, 0, -2):
        output *= value
    return output


def _effect_pairwise(region: dict[str, Any], effect: str) -> list[dict[str, Any]]:
    if effect == "interaction":
        return list(region.get("pairwise_direction_cosines") or [])
    effects = region.get("effect_direction_replication") or {}
    return list((effects.get(effect) or {}).get("pairwise") or [])


def _complete_cosine_map(
    labels: tuple[str, ...], pairwise: list[dict[str, Any]]
) -> dict[tuple[str, str], float]:
    label_set = set(labels)
    output = {}
    for record in pairwise:
        left = str(record["left"])
        right = str(record["right"])
        if left not in label_set or right not in label_set:
            raise ValueError(
                f"pairwise cosine references unknown labels: {left}, {right}"
            )
        key = tuple(sorted((left, right)))
        if key in output:
            raise ValueError(f"duplicate pairwise cosine: {left}, {right}")
        output[key] = float(record["cosine"])
    expected = len(labels) * (len(labels) - 1) // 2
    if len(output) != expected:
        raise ValueError(
            f"incomplete pairwise cosine matrix: {len(output)} != {expected}"
        )
    return output


def _permutation_test(
    observed: float, null_values: list[float], *, mode: str
) -> dict[str, Any]:
    extreme = sum(value >= observed - 1e-15 for value in null_values)
    p_value = (
        extreme / len(null_values)
        if mode == "exact"
        else (extreme + 1) / (len(null_values) + 1)
    )
    return {
        "statistic": observed,
        "mode": mode,
        "assignment_count": len(null_values),
        "alternative": "greater",
        "p_greater": p_value,
        "null": _null_summary(null_values),
    }


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


def _summary_text(record: dict[str, Any]) -> str:
    return f"{_fmt(record['mean'])} ({record['count']})"


def _interval(record: dict[str, Any]) -> str:
    return f"{_fmt(record['q025'])}-{_fmt(record['q975'])}"


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
