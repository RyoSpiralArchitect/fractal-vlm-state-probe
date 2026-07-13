from __future__ import annotations

from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

from .prompt_robustness_aggregate import analyze_prompt_robustness_aggregate
from .stimulus import write_json


AXES = ("spatial", "palette", "interaction")


def analyze_prompt_robustness_multi_pair_matrix(
    analyses: dict[str, dict[str, dict[str, Any]]],
    *,
    analysis_paths: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    if len(analyses) < 2:
        raise ValueError("at least two models are required")
    if set(analyses) != set(analysis_paths):
        raise ValueError("analysis_paths must align with analyses")

    pair_sets = {tuple(sorted(pair_records)) for pair_records in analyses.values()}
    if len(pair_sets) != 1:
        raise ValueError("every model must contain the same source pairs")
    source_pairs = pair_sets.pop()
    if len(source_pairs) < 3:
        raise ValueError("at least three source pairs are required")
    for model, pair_records in analyses.items():
        if set(pair_records) != set(analysis_paths.get(model) or {}):
            raise ValueError(f"analysis_paths must align for model {model!r}")

    phases = {
        str(analysis.get("phase"))
        for pair_records in analyses.values()
        for analysis in pair_records.values()
    }
    if len(phases) != 1:
        raise ValueError("all prompt-robustness analyses must use the same phase")

    models = []
    for model in sorted(analyses):
        aggregate = analyze_prompt_robustness_aggregate(
            analyses[model],
            analysis_paths=analysis_paths[model],
            comparison_axis="source_pair",
        )
        models.append(_model_summary(model, aggregate))

    source_pair_summaries = []
    for source_pair in source_pairs:
        pair_analyses = {
            model: analyses[model][source_pair] for model in sorted(analyses)
        }
        pair_paths = {
            model: analysis_paths[model][source_pair] for model in sorted(analyses)
        }
        aggregate = analyze_prompt_robustness_aggregate(
            pair_analyses,
            analysis_paths=pair_paths,
            comparison_axis="model",
        )
        source_pair_summaries.append(_source_pair_summary(source_pair, aggregate))

    all_pair_records = [record for model in models for record in model["records"]]
    pairwise_records = [
        {"model_label": model["model_label"], **comparison}
        for model in models
        for record in model["records"]
        for comparison in record["pairwise_comparisons"]
    ]
    source_pair_comparisons = [
        _source_pair_comparison_summary(left, right, pairwise_records)
        for left, right in combinations(source_pairs, 2)
    ]
    dominance = {
        axis: sum(
            summary["balanced_axis_dominance_counts"][axis]
            for summary in source_pair_summaries
        )
        for axis in AXES
    }

    all_pair_summary = _agreement_summary(
        all_pair_records,
        generated_key="generated_pattern_agrees_all_pairs",
        axis_key="balanced_axis_agrees_all_pairs",
    )
    pairwise_summary = _agreement_summary(pairwise_records)
    pairwise_summary["balanced_share_l1"] = _range_summary(
        [record["balanced_share_l1"] for record in pairwise_records]
    )
    return {
        "schema_version": 1,
        "analysis_kind": "prompt_robustness_model_multi_source_pair_matrix",
        "phase": phases.pop(),
        "model_count": len(models),
        "source_pair_count": len(source_pairs),
        "source_pairs": list(source_pairs),
        "source_pair_comparison_count": len(source_pair_comparisons),
        "model_source_pair_audit_count": len(models) * len(source_pairs),
        "model_family_variant_record_count": len(all_pair_records),
        "pairwise_model_family_variant_count": len(pairwise_records),
        "models": models,
        "source_pair_summaries": source_pair_summaries,
        "source_pair_comparisons": source_pair_comparisons,
        "global_all_pair_summary": all_pair_summary,
        "global_pairwise_summary": pairwise_summary,
        "global_source_pair_profile": {
            "balanced_axis_dominance_counts": dominance,
            "interaction_above_one_third_count": sum(
                summary["interaction_above_one_third_count"]
                for summary in source_pair_summaries
            ),
            "record_count": sum(
                summary["record_count"] for summary in source_pair_summaries
            ),
        },
        "interpretation_notes": [
            "All-pair agreement requires one model-family-variant record to repeat across every source pair.",
            "Pairwise agreement expands each record over all source-pair combinations and is descriptive repeated-measures evidence, not independent sampling.",
            "Generated-pattern agreement and balanced-axis agreement are separate replication criteria.",
            "Balanced shares use equal-scale full-vocabulary Hadamard contrasts.",
            "Source-pair agreement describes direct first-token readouts and does not establish cache or mechanism agreement.",
        ],
    }


def write_prompt_robustness_multi_pair_matrix_json(
    analysis: dict[str, Any], path: Path
) -> None:
    write_json(path, analysis)


def write_prompt_robustness_multi_pair_matrix_markdown(
    analysis: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_prompt_robustness_multi_pair_matrix_markdown(analysis),
        encoding="utf-8",
    )


def format_prompt_robustness_multi_pair_matrix_markdown(
    analysis: dict[str, Any],
) -> str:
    all_pair = analysis["global_all_pair_summary"]
    pairwise = analysis["global_pairwise_summary"]
    contingency = pairwise["agreement_contingency"]
    lines = [
        "# Model x Multi-Source-Pair Prompt Replication",
        "",
        f"- Phase: `{analysis['phase']}`",
        f"- Models: `{analysis['model_count']}`",
        f"- Source pairs: `{', '.join(analysis['source_pairs'])}`",
        f"- Model-family-variant records: `{analysis['model_family_variant_record_count']}`",
        f"- Pairwise repeated comparisons: `{analysis['pairwise_model_family_variant_count']}`",
        "",
        "## All-Source-Pair Agreement",
        "",
        "| Generated pattern same across all pairs | Balanced axis same across all pairs | Both same across all pairs |",
        "| ---: | ---: | ---: |",
        (
            f"| {all_pair['generated_pattern_agreement_count']}/{all_pair['record_count']} | "
            f"{all_pair['balanced_axis_agreement_count']}/{all_pair['record_count']} | "
            f"{all_pair['both_agreement_count']}/{all_pair['record_count']} |"
        ),
        "",
        "## Pairwise Agreement",
        "",
        "| Generated pattern same | Balanced axis same | Both same | Balanced-share L1 median | Balanced-share L1 range |",
        "| ---: | ---: | ---: | ---: | --- |",
        (
            f"| {pairwise['generated_pattern_agreement_count']}/{pairwise['record_count']} | "
            f"{pairwise['balanced_axis_agreement_count']}/{pairwise['record_count']} | "
            f"{pairwise['both_agreement_count']}/{pairwise['record_count']} | "
            f"{_fmt(pairwise['balanced_share_l1']['median'])} | "
            f"{_range_text(pairwise['balanced_share_l1'])} |"
        ),
        "",
        "| Generated pattern | Balanced axis | Records |",
        "| --- | --- | ---: |",
        f"| same | same | {contingency['generated_same_axis_same']} |",
        f"| same | different | {contingency['generated_same_axis_different']} |",
        f"| different | same | {contingency['generated_different_axis_same']} |",
        f"| different | different | {contingency['generated_different_axis_different']} |",
        "",
        "## Model Summary",
        "",
        "| Model | All-pair generated / axis / both | Pairwise generated / axis / both | Pairwise share-L1 median (range) |",
        "| --- | --- | --- | --- |",
    ]
    for model in analysis["models"]:
        all_summary = model["all_pair_summary"]
        pair_summary = model["pairwise_summary"]
        lines.append(
            f"| `{model['model_label']}` | "
            f"{_agreement_text(all_summary)} | "
            f"{_agreement_text(pair_summary)} | "
            f"{_fmt(pair_summary['balanced_share_l1']['median'])} "
            f"({_range_text(pair_summary['balanced_share_l1'])}) |"
        )

    lines.extend(
        [
            "",
            "## Source-Pair Comparisons",
            "",
            "| Source-pair comparison | Generated agreement | Balanced-axis agreement | Both agreement | Share-L1 median (range) |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for comparison in analysis["source_pair_comparisons"]:
        lines.append(
            f"| `{comparison['left_source_pair']}` vs `{comparison['right_source_pair']}` | "
            f"{comparison['generated_pattern_agreement_count']}/{comparison['record_count']} | "
            f"{comparison['balanced_axis_agreement_count']}/{comparison['record_count']} | "
            f"{comparison['both_agreement_count']}/{comparison['record_count']} | "
            f"{_fmt(comparison['balanced_share_l1']['median'])} "
            f"({_range_text(comparison['balanced_share_l1'])}) |"
        )

    lines.extend(
        [
            "",
            "## Source-Pair Profiles",
            "",
            "| Source pair | Changed non-baseline generated patterns | Cross-model shared variant patterns | Balanced dominant S / P / I | Interaction above 1/3 |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
    )
    for pair in analysis["source_pair_summaries"]:
        counts = pair["balanced_axis_dominance_counts"]
        lines.append(
            f"| `{pair['source_pair']}` | "
            f"{pair['generated_variant_change_count']}/{pair['generated_variant_comparison_count']} | "
            f"{pair['cross_model_generated_pattern_agreement_count']}/{pair['variant_count']} | "
            f"{counts['spatial']} / {counts['palette']} / {counts['interaction']} | "
            f"{pair['interaction_above_one_third_count']}/{pair['record_count']} |"
        )

    lines.extend(
        [
            "",
            "## All-Pair Records",
            "",
            "| Model | Family | Variant | Unique generated patterns | Unique balanced axes | Generated same across all | Axis same across all | Pairwise share-L1 median / max |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for model in analysis["models"]:
        for record in model["records"]:
            l1 = record["balanced_share_pairwise_l1"]
            lines.append(
                f"| `{model['model_label']}` | `{record['probe_family']}` | "
                f"`{record['prompt_variant']}` | "
                f"{record['unique_generated_pattern_count']} | "
                f"{record['unique_balanced_axis_count']} | "
                f"`{record['generated_pattern_agrees_all_pairs']}` | "
                f"`{record['balanced_axis_agrees_all_pairs']}` | "
                f"{_fmt(l1['median'])} / {_fmt(l1['max'])} |"
            )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _model_summary(model: str, aggregate: dict[str, Any]) -> dict[str, Any]:
    source_pair_units = list(aggregate["units"])
    source_pairs = [unit["unit_label"] for unit in source_pair_units]
    records = []
    for aggregate_record in aggregate["family_variants"]:
        patterns = aggregate_record["generated_patterns"]
        axes = aggregate_record["balanced_dominant_axes"]
        shares = aggregate_record["balanced_energy_shares"]
        pairwise_comparisons = []
        for left, right in combinations(source_pairs, 2):
            generated_agrees = _all_non_null_equal((patterns[left], patterns[right]))
            axis_agrees = _all_non_null_equal((axes[left], axes[right]))
            pairwise_comparisons.append(
                {
                    "probe_family": aggregate_record["probe_family"],
                    "prompt_variant": aggregate_record["prompt_variant"],
                    "left_source_pair": left,
                    "right_source_pair": right,
                    "generated_pattern_agrees": generated_agrees,
                    "balanced_axis_agrees": axis_agrees,
                    "both_agree": generated_agrees and axis_agrees,
                    "balanced_share_l1": _balanced_share_l1(
                        shares,
                        left,
                        right,
                        probe_family=aggregate_record["probe_family"],
                        prompt_variant=aggregate_record["prompt_variant"],
                    ),
                }
            )
        generated_all = _all_non_null_equal(patterns.values())
        axis_all = _all_non_null_equal(axes.values())
        records.append(
            {
                "probe_family": aggregate_record["probe_family"],
                "prompt_variant": aggregate_record["prompt_variant"],
                "generated_patterns": patterns,
                "generated_pattern_agrees_all_pairs": generated_all,
                "unique_generated_pattern_count": _unique_non_null_count(
                    patterns.values()
                ),
                "balanced_dominant_axes": axes,
                "balanced_axis_agrees_all_pairs": axis_all,
                "unique_balanced_axis_count": _unique_non_null_count(axes.values()),
                "both_agree_all_pairs": generated_all and axis_all,
                "balanced_energy_shares": shares,
                "semantic_interaction_l1": aggregate_record["semantic_interaction_l1"],
                "pairwise_comparisons": pairwise_comparisons,
                "balanced_share_pairwise_l1": _range_summary(
                    [record["balanced_share_l1"] for record in pairwise_comparisons]
                ),
            }
        )

    all_pair_summary = _agreement_summary(
        records,
        generated_key="generated_pattern_agrees_all_pairs",
        axis_key="balanced_axis_agrees_all_pairs",
    )
    pairwise_records = [
        comparison
        for record in records
        for comparison in record["pairwise_comparisons"]
    ]
    pairwise_summary = _agreement_summary(pairwise_records)
    pairwise_summary["balanced_share_l1"] = _range_summary(
        [record["balanced_share_l1"] for record in pairwise_records]
    )

    families = {}
    for family in sorted({record["probe_family"] for record in records}):
        family_records = [
            record for record in records if record["probe_family"] == family
        ]
        family_pairwise = [
            comparison
            for record in family_records
            for comparison in record["pairwise_comparisons"]
        ]
        family_pairwise_summary = _agreement_summary(family_pairwise)
        family_pairwise_summary["balanced_share_l1"] = _range_summary(
            [record["balanced_share_l1"] for record in family_pairwise]
        )
        families[family] = {
            "all_pair_summary": _agreement_summary(
                family_records,
                generated_key="generated_pattern_agrees_all_pairs",
                axis_key="balanced_axis_agrees_all_pairs",
            ),
            "pairwise_summary": family_pairwise_summary,
        }

    return {
        "model_label": model,
        "source_pairs": source_pairs,
        "analysis_paths": {
            unit["unit_label"]: unit["analysis_path"] for unit in source_pair_units
        },
        "record_count": len(records),
        "pairwise_record_count": len(pairwise_records),
        "all_pair_summary": all_pair_summary,
        "pairwise_summary": pairwise_summary,
        "families": families,
        "records": records,
    }


def _source_pair_summary(source_pair: str, aggregate: dict[str, Any]) -> dict[str, Any]:
    dominance = {axis: 0 for axis in AXES}
    interaction_above = 0
    generated_changes = 0
    generated_comparisons = 0
    record_count = 0
    for unit in aggregate["units"]:
        for family in unit["families"]:
            generated_changes += int(family["generated_variant_change_count"])
            generated_comparisons += int(family["variant_count"]) - 1
            for variant in family["variants"]:
                axis = variant["balanced_dominant_axis"]
                if axis in dominance:
                    dominance[axis] += 1
                share = variant["balanced_energy_shares"]["interaction"]
                interaction_above += share is not None and float(share) > (1.0 / 3.0)
                record_count += 1
    return {
        "source_pair": source_pair,
        "model_count": len(aggregate["units"]),
        "record_count": record_count,
        "variant_count": len(aggregate["family_variants"]),
        "generated_variant_change_count": generated_changes,
        "generated_variant_comparison_count": generated_comparisons,
        "cross_model_generated_pattern_agreement_count": sum(
            bool(record["generated_pattern_consistent_across_units"])
            for record in aggregate["family_variants"]
        ),
        "balanced_axis_dominance_counts": dominance,
        "interaction_above_one_third_count": interaction_above,
    }


def _source_pair_comparison_summary(
    left: str,
    right: str,
    pairwise_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records = [
        record
        for record in pairwise_records
        if record["left_source_pair"] == left and record["right_source_pair"] == right
    ]
    summary = _agreement_summary(records)
    return {
        "left_source_pair": left,
        "right_source_pair": right,
        **summary,
        "balanced_share_l1": _range_summary(
            [record["balanced_share_l1"] for record in records]
        ),
    }


def _agreement_summary(
    records: list[dict[str, Any]],
    *,
    generated_key: str = "generated_pattern_agrees",
    axis_key: str = "balanced_axis_agrees",
) -> dict[str, Any]:
    generated_same_axis_same = sum(
        bool(record[generated_key]) and bool(record[axis_key]) for record in records
    )
    generated_same_axis_different = sum(
        bool(record[generated_key]) and not bool(record[axis_key]) for record in records
    )
    generated_different_axis_same = sum(
        not bool(record[generated_key]) and bool(record[axis_key]) for record in records
    )
    generated_different_axis_different = len(records) - (
        generated_same_axis_same
        + generated_same_axis_different
        + generated_different_axis_same
    )
    return {
        "record_count": len(records),
        "generated_pattern_agreement_count": (
            generated_same_axis_same + generated_same_axis_different
        ),
        "balanced_axis_agreement_count": (
            generated_same_axis_same + generated_different_axis_same
        ),
        "both_agreement_count": generated_same_axis_same,
        "agreement_contingency": {
            "generated_same_axis_same": generated_same_axis_same,
            "generated_same_axis_different": generated_same_axis_different,
            "generated_different_axis_same": generated_different_axis_same,
            "generated_different_axis_different": generated_different_axis_different,
        },
    }


def _balanced_share_l1(
    shares: dict[str, dict[str, Any]],
    left: str,
    right: str,
    *,
    probe_family: str,
    prompt_variant: str,
) -> float:
    values = {
        (source_pair, axis): shares[source_pair].get(axis)
        for source_pair in (left, right)
        for axis in AXES
    }
    missing = [
        f"{source_pair}/{axis}"
        for (source_pair, axis), value in values.items()
        if value is None
    ]
    if missing:
        probe = f"{probe_family}/{prompt_variant}"
        raise ValueError(
            f"balanced energy shares are unavailable for {probe}: {', '.join(missing)}"
        )
    return sum(
        abs(float(values[(left, axis)]) - float(values[(right, axis)])) for axis in AXES
    )


def _range_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}


def _all_non_null_equal(values: Any) -> bool:
    values = list(values)
    return (
        bool(values)
        and all(value is not None for value in values)
        and len(set(values)) == 1
    )


def _unique_non_null_count(values: Any) -> int:
    return len({value for value in values if value is not None})


def _agreement_text(summary: dict[str, Any]) -> str:
    total = summary["record_count"]
    return (
        f"{summary['generated_pattern_agreement_count']}/{total} / "
        f"{summary['balanced_axis_agreement_count']}/{total} / "
        f"{summary['both_agreement_count']}/{total}"
    )


def _range_text(record: dict[str, Any]) -> str:
    return f"{_fmt(record.get('min'))}-{_fmt(record.get('max'))}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
