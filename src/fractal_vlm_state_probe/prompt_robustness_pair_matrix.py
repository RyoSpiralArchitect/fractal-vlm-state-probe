from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any

from .prompt_robustness_aggregate import analyze_prompt_robustness_aggregate
from .stimulus import write_json


AXES = ("spatial", "palette", "interaction")


def analyze_prompt_robustness_pair_matrix(
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
    if len(source_pairs) != 2:
        raise ValueError("exactly two source pairs are required")
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

    paired_records = [record for model in models for record in model["records"]]
    agreement = _agreement_summary(paired_records)
    share_l1_values = [record["balanced_share_l1"] for record in paired_records]
    dominance = {
        axis: sum(
            summary["balanced_axis_dominance_counts"][axis]
            for summary in source_pair_summaries
        )
        for axis in AXES
    }
    return {
        "schema_version": 1,
        "analysis_kind": "prompt_robustness_model_source_pair_matrix",
        "phase": phases.pop(),
        "model_count": len(models),
        "source_pair_count": len(source_pairs),
        "source_pairs": list(source_pairs),
        "model_source_pair_audit_count": len(models) * len(source_pairs),
        "paired_model_family_variant_count": len(paired_records),
        "models": models,
        "source_pair_summaries": source_pair_summaries,
        "global_summary": {
            **agreement,
            "balanced_share_l1": _range_summary(share_l1_values),
            "balanced_axis_dominance_counts": dominance,
            "interaction_above_one_third_count": sum(
                summary["interaction_above_one_third_count"]
                for summary in source_pair_summaries
            ),
        },
        "interpretation_notes": [
            "Generated-pattern agreement and balanced-axis agreement are separate replication criteria.",
            "Balanced shares use equal-scale full-vocabulary Hadamard contrasts.",
            "The paired records are repeated measurements within models, probe families, and prompt variants; they are not independent samples.",
            "Source-pair agreement describes direct first-token readouts and does not establish cache or mechanism agreement.",
        ],
    }


def write_prompt_robustness_pair_matrix_json(
    analysis: dict[str, Any], path: Path
) -> None:
    write_json(path, analysis)


def write_prompt_robustness_pair_matrix_markdown(
    analysis: dict[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_prompt_robustness_pair_matrix_markdown(analysis), encoding="utf-8"
    )


def format_prompt_robustness_pair_matrix_markdown(analysis: dict[str, Any]) -> str:
    source_pairs = analysis["source_pairs"]
    summary = analysis["global_summary"]
    contingency = summary["agreement_contingency"]
    lines = [
        "# Model x Source-Pair Prompt Replication",
        "",
        f"- Phase: `{analysis['phase']}`",
        f"- Models: `{analysis['model_count']}`",
        f"- Source pairs: `{', '.join(source_pairs)}`",
        f"- Paired model-family-variant records: `{analysis['paired_model_family_variant_count']}`",
        "",
        "## Global Agreement",
        "",
        "| Generated pattern same | Balanced dominant axis same | Both same | Balanced-share L1 median | Balanced-share L1 range |",
        "| ---: | ---: | ---: | ---: | --- |",
        (
            f"| {summary['generated_pattern_agreement_count']}/{summary['record_count']} | "
            f"{summary['balanced_axis_agreement_count']}/{summary['record_count']} | "
            f"{summary['both_agreement_count']}/{summary['record_count']} | "
            f"{_fmt(summary['balanced_share_l1']['median'])} | "
            f"{_range_text(summary['balanced_share_l1'])} |"
        ),
        "",
        "Agreement contingency:",
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
        "| Model | Generated pattern agreement | Balanced axis agreement | Both agreement | Family generated / axis | Frequency generated / axis | Balanced-share L1 median (range) |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for model in analysis["models"]:
        family = model["families"].get("family") or {}
        frequency = model["families"].get("frequency") or {}
        lines.append(
            f"| `{model['model_label']}` | "
            f"{model['generated_pattern_agreement_count']}/{model['record_count']} | "
            f"{model['balanced_axis_agreement_count']}/{model['record_count']} | "
            f"{model['both_agreement_count']}/{model['record_count']} | "
            f"{_family_agreement_text(family)} | "
            f"{_family_agreement_text(frequency)} | "
            f"{_fmt(model['balanced_share_l1']['median'])} ({_range_text(model['balanced_share_l1'])}) |"
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
            "## Paired Records",
            "",
            "| Model | Family | Variant | Generated patterns by pair | Pattern same | Dominant axes by pair | Axis same | Balanced-share L1 |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for model in analysis["models"]:
        for record in model["records"]:
            patterns = "; ".join(
                f"{pair}={pattern}"
                for pair, pattern in record["generated_patterns"].items()
            )
            axes = "; ".join(
                f"{pair}={axis}"
                for pair, axis in record["balanced_dominant_axes"].items()
            )
            lines.append(
                f"| `{model['model_label']}` | `{record['probe_family']}` | "
                f"`{record['prompt_variant']}` | {patterns} | "
                f"`{record['generated_pattern_agrees']}` | {axes} | "
                f"`{record['balanced_axis_agrees']}` | "
                f"{_fmt(record['balanced_share_l1'])} |"
            )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _model_summary(model: str, aggregate: dict[str, Any]) -> dict[str, Any]:
    source_pairs = list(aggregate["units"])
    source_pair_labels = [unit["unit_label"] for unit in source_pairs]
    records = []
    for record in aggregate["family_variants"]:
        generated_agrees = bool(record["generated_pattern_consistent_across_units"])
        balanced_axis_agrees = _all_non_null_equal(
            record["balanced_dominant_axes"].values()
        )
        left, right = source_pair_labels
        balanced_share_l1 = _balanced_share_l1(record, left, right)
        records.append(
            {
                "probe_family": record["probe_family"],
                "prompt_variant": record["prompt_variant"],
                "generated_patterns": record["generated_patterns"],
                "generated_pattern_agrees": generated_agrees,
                "balanced_dominant_axes": record["balanced_dominant_axes"],
                "balanced_axis_agrees": balanced_axis_agrees,
                "both_agree": generated_agrees and balanced_axis_agrees,
                "balanced_energy_shares": record["balanced_energy_shares"],
                "balanced_share_l1": balanced_share_l1,
                "semantic_interaction_l1": record["semantic_interaction_l1"],
            }
        )

    families = {}
    for family in sorted({record["probe_family"] for record in records}):
        family_records = [
            record for record in records if record["probe_family"] == family
        ]
        families[family] = _agreement_summary(family_records)

    agreement = _agreement_summary(records)
    return {
        "model_label": model,
        "source_pairs": source_pair_labels,
        "analysis_paths": {
            unit["unit_label"]: unit["analysis_path"] for unit in source_pairs
        },
        **agreement,
        "balanced_share_l1": _range_summary(
            [record["balanced_share_l1"] for record in records]
        ),
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


def _agreement_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    generated_same_axis_same = sum(
        record["generated_pattern_agrees"] and record["balanced_axis_agrees"]
        for record in records
    )
    generated_same_axis_different = sum(
        record["generated_pattern_agrees"] and not record["balanced_axis_agrees"]
        for record in records
    )
    generated_different_axis_same = sum(
        not record["generated_pattern_agrees"] and record["balanced_axis_agrees"]
        for record in records
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


def _range_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": median(values), "max": max(values)}


def _balanced_share_l1(record: dict[str, Any], left: str, right: str) -> float:
    shares = record["balanced_energy_shares"]
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
        probe = f"{record['probe_family']}/{record['prompt_variant']}"
        raise ValueError(
            f"balanced energy shares are unavailable for {probe}: {', '.join(missing)}"
        )
    return sum(
        abs(float(values[(left, axis)]) - float(values[(right, axis)])) for axis in AXES
    )


def _all_non_null_equal(values: Any) -> bool:
    values = list(values)
    return (
        bool(values)
        and all(value is not None for value in values)
        and len(set(values)) == 1
    )


def _family_agreement_text(summary: dict[str, Any]) -> str:
    if not summary:
        return "n/a"
    return (
        f"{summary['generated_pattern_agreement_count']}/{summary['record_count']} / "
        f"{summary['balanced_axis_agreement_count']}/{summary['record_count']}"
    )


def _range_text(record: dict[str, Any]) -> str:
    return f"{_fmt(record.get('min'))}-{_fmt(record.get('max'))}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
