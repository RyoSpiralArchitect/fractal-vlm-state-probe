from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Any

from .stimulus import write_json


FAMILY_ORDER = ("family", "frequency")
VARIANT_ORDER = ("baseline", "paraphrase", "reversed_order", "rotated_labels")
CELL_KEYS = ("mm", "jj", "mj", "jm")


def analyze_prompt_robustness_aggregate(
    analyses: dict[str, dict[str, Any]],
    *,
    analysis_paths: dict[str, Path],
    comparison_axis: str = "model",
) -> dict[str, Any]:
    if len(analyses) < 2:
        raise ValueError("at least two prompt-robustness analyses are required")
    if set(analyses) != set(analysis_paths):
        raise ValueError("analysis_paths must align with analyses")
    if comparison_axis not in {"model", "source_pair", "condition"}:
        raise ValueError(
            "comparison_axis must be one of: model, source_pair, condition"
        )

    phases = {str(analysis.get("phase")) for analysis in analyses.values()}
    if len(phases) != 1:
        raise ValueError("all prompt-robustness analyses must use the same phase")

    units = [
        _unit_summary(label, analyses[label], analysis_paths[label])
        for label in sorted(analyses)
    ]
    family_variants = _family_variant_summaries(units)
    result = {
        "schema_version": 1,
        "analysis_kind": f"prompt_robustness_cross_{comparison_axis}_aggregate",
        "comparison_axis": comparison_axis,
        "phase": phases.pop(),
        "unit_count": len(units),
        "units": units,
        "family_variants": family_variants,
        "interpretation_notes": [
            "Generated-semantic stability and semantic-probability stability are separate measurements.",
            "Candidate probabilities are aligned by declared meaning within each source analysis.",
            "Balanced S/P/I shares use equal-scale full-vocabulary Hadamard contrasts.",
            _comparison_axis_note(comparison_axis),
            "Prompt variants are diagnostic repeated measurements, not independent samples.",
        ],
    }
    if comparison_axis == "model":
        result["analysis_kind"] = "prompt_robustness_cross_model_aggregate"
        result["model_count"] = len(units)
        result["models"] = [
            {**unit, "model_label": unit["unit_label"]} for unit in units
        ]
        for record in family_variants:
            record["generated_pattern_consistent_across_models"] = record[
                "generated_pattern_consistent_across_units"
            ]
    return result


def write_prompt_robustness_aggregate_json(
    analysis: dict[str, Any],
    path: Path,
) -> None:
    write_json(path, analysis)


def write_prompt_robustness_aggregate_markdown(
    analysis: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_prompt_robustness_aggregate_markdown(analysis),
        encoding="utf-8",
    )


def format_prompt_robustness_aggregate_markdown(analysis: dict[str, Any]) -> str:
    comparison_axis = str(analysis.get("comparison_axis") or "model")
    units = analysis.get("units") or [
        {**model, "unit_label": model["model_label"]}
        for model in analysis.get("models") or []
    ]
    axis_title, unit_title, unit_title_plural = _axis_titles(comparison_axis)
    lines = [
        f"# Cross-{axis_title} Prompt Robustness",
        "",
        f"- Phase: `{analysis['phase']}`",
        f"- {unit_title_plural}: `{len(units)}`",
        "",
        f"## {unit_title} Summary",
        "",
        f"| {unit_title} | Family | Generated variants changed | Max semantic TV | Baseline semantic interaction L1 | Semantic interaction L1 range | Max interaction delta from baseline | Balanced dominant S / P / I | Balanced interaction share range |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for unit in units:
        for family in unit["families"]:
            counts = family["balanced_axis_dominance_counts"]
            lines.append(
                f"| `{unit['unit_label']}` | `{family['probe_family']}` | "
                f"{family['generated_variant_change_count']}/{family['variant_count'] - 1} | "
                f"{_fmt(family['max_semantic_tv_across_variant_pairs'])} | "
                f"{_fmt(family['baseline_semantic_interaction_l1'])} | "
                f"{_range_text(family['semantic_interaction_l1_range'])} | "
                f"{_fmt(family['max_interaction_vector_delta_from_baseline'])} | "
                f"{counts['spatial']} / {counts['palette']} / {counts['interaction']} | "
                f"{_range_text(family['balanced_interaction_share_range'])} |"
            )

    lines.extend(
        [
            "",
            "## Variant Table",
            "",
            f"| Family | Variant | Generated semantics by {unit_title.lower()} (MM/JJ/MJ/JM) | Semantic interaction L1 by {unit_title.lower()} | Max semantic TV from baseline by {unit_title.lower()} | Balanced S/P/I by {unit_title.lower()} | Cross-{axis_title.lower()} generated agreement |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in analysis["family_variants"]:
        generated = "; ".join(
            f"{label}={value}" for label, value in record["generated_patterns"].items()
        )
        interactions = "; ".join(
            f"{label}={_fmt(value)}"
            for label, value in record["semantic_interaction_l1"].items()
        )
        tvs = "; ".join(
            f"{label}={_fmt(value)}"
            for label, value in record["max_semantic_tv_from_baseline"].items()
        )
        balanced = "; ".join(
            f"{label}={_balanced_share_text(values)}"
            for label, values in record["balanced_energy_shares"].items()
        )
        lines.append(
            f"| `{record['probe_family']}` | `{record['prompt_variant']}` | "
            f"{generated} | {interactions} | {tvs} | {balanced} | "
            f"`{record['generated_pattern_consistent_across_units']}` |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _unit_summary(
    label: str,
    analysis: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    families = sorted(
        analysis.get("families") or [],
        key=lambda item: _ordered_key(str(item.get("probe_family")), FAMILY_ORDER),
    )
    if not families:
        raise ValueError(f"prompt-robustness analysis has no families: {label}")
    return {
        "unit_label": label,
        "analysis_path": str(path),
        "families": [_unit_family_summary(family) for family in families],
    }


def _unit_family_summary(family: dict[str, Any]) -> dict[str, Any]:
    variants = sorted(
        family.get("variants") or [],
        key=lambda item: _ordered_key(str(item.get("prompt_variant")), VARIANT_ORDER),
    )
    baseline = next(
        (item for item in variants if item.get("prompt_variant") == "baseline"),
        None,
    )
    if baseline is None:
        raise ValueError(f"family {family.get('probe_family')!r} has no baseline")
    variant_records = [_variant_record(item) for item in variants]
    interaction_values = [
        float(item["semantic_interaction_l1"]) for item in variant_records
    ]
    balanced_interaction_shares = [
        float(item["balanced_energy_shares"]["interaction"])
        for item in variant_records
        if item["balanced_energy_shares"]["interaction"] is not None
    ]
    balanced_axis_dominance_counts = {
        axis: sum(item["balanced_dominant_axis"] == axis for item in variant_records)
        for axis in ("spatial", "palette", "interaction")
    }
    changed = [
        item
        for item in variant_records
        if item["prompt_variant"] != "baseline"
        and item["generated_pattern"] != _generated_pattern(baseline)
    ]
    return {
        "probe_family": family["probe_family"],
        "variant_count": len(variant_records),
        "generated_semantics_invariant_across_variants": family[
            "generated_semantics_invariant_across_variants"
        ],
        "generated_variant_change_count": len(changed),
        "max_semantic_tv_across_variant_pairs": float(
            family["max_semantic_tv_across_variant_pairs"]
        ),
        "baseline_semantic_interaction_l1": float(
            baseline["semantic_contrasts"]["interaction"]["l1_norm"]
        ),
        "semantic_interaction_l1_range": {
            "min": min(interaction_values),
            "max": max(interaction_values),
        },
        "balanced_axis_dominance_counts": balanced_axis_dominance_counts,
        "balanced_interaction_share_range": {
            "min": min(balanced_interaction_shares)
            if balanced_interaction_shares
            else None,
            "max": max(balanced_interaction_shares)
            if balanced_interaction_shares
            else None,
        },
        "max_interaction_vector_delta_from_baseline": max(
            float(
                (item.get("comparison_to_baseline") or {}).get(
                    "interaction_vector_delta_l1", 0.0
                )
            )
            for item in variants
        ),
        "variants": variant_records,
    }


def _variant_record(variant: dict[str, Any]) -> dict[str, Any]:
    comparison = variant.get("comparison_to_baseline") or {}
    energy = variant.get("full_vocab_balanced_contrast_energy") or {}
    raw_shares = energy.get("energy_shares") or {}
    shares = {
        "spatial": _optional_float(raw_shares.get("spatial_contrast")),
        "palette": _optional_float(raw_shares.get("palette_contrast")),
        "interaction": _optional_float(raw_shares.get("interaction_contrast")),
    }
    return {
        "prompt_variant": variant["prompt_variant"],
        "generated_pattern": _generated_pattern(variant),
        "generated_semantic_agreement_with_baseline": float(
            comparison.get("generated_semantic_agreement_fraction", 0.0)
        ),
        "max_semantic_tv_from_baseline": float(
            comparison.get("max_cell_semantic_tv", 0.0)
        ),
        "mean_semantic_tv_from_baseline": float(
            comparison.get("mean_cell_semantic_tv", 0.0)
        ),
        "semantic_interaction_l1": float(
            variant["semantic_contrasts"]["interaction"]["l1_norm"]
        ),
        "full_vocab_interaction_l1": float(variant["full_vocab_interaction_l1"]),
        "interaction_vector_delta_l1": float(
            comparison.get("interaction_vector_delta_l1", 0.0)
        ),
        "balanced_energy_shares": shares,
        "balanced_dominant_axis": variant.get("full_vocab_balanced_axis_dominant"),
    }


def _family_variant_summaries(
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for unit in units:
        for family in unit["families"]:
            for variant in family["variants"]:
                key = (family["probe_family"], variant["prompt_variant"])
                indexed.setdefault(key, {})[unit["unit_label"]] = variant

    expected_units = {unit["unit_label"] for unit in units}
    summaries = []
    for (family, variant), records in indexed.items():
        if set(records) != expected_units:
            raise ValueError(f"unit coverage mismatch for {family}/{variant}")
        patterns = {
            label: record["generated_pattern"]
            for label, record in sorted(records.items())
        }
        interaction_values = {
            label: record["semantic_interaction_l1"]
            for label, record in sorted(records.items())
        }
        tv_values = {
            label: record["max_semantic_tv_from_baseline"]
            for label, record in sorted(records.items())
        }
        balanced_shares = {
            label: record["balanced_energy_shares"]
            for label, record in sorted(records.items())
        }
        balanced_dominant_axes = {
            label: record["balanced_dominant_axis"]
            for label, record in sorted(records.items())
        }
        summaries.append(
            {
                "probe_family": family,
                "prompt_variant": variant,
                "generated_patterns": patterns,
                "generated_pattern_consistent_across_units": len(set(patterns.values()))
                == 1,
                "semantic_interaction_l1": interaction_values,
                "semantic_interaction_l1_median": median(interaction_values.values()),
                "max_semantic_tv_from_baseline": tv_values,
                "max_semantic_tv_from_baseline_median": median(tv_values.values()),
                "balanced_energy_shares": balanced_shares,
                "balanced_dominant_axes": balanced_dominant_axes,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            _ordered_key(item["probe_family"], FAMILY_ORDER),
            _ordered_key(item["prompt_variant"], VARIANT_ORDER),
        ),
    )


def _generated_pattern(variant: dict[str, Any]) -> str:
    generated = variant.get("generated_semantics") or {}
    return "/".join(str(generated.get(cell)) for cell in CELL_KEYS)


def _ordered_key(value: str, order: tuple[str, ...]) -> tuple[int, str]:
    try:
        return order.index(value), value
    except ValueError:
        return len(order), value


def _comparison_axis_note(comparison_axis: str) -> str:
    if comparison_axis == "model":
        return (
            "The same source factorial across models supports an architecture "
            "comparison, not additional visual source-pair replication."
        )
    if comparison_axis == "source_pair":
        return (
            "The same model across source pairs supports visual replication, "
            "not an architecture comparison."
        )
    return (
        "Comparison units must be interpreted according to their declared "
        "condition labels."
    )


def _axis_titles(comparison_axis: str) -> tuple[str, str, str]:
    if comparison_axis == "source_pair":
        return "Source-Pair", "Source Pair", "Source Pairs"
    if comparison_axis == "condition":
        return "Condition", "Condition", "Conditions"
    return "Model", "Model", "Models"


def _balanced_share_text(shares: dict[str, Any]) -> str:
    values = [
        shares.get("spatial"),
        shares.get("palette"),
        shares.get("interaction"),
    ]
    if any(value is None for value in values):
        return "n/a"
    return "/".join(_fmt(value) for value in values)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _range_text(record: dict[str, Any]) -> str:
    return f"{_fmt(record.get('min'))}-{_fmt(record.get('max'))}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
