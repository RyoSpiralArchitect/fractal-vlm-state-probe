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
) -> dict[str, Any]:
    if len(analyses) < 2:
        raise ValueError("at least two prompt-robustness analyses are required")
    if set(analyses) != set(analysis_paths):
        raise ValueError("analysis_paths must align with analyses")

    phases = {str(analysis.get("phase")) for analysis in analyses.values()}
    if len(phases) != 1:
        raise ValueError("all prompt-robustness analyses must use the same phase")

    models = [
        _model_summary(label, analyses[label], analysis_paths[label])
        for label in sorted(analyses)
    ]
    family_variants = _family_variant_summaries(models)
    return {
        "schema_version": 1,
        "analysis_kind": "prompt_robustness_cross_model_aggregate",
        "phase": phases.pop(),
        "model_count": len(models),
        "models": models,
        "family_variants": family_variants,
        "interpretation_notes": [
            "Generated-semantic stability and semantic-probability stability are separate measurements.",
            "Candidate probabilities are aligned by declared meaning within each source analysis.",
            "The same source factorial across models supports an architecture comparison, not additional visual source-pair replication.",
            "Prompt variants are diagnostic repeated measurements, not independent samples.",
        ],
    }


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
    lines = [
        "# Cross-Model Prompt Robustness",
        "",
        f"- Phase: `{analysis['phase']}`",
        f"- Models: `{analysis['model_count']}`",
        "",
        "## Model Summary",
        "",
        "| Model | Family | Generated variants changed | Max semantic TV | Baseline semantic interaction L1 | Semantic interaction L1 range | Max interaction delta from baseline |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for model in analysis["models"]:
        for family in model["families"]:
            lines.append(
                f"| `{model['model_label']}` | `{family['probe_family']}` | "
                f"{family['generated_variant_change_count']}/{family['variant_count'] - 1} | "
                f"{_fmt(family['max_semantic_tv_across_variant_pairs'])} | "
                f"{_fmt(family['baseline_semantic_interaction_l1'])} | "
                f"{_range_text(family['semantic_interaction_l1_range'])} | "
                f"{_fmt(family['max_interaction_vector_delta_from_baseline'])} |"
            )

    lines.extend(
        [
            "",
            "## Variant Table",
            "",
            "| Family | Variant | Generated semantics by model (MM/JJ/MJ/JM) | Semantic interaction L1 by model | Max semantic TV from baseline by model | Cross-model generated agreement |",
            "| --- | --- | --- | --- | --- | --- |",
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
        lines.append(
            f"| `{record['probe_family']}` | `{record['prompt_variant']}` | "
            f"{generated} | {interactions} | {tvs} | "
            f"`{record['generated_pattern_consistent_across_models']}` |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _model_summary(
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
        "model_label": label,
        "analysis_path": str(path),
        "families": [_model_family_summary(family) for family in families],
    }


def _model_family_summary(family: dict[str, Any]) -> dict[str, Any]:
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
    }


def _family_variant_summaries(
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for model in models:
        for family in model["families"]:
            for variant in family["variants"]:
                key = (family["probe_family"], variant["prompt_variant"])
                indexed.setdefault(key, {})[model["model_label"]] = variant

    expected_models = {model["model_label"] for model in models}
    summaries = []
    for (family, variant), records in indexed.items():
        if set(records) != expected_models:
            raise ValueError(f"model coverage mismatch for {family}/{variant}")
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
        summaries.append(
            {
                "probe_family": family,
                "prompt_variant": variant,
                "generated_patterns": patterns,
                "generated_pattern_consistent_across_models": len(
                    set(patterns.values())
                )
                == 1,
                "semantic_interaction_l1": interaction_values,
                "semantic_interaction_l1_median": median(interaction_values.values()),
                "max_semantic_tv_from_baseline": tv_values,
                "max_semantic_tv_from_baseline_median": median(tv_values.values()),
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


def _range_text(record: dict[str, Any]) -> str:
    return f"{_fmt(record.get('min'))}-{_fmt(record.get('max'))}"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
