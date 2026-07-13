from __future__ import annotations

from pathlib import Path
from typing import Any

from .stimulus import write_json


CELL_KEYS = ("mm", "jj", "mj", "jm")
VARIANT_ORDER = ("baseline", "paraphrase", "reversed_order", "rotated_labels")


def analyze_prompt_robustness(
    full_vocab_analysis: dict[str, Any],
    *,
    source_path: Path,
    phase: str = "after",
) -> dict[str, Any]:
    records = [
        record
        for record in full_vocab_analysis.get("records") or []
        if record.get("phase") == phase and record.get("available") is True
    ]
    if not records:
        raise ValueError(f"no available full-vocabulary records for phase {phase!r}")

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        family = record.get("probe_family")
        variant = record.get("prompt_variant")
        candidates = record.get("forced_choice_candidates") or {}
        if not family or not variant or candidates.get("available") is not True:
            continue
        family_records = grouped.setdefault(str(family), {})
        if variant in family_records:
            raise ValueError(f"duplicate prompt variant for {family}: {variant}")
        family_records[str(variant)] = record
    if not grouped:
        raise ValueError("no metadata-rich forced-choice records were found")

    families = [
        _family_summary(family, variants)
        for family, variants in sorted(grouped.items())
    ]
    return {
        "schema_version": 1,
        "analysis_kind": "prompt_robustness",
        "source_full_vocab_analysis": str(source_path),
        "phase": phase,
        "family_count": len(families),
        "variant_count": sum(len(record["variants"]) for record in families),
        "families": families,
        "interpretation_notes": [
            "Candidate probabilities are aligned by declared meaning before variants are compared.",
            "Prompt variants change both wording and token context; this audit measures readout robustness, not a prompt-free visual representation.",
            "A stable generated label can coexist with a changing probability distribution, and the reverse can also occur near a decision boundary.",
            "Balanced spatial, palette, and interaction shares use equal-scale Hadamard contrasts over the complete first-token distribution.",
            "The four factorial cells are descriptive contrasts and are not independent samples.",
        ],
    }


def write_prompt_robustness_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_prompt_robustness_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_prompt_robustness_markdown(analysis), encoding="utf-8")


def format_prompt_robustness_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Prompt Robustness Audit",
        "",
        f"- Phase: `{analysis['phase']}`",
        f"- Source: `{analysis['source_full_vocab_analysis']}`",
    ]
    for family in analysis["families"]:
        lines.extend(
            [
                "",
                f"## {family['probe_family'].replace('_', ' ').title()}",
                "",
                f"- Generated semantics invariant across variants: `{family['generated_semantics_invariant_across_variants']}`",
                f"- Maximum semantic TV across variants: `{_fmt(family['max_semantic_tv_across_variant_pairs'])}`",
                "",
                "| Variant | Generated semantics (MM / JJ / MJ / JM) | Max pair JS | Full-vocab interaction L1 | Balanced S / P / I | Dominant balanced axis | Semantic interaction L1 | Semantic interaction argmax | Agreement with baseline | Max semantic TV from baseline | Interaction delta L1 |",
                "| --- | --- | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for variant in family["variants"]:
            generated = " / ".join(
                str(variant["generated_semantics"].get(cell)) for cell in CELL_KEYS
            )
            interaction = variant["semantic_contrasts"]["interaction"]
            comparison = variant.get("comparison_to_baseline") or {}
            lines.append(
                f"| `{variant['prompt_variant']}` | {generated} | "
                f"{_fmt(variant['max_pair_jensen_shannon'])} | "
                f"{_fmt(variant['full_vocab_interaction_l1'])} | "
                f"{_balanced_share_text(variant)} | "
                f"`{variant.get('full_vocab_balanced_axis_dominant') or 'n/a'}` | "
                f"{_fmt(interaction['l1_norm'])} | "
                f"`{interaction['argmax_semantic']}` | "
                f"{_fmt(comparison.get('generated_semantic_agreement_fraction'))} | "
                f"{_fmt(comparison.get('max_cell_semantic_tv'))} | "
                f"{_fmt(comparison.get('interaction_vector_delta_l1'))} |"
            )

        lines.extend(
            [
                "",
                "Semantic candidate conditionals:",
                "",
                "| Variant | Cell | Candidate mass | Semantic probabilities |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for variant in family["variants"]:
            for cell in CELL_KEYS:
                probabilities = ", ".join(
                    f"{semantic}={value:.6f}"
                    for semantic, value in variant["semantic_probabilities"][
                        cell
                    ].items()
                )
                lines.append(
                    f"| `{variant['prompt_variant']}` | `{cell}` | "
                    f"{_fmt(variant['candidate_probability_mass'][cell])} | "
                    f"{probabilities} |"
                )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _family_summary(
    family: str,
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variants = [
        _variant_summary(record)
        for _, record in sorted(
            records.items(), key=lambda item: _variant_sort_key(item[0])
        )
    ]
    baseline = next(
        (record for record in variants if record["prompt_variant"] == "baseline"),
        None,
    )
    if baseline is None:
        raise ValueError(f"prompt family {family!r} has no baseline variant")
    for variant in variants:
        variant["comparison_to_baseline"] = _compare_variants(baseline, variant)

    pairwise = []
    for index, left in enumerate(variants):
        for right in variants[index + 1 :]:
            comparison = _compare_variants(left, right)
            pairwise.append(
                {
                    "left": left["prompt_variant"],
                    "right": right["prompt_variant"],
                    **comparison,
                }
            )
    return {
        "probe_family": family,
        "variant_count": len(variants),
        "generated_semantics_invariant_across_variants": all(
            variant["generated_semantics"] == baseline["generated_semantics"]
            for variant in variants
        ),
        "max_semantic_tv_across_variant_pairs": max(
            (record["max_cell_semantic_tv"] for record in pairwise),
            default=0.0,
        ),
        "variants": variants,
        "variant_pairwise": pairwise,
    }


def _variant_summary(record: dict[str, Any]) -> dict[str, Any]:
    candidates = record["forced_choice_candidates"]
    semantics = list(dict.fromkeys((record.get("candidate_semantics") or {}).values()))
    cells = candidates["cells"]
    semantic_probabilities = {
        cell: {
            semantic: float(
                cells[cell]["semantic_conditional_probabilities"].get(semantic, 0.0)
            )
            for semantic in semantics
        }
        for cell in CELL_KEYS
    }
    pairwise = record.get("pairwise_distances") or []
    return {
        "probe_id": record["probe_id"],
        "probe_family": record["probe_family"],
        "prompt_variant": record["prompt_variant"],
        "candidate_order": record.get("candidate_order"),
        "candidate_semantics": record.get("candidate_semantics"),
        "generated_semantics": record.get("generated_semantics") or {},
        "candidate_probability_mass": {
            cell: float(cells[cell]["candidate_probability_mass"]) for cell in CELL_KEYS
        },
        "semantic_probabilities": semantic_probabilities,
        "semantic_contrasts": _semantic_contrasts(semantic_probabilities, semantics),
        "max_pair_jensen_shannon": max(
            (float(item["jensen_shannon"]) for item in pairwise),
            default=0.0,
        ),
        "full_vocab_interaction_l1": float(
            record["probability_contrasts"]["interaction"]["l1_norm"]
        ),
        "full_vocab_interaction_max_abs": float(
            record["probability_contrasts"]["interaction"]["max_abs"]
        ),
        "full_vocab_balanced_contrast_energy": record.get(
            "balanced_probability_contrast_energy"
        ),
        "full_vocab_balanced_axis_dominant": _balanced_axis_dominant(record),
    }


def _balanced_axis_dominant(record: dict[str, Any]) -> str | None:
    energy = record.get("balanced_probability_contrast_energy") or {}
    shares = energy.get("energy_shares") or {}
    named_shares = {
        "spatial": shares.get("spatial_contrast"),
        "palette": shares.get("palette_contrast"),
        "interaction": shares.get("interaction_contrast"),
    }
    available = {
        name: float(value) for name, value in named_shares.items() if value is not None
    }
    return max(available, key=available.get) if available else None


def _balanced_share_text(variant: dict[str, Any]) -> str:
    energy = variant.get("full_vocab_balanced_contrast_energy") or {}
    shares = energy.get("energy_shares") or {}
    values = [
        shares.get("spatial_contrast"),
        shares.get("palette_contrast"),
        shares.get("interaction_contrast"),
    ]
    if any(value is None for value in values):
        return "n/a"
    return " / ".join(_fmt(value) for value in values)


def _semantic_contrasts(
    probabilities: dict[str, dict[str, float]],
    semantics: list[str],
) -> dict[str, Any]:
    vectors = {
        "spatial_main_effect": {
            semantic: (
                probabilities["jm"][semantic]
                - probabilities["mm"][semantic]
                + probabilities["jj"][semantic]
                - probabilities["mj"][semantic]
            )
            / 2.0
            for semantic in semantics
        },
        "palette_main_effect": {
            semantic: (
                probabilities["mj"][semantic]
                - probabilities["mm"][semantic]
                + probabilities["jj"][semantic]
                - probabilities["jm"][semantic]
            )
            / 2.0
            for semantic in semantics
        },
        "interaction": {
            semantic: probabilities["jj"][semantic]
            - probabilities["jm"][semantic]
            - probabilities["mj"][semantic]
            + probabilities["mm"][semantic]
            for semantic in semantics
        },
    }
    return {name: _vector_summary(vector) for name, vector in vectors.items()}


def _vector_summary(vector: dict[str, float]) -> dict[str, Any]:
    argmax = max(vector, key=lambda key: abs(vector[key])) if vector else None
    return {
        "values": vector,
        "l1_norm": sum(abs(value) for value in vector.values()),
        "max_abs": abs(vector[argmax]) if argmax is not None else 0.0,
        "argmax_semantic": argmax,
    }


def _compare_variants(
    reference: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    cell_tvs = {
        cell: _total_variation(
            reference["semantic_probabilities"][cell],
            comparison["semantic_probabilities"][cell],
        )
        for cell in CELL_KEYS
    }
    left_interaction = reference["semantic_contrasts"]["interaction"]["values"]
    right_interaction = comparison["semantic_contrasts"]["interaction"]["values"]
    semantics = set(left_interaction) | set(right_interaction)
    generated_matches = {
        cell: reference["generated_semantics"].get(cell)
        == comparison["generated_semantics"].get(cell)
        for cell in CELL_KEYS
    }
    return {
        "generated_semantic_matches": generated_matches,
        "generated_semantic_agreement_fraction": sum(generated_matches.values())
        / len(CELL_KEYS),
        "cell_semantic_total_variation": cell_tvs,
        "mean_cell_semantic_tv": sum(cell_tvs.values()) / len(CELL_KEYS),
        "max_cell_semantic_tv": max(cell_tvs.values()),
        "interaction_vector_delta_l1": sum(
            abs(left_interaction.get(key, 0.0) - right_interaction.get(key, 0.0))
            for key in semantics
        ),
    }


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _variant_sort_key(variant: str) -> tuple[int, str]:
    try:
        return (VARIANT_ORDER.index(variant), variant)
    except ValueError:
        return (len(VARIANT_ORDER), variant)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
