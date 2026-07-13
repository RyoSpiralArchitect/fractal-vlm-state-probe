from __future__ import annotations

from pathlib import Path

import pytest

from fractal_vlm_state_probe.prompt_robustness_aggregate import (
    analyze_prompt_robustness_aggregate,
    format_prompt_robustness_aggregate_markdown,
)


def test_prompt_aggregate_separates_label_and_probability_stability() -> None:
    analyses = {
        "stable": _analysis(
            baseline_pattern="mandelbrot/mandelbrot/mandelbrot/mandelbrot",
            rotated_pattern="mandelbrot/mandelbrot/mandelbrot/mandelbrot",
            max_tv=0.1,
            rotated_interaction=0.4,
        ),
        "label_shift": _analysis(
            baseline_pattern="mandelbrot/mandelbrot/mandelbrot/mandelbrot",
            rotated_pattern="julia/julia/julia/julia",
            max_tv=0.8,
            rotated_interaction=1.2,
        ),
    }
    paths = {label: Path(f"{label}.json") for label in analyses}

    result = analyze_prompt_robustness_aggregate(
        analyses,
        analysis_paths=paths,
    )

    models = {record["model_label"]: record for record in result["models"]}
    stable_family = models["stable"]["families"][0]
    shifted_family = models["label_shift"]["families"][0]
    assert stable_family["generated_variant_change_count"] == 0
    assert shifted_family["generated_variant_change_count"] == 1
    assert shifted_family["max_semantic_tv_across_variant_pairs"] == pytest.approx(0.8)
    assert shifted_family["balanced_axis_dominance_counts"] == {
        "spatial": 0,
        "palette": 0,
        "interaction": 2,
    }

    rotated = next(
        record
        for record in result["family_variants"]
        if record["prompt_variant"] == "rotated_labels"
    )
    assert rotated["generated_pattern_consistent_across_models"] is False
    assert rotated["semantic_interaction_l1_median"] == pytest.approx(0.8)
    assert rotated["balanced_energy_shares"]["stable"]["interaction"] == pytest.approx(
        0.5
    )


def test_prompt_aggregate_names_source_pair_comparison_explicitly() -> None:
    analyses = {"b_c": _analysis(), "e_f": _analysis()}
    paths = {label: Path(f"{label}.json") for label in analyses}

    result = analyze_prompt_robustness_aggregate(
        analyses,
        analysis_paths=paths,
        comparison_axis="source_pair",
    )

    assert result["analysis_kind"] == "prompt_robustness_cross_source_pair_aggregate"
    assert result["comparison_axis"] == "source_pair"
    assert result["unit_count"] == 2
    assert {unit["unit_label"] for unit in result["units"]} == {"b_c", "e_f"}
    assert any("visual replication" in note for note in result["interpretation_notes"])
    markdown = format_prompt_robustness_aggregate_markdown(result)
    assert markdown.startswith("# Cross-Source-Pair Prompt Robustness")
    assert "## Source Pair Summary" in markdown
    assert "Balanced S/P/I by source pair" in markdown


def test_prompt_robustness_aggregate_requires_aligned_paths() -> None:
    with pytest.raises(ValueError, match="analysis_paths"):
        analyze_prompt_robustness_aggregate(
            {"a": _analysis(), "b": _analysis()},
            analysis_paths={"a": Path("a.json")},
        )


def test_prompt_robustness_aggregate_rejects_unknown_comparison_axis() -> None:
    with pytest.raises(ValueError, match="comparison_axis"):
        analyze_prompt_robustness_aggregate(
            {"a": _analysis(), "b": _analysis()},
            analysis_paths={"a": Path("a.json"), "b": Path("b.json")},
            comparison_axis="architecture",
        )


def _analysis(
    *,
    baseline_pattern: str = "mandelbrot/mandelbrot/mandelbrot/mandelbrot",
    rotated_pattern: str = "mandelbrot/mandelbrot/mandelbrot/mandelbrot",
    max_tv: float = 0.2,
    rotated_interaction: float = 0.4,
) -> dict:
    return {
        "phase": "after",
        "families": [
            {
                "probe_family": "family",
                "generated_semantics_invariant_across_variants": (
                    baseline_pattern == rotated_pattern
                ),
                "max_semantic_tv_across_variant_pairs": max_tv,
                "variants": [
                    _variant("baseline", baseline_pattern, 0.2, 0.0, 0.0),
                    _variant(
                        "rotated_labels",
                        rotated_pattern,
                        rotated_interaction,
                        max_tv,
                        abs(rotated_interaction - 0.2),
                    ),
                ],
            }
        ],
    }


def _variant(
    name: str,
    pattern: str,
    interaction: float,
    max_tv: float,
    interaction_delta: float,
) -> dict:
    generated = dict(zip(("mm", "jj", "mj", "jm"), pattern.split("/"), strict=True))
    return {
        "prompt_variant": name,
        "generated_semantics": generated,
        "semantic_contrasts": {"interaction": {"l1_norm": interaction}},
        "full_vocab_interaction_l1": interaction,
        "full_vocab_balanced_contrast_energy": {
            "energy_shares": {
                "spatial_contrast": 0.2,
                "palette_contrast": 0.3,
                "interaction_contrast": 0.5,
            }
        },
        "full_vocab_balanced_axis_dominant": "interaction",
        "comparison_to_baseline": {
            "generated_semantic_agreement_fraction": (
                1.0
                if name == "baseline"
                else sum(value == "mandelbrot" for value in generated.values()) / 4
            ),
            "max_cell_semantic_tv": max_tv,
            "mean_cell_semantic_tv": max_tv / 2,
            "interaction_vector_delta_l1": interaction_delta,
        },
    }
