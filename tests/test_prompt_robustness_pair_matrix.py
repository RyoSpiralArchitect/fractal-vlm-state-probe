from __future__ import annotations

from pathlib import Path

import pytest

from fractal_vlm_state_probe.prompt_robustness_pair_matrix import (
    analyze_prompt_robustness_pair_matrix,
    format_prompt_robustness_pair_matrix_markdown,
)


def test_pair_matrix_separates_generated_and_balanced_axis_agreement() -> None:
    analyses = {
        "label_stable": {
            "b_c": _analysis(("mandelbrot", "julia"), ("spatial", "interaction")),
            "e_f": _analysis(("mandelbrot", "julia"), ("palette", "spatial")),
        },
        "fully_stable": {
            "b_c": _analysis(("mandelbrot", "julia"), ("interaction", "palette")),
            "e_f": _analysis(("mandelbrot", "julia"), ("interaction", "palette")),
        },
    }
    paths = {
        model: {pair: Path(f"{model}_{pair}.json") for pair in pair_records}
        for model, pair_records in analyses.items()
    }

    result = analyze_prompt_robustness_pair_matrix(
        analyses,
        analysis_paths=paths,
    )

    summary = result["global_summary"]
    assert result["paired_model_family_variant_count"] == 4
    assert summary["generated_pattern_agreement_count"] == 4
    assert summary["balanced_axis_agreement_count"] == 2
    assert summary["both_agreement_count"] == 2
    assert summary["agreement_contingency"] == {
        "generated_same_axis_same": 2,
        "generated_same_axis_different": 2,
        "generated_different_axis_same": 0,
        "generated_different_axis_different": 0,
    }
    assert summary["balanced_share_l1"]["median"] == pytest.approx(0.4)

    models = {record["model_label"]: record for record in result["models"]}
    assert models["label_stable"]["generated_pattern_agreement_count"] == 2
    assert models["label_stable"]["balanced_axis_agreement_count"] == 0
    assert models["fully_stable"]["both_agreement_count"] == 2

    markdown = format_prompt_robustness_pair_matrix_markdown(result)
    assert markdown.startswith("# Model x Source-Pair Prompt Replication")
    assert "| same | different | 2 |" in markdown
    assert "`label_stable`" in markdown


def test_pair_matrix_requires_common_pair_coverage() -> None:
    analyses = {
        "a": {"b_c": _analysis(), "e_f": _analysis()},
        "b": {"b_c": _analysis(), "d_e": _analysis()},
    }
    paths = {
        model: {pair: Path(f"{model}_{pair}.json") for pair in pair_records}
        for model, pair_records in analyses.items()
    }

    with pytest.raises(ValueError, match="same source pairs"):
        analyze_prompt_robustness_pair_matrix(analyses, analysis_paths=paths)


def test_pair_matrix_requires_two_models() -> None:
    analyses = {"only": {"b_c": _analysis(), "e_f": _analysis()}}
    paths = {
        "only": {
            "b_c": Path("only_b_c.json"),
            "e_f": Path("only_e_f.json"),
        }
    }

    with pytest.raises(ValueError, match="at least two models"):
        analyze_prompt_robustness_pair_matrix(analyses, analysis_paths=paths)


def test_pair_matrix_requires_balanced_energy_shares() -> None:
    analyses = {
        "a": {"b_c": _analysis(), "e_f": _analysis()},
        "b": {"b_c": _analysis(), "e_f": _analysis()},
    }
    del analyses["a"]["e_f"]["families"][0]["variants"][0][
        "full_vocab_balanced_contrast_energy"
    ]
    paths = {
        model: {pair: Path(f"{model}_{pair}.json") for pair in pair_records}
        for model, pair_records in analyses.items()
    }

    with pytest.raises(ValueError, match="balanced energy shares"):
        analyze_prompt_robustness_pair_matrix(analyses, analysis_paths=paths)


def _analysis(
    patterns: tuple[str, str] = ("mandelbrot", "julia"),
    axes: tuple[str, str] = ("spatial", "interaction"),
) -> dict:
    variants = [
        _variant("baseline", patterns[0], axes[0], comparison_tv=0.0),
        _variant("rotated_labels", patterns[1], axes[1], comparison_tv=0.2),
    ]
    return {
        "phase": "after",
        "families": [
            {
                "probe_family": "family",
                "generated_semantics_invariant_across_variants": (
                    patterns[0] == patterns[1]
                ),
                "max_semantic_tv_across_variant_pairs": 0.2,
                "variants": variants,
            }
        ],
    }


def _variant(
    name: str,
    semantic: str,
    dominant_axis: str,
    *,
    comparison_tv: float,
) -> dict:
    generated = {cell: semantic for cell in ("mm", "jj", "mj", "jm")}
    shares = {axis: 0.2 for axis in ("spatial", "palette", "interaction")}
    shares[dominant_axis] = 0.6
    return {
        "prompt_variant": name,
        "generated_semantics": generated,
        "semantic_contrasts": {"interaction": {"l1_norm": 0.2}},
        "full_vocab_interaction_l1": 0.2,
        "full_vocab_balanced_contrast_energy": {
            "energy_shares": {
                "spatial_contrast": shares["spatial"],
                "palette_contrast": shares["palette"],
                "interaction_contrast": shares["interaction"],
            }
        },
        "full_vocab_balanced_axis_dominant": dominant_axis,
        "comparison_to_baseline": {
            "generated_semantic_agreement_fraction": 1.0 if name == "baseline" else 0.0,
            "max_cell_semantic_tv": comparison_tv,
            "mean_cell_semantic_tv": comparison_tv,
            "interaction_vector_delta_l1": comparison_tv,
        },
    }
