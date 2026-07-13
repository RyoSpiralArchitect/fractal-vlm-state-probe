from __future__ import annotations

from pathlib import Path

import pytest

from fractal_vlm_state_probe.prompt_robustness_multi_pair_matrix import (
    analyze_prompt_robustness_multi_pair_matrix,
    format_prompt_robustness_multi_pair_matrix_markdown,
)


def test_multi_pair_matrix_separates_all_pair_and_pairwise_agreement() -> None:
    analyses = {
        "label_stable": {
            "b_c": _analysis(("mandelbrot", "julia"), ("spatial", "palette")),
            "c_d": _analysis(("mandelbrot", "julia"), ("palette", "spatial")),
            "d_e": _analysis(("mandelbrot", "julia"), ("spatial", "palette")),
        },
        "axis_stable": {
            "b_c": _analysis(("mandelbrot", "julia"), ("interaction", "palette")),
            "c_d": _analysis(("mandelbrot", "julia"), ("interaction", "palette")),
            "d_e": _analysis(("julia", "mandelbrot"), ("interaction", "palette")),
        },
    }
    paths = {
        model: {pair: Path(f"{model}_{pair}.json") for pair in pair_records}
        for model, pair_records in analyses.items()
    }

    result = analyze_prompt_robustness_multi_pair_matrix(
        analyses,
        analysis_paths=paths,
    )

    all_pair = result["global_all_pair_summary"]
    assert result["model_family_variant_record_count"] == 4
    assert all_pair["generated_pattern_agreement_count"] == 2
    assert all_pair["balanced_axis_agreement_count"] == 2
    assert all_pair["both_agreement_count"] == 0

    pairwise = result["global_pairwise_summary"]
    assert result["pairwise_model_family_variant_count"] == 12
    assert pairwise["generated_pattern_agreement_count"] == 8
    assert pairwise["balanced_axis_agreement_count"] == 8
    assert pairwise["both_agreement_count"] == 4
    assert pairwise["agreement_contingency"] == {
        "generated_same_axis_same": 4,
        "generated_same_axis_different": 4,
        "generated_different_axis_same": 4,
        "generated_different_axis_different": 0,
    }
    assert pairwise["balanced_share_l1"]["median"] == pytest.approx(0.0)
    assert pairwise["balanced_share_l1"]["max"] == pytest.approx(0.8)

    models = {record["model_label"]: record for record in result["models"]}
    assert (
        models["label_stable"]["all_pair_summary"]["generated_pattern_agreement_count"]
        == 2
    )
    assert (
        models["label_stable"]["all_pair_summary"]["balanced_axis_agreement_count"] == 0
    )
    assert (
        models["axis_stable"]["all_pair_summary"]["balanced_axis_agreement_count"] == 2
    )

    markdown = format_prompt_robustness_multi_pair_matrix_markdown(result)
    assert markdown.startswith("# Model x Multi-Source-Pair Prompt Replication")
    assert "| same | different | 4 |" in markdown
    assert "`b_c` vs `c_d`" in markdown


def test_multi_pair_matrix_requires_common_pair_coverage() -> None:
    analyses = {
        "a": {"b_c": _analysis(), "c_d": _analysis(), "d_e": _analysis()},
        "b": {"b_c": _analysis(), "c_d": _analysis(), "e_f": _analysis()},
    }
    paths = {
        model: {pair: Path(f"{model}_{pair}.json") for pair in pair_records}
        for model, pair_records in analyses.items()
    }

    with pytest.raises(ValueError, match="same source pairs"):
        analyze_prompt_robustness_multi_pair_matrix(analyses, analysis_paths=paths)


def test_multi_pair_matrix_requires_three_source_pairs() -> None:
    analyses = {
        "a": {"b_c": _analysis(), "c_d": _analysis()},
        "b": {"b_c": _analysis(), "c_d": _analysis()},
    }
    paths = {
        model: {pair: Path(f"{model}_{pair}.json") for pair in pair_records}
        for model, pair_records in analyses.items()
    }

    with pytest.raises(ValueError, match="at least three source pairs"):
        analyze_prompt_robustness_multi_pair_matrix(analyses, analysis_paths=paths)


def test_multi_pair_matrix_requires_balanced_energy_shares() -> None:
    analyses = {
        model: {
            "b_c": _analysis(),
            "c_d": _analysis(),
            "d_e": _analysis(),
        }
        for model in ("a", "b")
    }
    del analyses["a"]["d_e"]["families"][0]["variants"][0][
        "full_vocab_balanced_contrast_energy"
    ]
    paths = {
        model: {pair: Path(f"{model}_{pair}.json") for pair in pair_records}
        for model, pair_records in analyses.items()
    }

    with pytest.raises(ValueError, match="balanced energy shares"):
        analyze_prompt_robustness_multi_pair_matrix(analyses, analysis_paths=paths)


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
