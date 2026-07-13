from __future__ import annotations

from pathlib import Path

import pytest

from fractal_vlm_state_probe.prompt_robustness import analyze_prompt_robustness


def test_prompt_robustness_aligns_labels_by_semantics() -> None:
    analysis = {
        "records": [
            _record(
                "baseline",
                {"A": "mandelbrot", "B": "julia", "C": "unclear"},
                {
                    "mm": {"mandelbrot": 0.8, "julia": 0.1, "unclear": 0.1},
                    "jj": {"mandelbrot": 0.2, "julia": 0.7, "unclear": 0.1},
                    "mj": {"mandelbrot": 0.4, "julia": 0.5, "unclear": 0.1},
                    "jm": {"mandelbrot": 0.6, "julia": 0.3, "unclear": 0.1},
                },
            ),
            _record(
                "rotated_labels",
                {"A": "unclear", "B": "mandelbrot", "C": "julia"},
                {
                    "mm": {"unclear": 0.1, "mandelbrot": 0.8, "julia": 0.1},
                    "jj": {"unclear": 0.1, "mandelbrot": 0.2, "julia": 0.7},
                    "mj": {"unclear": 0.1, "mandelbrot": 0.4, "julia": 0.5},
                    "jm": {"unclear": 0.1, "mandelbrot": 0.6, "julia": 0.3},
                },
            ),
        ]
    }

    result = analyze_prompt_robustness(
        analysis,
        source_path=Path("full_vocab.json"),
    )

    family = result["families"][0]
    rotated = family["variants"][1]
    assert rotated["comparison_to_baseline"]["max_cell_semantic_tv"] == pytest.approx(0)
    assert rotated["comparison_to_baseline"][
        "interaction_vector_delta_l1"
    ] == pytest.approx(0)
    assert family["generated_semantics_invariant_across_variants"] is True
    assert rotated["full_vocab_balanced_axis_dominant"] == "interaction"
    assert rotated["full_vocab_balanced_contrast_energy"]["energy_shares"] == {
        "spatial_contrast": 0.2,
        "palette_contrast": 0.3,
        "interaction_contrast": 0.5,
    }


def test_prompt_robustness_detects_prompt_sensitive_readout() -> None:
    baseline = {
        cell: {"mandelbrot": 0.99, "julia": 0.005, "unclear": 0.005}
        for cell in ("mm", "jj", "mj", "jm")
    }
    reversed_order = {
        "mm": {"mandelbrot": 0.99, "julia": 0.005, "unclear": 0.005},
        "jj": {"mandelbrot": 0.01, "julia": 0.985, "unclear": 0.005},
        "mj": {"mandelbrot": 0.01, "julia": 0.985, "unclear": 0.005},
        "jm": {"mandelbrot": 0.01, "julia": 0.985, "unclear": 0.005},
    }
    analysis = {
        "records": [
            _record(
                "baseline",
                {"A": "mandelbrot", "B": "julia", "C": "unclear"},
                baseline,
            ),
            _record(
                "reversed_order",
                {"A": "mandelbrot", "B": "julia", "C": "unclear"},
                reversed_order,
            ),
        ]
    }

    result = analyze_prompt_robustness(
        analysis,
        source_path=Path("full_vocab.json"),
    )

    family = result["families"][0]
    reversed_summary = family["variants"][1]
    comparison = reversed_summary["comparison_to_baseline"]
    assert comparison["generated_semantic_agreement_fraction"] == pytest.approx(0.25)
    assert comparison["max_cell_semantic_tv"] == pytest.approx(0.98)
    assert comparison["interaction_vector_delta_l1"] > 1.9
    assert family["generated_semantics_invariant_across_variants"] is False


def _record(
    variant: str,
    candidate_semantics: dict[str, str],
    semantic_probabilities: dict[str, dict[str, float]],
) -> dict:
    inverse = {semantic: label for label, semantic in candidate_semantics.items()}
    generated = {
        cell: max(probabilities, key=probabilities.get)
        for cell, probabilities in semantic_probabilities.items()
    }
    return {
        "phase": "after",
        "probe_id": f"forced_family_choice_{variant}",
        "probe_family": "family",
        "prompt_variant": variant,
        "candidate_order": list(candidate_semantics),
        "candidate_semantics": candidate_semantics,
        "available": True,
        "generated_semantics": generated,
        "pairwise_distances": [{"jensen_shannon": 0.1}],
        "probability_contrasts": {"interaction": {"l1_norm": 0.2, "max_abs": 0.1}},
        "balanced_probability_contrast_energy": {
            "energy_shares": {
                "spatial_contrast": 0.2,
                "palette_contrast": 0.3,
                "interaction_contrast": 0.5,
            }
        },
        "forced_choice_candidates": {
            "available": True,
            "cells": {
                cell: {
                    "candidate_probability_mass": 1.0,
                    "semantic_conditional_probabilities": probabilities,
                    "conditional_probabilities": {
                        inverse[semantic]: value
                        for semantic, value in probabilities.items()
                    },
                }
                for cell, probabilities in semantic_probabilities.items()
            },
        },
    }
