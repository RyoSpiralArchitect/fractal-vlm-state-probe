from __future__ import annotations

from fractal_vlm_state_probe.prompts import (
    available_probe_presets,
    probe_metadata,
    resolve_probe_preset,
)


def test_forced_choice_robustness_preset_covers_four_variants() -> None:
    probes = resolve_probe_preset("forced_choice_robustness")

    assert len(probes) == 8
    assert len({probe["id"] for probe in probes}) == 8
    assert {probe["prompt_variant"] for probe in probes} == {
        "baseline",
        "paraphrase",
        "reversed_order",
        "rotated_labels",
    }
    assert "forced_choice_robustness" in available_probe_presets()


def test_rotated_labels_keep_explicit_semantic_mapping() -> None:
    probes = resolve_probe_preset("forced_choice_rotated_labels")
    family = next(probe for probe in probes if probe["probe_family"] == "family")
    frequency = next(probe for probe in probes if probe["probe_family"] == "frequency")

    assert family["candidate_semantics"] == {
        "A": "unclear",
        "B": "mandelbrot",
        "C": "julia",
    }
    assert frequency["candidate_semantics"] == {
        "L": "unclear",
        "H": "low_frequency",
        "C": "high_frequency",
    }
    assert probe_metadata(family)["prompt_variant"] == "rotated_labels"


def test_resolve_probe_preset_deep_copies_nested_metadata() -> None:
    first = resolve_probe_preset("forced_choice_rotated_labels")
    first[0]["candidate_semantics"]["A"] = "changed"

    second = resolve_probe_preset("forced_choice_rotated_labels")

    assert second[0]["candidate_semantics"]["A"] == "unclear"
