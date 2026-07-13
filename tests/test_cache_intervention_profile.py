from __future__ import annotations

from fractal_vlm_state_probe.cache_intervention_profile import (
    compare_cache_intervention_profiles,
    format_profile_comparison_markdown,
)


def test_profile_comparison_aligns_layers_and_reports_correlations() -> None:
    result = compare_cache_intervention_profiles(
        {
            "pair_a": _analysis([0.1, 0.2, 0.4]),
            "pair_b": _analysis([0.2, 0.4, 0.8], source="mm_b", donor="jj_b"),
        }
    )

    comparison = result["comparisons"][0]
    assert comparison["common_layers"] == [8, 9, 10]
    assert comparison["pearson"] == 1.0
    assert comparison["spearman"] == 1.0
    assert comparison["argmax_same"] is True
    assert comparison["top_3_overlap_count"] == 3
    assert result["profiles"]["pair_a"]["argmax_layer"] == 10
    directional = result["profiles"]["pair_a"]["directional_comparison"]
    assert directional["pearson"] == 1.0
    assert directional["argmax_same"] is True
    assert "Cache Intervention Layer Profiles" in format_profile_comparison_markdown(result)
    assert "Reciprocal Direction Checks" in format_profile_comparison_markdown(result)


def test_profile_comparison_requires_one_identity_per_input() -> None:
    analysis = _analysis([0.1, 0.2, 0.3])
    analysis["group_summaries"][1]["donor_condition"] = "different"

    try:
        compare_cache_intervention_profiles({"a": analysis, "b": _analysis([0.1, 0.2, 0.3])})
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("expected mixed profile identity to be rejected")


def _analysis(values: list[float], *, source: str = "mm", donor: str = "jj") -> dict:
    return {
        "group_summaries": [
            {
                "model_id": "test/model",
                "source_condition": source,
                "donor_condition": donor,
                "probe_phase": "mid",
                "layer_index": layer,
                "tensor": "values",
                "trial_count": 1,
                "probe_seeds": [0],
                "metrics": {
                    "top_k_effect_to_baseline_ratio": {"mean": ratio},
                    "source_intervention_top_k_effect": {"mean": ratio / 10},
                    "top_k_donor_pull_index": {"mean": -1 + ratio},
                    "reciprocal_top_k_effect_to_baseline_ratio": {
                        "mean": ratio * 0.9
                    },
                    "top_k_source_pull_index": {"mean": -1 + ratio * 0.9},
                    "self_sham_top_k_effect": {"mean": 0.0},
                },
            }
            for layer, ratio in zip((8, 9, 10), values)
        ]
    }
