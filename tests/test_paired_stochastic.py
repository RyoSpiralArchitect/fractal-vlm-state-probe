from __future__ import annotations

from pathlib import Path

import pytest

from fractal_vlm_state_probe.paired_stochastic import (
    analyze_paired_stochastic_batch,
    format_paired_stochastic_markdown,
    lexical_distance,
)
from fractal_vlm_state_probe.stimulus import write_json


def test_lexical_distance_uses_token_sets() -> None:
    assert lexical_distance("Blue silence", "Blue silence") == 0.0
    assert lexical_distance("blue silence", "red motion") == 1.0
    assert lexical_distance("blue silence", "blue motion") == pytest.approx(2 / 3)


def test_paired_stochastic_analysis_subtracts_before_variance(tmp_path: Path) -> None:
    records = []
    for seed in (0, 1):
        runs = {}
        for condition, mid_word in (
            ("null_blank", "plain"),
            ("julia", "color"),
        ):
            path = tmp_path / f"seed_{seed}_{condition}.json"
            write_json(
                path,
                _run_json(
                    condition=condition,
                    before=f"same before {seed}",
                    mid=f"{mid_word} mid {seed}",
                    after=f"{mid_word} after {seed}",
                ),
            )
            runs[condition] = str(path)
        comparison_path = tmp_path / f"comparison_{seed}.json"
        write_json(comparison_path, _comparison_json())
        records.append(
            {
                "probe_seed": seed,
                "runs": runs,
                "comparisons": {
                    "null_vs_julia": {
                        "json": str(comparison_path),
                    }
                },
            }
        )

    analysis = analyze_paired_stochastic_batch(records)

    assert analysis["before_alignment"]["all_conditions_same_count"] == 2
    pairwise_mid = [
        item
        for item in analysis["pairwise_phase_distances"]
        if item["left"] == "julia" and item["right"] == "null_blank" and item["phase"] == "mid"
    ][0]
    assert pairwise_mid["mean_before_adjusted_distance"] > 0
    assert analysis["pairwise_cache_distances"][0]["mean_max_abs_l2_delta"] == 2.0
    markdown = format_paired_stochastic_markdown(analysis)
    assert "Paired Stochastic Probe Batch" in markdown
    assert "Before probes identical" in markdown


def _run_json(*, condition: str, before: str, mid: str, after: str) -> dict:
    return {
        "stimulus": {"condition": {"condition_id": condition}},
        "probes": {
            "before": [_probe(before)],
            "mid": [_probe(mid)],
            "after": [_probe(after)],
        },
    }


def _probe(text: str) -> dict:
    return {
        "probe_id": "blue_silence_poem",
        "assistant_text": text,
    }


def _comparison_json() -> dict:
    return {
        "probe_source_cache_comparison": [
            {
                "phase": "mid",
                "source_cache_delta": {
                    "available": True,
                    "max_abs_l2_delta": 2.0,
                    "mean_abs_l2_delta": 1.0,
                },
            }
        ]
    }
