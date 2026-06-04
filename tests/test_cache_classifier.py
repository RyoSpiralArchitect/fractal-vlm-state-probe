from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.cache_classifier import (
    analyze_cache_feature_records,
    format_classifier_markdown,
    load_run_records,
)
from fractal_vlm_state_probe.stimulus import write_json


def test_cache_classifier_separates_simple_summary_features(tmp_path: Path) -> None:
    paths = []
    for seed in (1, 2):
        for label, value in (("blank", 1.0), ("julia", 9.0)):
            path = tmp_path / f"seed_{seed}_{label}_mlx.json"
            write_json(path, _run_json(label=label, seed=seed, value=value))
            paths.append(path)

    analysis = analyze_cache_feature_records(load_run_records(paths))

    assert analysis["nearest_centroid_leave_one_seed_out"]["accuracy"] == 1.0
    assert len(analysis["duplicate_feature_groups"]) == 2
    markdown = format_classifier_markdown(analysis)
    assert "Cache Summary Condition Classifier" in markdown
    assert "deterministic duplicate seeds" in markdown


def _run_json(*, label: str, seed: int, value: float) -> dict:
    summary = {
        "available": True,
        "layers": [
            {
                "layer_index": 0,
                "keys": {
                    "mean": value,
                    "variance": value + 0.1,
                    "std": value + 0.2,
                    "abs_mean": value + 0.3,
                    "l2_norm": value + 0.4,
                },
                "values": {
                    "mean": value + 0.5,
                    "variance": value + 0.6,
                    "std": value + 0.7,
                    "abs_mean": value + 0.8,
                    "l2_norm": value + 0.9,
                },
            }
        ],
    }
    return {
        "stimulus": {"condition": {"condition_id": label}},
        "reproducibility": {"seed": seed},
        "stream_events": [{"frame_index": 0, "cache_summary": summary}],
        "probes": {
            "mid": [
                {
                    "probe_id": "p",
                    "source_cache_summary_before_probe": summary,
                }
            ],
            "after": [
                {
                    "probe_id": "p",
                    "source_cache_summary_before_probe": summary,
                }
            ],
        },
    }
