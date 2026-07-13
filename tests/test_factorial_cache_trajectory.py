from __future__ import annotations

import json
from pathlib import Path

from fractal_vlm_state_probe.factorial_cache_trajectory import (
    analyze_factorial_cache_trajectory,
    format_factorial_cache_trajectory_markdown,
)


def test_factorial_cache_trajectory_tracks_scalar_and_position_stability(
    tmp_path: Path,
) -> None:
    analyses = {}
    paths = {}
    for frames, effect, position in ((1, -10.0, 43), (2, -20.0, 144)):
        root = tmp_path / f"{frames}f"
        root.mkdir()
        run_path = root / "mm.json"
        _write_json(run_path, _run(frames, position))
        _write_json(root / "probe_readout_contrast.json", _readout())
        analysis_path = root / "factorial_cache_contrast.json"
        analysis = _analysis(run_path, effect=effect, position=position)
        _write_json(analysis_path, analysis)
        label = f"point_{frames}f"
        analyses[label] = analysis
        paths[label] = analysis_path

    result = analyze_factorial_cache_trajectory(
        analyses,
        analysis_paths=paths,
    )

    series = result["series"][0]
    assert series["frame_counts"] == [1, 2]
    assert series["scalar_argmax_location_consistent"] is True
    assert series["position_argmax_location_consistent"] is False
    assert series["readout_cell_invariant_at_all_points"] is True
    assert series["frame_count_vs_abs_scalar_interaction_pearson"] == 1.0
    assert result["points"][1]["position_argmax"]["roles"] == ["image_run_1_start"]
    assert "Factorial Cache Trajectory" in format_factorial_cache_trajectory_markdown(result)


def _analysis(run_path: Path, *, effect: float, position: int) -> dict:
    scalar = {
        "phase": "after",
        "probe_id": "forced_family_choice",
        "layer_index": 33,
        "tensor": "values",
        "interaction_effect": effect,
        "abs_interaction_effect": abs(effect),
    }
    positioned = {**scalar, "position": position}
    return {
        "cells": {
            "mm": {
                "source_path": str(run_path),
                "condition_id": "mandelbrot",
                "model": "test/model",
            },
            "jj": {"condition_id": "julia", "model": "test/model"},
        },
        "interaction_argmax_summary": [
            {
                "record_type": "scalar",
                "probe_id": "forced_family_choice",
                "after": scalar,
            },
            {
                "record_type": "sequence_position",
                "probe_id": "forced_family_choice",
                "after": positioned,
            },
        ],
        "scalar_records": [
            {
                **scalar,
                "field": "l2_norm",
                "relative_abs_interaction_to_grand_mean": 0.5,
            }
        ],
        "sequence_position_records": [
            {
                **positioned,
                "field": "l2_norm",
                "relative_abs_interaction_to_grand_mean": 0.25,
            }
        ],
    }


def _run(frames: int, position: int) -> dict:
    return {
        "stimulus": {"frame_count_selected": frames},
        "context_policy": {
            "visual_context_protocol": "single_turn_ordered_multi_image_replay"
        },
        "stream_events": [
            {
                "cache_token_layout": {
                    "token_count": 100 * frames,
                    "image_token_count": 99 * frames,
                    "image_token_runs": [],
                    "sequence_position_plan": [
                        {"position": position, "roles": [f"image_run_{frames - 1}_start"]}
                    ],
                }
            }
        ],
    }


def _readout() -> dict:
    return {
        "records": [
            {
                "phase": "after",
                "probe_id": "forced_family_choice",
                "generated_tokens": {
                    cell: {"token": "C"} for cell in ("mm", "jj", "mj", "jm")
                },
                "mean_top_k_jaccard": 1.0,
                "top_common_token_effects": [{"interaction_effect": 0.0}],
            }
        ]
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")
