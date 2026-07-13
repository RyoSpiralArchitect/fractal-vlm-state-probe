from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.fractals import FractalSpec
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    cumulative_replay_prompt,
    cumulative_replay_probe_prompt,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.stimulus import render_stimulus


def test_cumulative_replay_prompt_records_order_and_timecodes() -> None:
    prompt = cumulative_replay_prompt(
        [
            {"index": 3, "t_seconds": 0.5},
            {"index": 9, "t_seconds": 1.25},
        ]
    )

    assert "ordered cumulative replay of 2 frames" in prompt
    assert prompt.index("frame 000003") < prompt.index("frame 000009")
    assert "00:00.500" in prompt
    assert "00:01.250" in prompt


def test_direct_replay_probe_prompt_keeps_order_and_measurement_question() -> None:
    prompt = cumulative_replay_probe_prompt(
        [
            {"index": 3, "t_seconds": 0.5},
            {"index": 9, "t_seconds": 1.25},
        ],
        "Answer exactly A, B, or C.",
    )

    assert prompt.index("frame 000003") < prompt.index("frame 000009")
    assert "Answer exactly A, B, or C." in prompt
    assert "Return ACK only" not in prompt


def test_cumulative_replay_dry_run_records_distinct_protocol(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "replay.json"

    result = run_cumulative_replay_probe(
        CumulativeReplayRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            max_frames=2,
            probe_seed=5,
            dry_run=True,
            include_frame_artifacts=False,
        )
    )

    assert result["run_kind"] == "mlx_vlm_cumulative_visual_replay_probe"
    assert (
        result["context_policy"]["visual_context_protocol"]
        == "single_turn_ordered_multi_image_replay"
    )
    assert result["probe_schedule"]["mid"] is None
    assert result["context_policy"]["after_probe_protocol"] == "direct_multimodal_replay"
    assert result["context_policy"]["capture_direct_probe_cache"] is False
    assert "fresh direct multimodal" in result["probe_schedule"]["after"]
    assert result["reproducibility"]["probe_phase_seeds"]["after"] == 7
    assert result["probes"]["before"][0]["prompt_variant"] == "baseline"
    assert result["probes"]["before"][0]["candidate_semantics"]["A"] == "mandelbrot"
    event = result["stream_events"][0]
    assert event["image_count"] == 2
    assert event["frame_indices"] == [0, 1]
    assert event["frame_index"] == 1
    assert event["cache_summary_captured"] is False
    assert output_path.exists()


def _render_test_manifest(output_dir: Path) -> Path:
    render_stimulus(
        FractalSpec(
            kind="mandelbrot",
            width=32,
            height=24,
            total_frames=3,
            fps=2.0,
            max_iter=12,
            center_start=(-0.6, 0.0),
            center_end=(-0.58, 0.02),
            scale_start=3.0,
            scale_end=2.0,
            color_seed=17,
        ),
        output_dir,
        overwrite=True,
    )
    return output_dir / "manifest.json"
