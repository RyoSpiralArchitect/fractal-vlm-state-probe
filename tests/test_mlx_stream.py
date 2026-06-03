from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.fractals import FractalSpec
from fractal_vlm_state_probe.mlx_stream import (
    StreamRunConfig,
    _mid_probe_after_position,
    _should_summarize_cache,
    run_stream_probe,
)
from fractal_vlm_state_probe.stimulus import render_stimulus


def test_mid_probe_after_position_consumes_half_stream() -> None:
    assert _mid_probe_after_position(0) is None
    assert _mid_probe_after_position(1) == 0
    assert _mid_probe_after_position(11) == 5
    assert _mid_probe_after_position(12) == 5


def test_should_summarize_cache_keeps_key_positions() -> None:
    assert _should_summarize_cache(0, frame_count=12, every=10)
    assert _should_summarize_cache(5, frame_count=12, every=10)
    assert _should_summarize_cache(10, frame_count=12, every=10)
    assert _should_summarize_cache(11, frame_count=12, every=10)
    assert not _should_summarize_cache(4, frame_count=12, every=10)
    assert not _should_summarize_cache(0, frame_count=12, every=0)


def test_mlx_dry_run_records_text_only_delivery(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "text_only.json"
    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            dry_run=True,
            delivery_mode="text_only_stream",
            max_frames=2,
        )
    )
    assert result["stimulus_delivery"]["mode"] == "text_only_stream"
    assert result["stream_events"][0]["delivery"]["num_images"] == 0
    assert "No image is attached" in result["stream_events"][0]["planned_prompt"]
    assert result["frame_artifacts"] == []


def test_mlx_dry_run_records_blank_visual_delivery(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "blank.json"
    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            dry_run=True,
            delivery_mode="blank_visual_stream",
            max_frames=1,
            blank_rgb=(5, 6, 7),
        )
    )
    assert result["stimulus_delivery"]["mode"] == "blank_visual_stream"
    assert result["stream_events"][0]["delivery"]["num_images"] == 1
    assert result["frame_artifacts"][0]["generated_control"]
    assert (tmp_path / result["frame_artifacts"][0]["path"]).exists()


def test_mlx_dry_run_probe_only_omits_stream_turns(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "probe_only.json"
    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            dry_run=True,
            delivery_mode="probe_only",
            max_frames=2,
        )
    )
    assert result["stimulus_delivery"]["mode"] == "probe_only"
    assert result["stimulus"]["source_frame_count_selected"] == 2
    assert result["stimulus"]["frame_count_selected"] == 0
    assert result["stream_events"] == []


def _render_test_manifest(output_dir: Path) -> Path:
    render_stimulus(
        FractalSpec(
            kind="mandelbrot",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            center_start=(-0.75, 0.1),
            center_end=(-0.74, 0.12),
            scale_start=2.0,
            scale_end=1.0,
            max_iter=24,
        ),
        output_dir,
    )
    return output_dir / "manifest.json"
