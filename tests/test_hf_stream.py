from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.fractals import FractalSpec
from fractal_vlm_state_probe.hf_stream import HFStreamRunConfig, run_hf_stream_probe
from fractal_vlm_state_probe.stimulus import render_stimulus


def test_hf_stream_dry_run_records_adapter_and_condition(tmp_path: Path) -> None:
    stimulus_dir = tmp_path / "stimulus"
    output_path = tmp_path / "hf_dry_run.json"
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
        stimulus_dir,
    )
    result = run_hf_stream_probe(
        HFStreamRunConfig(
            manifest_path=stimulus_dir / "manifest.json",
            output_path=output_path,
            model_ref="example/model",
            dry_run=True,
            seed=123,
        )
    )
    assert result["adapter_capabilities"]["adapter_id"] == "hf_transformers"
    assert result["reproducibility"]["seed"] == 123
    assert result["stimulus_delivery"]["mode"] == "visual_stream"
    assert result["stimulus"]["condition"]["condition_family"] == "fractal"
    assert len(result["stream_events"]) == 2
    assert len(result["frame_artifacts"]) == 2
    assert result["stream_events"][0]["frame_artifact"]["path"].endswith(".png")
    assert output_path.exists()
