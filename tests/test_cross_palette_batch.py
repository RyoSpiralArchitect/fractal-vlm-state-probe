from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.cross_palette_batch import (
    CrossPalettePairSpec,
    parse_pair_spec,
    prepare_cross_palette_factorial_batch,
)
from fractal_vlm_state_probe.fractals import FractalSpec
from fractal_vlm_state_probe.stimulus import render_stimulus, validate_manifest


def test_parse_pair_spec_requires_pair_id_and_two_paths() -> None:
    parsed = parse_pair_spec("pair_a=runs/m/manifest.json,runs/j/manifest.json")

    assert parsed.pair_id == "pair_a"
    assert str(parsed.mandelbrot_manifest) == "runs/m/manifest.json"
    assert str(parsed.julia_manifest) == "runs/j/manifest.json"


def test_prepare_cross_palette_factorial_batch_writes_manifests_and_raw_contrast(tmp_path: Path) -> None:
    mandelbrot_manifest = _render_fractal(tmp_path / "mandelbrot", "mandelbrot")
    julia_manifest = _render_fractal(tmp_path / "julia", "julia")

    summary = prepare_cross_palette_factorial_batch(
        pair_specs=[
            CrossPalettePairSpec(
                pair_id="m_a_j_b",
                mandelbrot_manifest=mandelbrot_manifest,
                julia_manifest=julia_manifest,
            )
        ],
        output_root=tmp_path / "batch",
        max_frames=2,
    )

    record = summary["records"][0]
    mj_manifest = Path(record["manifests"]["mj"]["path"])
    jm_manifest = Path(record["manifests"]["jm"]["path"])

    assert summary["pair_count"] == 1
    assert validate_manifest(mj_manifest) == []
    assert validate_manifest(jm_manifest) == []
    assert Path(record["analyses"]["raw_image_stats_json"]).exists()
    assert Path(record["analyses"]["raw_factorial_image_contrast_json"]).exists()
    assert "run_mlx_manifest_probe_batch.py" in record["suggested_commands"]["mlx_manifest_probe_batch"]
    assert (tmp_path / "batch" / "cross_palette_factorial_batch_summary.md").exists()


def _render_fractal(output_dir: Path, kind: str) -> Path:
    render_stimulus(
        FractalSpec(
            kind=kind,  # type: ignore[arg-type]
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            center_start=(-0.5, 0.0),
            center_end=(-0.4, 0.0),
            scale_start=2.5,
            scale_end=2.0,
            julia_c=(0.285, 0.01),
            max_iter=24,
            color_seed=7 if kind == "mandelbrot" else 11,
            extra={
                "stimulus_condition": {
                    "condition_id": f"{kind}_test",
                    "condition_family": "fractal",
                    "temporal_policy": "ordered",
                    "semantic_load": "low",
                    "deterministic": True,
                    "source_kind": "generated",
                    "comparison_role": "source_variant",
                    "description": f"{kind} test source",
                }
            },
        ),
        output_dir,
        overwrite=True,
    )
    return output_dir / "manifest.json"
