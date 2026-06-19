from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.image_cache_correlation import (
    analyze_image_cache_correlations,
    format_image_cache_correlation_markdown,
)
from fractal_vlm_state_probe.stimulus import write_json


def test_image_cache_correlation_joins_batch_keys_to_condition_ids(tmp_path: Path) -> None:
    image_stats_path = tmp_path / "image_stats.json"
    cache_path = tmp_path / "cache.json"
    batch_path = tmp_path / "batch.json"

    write_json(
        image_stats_path,
        {
            "schema_version": 1,
            "analysis_kind": "image_statistics_batch",
            "records": [
                _image_record("mandelbrot_zoom_a", "runs/stimuli/mandelbrot/manifest.json"),
                _image_record("julia_zoom_b", "runs/stimuli/julia/manifest.json"),
                _image_record("blank_visual_null", "runs/stimuli/null_blank/manifest.json"),
            ],
            "pairwise_deltas": [
                _pair("mandelbrot_zoom_a", "julia_zoom_b", entropy=0.4, edge=0.8),
                _pair("mandelbrot_zoom_a", "blank_visual_null", entropy=0.1, edge=0.2),
                _pair("julia_zoom_b", "blank_visual_null", entropy=0.3, edge=0.6),
            ],
        },
    )
    write_json(
        batch_path,
        {
            "schema_version": 1,
            "batch_kind": "mlx_manifest_paired_stochastic_probe_batch",
            "conditions": {
                "mandelbrot": "runs/stimuli/mandelbrot/manifest.json",
                "julia": "runs/stimuli/julia/manifest.json",
                "null_blank": "runs/stimuli/null_blank/manifest.json",
            },
        },
    )
    write_json(
        cache_path,
        {
            "schema_version": 1,
            "analysis_kind": "paired_stochastic_probe_batch",
            "pairwise_cache_distances": [
                _cache("mandelbrot_vs_julia", "mid", 8.0, 3.0),
                _cache("mandelbrot_vs_null_blank", "mid", 2.0, 1.0),
                _cache("julia_vs_null_blank", "mid", 6.0, 2.0),
            ],
        },
    )

    analysis = analyze_image_cache_correlations(
        image_stats_paths=[image_stats_path],
        cache_analysis_paths=[cache_path],
        batch_summary_paths=[batch_path],
        min_samples=3,
    )

    assert analysis["joined_row_count"] == 3
    assert not analysis["unmatched_cache_rows"]
    assert analysis["condition_aliases"]["julia"] == "julia_zoom_b"
    correlations = analysis["correlations"]
    entropy = [
        item
        for item in correlations
        if item["phase"] == "mid"
        and item["cache_metric"] == "max_abs_l2_delta"
        and item["image_delta_field"] == "luminance_entropy_bits"
    ][0]
    assert entropy["sample_count"] == 3
    assert entropy["pearson"] > 0.99

    markdown = format_image_cache_correlation_markdown(analysis)
    assert "Image-Stat / Cache-Distance Correlation" in markdown
    assert "mandelbrot" in markdown


def _image_record(condition_id: str, manifest_path: str) -> dict:
    return {
        "analysis_kind": "manifest_image_statistics",
        "manifest_path": manifest_path,
        "condition": {"condition_id": condition_id},
        "aggregate": {},
    }


def _pair(left: str, right: str, *, entropy: float, edge: float) -> dict:
    return {
        "left_condition_id": left,
        "right_condition_id": right,
        "aggregate_mean_abs_deltas": {
            "luminance_mean": 0.0,
            "luminance_std": 0.0,
            "luminance_entropy_bits": entropy,
            "edge_density": edge,
            "edge_strength_mean": edge,
            "spectral_centroid": edge,
            "high_frequency_energy_ratio": edge,
            "colorfulness": 0.0,
            "mean_abs_luminance_delta_from_previous": edge,
        },
    }


def _cache(comparison_id: str, phase: str, max_l2: float, mean_l2: float) -> dict:
    return {
        "comparison_id": comparison_id,
        "phase": phase,
        "mean_max_abs_l2_delta": max_l2,
        "mean_mean_abs_l2_delta": mean_l2,
    }
