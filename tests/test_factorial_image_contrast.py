from __future__ import annotations

from fractal_vlm_state_probe.factorial_image_contrast import (
    analyze_factorial_image_contrast,
    format_factorial_image_contrast_markdown,
)


def test_factorial_image_contrast_reports_metric_interactions() -> None:
    analysis = analyze_factorial_image_contrast(
        stats={
            "analysis_kind": "image_statistics_batch",
            "records": [
                _record("mm", "luminance_mean", 1.0),
                _record("jj", "luminance_mean", 7.0),
                _record("mj", "luminance_mean", 4.0),
                _record("jm", "luminance_mean", 2.0),
            ],
        },
        mm_condition_id="mm",
        jj_condition_id="jj",
        mj_condition_id="mj",
        jm_condition_id="jm",
    )

    row = analysis["records"][0]

    assert row["field"] == "luminance_mean"
    assert row["spatial_main_effect"] == 2.0
    assert row["palette_main_effect"] == 4.0
    assert row["interaction_effect"] == 2.0
    assert analysis["top_interaction_effects"][0]["field"] == "luminance_mean"
    markdown = format_factorial_image_contrast_markdown(analysis)
    assert "2x2 Factorial Image-Statistic Contrast" in markdown
    assert "luminance_mean" in markdown
    assert "source A spatial rank x source A palette" in markdown
    assert "Mandelbrot spatial rank" not in markdown


def _record(condition_id: str, field: str, mean: float) -> dict:
    return {
        "condition": {"condition_id": condition_id, "condition_family": "control"},
        "frame_count_analyzed": 2,
        "manifest_path": f"runs/{condition_id}/manifest.json",
        "aggregate": {
            field: {
                "mean": mean,
                "std": 0.0,
                "min": mean,
                "max": mean,
            }
        },
    }
