from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.control_stimulus import (
    GeneratedControlSpec,
    render_blank_stimulus,
    render_generated_control_stimulus,
    render_manifest_transform,
)
from fractal_vlm_state_probe.image_stats import (
    analyze_manifest_batch,
    analyze_manifest_image_stats,
    format_image_stats_markdown,
)


def test_blank_image_stats_are_low_entropy_and_edge_free(tmp_path: Path) -> None:
    render_blank_stimulus(
        tmp_path / "blank",
        width=32,
        height=24,
        total_frames=2,
        fps=1.0,
        rgb=(0, 0, 0),
    )

    stats = analyze_manifest_image_stats(tmp_path / "blank" / "manifest.json")

    aggregate = stats["aggregate"]
    assert aggregate["luminance_std"]["mean"] == 0.0
    assert aggregate["luminance_entropy_bits"]["mean"] == 0.0
    assert aggregate["edge_density"]["mean"] == 0.0
    assert aggregate["high_frequency_energy_ratio"]["mean"] == 0.0


def test_checkerboard_has_more_edges_than_blank(tmp_path: Path) -> None:
    render_blank_stimulus(
        tmp_path / "blank",
        width=32,
        height=24,
        total_frames=2,
        fps=1.0,
    )
    render_generated_control_stimulus(
        GeneratedControlSpec(
            kind="checkerboard",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            seed=7,
            cell_size=8,
        ),
        tmp_path / "checkerboard",
    )

    batch = analyze_manifest_batch(
        [
            tmp_path / "blank" / "manifest.json",
            tmp_path / "checkerboard" / "manifest.json",
        ]
    )
    blank, checkerboard = batch["records"]

    assert checkerboard["aggregate"]["edge_density"]["mean"] > blank["aggregate"]["edge_density"]["mean"]
    assert checkerboard["aggregate"]["luminance_entropy_bits"]["mean"] > blank["aggregate"]["luminance_entropy_bits"]["mean"]
    markdown = format_image_stats_markdown(batch)
    assert "Image Statistics" in markdown
    assert "blank_visual_null" in markdown
    assert "checkerboard_seed_7" in markdown


def test_phase_scramble_preserves_luminance_scale_without_matching_geometry(tmp_path: Path) -> None:
    source_manifest = render_generated_control_stimulus(
        GeneratedControlSpec(
            kind="quasicrystal",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            seed=11,
            cell_size=8,
        ),
        tmp_path / "source",
    )
    render_manifest_transform(
        tmp_path / "phase_scrambled",
        source_manifest_path=tmp_path / "source" / "manifest.json",
        transform_kind="phase_scrambled",
        seed=99,
    )

    batch = analyze_manifest_batch(
        [
            tmp_path / "source" / "manifest.json",
            tmp_path / "phase_scrambled" / "manifest.json",
        ],
        include_frame_stats=True,
    )
    source, scrambled = batch["records"]

    assert scrambled["condition"]["condition_id"].endswith("_phase_scrambled")
    assert abs(source["aggregate"]["luminance_mean"]["mean"] - scrambled["aggregate"]["luminance_mean"]["mean"]) < 0.05
    assert source_manifest["frames"][0]["sha256"] != scrambled["frame_stats"][0]["sha256"]
