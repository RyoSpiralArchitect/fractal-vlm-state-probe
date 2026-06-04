from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.control_stimulus import (
    GeneratedControlSpec,
    render_blank_stimulus,
    render_generated_control_stimulus,
    render_manifest_transform,
)
from fractal_vlm_state_probe.stimulus import validate_manifest


def test_render_blank_stimulus_writes_control_manifest(tmp_path: Path) -> None:
    manifest = render_blank_stimulus(
        tmp_path,
        width=32,
        height=24,
        total_frames=3,
        fps=1.0,
        rgb=(1, 2, 3),
    )
    assert len(manifest["frames"]) == 3
    assert manifest["stimulus_condition"]["condition_family"] == "control"
    assert manifest["stimulus_condition"]["condition_id"] == "blank_visual_null"
    assert manifest["stimulus_config"]["rgb"] == [1, 2, 3]
    assert validate_manifest(tmp_path / "manifest.json") == []


def test_render_generated_pattern_controls_write_valid_manifests(tmp_path: Path) -> None:
    for kind in ("white_noise", "blue_noise", "checkerboard", "voronoi", "quasicrystal"):
        output_dir = tmp_path / kind
        manifest = render_generated_control_stimulus(
            GeneratedControlSpec(
                kind=kind,
                width=32,
                height=24,
                total_frames=2,
                fps=1.0,
                seed=7,
                cell_size=8,
                sites=5,
            ),
            output_dir,
        )

        assert len(manifest["frames"]) == 2
        assert manifest["stimulus_condition"]["deterministic"] is True
        assert validate_manifest(output_dir / "manifest.json") == []


def test_manifest_transforms_preserve_source_order_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    render_generated_control_stimulus(
        GeneratedControlSpec(
            kind="checkerboard",
            width=32,
            height=24,
            total_frames=4,
            fps=1.0,
            seed=11,
            cell_size=8,
        ),
        source_dir,
    )

    reversed_manifest = render_manifest_transform(
        tmp_path / "reversed",
        source_manifest_path=source_dir / "manifest.json",
        transform_kind="reversed",
    )
    assert [frame["source_index"] for frame in reversed_manifest["frames"]] == [3, 2, 1, 0]
    assert reversed_manifest["stimulus_condition"]["temporal_policy"] == "reversed"
    assert validate_manifest(tmp_path / "reversed" / "manifest.json") == []

    repeat_manifest = render_manifest_transform(
        tmp_path / "repeat",
        source_manifest_path=source_dir / "manifest.json",
        transform_kind="static_repeat",
        source_frame_index=1,
    )
    assert [frame["source_index"] for frame in repeat_manifest["frames"]] == [1, 1, 1, 1]
    assert repeat_manifest["stimulus_condition"]["temporal_policy"] == "static_repeat"
    assert validate_manifest(tmp_path / "repeat" / "manifest.json") == []


def test_phase_scrambled_transform_writes_control_condition(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_manifest = render_generated_control_stimulus(
        GeneratedControlSpec(
            kind="quasicrystal",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            seed=13,
            cell_size=8,
        ),
        source_dir,
    )

    scrambled = render_manifest_transform(
        tmp_path / "phase_scrambled",
        source_manifest_path=source_dir / "manifest.json",
        transform_kind="phase_scrambled",
        seed=99,
    )

    assert scrambled["stimulus_condition"]["condition_family"] == "control"
    assert scrambled["stimulus_condition"]["condition_id"].endswith("_phase_scrambled")
    assert scrambled["frames"][0]["sha256"] != source_manifest["frames"][0]["sha256"]
    assert validate_manifest(tmp_path / "phase_scrambled" / "manifest.json") == []
