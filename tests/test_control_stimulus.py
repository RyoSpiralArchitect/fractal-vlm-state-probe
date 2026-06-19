from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.control_stimulus import (
    GeneratedControlSpec,
    frequency_filter_rgb,
    luminance_quantile_match_rgb,
    quantile_match_rgb,
    render_cross_palette_manifest_transform,
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


def test_quantile_match_rgb_preserves_per_channel_reference_distribution() -> None:
    source = np.array(
        [
            [[0, 8, 16], [32, 40, 48]],
            [[64, 72, 80], [96, 104, 112]],
        ],
        dtype=np.uint8,
    )
    reference = np.array(
        [
            [[255, 200, 100], [128, 30, 80]],
            [[0, 10, 60], [64, 20, 40]],
        ],
        dtype=np.uint8,
    )

    matched = quantile_match_rgb(source, reference=reference)

    assert matched.shape == source.shape
    for channel in range(3):
        assert np.array_equal(
            np.sort(matched[:, :, channel].reshape(-1)),
            np.sort(reference[:, :, channel].reshape(-1)),
        )


def test_luminance_quantile_match_rgb_preserves_reference_pixel_multiset() -> None:
    source = np.array(
        [
            [[0, 8, 16], [32, 40, 48]],
            [[64, 72, 80], [96, 104, 112]],
        ],
        dtype=np.uint8,
    )
    reference = np.array(
        [
            [[255, 200, 100], [128, 30, 80]],
            [[0, 10, 60], [64, 20, 40]],
        ],
        dtype=np.uint8,
    )

    matched = luminance_quantile_match_rgb(source, reference=reference)

    assert matched.shape == source.shape
    assert sorted(map(tuple, matched.reshape(-1, 3).tolist())) == sorted(
        map(tuple, reference.reshape(-1, 3).tolist())
    )


def test_frequency_filter_rgb_splits_low_and_high_frequency_views() -> None:
    yy, xx = np.mgrid[0:32, 0:32]
    gradient = ((xx + yy) / 62.0 * 255.0).astype(np.uint8)
    checker = (((xx // 2 + yy // 2) % 2) * 80).astype(np.uint8)
    source = np.stack(
        [
            np.clip(gradient + checker, 0, 255),
            np.clip(gradient // 2 + checker, 0, 255),
            np.clip(255 - gradient + checker, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)

    low = frequency_filter_rgb(source, mode="low_pass", cutoff=0.2)
    high = frequency_filter_rgb(source, mode="high_pass", cutoff=0.2)

    assert low.shape == source.shape
    assert high.shape == source.shape
    assert not np.array_equal(low, source)
    assert not np.array_equal(high, source)
    assert not np.array_equal(low, high)


def test_phase_scrambled_quantile_matched_transform_matches_rgb_histograms(tmp_path: Path) -> None:
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

    matched_manifest = render_manifest_transform(
        tmp_path / "phase_scrambled_quantile_matched",
        source_manifest_path=source_dir / "manifest.json",
        transform_kind="phase_scrambled_quantile_matched",
        seed=99,
    )

    source_path = source_dir / source_manifest["frames"][0]["path"]
    matched_path = tmp_path / "phase_scrambled_quantile_matched" / matched_manifest["frames"][0]["path"]
    source_image = np.asarray(Image.open(source_path).convert("RGB"))
    matched_image = np.asarray(Image.open(matched_path).convert("RGB"))

    assert matched_manifest["stimulus_condition"]["condition_family"] == "control"
    assert matched_manifest["stimulus_condition"]["condition_id"].endswith("_phase_scrambled_quantile_matched")
    assert matched_manifest["stimulus_condition"]["comparison_role"] == "phase_scrambled_quantile_matched_visual_control"
    assert matched_manifest["frames"][0]["sha256"] != source_manifest["frames"][0]["sha256"]
    for channel in range(3):
        assert np.array_equal(
            np.sort(source_image[:, :, channel].reshape(-1)),
            np.sort(matched_image[:, :, channel].reshape(-1)),
        )
    assert validate_manifest(tmp_path / "phase_scrambled_quantile_matched" / "manifest.json") == []


def test_phase_scrambled_luminance_quantile_matched_transform_preserves_source_pixels(tmp_path: Path) -> None:
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

    matched_manifest = render_manifest_transform(
        tmp_path / "phase_scrambled_luminance_quantile_matched",
        source_manifest_path=source_dir / "manifest.json",
        transform_kind="phase_scrambled_luminance_quantile_matched",
        seed=99,
    )

    source_path = source_dir / source_manifest["frames"][0]["path"]
    matched_path = tmp_path / "phase_scrambled_luminance_quantile_matched" / matched_manifest["frames"][0]["path"]
    source_image = np.asarray(Image.open(source_path).convert("RGB"))
    matched_image = np.asarray(Image.open(matched_path).convert("RGB"))

    assert matched_manifest["stimulus_condition"]["condition_id"].endswith(
        "_phase_scrambled_luminance_quantile_matched"
    )
    assert matched_manifest["stimulus_condition"]["comparison_role"] == (
        "phase_scrambled_luminance_quantile_matched_visual_control"
    )
    assert matched_manifest["frames"][0]["sha256"] != source_manifest["frames"][0]["sha256"]
    assert sorted(map(tuple, matched_image.reshape(-1, 3).tolist())) == sorted(
        map(tuple, source_image.reshape(-1, 3).tolist())
    )
    assert validate_manifest(tmp_path / "phase_scrambled_luminance_quantile_matched" / "manifest.json") == []


def test_frequency_transform_writes_control_condition(tmp_path: Path) -> None:
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

    low_manifest = render_manifest_transform(
        tmp_path / "low_pass",
        source_manifest_path=source_dir / "manifest.json",
        transform_kind="low_pass",
        frequency_cutoff=0.2,
        seed=99,
    )
    high_manifest = render_manifest_transform(
        tmp_path / "high_pass_luminance_quantile_matched",
        source_manifest_path=source_dir / "manifest.json",
        transform_kind="high_pass_luminance_quantile_matched",
        frequency_cutoff=0.2,
        seed=99,
    )

    source_path = source_dir / source_manifest["frames"][0]["path"]
    high_path = tmp_path / "high_pass_luminance_quantile_matched" / high_manifest["frames"][0]["path"]
    source_image = np.asarray(Image.open(source_path).convert("RGB"))
    high_image = np.asarray(Image.open(high_path).convert("RGB"))

    assert low_manifest["stimulus_condition"]["condition_id"].endswith("_low_pass")
    assert low_manifest["stimulus_condition"]["comparison_role"] == "low_pass_frequency_control"
    assert low_manifest["stimulus_config"]["frequency_cutoff"] == 0.2
    assert low_manifest["frames"][0]["sha256"] != source_manifest["frames"][0]["sha256"]
    assert high_manifest["stimulus_condition"]["condition_id"].endswith("_high_pass_luminance_quantile_matched")
    assert high_manifest["stimulus_condition"]["comparison_role"] == (
        "high_pass_luminance_quantile_matched_frequency_control"
    )
    assert sorted(map(tuple, high_image.reshape(-1, 3).tolist())) == sorted(
        map(tuple, source_image.reshape(-1, 3).tolist())
    )
    assert validate_manifest(tmp_path / "low_pass" / "manifest.json") == []
    assert validate_manifest(tmp_path / "high_pass_luminance_quantile_matched" / "manifest.json") == []


def test_cross_palette_transform_preserves_palette_pixels_with_source_geometry(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    palette_dir = tmp_path / "palette"
    source_manifest = render_generated_control_stimulus(
        GeneratedControlSpec(
            kind="checkerboard",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            seed=3,
            cell_size=8,
        ),
        source_dir,
    )
    palette_manifest = render_generated_control_stimulus(
        GeneratedControlSpec(
            kind="quasicrystal",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            seed=17,
            cell_size=8,
        ),
        palette_dir,
    )

    matched_manifest = render_cross_palette_manifest_transform(
        tmp_path / "matched",
        source_manifest_path=source_dir / "manifest.json",
        palette_manifest_path=palette_dir / "manifest.json",
    )

    matched_path = tmp_path / "matched" / matched_manifest["frames"][0]["path"]
    palette_path = palette_dir / palette_manifest["frames"][0]["path"]
    matched_image = np.asarray(Image.open(matched_path).convert("RGB"))
    palette_image = np.asarray(Image.open(palette_path).convert("RGB"))

    assert matched_manifest["stimulus_condition"]["condition_family"] == "control"
    assert matched_manifest["stimulus_condition"]["comparison_role"] == "cross_family_palette_matched_control"
    assert matched_manifest["frames"][0]["source_index"] == 0
    assert matched_manifest["frames"][0]["palette_index"] == 0
    assert matched_manifest["frames"][0]["sha256"] != source_manifest["frames"][0]["sha256"]
    assert matched_manifest["frames"][0]["source_sha256"] == source_manifest["frames"][0]["sha256"]
    assert matched_manifest["frames"][0]["palette_sha256"] == palette_manifest["frames"][0]["sha256"]
    assert sorted(map(tuple, matched_image.reshape(-1, 3).tolist())) == sorted(
        map(tuple, palette_image.reshape(-1, 3).tolist())
    )
    assert validate_manifest(tmp_path / "matched" / "manifest.json") == []
