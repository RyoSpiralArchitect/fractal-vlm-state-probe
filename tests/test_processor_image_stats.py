from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.control_stimulus import (
    GeneratedControlSpec,
    render_blank_stimulus,
    render_generated_control_stimulus,
)
from fractal_vlm_state_probe.processor_image_stats import (
    analyze_manifest_processor_image_stats,
    analyze_processor_manifest_batch,
    extract_processor_pixel_tensor,
    format_processor_image_stats_markdown,
)


class FakeProcessor:
    def __call__(self, *, images: Image.Image, return_tensors: str) -> dict[str, np.ndarray]:
        assert return_tensors == "np"
        array = np.asarray(images.resize((16, 16)).convert("RGB"), dtype=np.float32) / 255.0
        normalized = (array - 0.5) / 0.25
        return {"pixel_values": np.moveaxis(normalized, -1, 0)[None, :, :, :]}


def test_extract_processor_pixel_tensor_normalizes_batch_channel_first() -> None:
    image = Image.new("RGB", (8, 8), (128, 64, 32))
    tensor = extract_processor_pixel_tensor(FakeProcessor(), image)

    assert tensor.shape == (3, 16, 16)
    assert float(tensor.std()) > 0.0


def test_processor_image_stats_show_checkerboard_energy_after_resize(tmp_path: Path) -> None:
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

    batch = analyze_processor_manifest_batch(
        [
            tmp_path / "blank" / "manifest.json",
            tmp_path / "checkerboard" / "manifest.json",
        ],
        processor=FakeProcessor(),
        patch_size=4,
        include_frame_stats=True,
    )
    blank, checkerboard = batch["records"]

    assert blank["aggregate"]["high_frequency_energy_ratio"]["mean"] == 0.0
    assert checkerboard["aggregate"]["high_frequency_energy_ratio"]["mean"] > 0.0
    assert checkerboard["aggregate"]["spectral_centroid_cycles_per_patch"]["mean"] > 0.0
    assert checkerboard["frame_stats"][0]["pixel_values_shape"] == [3, 16, 16]

    markdown = format_processor_image_stats_markdown(batch)
    assert "Processor Image Statistics" in markdown
    assert "blank_visual_null" in markdown
    assert "checkerboard_seed_7" in markdown


def test_single_processor_image_stats_can_omit_frame_records(tmp_path: Path) -> None:
    render_generated_control_stimulus(
        GeneratedControlSpec(
            kind="quasicrystal",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            seed=9,
            cell_size=8,
        ),
        tmp_path / "quasicrystal",
    )

    stats = analyze_manifest_processor_image_stats(
        tmp_path / "quasicrystal" / "manifest.json",
        processor=FakeProcessor(),
        patch_size=4,
        include_frame_stats=False,
    )

    assert "frame_stats" not in stats
    assert stats["processor_patch_size"] == 4
    assert stats["aggregate"]["energy_ratio_above_half_cycle_per_patch"]["mean"] is not None
