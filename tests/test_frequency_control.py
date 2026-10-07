from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fractal_vlm_state_probe.conditions import StimulusCondition
from fractal_vlm_state_probe.frequency_control import (
    CELLS,
    cell_statistics,
    frequency_gate,
    low_pass_rank_cells,
    permute_cells,
    prepare_frequency_controls,
    rgb_multiset_sha256,
    validate_frequency_panel,
)
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def _stats(centroid=0.1, hf=0.01):
    return {
        c: {
            "processor": {
                "spectral_centroid": centroid,
                "high_frequency_energy_ratio": hf,
                "luminance_std": 0.2,
            },
            "rgb_multiset_preserved": True,
            "pixel_values_shape": [3, 8, 8],
        }
        for c in CELLS
    }


def test_gate_requires_all_four_nondegenerate_cells():
    stats = _stats()
    assert frequency_gate(stats)["accepted"]
    stats["jm"]["processor"]["spectral_centroid"] = 0.14
    assert not frequency_gate(stats)["accepted"]
    with pytest.raises(ValueError, match="all four"):
        frequency_gate({c: s for c, s in stats.items() if c != "jm"})
    assert not frequency_gate(_stats(0.0))["accepted"]
    assert not frequency_gate(_stats(float("nan")))["accepted"]


@pytest.mark.parametrize("change", ["hf", "marginal", "shape", "variance"])
def test_gate_retains_nonfrequency_failure_reasons(change):
    stats = _stats()
    if change == "hf":
        stats["mj"]["processor"]["high_frequency_energy_ratio"] = 0.031
    elif change == "marginal":
        stats["jj"]["rgb_multiset_preserved"] = False
    elif change == "shape":
        stats["jj"]["pixel_values_shape"] = [3, 9, 8]
    else:
        stats["mm"]["processor"]["luminance_std"] = 0.0
    assert not frequency_gate(stats)["accepted"]


def test_rank_filter_preserves_each_joint_rgb_multiset_and_identity():
    rng = np.random.default_rng(14)
    a = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    cells = {"mm": a, "jj": b}
    cells = low_pass_rank_cells(cells, 1.0, 1.0)
    np.testing.assert_array_equal(cells["mm"], a)
    np.testing.assert_array_equal(cells["jj"], b)
    filtered = low_pass_rank_cells(cells, 0.2, 0.4)
    for c, donor in (("mm", a), ("jm", a), ("mj", b), ("jj", b)):
        assert rgb_multiset_sha256(filtered[c]) == rgb_multiset_sha256(donor)
    assert not np.array_equal(filtered["mm"], a)


def test_shared_permutation_commutes_with_interaction_and_roundtrip():
    rng = np.random.default_rng(15)
    cells = {c: rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for c in CELLS}
    permutation = rng.permutation(64)
    shuffled = permute_cells(cells, permutation)
    def contrast(x):
        return sum(
            sign * x[c].astype(float)
            for c, sign in (("jj", 1), ("jm", -1), ("mj", -1), ("mm", 1))
        )
    np.testing.assert_array_equal(
        contrast(shuffled), contrast(cells).reshape(-1, 3)[permutation].reshape(8, 8, 3)
    )
    restored = permute_cells(shuffled, np.argsort(permutation))
    for cell in CELLS:
        np.testing.assert_array_equal(restored[cell], cells[cell])
    with pytest.raises(ValueError, match="bijection"):
        permute_cells(cells, np.zeros(64, dtype=int))


def test_processor_stats_use_luminance_not_channel_average():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, ::2, 0] = 255
    image[::2, :, 1] = 255

    def processor(*, images, return_tensors):
        return {
            "pixel_values": np.moveaxis(np.asarray(images, dtype=float) / 255, -1, 0)[
                None
            ]
        }

    stats = cell_statistics(
        image,
        processor=processor,
        normalization={"image_mean": [0, 0, 0], "image_std": [1, 1, 1]},
        donor_multiset_sha256=rgb_multiset_sha256(image),
    )
    assert stats["processor"]["luminance_mean"] == pytest.approx((0.2126 + 0.7152) / 2)
    assert stats["tensor_mean"] == pytest.approx(1 / 3)
    assert stats["rgb_multiset_preserved"]


def _panel(root: Path, label: str, replicate: int):
    rgb = np.random.default_rng(16).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    cells = {
        "mm": rgb,
        "jj": np.roll(rgb, 2, axis=1),
        "mj": rgb,
        "jm": np.roll(rgb, 2, axis=1),
    }
    manifests = {}
    for cell, array in cells.items():
        directory = root / label / cell
        directory.mkdir(parents=True)
        Image.fromarray(array).save(directory / "frame.png")
        path = directory / "manifest.json"
        write_json(
            path,
            {
                "schema_version": 1,
                "stimulus_condition": {
                    "condition_id": f"{label}_{cell}",
                    "condition_family": "control",
                    "temporal_policy": "ordered",
                    "semantic_load": "none",
                    "deterministic": True,
                    "source_kind": "generated",
                },
                "frames": [
                    {
                        "index": 0,
                        "t_seconds": 0.0,
                        "path": "frame.png",
                        "width": 16,
                        "height": 16,
                        "sha256": sha256_file(directory / "frame.png"),
                    }
                ],
            },
        )
        manifests[cell] = {"path": str(path)}
    record = {
        "pair_id": label,
        "pairing_family": "f",
        "broad_class": "b",
        "replicate": replicate,
        "source_a": {},
        "source_b": {},
        "factorial": {"manifests": manifests},
    }
    path = root / f"{label}_panel.json"
    write_json(
        path,
        {"analysis_kind": "generator_pairing_factorial_panel", "records": [record]},
    )
    return path


def test_input_panel_freeze_and_cache_gate(tmp_path):
    reference = _panel(tmp_path, "r1", 1)
    test = _panel(tmp_path, "r3", 3)

    def processor(*, images, return_tensors):
        return {
            "pixel_values": np.moveaxis(np.asarray(images, dtype=float) / 255, -1, 0)[
                None
            ]
        }

    root = tmp_path / "output"
    config = {
        "schema_version": 1,
        "study_id": "test",
        "arms": ["rank_low_pass", "shared_pixel_permutation"],
        "low_pass_cutoffs": [0.2, 1.0],
        "permutation_seed": 8,
        "centroid_relative_tolerance": 0.05,
        "hf_absolute_tolerance": 0.02,
    }
    summary = prepare_frequency_controls(
        reference_panel=reference,
        test_panel=test,
        config=config,
        processor=processor,
        provenance={"qualified": True},
        normalization={"image_mean": [0, 0, 0], "image_std": [1, 1, 1]},
        output_root=root,
    )
    assert summary["cache_forwards"] == 0
    assert summary["arms"]["rank_low_pass"]["accepted_blocks"] == 2
    assert all(all(r["sham_pixels_equal"].values()) for r in summary["records"])
    panel = json.loads((root / "rank_low_pass" / "reference_panel.json").read_text())
    validate_frequency_panel(panel)
    condition_path = Path(panel["records"][0]["factorial"]["manifests"]["mm"]["path"])
    StimulusCondition.from_dict(
        json.loads(condition_path.read_text())["stimulus_condition"]
    )
    bad = copy.deepcopy(panel)
    bad["frequency_control"]["all_blocks_accepted"] = False
    with pytest.raises(ValueError, match="unmatched"):
        validate_frequency_panel(bad)
    bad = copy.deepcopy(panel)
    bad["frequency_control"]["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt hash"):
        validate_frequency_panel(bad)
    image_path = condition_path.parent / "frame.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image_path)
    with pytest.raises(ValueError, match="invalid source manifest"):
        validate_frequency_panel(panel)
    with pytest.raises(FileExistsError, match="new frequency"):
        prepare_frequency_controls(
            reference_panel=reference,
            test_panel=test,
            config=config,
            processor=processor,
            provenance={},
            normalization={},
            output_root=root,
        )
