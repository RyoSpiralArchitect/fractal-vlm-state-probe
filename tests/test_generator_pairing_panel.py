from __future__ import annotations

import json
from pathlib import Path

import pytest

from fractal_vlm_state_probe.generator_pairing_panel import (
    prepare_generator_pairing_panel,
)
from fractal_vlm_state_probe.stimulus import validate_manifest


def test_prepare_generator_pairing_panel_writes_hierarchy_and_factorials(
    tmp_path: Path,
) -> None:
    config = {
        "schema_version": 1,
        "panel_id": "test_panel",
        "max_frames": 1,
        "source_defaults": {
            "width": 32,
            "height": 24,
            "total_frames": 1,
            "fps": 1.0,
            "cell_size": 8,
            "sites": 5,
        },
        "pairs": [
            {
                "pair_id": "geometry_vq_r1",
                "broad_class": "geometry",
                "pairing_family": "geometry_voronoi_quasicrystal",
                "replicate": 1,
                "source_a": {"kind": "voronoi", "seed": 7},
                "source_b": {"kind": "quasicrystal", "seed": 11},
            },
            {
                "pair_id": "geometry_vq_r2",
                "broad_class": "geometry",
                "pairing_family": "geometry_voronoi_quasicrystal",
                "replicate": 2,
                "source_a": {"kind": "voronoi", "seed": 13},
                "source_b": {"kind": "quasicrystal", "seed": 17},
            },
        ],
    }

    summary = prepare_generator_pairing_panel(
        config=config,
        output_root=tmp_path / "panel",
    )

    assert summary["source_pair_count"] == 2
    assert summary["source_count"] == 4
    assert summary["unique_source_first_frame_hash_count"] == 4
    assert summary["broad_class_counts"] == {"geometry": 2}
    assert summary["pairing_family_counts"] == {"geometry_voronoi_quasicrystal": 2}
    assert summary["raw_marginal_audit"]["field_record_count"] == 8
    assert (
        summary["raw_marginal_audit"]["max_abs_spatial_or_interaction_effect"] < 1e-12
    )
    for record in summary["records"]:
        assert validate_manifest(Path(record["source_a"]["manifest_path"])) == []
        assert validate_manifest(Path(record["source_b"]["manifest_path"])) == []
        for cell in ("mm", "jj", "mj", "jm"):
            assert (
                validate_manifest(Path(record["factorial"]["manifests"][cell]["path"]))
                == []
            )

    saved = json.loads(
        (tmp_path / "panel" / "generator_pairing_panel_summary.json").read_text()
    )
    assert saved["panel_id"] == "test_panel"
    markdown = (tmp_path / "panel" / "generator_pairing_panel_summary.md").read_text()
    assert "Simultaneously exchanging source A" in markdown


def test_generator_pairing_panel_rejects_duplicate_pair_ids(tmp_path: Path) -> None:
    pair = {
        "pair_id": "duplicate",
        "broad_class": "geometry",
        "pairing_family": "geometry_vq",
        "replicate": 1,
        "source_a": {"kind": "voronoi", "seed": 7},
        "source_b": {"kind": "quasicrystal", "seed": 11},
    }
    config = {
        "schema_version": 1,
        "panel_id": "bad",
        "source_defaults": {
            "width": 32,
            "height": 24,
            "total_frames": 1,
            "fps": 1.0,
        },
        "pairs": [pair, {**pair, "replicate": 2}],
    }

    with pytest.raises(ValueError, match="duplicate pair_id"):
        prepare_generator_pairing_panel(
            config=config,
            output_root=tmp_path / "panel",
        )


def test_generator_pairing_panel_rejects_cross_class_pairing_family(
    tmp_path: Path,
) -> None:
    config = {
        "schema_version": 1,
        "panel_id": "bad",
        "source_defaults": {
            "width": 32,
            "height": 24,
            "total_frames": 1,
            "fps": 1.0,
        },
        "pairs": [
            {
                "pair_id": "a",
                "broad_class": "geometry",
                "pairing_family": "shared_family",
                "replicate": 1,
                "source_a": {"kind": "voronoi", "seed": 7},
                "source_b": {"kind": "quasicrystal", "seed": 11},
            },
            {
                "pair_id": "b",
                "broad_class": "stochastic",
                "pairing_family": "shared_family",
                "replicate": 2,
                "source_a": {"kind": "white_noise", "seed": 13},
                "source_b": {"kind": "blue_noise", "seed": 17},
            },
        ],
    }

    with pytest.raises(ValueError, match="multiple broad classes"):
        prepare_generator_pairing_panel(
            config=config,
            output_root=tmp_path / "panel",
        )
