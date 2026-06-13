from __future__ import annotations

import json
import sys
from pathlib import Path

from fractal_vlm_state_probe.cli.run_mlx_pattern_probe_batch import (
    main,
    prepare_pattern_manifests,
)
from fractal_vlm_state_probe.stimulus import validate_manifest


def test_prepare_pattern_manifests_writes_fractal_and_geometric_controls(tmp_path: Path) -> None:
    manifests = prepare_pattern_manifests(
        output_root=tmp_path,
        conditions=["null_blank", "mandelbrot", "julia", "checkerboard", "voronoi"],
        mandelbrot_config=Path("configs/mandelbrot_smoke.json"),
        julia_config=Path("configs/julia_smoke.json"),
        frames=2,
        width=32,
        height=24,
        fps=1.0,
        stimulus_seed=7,
        cell_size=8,
        sites=5,
        overwrite=True,
    )

    assert list(manifests) == ["null_blank", "mandelbrot", "julia", "checkerboard", "voronoi"]
    for manifest_path in manifests.values():
        assert validate_manifest(manifest_path) == []

    checkerboard = _load_json(manifests["checkerboard"])
    assert checkerboard["stimulus_condition"]["condition_family"] == "geometric"
    assert checkerboard["stimulus_config"]["kind"] == "checkerboard"
    assert checkerboard["frames"][0]["width"] == 32

    null_blank = _load_json(manifests["null_blank"])
    assert null_blank["stimulus_condition"]["condition_id"] == "blank_visual_null"


def test_pattern_batch_dry_run_writes_summary_analysis_and_all_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "pattern_batch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mlx_pattern_probe_batch.py",
            "--output-root",
            str(output_root),
            "--conditions",
            "null_blank",
            "checkerboard",
            "voronoi",
            "--probe-seeds",
            "0",
            "--frames",
            "2",
            "--width",
            "32",
            "--height",
            "24",
            "--cell-size",
            "8",
            "--sites",
            "5",
            "--dry-run",
            "--overwrite",
        ],
    )

    main()

    summary = _load_json(output_root / "pattern_batch_summary.json")
    assert summary["batch_kind"] == "mlx_pattern_paired_stochastic_probe_batch"
    assert list(summary["conditions"]) == ["null_blank", "checkerboard", "voronoi"]
    comparisons = summary["records"][0]["comparisons"]
    assert sorted(comparisons) == [
        "checkerboard_vs_voronoi",
        "null_blank_vs_checkerboard",
        "null_blank_vs_voronoi",
    ]

    analysis = _load_json(output_root / "paired_stochastic_analysis.json")
    assert analysis["analysis_kind"] == "paired_stochastic_probe_batch"
    assert analysis["condition_keys"] == ["checkerboard", "null_blank", "voronoi"]
    assert analysis["probe_seed_count"] == 1


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
