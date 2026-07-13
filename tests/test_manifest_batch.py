from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fractal_vlm_state_probe.cli.run_mlx_manifest_probe_batch import (
    main,
    parse_manifest_specs,
)
from fractal_vlm_state_probe.control_stimulus import (
    GeneratedControlSpec,
    render_blank_stimulus,
    render_generated_control_stimulus,
)


def test_parse_manifest_specs_requires_key_value_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    parsed = parse_manifest_specs([f"blank={manifest}", f"dot.control={manifest}"])

    assert list(parsed) == ["blank", "dot.control"]
    with pytest.raises(ValueError, match="KEY=PATH"):
        parse_manifest_specs([str(manifest)])
    with pytest.raises(ValueError, match="duplicate"):
        parse_manifest_specs([f"blank={manifest}", f"blank={manifest}"])
    with pytest.raises(ValueError, match="at least two"):
        parse_manifest_specs([f"blank={manifest}"])


def test_manifest_batch_dry_run_writes_summary_analysis_and_all_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    blank_manifest = _render_blank(tmp_path / "blank")
    checker_manifest = _render_control(tmp_path / "checkerboard", "checkerboard")
    voronoi_manifest = _render_control(tmp_path / "voronoi", "voronoi")
    output_root = tmp_path / "manifest_batch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mlx_manifest_probe_batch.py",
            "--output-root",
            str(output_root),
            "--manifest",
            f"blank={blank_manifest}",
            "--manifest",
            f"checkerboard={checker_manifest}",
            "--manifest",
            f"voronoi={voronoi_manifest}",
            "--probe-seeds",
            "0",
            "--max-frames",
            "2",
            "--dry-run",
            "--overwrite",
        ],
    )

    main()

    summary = _load_json(output_root / "manifest_batch_summary.json")
    assert summary["batch_kind"] == "mlx_manifest_paired_stochastic_probe_batch"
    assert list(summary["conditions"]) == ["blank", "checkerboard", "voronoi"]
    comparisons = summary["records"][0]["comparisons"]
    assert sorted(comparisons) == [
        "blank_vs_checkerboard",
        "blank_vs_voronoi",
        "checkerboard_vs_voronoi",
    ]

    analysis = _load_json(output_root / "paired_stochastic_analysis.json")
    assert analysis["analysis_kind"] == "paired_stochastic_probe_batch"
    assert analysis["condition_keys"] == ["blank", "checkerboard", "voronoi"]
    assert analysis["probe_seed_count"] == 1


def test_manifest_batch_supports_cumulative_replay_dry_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    blank_manifest = _render_blank(tmp_path / "blank")
    checker_manifest = _render_control(tmp_path / "checkerboard", "checkerboard")
    output_root = tmp_path / "cumulative_batch"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mlx_manifest_probe_batch.py",
            "--output-root",
            str(output_root),
            "--manifest",
            f"blank={blank_manifest}",
            "--manifest",
            f"checkerboard={checker_manifest}",
            "--probe-seeds",
            "0",
            "--max-frames",
            "2",
            "--context-protocol",
            "cumulative_replay",
            "--dry-run",
            "--overwrite",
        ],
    )

    main()

    summary = _load_json(output_root / "manifest_batch_summary.json")
    assert summary["context_protocol"] == "cumulative_replay"
    run = _load_json(output_root / "probe_seed_0" / "blank_mlx.json")
    assert run["run_kind"] == "mlx_vlm_cumulative_visual_replay_probe"
    assert run["stream_events"][0]["image_count"] == 2


def _render_blank(output_dir: Path) -> Path:
    render_blank_stimulus(
        output_dir,
        width=32,
        height=24,
        total_frames=2,
        fps=1.0,
        overwrite=True,
    )
    return output_dir / "manifest.json"


def _render_control(output_dir: Path, kind: str) -> Path:
    render_generated_control_stimulus(
        GeneratedControlSpec(
            kind=kind,  # type: ignore[arg-type]
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            seed=7,
            cell_size=8,
            sites=5,
        ),
        output_dir,
        overwrite=True,
    )
    return output_dir / "manifest.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
