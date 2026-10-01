from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_tensor_artifact import (
    load_cache_tensor_artifact,
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cache_tensor_factorial import (
    analyze_cache_tensor_factorial,
    cache_tensor_regions,
    write_cache_tensor_factorial_markdown,
)
from fractal_vlm_state_probe.cache_tensor_replication import (
    analyze_cache_tensor_replication,
    write_cache_tensor_replication_markdown,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _read,
    _response_counts,
    _sha,
    _validate_panel_split,
    _validate_run,
)
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.mlx_stream import _load_mlx_runtime
from fractal_vlm_state_probe.stimulus import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a frozen pairing reference panel after historical calibration."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-panel", required=True, type=Path)
    parser.add_argument("--test-panel", required=True, type=Path)
    parser.add_argument(
        "--historical-factorial",
        required=True,
        action="append",
        metavar="LAYER:TENSOR=PATH",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    panel, test = _read(args.reference_panel), _read(args.test_panel)
    metadata, hashes = _validate_panel_split(panel, test)
    historical, captures = {}, []
    for spec in args.historical_factorial:
        target, raw_path = spec.split("=", 1)
        capture = parse_cache_tensor_capture_spec(target)
        if capture.identifier in historical:
            raise ValueError(f"duplicate target: {target}")
        analysis = _read(Path(raw_path))
        if (analysis["model_id"], analysis["layer_index"], analysis["tensor"]) != (
            args.model,
            capture.layer_index,
            capture.tensor,
        ):
            raise ValueError(f"historical target differs: {target}")
        historical[capture.identifier] = analysis
        captures.append(capture)
    first = historical[captures[0].identifier]
    source_paths = {c: Path(r["source_path"]) for c, r in first["cells"].items()}
    reference_run = _read(source_paths["mm"])
    for analysis in historical.values():
        if {
            c: Path(r["source_path"]) for c, r in analysis["cells"].items()
        } != source_paths:
            raise ValueError("historical targets must use the same four source cells")
        for cell, record in analysis["cells"].items():
            original = _read(source_paths[cell])
            _validate_run(
                original,
                source_paths[cell],
                Path(original["manifest_path"]),
                reference_run,
            )
            load_cache_tensor_artifact(
                source_paths[cell], record["cache_tensor_artifact"]
            )
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": 1,
        "analysis_kind": "pairing_reference_capture_frozen_specification",
        "model_id": args.model,
        "reference_panel_sha256": _sha(args.reference_panel),
        "test_panel_sha256": _sha(args.test_panel),
        "hierarchy_metadata": metadata,
        "source_hash_count": len(hashes),
        "unique_source_hash_count": len(set(hashes)),
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "stream_seed": 20260604,
        "historical_artifact_hashes": {
            target: {
                c: r["cache_tensor_artifact"]["sha256"] for c, r in a["cells"].items()
            }
            for target, a in historical.items()
        },
    }
    frozen = root / "frozen_specification.json"
    if frozen.exists() and _read(frozen) != signature:
        raise ValueError(
            "frozen reference specification differs; use a new output root"
        )
    write_json(frozen, signature)
    runtime = None
    snapshot_path = root / "model_snapshot.json"

    def capture_cell(manifest: Path, path: Path) -> dict:
        nonlocal runtime
        if not path.exists():
            if args.analysis_only:
                raise FileNotFoundError(f"missing source run: {path}")
            if runtime is None:
                runtime = _load_mlx_runtime(args.model)
                _freeze_model_snapshot(args.model, snapshot_path)
                processor_type = type(runtime["processor"].image_processor)
                write_json(
                    root / "processor_implementation.json",
                    {
                        "implementation": f"{processor_type.__module__}.{processor_type.__name__}",
                        "implementation_sha256": _sha(
                            Path(inspect.getfile(processor_type))
                        ),
                    },
                )
            run_cumulative_replay_probe(
                CumulativeReplayRunConfig(
                    manifest_path=manifest,
                    output_path=path,
                    model_id=args.model,
                    max_frames=1,
                    max_tokens=2,
                    temperature=0,
                    probe_temperature=0,
                    probe_preset="default",
                    cache_summary_max_layers=None,
                    source_cache_only=True,
                    cache_tensor_captures=tuple(captures),
                    include_frame_artifacts=False,
                ),
                mlx_runtime=runtime,
            )
        run = _read(path)
        _validate_run(run, path, manifest, reference_run)
        for capture in captures:
            _validate_target_contract(run, path, historical[capture.identifier])
        return run

    checks = []
    for cell in ("mm", "jj", "mj", "jm"):
        original = _read(source_paths[cell])
        path = root / "historical_recheck" / f"{cell}_mlx.json"
        print(f"historical calibration {cell}", flush=True)
        run = capture_cell(Path(original["manifest_path"]), path)
        for capture in captures:
            actual = _validate_target_contract(
                run, path, historical[capture.identifier]
            )
            expected = load_cache_tensor_artifact(
                source_paths[cell],
                historical[capture.identifier]["cells"][cell]["cache_tensor_artifact"],
            )
            checks.append(
                {
                    "cell": cell,
                    "target": capture.identifier,
                    "bitwise_equal": actual.shape == expected.shape
                    and actual.tobytes() == expected.tobytes(),
                    "max_abs_difference": float(np.max(np.abs(actual - expected))),
                }
            )
    write_json(root / "historical_recheck.json", {"checks": checks})
    if not all(c["bitwise_equal"] for c in checks):
        raise ValueError(
            "historical calibration differs; stop before pairing references"
        )

    run_paths = {}
    for index, record in enumerate(panel["records"]):
        label = record["pair_id"]
        run_paths[label] = {}
        for cell in ("mm", "jj", "mj", "jm"):
            print(
                f"reference pair {index + 1}/{len(panel['records'])} {label}/{cell}",
                flush=True,
            )
            path = root / "source_runs" / f"{label}_{cell}_mlx.json"
            capture_cell(Path(record["factorial"]["manifests"][cell]["path"]), path)
            run_paths[label][cell] = path
    _freeze_model_snapshot(args.model, snapshot_path)
    replications = {}
    for capture in captures:
        analyses, paths = {}, {}
        for label, cells in run_paths.items():
            analysis = analyze_cache_tensor_factorial(
                runs={c: _read(p) for c, p in cells.items()},
                run_paths=cells,
                layer_index=capture.layer_index,
                tensor=capture.tensor,
            )
            path = (
                root
                / "factorials"
                / label
                / capture.identifier
                / "cache_tensor_factorial.json"
            )
            write_json(path, analysis)
            write_cache_tensor_factorial_markdown(analysis, path.with_suffix(".md"))
            analyses[label], paths[label] = analysis, path
        replication = analyze_cache_tensor_replication(analyses, analysis_paths=paths)
        path = root / "replication" / f"{capture.identifier}.json"
        write_json(path, replication)
        write_cache_tensor_replication_markdown(replication, path.with_suffix(".md"))
        replications[f"{capture.layer_index}:{capture.tensor}"] = str(path)
    write_json(
        root / "reference_capture_summary.json",
        {
            **signature,
            "analysis_kind": "pairing_reference_capture_execution",
            "new_reference_source_cells": len(run_paths) * 4,
            "new_reference_tensor_sidecars": len(run_paths) * 4 * len(captures),
            "new_reference_factorial_analyses": len(run_paths) * len(captures),
            "historical_recheck_cells": 4,
            "historical_recheck": checks,
            "reference_replications": replications,
            "model_snapshot": _read(snapshot_path),
            "processor_implementation": _read(root / "processor_implementation.json"),
            "source_response_counts": _response_counts(run_paths),
            "source_suffix_token_ids": [
                s["token_id"]
                for s in reference_run["stream_events"][0]["generation"]["steps"]
            ],
            "runtime": reference_run["runtime"],
        },
    )
    print(f"wrote frozen pairing references to {root}", flush=True)


def _validate_target_contract(run: dict, path: Path, historical: dict) -> np.ndarray:
    event = run["stream_events"][0]
    artifacts = [
        a
        for a in event["cache_tensor_artifacts"]
        if (a["layer_index"], a["tensor"])
        == (historical["layer_index"], historical["tensor"])
    ]
    if len(artifacts) != 1:
        raise ValueError(f"missing or duplicate target tensor: {path}")
    array = load_cache_tensor_artifact(path, artifacts[0])
    if list(array.shape) != historical["tensor_shape"]:
        raise ValueError(f"target tensor shape differs: {path}")
    actual = cache_tensor_regions(
        event["cache_token_layout"], sequence_length=array.shape[-2]
    )
    expected = cache_tensor_regions(
        historical["cells"]["mm"]["cache_token_layout"], sequence_length=array.shape[-2]
    )
    if not actual["image_tokens"] or actual != expected:
        raise ValueError(f"target token layout differs or is unresolved: {path}")
    return array


if __name__ == "__main__":
    main()
