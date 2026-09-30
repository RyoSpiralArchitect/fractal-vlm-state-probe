from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_direction_holdout import (
    analyze_pairing_seed_holdout,
    format_pairing_seed_holdout,
    validate_pairing_holdout_metadata,
)
from fractal_vlm_state_probe.cache_tensor_artifact import (
    load_cache_tensor_artifact,
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cache_tensor_factorial import (
    analyze_cache_tensor_factorial,
    write_cache_tensor_factorial_markdown,
)
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.mlx_stream import _load_mlx_runtime
from fractal_vlm_state_probe.stimulus import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate fixed pairing directions on new visual seeds."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-panel", required=True, type=Path)
    parser.add_argument("--test-panel", required=True, type=Path)
    parser.add_argument(
        "--reference-replication",
        required=True,
        action="append",
        metavar="LAYER:TENSOR=PATH",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    reference_panel = _read(args.reference_panel)
    test_panel = _read(args.test_panel)
    metadata = {}
    for panel in (reference_panel, test_panel):
        if panel.get("analysis_kind") != "generator_pairing_factorial_panel":
            raise ValueError("expected generated panel summaries")
        for record in panel["records"]:
            label = record["pair_id"]
            if label in metadata:
                raise ValueError(f"duplicate reference/test label: {label}")
            metadata[label] = {
                k: record[k] for k in ("pairing_family", "broad_class", "replicate")
            }
    hashes = [
        r[role]["first_frame_sha256"]
        for p in (reference_panel, test_panel)
        for r in p["records"]
        for role in ("source_a", "source_b")
    ]
    if len(hashes) != len(set(hashes)):
        raise ValueError(
            "reference/test source images contain repeated first-frame hashes"
        )
    validate_pairing_holdout_metadata(metadata)
    families = {r["pairing_family"]: r["broad_class"] for r in metadata.values()}
    if len(families) != 8 or any(
        list(families.values()).count(broad) != 4 for broad in set(families.values())
    ):
        raise ValueError("registered validation requires four families per broad class")
    captures = []
    references = {}
    for spec in args.reference_replication:
        target, raw_path = spec.split("=", 1)
        capture = parse_cache_tensor_capture_spec(target)
        if capture.identifier in references:
            raise ValueError(f"duplicate target: {target}")
        captures.append(capture)
        path = Path(raw_path)
        replication = _read(path)
        if len(replication["groups"]) != 1:
            raise ValueError("reference replication must contain one target")
        group = replication["groups"][0]
        if (group["model_id"], group["layer_index"], group["tensor"]) != (
            args.model,
            capture.layer_index,
            capture.tensor,
        ):
            raise ValueError(f"reference target differs: {target}")
        paths = {p["label"]: Path(p["analysis_path"]) for p in group["points"]}
        if set(paths) != {r["pair_id"] for r in reference_panel["records"]}:
            raise ValueError("reference panel and replication labels differ")
        references[capture.identifier] = {label: _read(p) for label, p in paths.items()}
        references[capture.identifier + "_paths"] = paths

    args.output_root.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": 1,
        "analysis_kind": "pairing_seed_validation_frozen_specification",
        "model_id": args.model,
        "reference_panel_sha256": _sha(args.reference_panel),
        "test_panel_sha256": _sha(args.test_panel),
        "source_hash_count": len(hashes),
        "unique_source_hash_count": len(set(hashes)),
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "stream_seed": 20260604,
        "reference_artifact_hashes": {
            c.identifier: {
                label: {
                    cell: a["cells"][cell]["cache_tensor_artifact"]["sha256"]
                    for cell in ("mm", "jj", "mj", "jm")
                }
                for label, a in references[c.identifier].items()
            }
            for c in captures
        },
    }
    frozen = args.output_root / "frozen_specification.json"
    if frozen.exists() and _read(frozen) != signature:
        raise ValueError("existing frozen specification differs; use a new output root")
    write_json(frozen, signature)
    reference_first = next(iter(references[captures[0].identifier].values()))
    reference_run = _read(Path(reference_first["cells"]["mm"]["source_path"]))
    checked_reference_paths = set()
    for capture in captures:
        for analysis in references[capture.identifier].values():
            for record in analysis["cells"].values():
                source_path = Path(record["source_path"])
                load_cache_tensor_artifact(source_path, record["cache_tensor_artifact"])
                if source_path not in checked_reference_paths:
                    source_run = _read(source_path)
                    _validate_run(
                        source_run,
                        source_path,
                        Path(source_run["manifest_path"]),
                        reference_run,
                    )
                    checked_reference_paths.add(source_path)
    runtime = None
    model_fingerprint_path = args.output_root / "model_snapshot.json"
    anchor_label = sorted(references[captures[0].identifier])[0]
    anchor_checks = []
    for cell in ("mm", "jj", "mj", "jm"):
        reference_cell = references[captures[0].identifier][anchor_label]["cells"][cell]
        original = _read(Path(reference_cell["source_path"]))
        manifest_path = Path(original["manifest_path"])
        path = (
            args.output_root / "reference_recheck" / f"{anchor_label}_{cell}_mlx.json"
        )
        if not path.exists():
            if args.analysis_only:
                raise FileNotFoundError(f"missing reference recalibration: {path}")
            if runtime is None:
                runtime = _load_mlx_runtime(args.model)
                _freeze_model_snapshot(args.model, model_fingerprint_path)
            print(f"reference recalibration {anchor_label}/{cell}", flush=True)
            run_cumulative_replay_probe(
                CumulativeReplayRunConfig(
                    manifest_path=manifest_path,
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
        _validate_run(run, path, manifest_path, reference_run)
        artifacts = {
            (a["layer_index"], a["tensor"]): a
            for a in run["stream_events"][0]["cache_tensor_artifacts"]
        }
        for capture in captures:
            record = references[capture.identifier][anchor_label]["cells"][cell]
            expected = load_cache_tensor_artifact(
                Path(record["source_path"]), record["cache_tensor_artifact"]
            )
            actual = load_cache_tensor_artifact(
                path, artifacts[(capture.layer_index, capture.tensor)]
            )
            equal = (
                actual.shape == expected.shape
                and actual.tobytes() == expected.tobytes()
            )
            anchor_checks.append(
                {
                    "cell": cell,
                    "target": capture.identifier,
                    "bitwise_equal": equal,
                    "max_abs_difference": float(np.max(np.abs(actual - expected)))
                    if actual.shape == expected.shape
                    else None,
                }
            )
    write_json(
        args.output_root / "reference_recheck.json",
        {
            "schema_version": 1,
            "analysis_kind": "source_cache_reference_recalibration",
            "anchor_label": anchor_label,
            "checks": anchor_checks,
        },
    )
    if not all(c["bitwise_equal"] for c in anchor_checks):
        raise ValueError(
            "historical reference recalibration differs; audit before pooling new seeds"
        )
    run_paths = {}
    for pair_index, record in enumerate(test_panel["records"]):
        label = record["pair_id"]
        run_paths[label] = {}
        for cell in ("mm", "jj", "mj", "jm"):
            path = args.output_root / "source_runs" / f"{label}_{cell}_mlx.json"
            manifest_path = Path(record["factorial"]["manifests"][cell]["path"])
            if not path.exists():
                if args.analysis_only:
                    raise FileNotFoundError(f"missing source run: {path}")
                if runtime is None:
                    runtime = _load_mlx_runtime(args.model)
                    _freeze_model_snapshot(args.model, model_fingerprint_path)
                print(
                    f"pair {pair_index + 1}/{len(test_panel['records'])} {label}/{cell}",
                    flush=True,
                )
                run_cumulative_replay_probe(
                    CumulativeReplayRunConfig(
                        manifest_path=manifest_path,
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
            _validate_run(_read(path), path, manifest_path, reference_run)
            run_paths[label][cell] = path

    results = []
    for capture in captures:
        paths = dict(references[capture.identifier + "_paths"])
        analyses = dict(references[capture.identifier])
        for label, cell_paths in run_paths.items():
            analysis = analyze_cache_tensor_factorial(
                runs={c: _read(p) for c, p in cell_paths.items()},
                run_paths=cell_paths,
                layer_index=capture.layer_index,
                tensor=capture.tensor,
            )
            path = (
                args.output_root
                / "factorials"
                / label
                / capture.identifier
                / "cache_tensor_factorial.json"
            )
            write_json(path, analysis)
            write_cache_tensor_factorial_markdown(analysis, path.with_suffix(".md"))
            paths[label], analyses[label] = path, analysis
        result = analyze_pairing_seed_holdout(
            analyses, analysis_paths=paths, metadata=metadata
        )
        path = args.output_root / "holdout" / f"{capture.identifier}.json"
        write_json(path, result)
        path.with_suffix(".md").write_text(
            format_pairing_seed_holdout(result), encoding="utf-8"
        )
        results.append(str(path))
    write_json(
        args.output_root / "validation_summary.json",
        {
            **signature,
            "analysis_kind": "pairing_seed_validation_execution",
            "new_source_cells": len(run_paths) * 4,
            "new_tensor_sidecars": len(run_paths) * 4 * len(captures),
            "new_factorial_analyses": len(run_paths) * len(captures),
            "holdout_analyses": results,
            "reference_recheck_cells": 4,
            "reference_recheck": anchor_checks,
            "model_snapshot": _read(model_fingerprint_path),
            "source_response_counts": _response_counts(run_paths),
            "source_suffix_token_ids": [
                step["token_id"]
                for step in reference_run["stream_events"][0]["generation"]["steps"]
            ],
            "runtime": reference_run["runtime"],
        },
    )
    print(f"wrote pairing seed validation to {args.output_root}", flush=True)


def _validate_run(run: dict, path: Path, manifest_path: Path, reference: dict) -> None:
    if run["model_id"] != reference["model_id"] or run.get("dry_run"):
        raise ValueError(f"model/protocol differs for {path}")
    policy = run["context_policy"]
    if (
        not policy["source_cache_only"]
        or policy["stream_temperature"] != 0
        or run["reproducibility"]["seed"] != 20260604
    ):
        raise ValueError(f"source protocol differs for {path}")
    for key in ("mlx_version", "mlx_vlm_version"):
        if run["runtime"][key] != reference["runtime"][key]:
            raise ValueError(f"runtime {key} differs for {path}")
    event = run["stream_events"][0]
    expected = _read(manifest_path)["frames"][0]["sha256"]
    if (
        event["frame_sha256s"] != [expected]
        or event["prompt"] != reference["stream_events"][0]["prompt"]
    ):
        raise ValueError(f"image hash/prompt differs for {path}")
    if event["assistant_text"] != reference["stream_events"][0]["assistant_text"]:
        raise ValueError(
            f"generated source suffix differs for {path}; audit before pooling"
        )
    steps = event.get("generation", {}).get("steps") or []
    reference_steps = (
        reference["stream_events"][0].get("generation", {}).get("steps") or []
    )
    if not steps or not reference_steps:
        raise ValueError(
            f"generated source suffix token IDs are unavailable for {path}"
        )
    if [step["token_id"] for step in steps] != [
        step["token_id"] for step in reference_steps
    ]:
        raise ValueError(f"generated source suffix token IDs differ for {path}")
    for artifact in event["cache_tensor_artifacts"]:
        load_cache_tensor_artifact(path, artifact)


def _response_counts(paths: dict) -> dict[str, int]:
    counts = {}
    for cells in paths.values():
        for path in cells.values():
            response = _read(path)["stream_events"][0]["assistant_text"]
            counts[response] = counts.get(response, 0) + 1
    return counts


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze_model_snapshot(model_id: str, output_path: Path) -> None:
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(cached, str):
        raise ValueError("cannot identify the cached model snapshot")
    snapshot = Path(cached).parent
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise ValueError("cached model snapshot contains no weights")
    files = sorted(
        {
            path
            for pattern in (
                "*.json",
                "*.jinja",
                "*.safetensors",
                "*.py",
                "*.txt",
                "*.model",
                "*.tiktoken",
            )
            for path in snapshot.glob(pattern)
        }
    )
    record = {
        "schema_version": 1,
        "analysis_kind": "model_snapshot_fingerprint",
        "model_id": model_id,
        "revision": snapshot.name,
        "file_sha256": {p.name: _sha(p) for p in files},
    }
    if output_path.exists() and _read(output_path) != record:
        raise ValueError("model snapshot differs from the study fingerprint")
    write_json(output_path, record)


if __name__ == "__main__":
    main()
