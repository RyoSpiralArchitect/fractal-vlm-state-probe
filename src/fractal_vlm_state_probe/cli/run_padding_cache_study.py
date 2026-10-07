from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_direction_holdout import (
    analyze_pairing_seed_holdout,
    holm_adjust,
    validate_pairing_holdout_metadata,
)
from fractal_vlm_state_probe.cache_tensor_artifact import (
    load_cache_tensor_artifact,
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cache_tensor_factorial import (
    analyze_cache_tensor_factorial,
    cache_tensor_regions,
)
from fractal_vlm_state_probe.cli.capture_pairing_references import (
    _validate_target_contract,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _read,
    _sha,
    _validate_run,
)
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.mlx_stream import _load_mlx_runtime
from fractal_vlm_state_probe.padding_cache import (
    changed_input_keys,
    prepared_input_audit,
    require_calibration,
    tensor_difference,
)
from fractal_vlm_state_probe.padding_policy import _checked_artifact, padding_gate
from fractal_vlm_state_probe.stimulus import write_json


def execute(args) -> dict:
    root = args.output_root
    config = _read(args.config)
    if (
        config["model_id"] != "mlx-community/FastVLM-0.5B-bf16"
        or config["targets"] != ["1:keys", "12:keys", "23:values"]
        or config["conditions"]
        != ["shared_pixel_permutation/black", "shared_pixel_permutation/palette_mean"]
        or config["primary_test_family_size"] != 12
        or config["metadata_cross_anchor"] != "geometry_checker_hex_r1"
        or config["reference_replicates"] != [1, 2]
        or config["test_replicates"] != [3, 4]
        or config["expected_calibration_cells"] != 264
        or config["expected_panel_cells"] != 256
        or config["expected_tensor_shape"] != [1, 2, 397, 64]
        or config["expected_source_response"] != "The image"
        or config["expected_generation_step_ids"] != [785, 2168, 2168]
    ):
        raise ValueError("cache design differs from registered specification")
    study, published = _read(args.input_summary), _read(args.published_input)
    if _sha(args.input_summary) != published["source_report_sha256"]:
        raise ValueError("input receipt differs from published Note 0045")
    records = study["records"]
    metadata = {
        r["pair_id"]: {k: r[k] for k in ("pairing_family", "broad_class", "replicate")}
        for r in records
    }
    validate_pairing_holdout_metadata(metadata)
    if len(metadata) != len(records) or len(records) != 32:
        raise ValueError("all 32 unique input blocks are required")
    captures = tuple(parse_cache_tensor_capture_spec(t) for t in config["targets"])
    historical, historical_hashes, manifests = {}, {}, {}
    for record in records:
        label = record["pair_id"]
        directory = (
            "references_model_path" if record["replicate"] in (1, 2) else "tests"
        )
        historical[label] = {}
        for cell in ("mm", "jj", "mj", "jm"):
            path = (
                args.historical_root
                / directory
                / "source_runs"
                / f"{label}_{cell}_mlx.json"
            )
            run = _read(path)
            historical[label][cell] = path
            historical_hashes[str(path)] = _sha(path)
            historical_manifest = Path(run["manifest_path"])
            manifests[str(historical_manifest)] = _sha(historical_manifest)
            for artifact in run["stream_events"][0]["cache_tensor_artifacts"]:
                load_cache_tensor_artifact(path, artifact)
        for condition in ("original/black", *config["conditions"]):
            measured = record["conditions"][condition]
            if padding_gate(measured["cells"]) != measured["gate"]:
                raise ValueError("saved input gate does not reproduce")
            if condition.endswith("palette_mean") and not measured["gate"]["accepted"]:
                raise ValueError("mean panel contains an unmatched block")
            for cell, artifact in measured["artifacts"].items():
                _checked_artifact(artifact)
                manifests[artifact["manifest_path"]] = artifact["manifest_sha256"]
                if condition == "original/black":
                    old = _read(historical[label][cell])["stream_events"][0]
                    if old["frame_sha256s"] != [
                        measured["cells"][cell]["content_frame_sha256"]
                    ]:
                        raise ValueError(
                            "historical original image differs from input study"
                        )
    reference = _read(historical[config["metadata_cross_anchor"]]["mm"])
    if reference["model_id"] != config["model_id"]:
        raise ValueError("historical model differs")
    contracts = {}
    event = reference["stream_events"][0]
    if (
        event["assistant_text"] != config["expected_source_response"]
        or [s["token_id"] for s in event["generation"]["steps"]]
        != config["expected_generation_step_ids"]
    ):
        raise ValueError("historical response differs from registration")
    for target in captures:
        contracts[target.identifier] = {
            "layer_index": target.layer_index,
            "tensor": target.tensor,
            "tensor_shape": config["expected_tensor_shape"],
            "cells": {"mm": {"cache_token_layout": event["cache_token_layout"]}},
        }
    regions = cache_tensor_regions(event["cache_token_layout"], sequence_length=397)
    if [len(regions[r]) for r in ("pre_image", "image_tokens", "post_image")] != [
        42,
        256,
        99,
    ]:
        raise ValueError("historical cache partition differs")
    for cells in historical.values():
        for path in cells.values():
            run = _read(path)
            _validate_run(run, path, Path(run["manifest_path"]), reference)
            for contract in contracts.values():
                _validate_target_contract(run, path, contract)
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    code_paths = [
        Path(__file__),
        package / "padding_cache.py",
        package / "padding_policy.py",
        package / "cache_tensor_artifact.py",
        package / "cache_tensor_factorial.py",
        package / "cache_direction_holdout.py",
        package / "mlx_stream.py",
        package / "mlx_cumulative_replay.py",
        package / "cli/capture_pairing_references.py",
        package / "cli/run_pairing_seed_validation.py",
    ]
    signature = {
        "schema_version": 1,
        "analysis_kind": "padding_cache_frozen_specification",
        "config": config,
        "input_summary_sha256": _sha(args.input_summary),
        "published_input_sha256": _sha(args.published_input),
        "historical_run_sha256": historical_hashes,
        "manifest_sha256": manifests,
        "hierarchy_metadata": metadata,
        "code_sha256": {str(p.relative_to(repo)): _sha(p) for p in code_paths},
        "historical_model_snapshot_sha256": _sha(
            args.historical_root / "references_model_path/model_snapshot.json"
        ),
    }
    frozen = root / "frozen_specification.json"
    if root.exists():
        if not args.resume or not frozen.exists() or _read(frozen) != signature:
            raise ValueError(
                "use a fresh output root or resume the exact frozen specification"
            )
    else:
        root.mkdir(parents=True)
        write_json(frozen, signature)
    args.output_owned = True
    _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
    if _read(root / "model_snapshot.json") != _read(
        args.historical_root / "references_model_path/model_snapshot.json"
    ):
        raise ValueError("model snapshot differs from qualified historical model")
    if args.preflight_only:
        return {
            "schema_version": 1,
            "analysis_kind": "padding_cache_preflight",
            "status": "PASS",
            "planned_calibration_cells": 264,
            "planned_panel_cells": 256,
        }

    runtime = _load_mlx_runtime(config["model_id"])
    generator = importlib.import_module("mlx_vlm.generate")
    modules = [
        generator,
        importlib.import_module("mlx_vlm.utils"),
        inspect.getmodule(type(runtime["model"])),
        inspect.getmodule(type(runtime["model"].language_model)),
    ]
    processor_type = type(runtime["processor"].image_processor)
    processor_source = Path(inspect.getfile(processor_type))
    expected_processor = study["frozen_specification"]["processor_provenance"]
    if (
        _sha(processor_source) != expected_processor["implementation_sha256"]
        or f"{processor_type.__module__}.{processor_type.__name__}"
        != expected_processor["implementation"]
    ):
        raise ValueError("actual processor differs from qualified input path")
    runtime_record = {
        "schema_version": 1,
        "analysis_kind": "padding_cache_runtime_identity",
        "runtime": runtime["runtime"],
        "processor_source_sha256": _sha(processor_source),
        "modules": {
            m.__name__: {
                "path": inspect.getfile(m),
                "sha256": _sha(Path(inspect.getfile(m))),
            }
            for m in modules
        },
    }
    identity_path = root / "runtime_identity.json"
    if identity_path.exists() and _read(identity_path) != runtime_record:
        raise ValueError("runtime implementation changed on resume")
    write_json(identity_path, runtime_record)
    registry = {}

    def capture(manifest, path, measured, *, rectangular=False, override=None):
        expected_sizes = [[320, 240]] if rectangular else [[320, 320]]
        audit_path = path.with_suffix(".input_audit.json")
        context = {
            "manifest_path": str(manifest),
            "manifest_sha256": _sha(manifest),
            "override_sizes": override,
            "frozen_specification_sha256": _sha(frozen),
        }
        if not path.exists():
            if (path.parent / f"{path.stem}_cache_tensors").exists():
                raise ValueError(
                    f"incomplete tensor capture requires separate audit: {path}"
                )
            with prepared_input_audit(
                generator,
                audit_path=audit_path,
                context=context,
                expected_pixel_sha256=measured["pixel_values_sha256"],
                expected_sizes=expected_sizes,
                override_sizes=override,
            ):
                run_cumulative_replay_probe(
                    CumulativeReplayRunConfig(
                        manifest_path=manifest,
                        output_path=path,
                        model_id=config["model_id"],
                        max_frames=1,
                        max_tokens=2,
                        temperature=0,
                        probe_temperature=0,
                        probe_preset="default",
                        cache_summary_max_layers=None,
                        source_cache_only=True,
                        cache_tensor_captures=captures,
                        include_frame_artifacts=False,
                    ),
                    mlx_runtime=runtime,
                )
        run, audit = _read(path), _read(audit_path)
        if audit["context"] != context or audit["override_image_sizes"] != override:
            raise ValueError("prepared input audit belongs to another capture")
        if (
            audit["native"]["pixel_values"]["sha256"] != measured["pixel_values_sha256"]
            or audit["native"]["image_sizes"]["values"] != expected_sizes
        ):
            raise ValueError("saved prepared input differs from frozen input")
        expected_changes = [] if override is None else ["image_sizes"]
        if changed_input_keys(audit["native"], audit["effective"]) != expected_changes:
            raise ValueError("unexpected effective input mutation")
        _validate_run(run, path, manifest, reference)
        for contract in contracts.values():
            _validate_target_contract(run, path, contract)
        if any(
            a["source_dtype"] != "mlx.core.bfloat16"
            for a in run["stream_events"][0]["cache_tensor_artifacts"]
        ):
            raise ValueError("native tensor dtype differs")
        registry[str(path.relative_to(root))] = {
            "run_sha256": _sha(path),
            "input_audit_path": str(audit_path),
            "input_audit_sha256": _sha(audit_path),
            "tensor_artifacts": run["stream_events"][0]["cache_tensor_artifacts"],
        }
        return path, audit["effective"]

    def compare(before, after, label, cell):
        a, b = _read(before), _read(after)
        out = []
        for target in captures:
            contract = contracts[target.identifier]
            x = _validate_target_contract(a, before, contract)
            y = _validate_target_contract(b, after, contract)
            out.append(
                {
                    "pair_id": label,
                    "cell": cell,
                    "target": target.identifier,
                    "before_path": str(before),
                    "after_path": str(after),
                    **tensor_difference(x, y),
                    "regions": {
                        name: tensor_difference(
                            np.take(x, positions, axis=-2),
                            np.take(y, positions, axis=-2),
                        )
                        for name, positions in regions.items()
                        if positions
                    },
                }
            )
        return out

    calibration = {
        "schema_version": 1,
        "analysis_kind": "padding_cache_calibration",
        "status": "INCOMPLETE",
        "planned_cells": 264,
        "historical_checks": [],
        "black_square_checks": [],
        "crossed_metadata_checks": [],
        "prepared_input_checks": [],
    }
    native_paths, native_inputs = {}, {}
    for index, record in enumerate(records):
        label = record["pair_id"]
        print(f"calibration {index + 1}/32 {label}", flush=True)
        measured = record["conditions"]["original/black"]
        for cell in ("mm", "jj", "mj", "jm"):
            old = historical[label][cell]
            manifest = Path(_read(old)["manifest_path"])
            rect_path, rect_inputs = capture(
                manifest,
                root / "calibration/rectangular" / f"{label}_{cell}.json",
                measured["cells"][cell],
                rectangular=True,
            )
            checks = compare(old, rect_path, label, cell)
            calibration["historical_checks"].extend(checks)
            write_json(root / "calibration.json", calibration)
            require_calibration(checks, expected_count=3)
            square_manifest = Path(measured["artifacts"][cell]["manifest_path"])
            square_path, square_inputs = capture(
                square_manifest,
                root / "calibration/black_square" / f"{label}_{cell}.json",
                measured["cells"][cell],
            )
            keys = changed_input_keys(rect_inputs, square_inputs)
            calibration["prepared_input_checks"].append(
                {"pair_id": label, "cell": cell, "changed_keys": keys}
            )
            checks = compare(rect_path, square_path, label, cell)
            calibration["black_square_checks"].extend(checks)
            write_json(root / "calibration.json", calibration)
            if keys != ["image_sizes"]:
                raise ValueError("natural sham changes more than size metadata")
            require_calibration(checks, expected_count=3)
            native_paths[(label, cell)] = (rect_path, square_path)
            native_inputs[(label, cell)] = (rect_inputs, square_inputs)
    anchor = next(r for r in records if r["pair_id"] == config["metadata_cross_anchor"])
    for cell in ("mm", "jj", "mj", "jm"):
        label = anchor["pair_id"]
        rect_path, square_path = native_paths[(label, cell)]
        for index, (base, rectangular, override) in enumerate(
            ((rect_path, True, [[320, 320]]), (square_path, False, [[320, 240]]))
        ):
            print(f"metadata-only cross {cell} {override}", flush=True)
            path, inputs = capture(
                Path(_read(base)["manifest_path"]),
                root / "calibration/metadata_cross" / f"{cell}_{index}.json",
                anchor["conditions"]["original/black"]["cells"][cell],
                rectangular=rectangular,
                override=override,
            )
            if inputs != native_inputs[(label, cell)][1 - index]:
                raise ValueError(
                    "crossed prepared inputs differ from natural counterpart"
                )
            checks = compare(base, path, label, cell)
            calibration["crossed_metadata_checks"].extend(checks)
            write_json(root / "calibration.json", calibration)
            require_calibration(checks, expected_count=3)
    for name, count in (
        ("historical_checks", 384),
        ("black_square_checks", 384),
        ("crossed_metadata_checks", 24),
    ):
        require_calibration(calibration[name], expected_count=count)
    if len(registry) != 264:
        raise ValueError("calibration capture denominator differs")
    calibration.update(status="PASS", completed_cells=264, tensor_sidecars=792)
    write_json(root / "calibration.json", calibration)
    write_json(root / "capture_registry.json", registry)
    print("calibration PASS; beginning fixed permutation panels", flush=True)

    panel_paths = {condition: {} for condition in config["conditions"]}
    for phase, replicates in (("reference", (1, 2)), ("test", (3, 4))):
        if phase == "test":
            reference_registry = {
                k: v for k, v in registry.items() if k.startswith("panel/reference/")
            }
            reference_path = root / "frozen_panel_references.json"
            if reference_path.exists() and _read(reference_path) != reference_registry:
                raise ValueError("frozen panel reference tensor registry changed")
            write_json(reference_path, reference_registry)
        for condition in config["conditions"]:
            policy = condition.split("/")[1]
            for record in records:
                if record["replicate"] not in replicates:
                    continue
                label = record["pair_id"]
                print(f"{phase} {policy} {label}", flush=True)
                measured = record["conditions"][condition]
                panel_paths[condition][label] = {}
                for cell in ("mm", "jj", "mj", "jm"):
                    path, _ = capture(
                        Path(measured["artifacts"][cell]["manifest_path"]),
                        root / "panel" / phase / policy / f"{label}_{cell}.json",
                        measured["cells"][cell],
                    )
                    panel_paths[condition][label][cell] = path
                write_json(root / "capture_registry.json", registry)
    if len(registry) != 520:
        raise ValueError("completed capture denominator differs")
    result = analyze_panels(panel_paths, metadata, captures, root)
    _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
    result.update(
        schema_version=1,
        analysis_kind="padding_cache_calibrated_panel",
        frozen_specification_sha256=_sha(frozen),
        calibration_summary_sha256=_sha(root / "calibration.json"),
        capture_registry_sha256=_sha(root / "capture_registry.json"),
        frozen_reference_registry_sha256=_sha(root / "frozen_panel_references.json"),
        calibration_cells=264,
        calibration_tensors=792,
        panel_source_cells=256,
        panel_tensor_sidecars=768,
        panel_factorial_analyses=192,
        previous_selected_counts={
            "source_cells": 616,
            "tensors": 1704,
            "factorials": 426,
        },
        updated_selected_counts={
            "source_cells": 872,
            "tensors": 2472,
            "factorials": 618,
        },
        new_direct_probe_cells=0,
        source_response_counts={"The image": len(registry)},
    )
    write_json(root / "study_summary.json", result)
    return result


def analyze_panels(panel_paths, metadata, captures, root):
    primary, exploratory, factorial_paths, holdout_paths = [], [], {}, []
    for condition, blocks in panel_paths.items():
        policy = condition.split("/")[1]
        for target in captures:
            analyses, paths = {}, {}
            for label, cells in blocks.items():
                analysis = analyze_cache_tensor_factorial(
                    runs={c: _read(p) for c, p in cells.items()},
                    run_paths=cells,
                    layer_index=target.layer_index,
                    tensor=target.tensor,
                )
                path = (
                    root / "factorials" / policy / label / f"{target.identifier}.json"
                )
                write_json(path, analysis)
                analyses[label], paths[label] = analysis, path
            holdout = analyze_pairing_seed_holdout(
                analyses, analysis_paths=paths, metadata=metadata
            )
            path = root / "holdout" / policy / f"{target.identifier}.json"
            write_json(path, holdout)
            holdout_paths.append(str(path))
            factorial_paths[f"{condition}/{target.identifier}"] = {
                label: str(p) for label, p in paths.items()
            }
            for view in holdout["views"]:
                row = {"condition": condition, "target": target.identifier, **view}
                (primary if view["primary"] else exploratory).append(row)
    if len(primary) != 12 or len(exploratory) != 48:
        raise ValueError("primary or exploratory test denominator differs")
    for row, corrected in zip(
        primary,
        holm_adjust(
            [
                r["exact_block_test"]["p_greater"] if r["available"] else 1.0
                for r in primary
            ]
        ),
    ):
        row.update(holm_p_greater=corrected, holm_family_size=12)
    contrasts = []
    for row in primary:
        if not row["condition"].endswith("palette_mean"):
            continue
        baseline = next(
            r
            for r in primary
            if r["condition"].endswith("/black")
            and r["target"] == row["target"]
            and r["view"] == row["view"]
        )
        available = row["available"] and baseline["available"]
        contrasts.append(
            {
                "target": row["target"],
                "view": row["view"],
                "available": available,
                "mean_minus_black_margin": row["mean_family_margin"]
                - baseline["mean_family_margin"]
                if available
                else None,
                "family_margin_differences": {
                    f: row["family_margins"][f] - baseline["family_margins"][f]
                    for f in row["family_margins"]
                }
                if available
                else None,
                "interpretation": "descriptive paired difference; not an independent or frequency-only causal test",
            }
        )
    return {
        "primary_test_count": len(primary),
        "exploratory_view_count": len(exploratory),
        "primary_tests": primary,
        "exploratory_views": exploratory,
        "paired_margin_differences": contrasts,
        "factorial_paths": factorial_paths,
        "holdout_paths": holdout_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate black padding and measure the registered permutation panels."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--published-input", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        result = execute(args)
    except Exception as exc:
        if getattr(args, "output_owned", False):
            failure_root = args.output_root / "execution_failures"
            index = len(list(failure_root.glob("*.json"))) + 1
            write_json(
                failure_root / f"{index:04d}.json",
                {
                    "schema_version": 1,
                    "analysis_kind": "padding_cache_execution_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(f"completed {result['analysis_kind']} at {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
