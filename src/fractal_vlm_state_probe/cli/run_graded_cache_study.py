from __future__ import annotations

import argparse
import importlib
import inspect
import subprocess
from pathlib import Path

from fractal_vlm_state_probe.cache_direction_holdout import (
    validate_pairing_holdout_metadata,
)
from fractal_vlm_state_probe.cache_tensor_artifact import (
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cache_tensor_factorial import cache_tensor_regions
from fractal_vlm_state_probe.cli.capture_pairing_references import (
    _validate_target_contract,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _read,
    _sha,
    _validate_run,
)
from fractal_vlm_state_probe.graded_cache import (
    LEVELS,
    analyze_graded_panels,
    input_frequency_context,
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


def validate_config(config):
    required = {
        "schema_version": 1,
        "model_id": "mlx-community/FastVLM-0.5B-bf16",
        "targets": ["1:keys", "12:keys", "23:values"],
        "levels": list(LEVELS),
        "endpoint": LEVELS[-1],
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "primary_test_family_size": 18,
        "exploratory_head_band_views": 72,
        "descriptive_endpoint_transfer_views": 12,
        "expected_calibration_cells": 128,
        "expected_panel_cells": 384,
        "expected_tensor_shape": [1, 2, 397, 64],
        "expected_source_response": "The image",
        "expected_generation_step_ids": [785, 2168, 2168],
    }
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("graded cache specification differs from registration")


def execute(args):
    root, config = args.output_root, _read(args.config)
    validate_config(config)
    study, published, old_result, old_publication = (
        _read(p)
        for p in (
            args.input_summary,
            args.published_input,
            args.historical_root / "study_summary.json",
            args.published_endpoint,
        )
    )
    if (
        _sha(args.input_summary) != published["source_report_sha256"]
        or _sha(args.historical_root / "study_summary.json")
        != old_publication["source_summary_sha256"]
        or old_publication["calibration_status"] != "PASS"
    ):
        raise ValueError("input or historical cache receipt differs from publication")
    registry_path = args.historical_root / "capture_registry.json"
    if _sha(registry_path) != old_result["capture_registry_sha256"]:
        raise ValueError("historical capture registry changed")
    old_registry = _read(registry_path)
    if (
        _read(args.historical_root / "model_snapshot.json")
        != old_publication["model_snapshot"]
        or _read(args.historical_root / "runtime_identity.json")
        != old_publication["runtime_identity"]
    ):
        raise ValueError("historical model/runtime identity differs from publication")
    records = study["records"]
    metadata = {
        r["pair_id"]: {k: r[k] for k in ("pairing_family", "broad_class", "replicate")}
        for r in records
    }
    validate_pairing_holdout_metadata(metadata)
    if len(records) != len(metadata) or len(metadata) != 32:
        raise ValueError("all 32 unique blocks are required")
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    for prior in (
        study["frozen_specification"],
        old_publication["frozen_specification"],
    ):
        for path, digest in prior["code_sha256"].items():
            if _sha(repo / path) != digest:
                raise ValueError(f"qualified measurement code changed: {path}")
    historical, manifests, historical_hashes, historical_audit_hashes = {}, {}, {}, {}
    for record in records:
        label = record["pair_id"]
        phase = "reference" if record["replicate"] in (1, 2) else "test"
        historical[label] = {}
        for level in LEVELS:
            measured = record["levels"][level]
            if (
                padding_gate(measured["cells"]) != measured["gate"]
                or not measured["gate"]["accepted"]
            ):
                raise ValueError("all three complete input panels must remain eligible")
            for cell, artifact in measured["artifacts"].items():
                _checked_artifact(artifact)
                manifests[artifact["manifest_path"]] = artifact["manifest_sha256"]
                if level == LEVELS[-1]:
                    path = (
                        args.historical_root
                        / "panel"
                        / phase
                        / "palette_mean"
                        / f"{label}_{cell}.json"
                    )
                    run = _read(path)
                    audit_path = path.with_suffix(".input_audit.json")
                    audit = _read(audit_path)
                    receipt = old_registry[str(path.relative_to(args.historical_root))]
                    if (
                        _sha(path) != receipt["run_sha256"]
                        or _sha(audit_path) != receipt["input_audit_sha256"]
                        or run["stream_events"][0]["cache_tensor_artifacts"]
                        != receipt["tensor_artifacts"]
                    ):
                        raise ValueError("historical endpoint capture changed")
                    if (
                        run["stream_events"][0]["frame_sha256s"]
                        != [artifact["frame_sha256"]]
                        or audit["effective"]["pixel_values"]["sha256"]
                        != measured["cells"][cell]["pixel_values_sha256"]
                    ):
                        raise ValueError("full graded endpoint differs from Note 0046")
                    historical[label][cell] = path
                    historical_hashes[str(path)] = _sha(path)
                    historical_audit_hashes[str(audit_path)] = _sha(audit_path)
    reference = _read(next(iter(historical.values()))["mm"])
    event = reference["stream_events"][0]
    if (
        reference["model_id"] != config["model_id"]
        or event["assistant_text"] != config["expected_source_response"]
        or [s["token_id"] for s in event["generation"]["steps"]]
        != config["expected_generation_step_ids"]
    ):
        raise ValueError("historical model or source response differs")
    captures = tuple(parse_cache_tensor_capture_spec(t) for t in config["targets"])
    contracts = {
        t.identifier: {
            "layer_index": t.layer_index,
            "tensor": t.tensor,
            "tensor_shape": config["expected_tensor_shape"],
            "cells": {"mm": {"cache_token_layout": event["cache_token_layout"]}},
        }
        for t in captures
    }
    regions = cache_tensor_regions(event["cache_token_layout"], sequence_length=397)
    if [len(regions[r]) for r in ("pre_image", "image_tokens", "post_image")] != [
        42,
        256,
        99,
    ]:
        raise ValueError("qualified cache partition differs")
    for cells in historical.values():
        for path in cells.values():
            run = _read(path)
            _validate_run(run, path, Path(run["manifest_path"]), reference)
            for contract in contracts.values():
                _validate_target_contract(run, path, contract)
    code_paths = [
        Path(__file__),
        package / "graded_cache.py",
        *[repo / p for p in old_publication["frozen_specification"]["code_sha256"]],
    ]
    signature = {
        "schema_version": 1,
        "analysis_kind": "graded_cache_frozen_specification",
        "config": config,
        "input_summary_path": str(args.input_summary.resolve()),
        "input_summary_sha256": _sha(args.input_summary),
        "published_input_sha256": _sha(args.published_input),
        "historical_summary_sha256": _sha(args.historical_root / "study_summary.json"),
        "historical_publication_sha256": _sha(args.published_endpoint),
        "historical_root": str(args.historical_root.resolve()),
        "historical_run_sha256": historical_hashes,
        "historical_audit_sha256": historical_audit_hashes,
        "manifest_sha256": manifests,
        "hierarchy_metadata": metadata,
        "code_sha256": {str(p.relative_to(repo)): _sha(p) for p in code_paths},
    }
    frozen_path = root / "frozen_specification.json"
    if root.exists():
        if (
            not args.resume
            or not frozen_path.exists()
            or _read(frozen_path) != signature
        ):
            raise ValueError(
                "use a fresh output root or resume its exact frozen specification"
            )
    else:
        root.mkdir(parents=True)
        write_json(frozen_path, signature)
        write_json(
            root / "registration.json",
            {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
                "date": config["registered_date"],
            },
        )
    args.output_owned = True
    _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
    if _read(root / "model_snapshot.json") != _read(
        args.historical_root / "model_snapshot.json"
    ):
        raise ValueError("model snapshot differs from qualified endpoint")
    if args.preflight_only:
        return {
            "status": "PASS",
            "planned_calibration_cells": 128,
            "planned_panel_cells": 384,
        }
    runtime = _load_mlx_runtime(config["model_id"])
    generator = importlib.import_module("mlx_vlm.generate")
    modules = [
        generator,
        importlib.import_module("mlx_vlm.utils"),
        inspect.getmodule(type(runtime["model"])),
        inspect.getmodule(type(runtime["model"].language_model)),
    ]
    cls = type(runtime["processor"].image_processor)
    expected_processor = study["frozen_specification"]["processor_provenance"]
    if (
        _sha(Path(inspect.getfile(cls))) != expected_processor["implementation_sha256"]
        or f"{cls.__module__}.{cls.__name__}" != expected_processor["implementation"]
    ):
        raise ValueError("actual processor identity differs")
    identity = {
        "runtime": runtime["runtime"],
        "processor_source_sha256": _sha(Path(inspect.getfile(cls))),
        "modules": {
            m.__name__: {
                "path": inspect.getfile(m),
                "sha256": _sha(Path(inspect.getfile(m))),
            }
            for m in modules
        },
    }
    old_identity = _read(args.historical_root / "runtime_identity.json")
    if any(identity[k] != old_identity[k] for k in identity):
        raise ValueError("actual model-path runtime differs from historical endpoint")
    if (root / "runtime_identity.json").exists() and _read(
        root / "runtime_identity.json"
    ) != identity:
        raise ValueError("runtime changed on resume")
    write_json(root / "runtime_identity.json", identity)
    registry, panel_paths = {}, {level: {} for level in LEVELS}

    def capture(manifest, path, measured):
        audit_path = path.with_suffix(".input_audit.json")
        context = {
            "manifest_path": str(manifest),
            "manifest_sha256": _sha(manifest),
            "frozen_specification_sha256": _sha(frozen_path),
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
                expected_sizes=[[320, 320]],
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
        if (
            audit["context"] != context
            or audit["override_image_sizes"] is not None
            or audit["native"] != audit["effective"]
            or audit["native"]["pixel_values"]["sha256"]
            != measured["pixel_values_sha256"]
            or audit["native"]["image_sizes"]["values"] != [[320, 320]]
        ):
            raise ValueError("actual prepared-input receipt differs")
        _validate_run(run, path, manifest, reference)
        for contract in contracts.values():
            _validate_target_contract(run, path, contract)
        artifacts = run["stream_events"][0]["cache_tensor_artifacts"]
        if len(artifacts) != 3 or any(
            a["source_dtype"] != "mlx.core.bfloat16" for a in artifacts
        ):
            raise ValueError("target count or native dtype differs")
        registry[str(path.relative_to(root))] = {
            "run_sha256": _sha(path),
            "input_audit_path": str(audit_path),
            "input_audit_sha256": _sha(audit_path),
            "tensor_artifacts": artifacts,
        }
        return path, audit["effective"]

    def compare(old, new, label, cell):
        checks = []
        for target, contract in contracts.items():
            before = _validate_target_contract(_read(old), old, contract)
            after = _validate_target_contract(_read(new), new, contract)
            checks.append(
                {
                    "pair_id": label,
                    "cell": cell,
                    "target": target,
                    "before_path": str(old),
                    "after_path": str(new),
                    **tensor_difference(before, after),
                }
            )
        return checks

    calibration = {"status": "INCOMPLETE", "planned_cells": 128, "checks": []}
    for record in records:
        label = record["pair_id"]
        measured = record["levels"][LEVELS[-1]]
        print(f"endpoint calibration {label}", flush=True)
        for cell in ("mm", "jj", "mj", "jm"):
            path, inputs = capture(
                Path(measured["artifacts"][cell]["manifest_path"]),
                root / "calibration" / f"{label}_{cell}.json",
                measured["cells"][cell],
            )
            old = historical[label][cell]
            if inputs != _read(old.with_suffix(".input_audit.json"))["effective"]:
                raise ValueError("endpoint prepared inputs differ from historical run")
            checks = compare(old, path, label, cell)
            calibration["checks"].extend(checks)
            write_json(root / "calibration.json", calibration)
            require_calibration(checks, expected_count=3)
        write_json(root / "capture_registry.json", registry)
    require_calibration(calibration["checks"], expected_count=384)
    calibration.update(status="PASS", completed_cells=128, tensor_sidecars=384)
    write_json(root / "calibration.json", calibration)
    print("endpoint calibration PASS; beginning three registered panels", flush=True)
    endpoint_rechecks = []
    for phase, replicates in (("reference", (1, 2)), ("test", (3, 4))):
        if phase == "test":
            refs = {
                k: v for k, v in registry.items() if k.startswith("panel/reference/")
            }
            path = root / "frozen_panel_references.json"
            if len(refs) != 192 or (path.exists() and _read(path) != refs):
                raise ValueError("frozen reference registry differs")
            write_json(path, refs)
        for level in LEVELS:
            for record in records:
                if record["replicate"] not in replicates:
                    continue
                label = record["pair_id"]
                print(f"{phase} {level} {label}", flush=True)
                measured = record["levels"][level]
                panel_paths[level][label] = {}
                for cell in ("mm", "jj", "mj", "jm"):
                    path, inputs = capture(
                        Path(measured["artifacts"][cell]["manifest_path"]),
                        root / "panel" / phase / level / f"{label}_{cell}.json",
                        measured["cells"][cell],
                    )
                    old_inputs = _read(
                        historical[label][cell].with_suffix(".input_audit.json")
                    )["effective"]
                    if changed_input_keys(old_inputs, inputs) != (
                        [] if level == LEVELS[-1] else ["pixel_values"]
                    ):
                        raise ValueError("graded panel changes non-pixel model inputs")
                    if level == LEVELS[-1]:
                        checks = compare(historical[label][cell], path, label, cell)
                        endpoint_rechecks.extend(checks)
                        write_json(
                            root / "panel_endpoint_rechecks.json", endpoint_rechecks
                        )
                        require_calibration(checks, expected_count=3)
                    panel_paths[level][label][cell] = path
                write_json(root / "capture_registry.json", registry)
    if len(registry) != 512:
        raise ValueError("completed capture denominator differs")
    require_calibration(endpoint_rechecks, expected_count=384)
    result = analyze_graded_panels(panel_paths, metadata, captures, root)
    _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
    result.update(
        schema_version=1,
        analysis_kind="graded_cache_direction_decomposition",
        frozen_specification_sha256=_sha(frozen_path),
        calibration_summary_sha256=_sha(root / "calibration.json"),
        capture_registry_sha256=_sha(root / "capture_registry.json"),
        frozen_reference_registry_sha256=_sha(root / "frozen_panel_references.json"),
        panel_endpoint_rechecks_sha256=_sha(root / "panel_endpoint_rechecks.json"),
        input_frequency_context=input_frequency_context(study),
        calibration_cells=128,
        calibration_tensors=384,
        panel_source_cells=384,
        panel_tensor_sidecars=1152,
        panel_factorial_analyses=288,
        previous_selected_counts={
            "source_cells": 872,
            "tensors": 2472,
            "factorials": 618,
        },
        updated_selected_counts={
            "source_cells": 1256,
            "tensors": 3624,
            "factorials": 906,
        },
        repeated_endpoint_panel_counts={
            "source_cells": 128,
            "tensors": 384,
            "factorials": 96,
        },
        new_intermediate_counts={
            "source_cells": 256,
            "tensors": 768,
            "factorials": 192,
        },
        new_direct_probe_cells=0,
    )
    write_json(root / "study_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate and decompose three graded common-permutation cache panels."
    )
    for name in (
        "config",
        "input-summary",
        "published-input",
        "historical-root",
        "published-endpoint",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = execute(args)
    except Exception as error:
        if getattr(args, "output_owned", False):
            failures = args.output_root / "execution_failures.json"
            rows = _read(failures) if failures.exists() else []
            rows.append({"type": type(error).__name__, "message": str(error)})
            write_json(failures, rows)
        raise
    print(
        f"wrote graded cache study to {args.output_root}; {result.get('status', 'COMPLETE')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
