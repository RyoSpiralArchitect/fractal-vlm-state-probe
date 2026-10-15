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
from fractal_vlm_state_probe.map_seed_cache import analyze_map_panels
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
from fractal_vlm_state_probe.permutation_maps import ANCHOR, MAPS, summarize_input_maps
from fractal_vlm_state_probe.stimulus import write_json


def validate_config(config):
    required = {
        "schema_version": 1,
        "study_id": "map_seed_cache_fastvlm_v1",
        "model_id": "mlx-community/FastVLM-0.5B-bf16",
        "maps": list(MAPS),
        "targets": ["1:keys", "12:keys", "23:values"],
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "primary_test_family_size": 12,
        "primary_regimes": ["same_map", "cross_map"],
        "primary_map_scope": "three_new_maps_only",
        "descriptive_matrix_views": 96,
        "expected_calibration_cells": 128,
        "expected_panel_cells": 512,
        "expected_source_response": "The image",
        "expected_generation_step_ids": [785, 2168, 2168],
        "expected_tensor_shape": [1, 2, 397, 64],
    }
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("map cache specification differs from registration")


def execute(args):
    root, config = args.output_root, _read(args.config)
    validate_config(config)
    study, published = _read(args.input_summary), _read(args.published_input)
    previous = _read(args.historical_root / "study_summary.json")
    old_publication = _read(args.published_anchor)
    if (
        _sha(args.input_summary) != published["source_report_sha256"]
        or _sha(args.historical_root / "study_summary.json")
        != old_publication["source_summary_sha256"]
        or old_publication["calibration_status"] != "PASS"
        or study["frozen_specification"] != published["frozen_specification"]
        or not study["cache_panel_eligible"]
        or not published["cache_panel_eligible"]
        or summarize_input_maps(study["records"]) != study["maps"]
        or any(v["accepted_blocks"] != 32 for v in study["maps"].values())
    ):
        raise ValueError("input eligibility or historical publication differs")
    registry_path = args.historical_root / "capture_registry.json"
    if _sha(registry_path) != previous["capture_registry_sha256"]:
        raise ValueError("historical capture registry changed")
    old_registry = _read(registry_path)
    for name, field in (
        ("model_snapshot.json", "model_snapshot"),
        ("runtime_identity.json", "runtime_identity"),
    ):
        if _read(args.historical_root / name) != old_publication[field]:
            raise ValueError("historical runtime/model publication differs")
    repo = Path(__file__).resolve().parents[3]
    for specification in (
        study["frozen_specification"],
        old_publication["frozen_specification"],
    ):
        for name, digest in specification["code_sha256"].items():
            if _sha(repo / name) != digest:
                raise ValueError(f"qualified code changed: {name}")
    records = study["records"]
    metadata = {
        r["pair_id"]: {k: r[k] for k in ("pairing_family", "broad_class", "replicate")}
        for r in records
    }
    validate_pairing_holdout_metadata(metadata)
    if len(records) != 32 or len(metadata) != 32:
        raise ValueError("all 32 unique source blocks required")
    historical, manifests, historical_hashes, historical_audits = {}, {}, {}, {}
    for record in records:
        label = record["pair_id"]
        phase = "reference" if record["replicate"] in (1, 2) else "test"
        historical[label] = {}
        if set(record["maps"]) != set(MAPS):
            raise ValueError("map coverage changed")
        for key in MAPS:
            measured = record["maps"][key]
            if (
                padding_gate(measured["cells"]) != measured["gate"]
                or not measured["gate"]["accepted"]
            ):
                raise ValueError("input gate changed or failed")
            for cell, artifact in measured["artifacts"].items():
                _checked_artifact(artifact)
                manifests[artifact["manifest_path"]] = artifact["manifest_sha256"]
                if key != ANCHOR:
                    continue
                path = (
                    args.historical_root
                    / "panel"
                    / phase
                    / "selected_06_of_08"
                    / f"{label}_{cell}.json"
                )
                audit_path = path.with_suffix(".input_audit.json")
                run, audit = _read(path), _read(audit_path)
                receipt = old_registry[str(path.relative_to(args.historical_root))]
                if (
                    _sha(path) != receipt["run_sha256"]
                    or _sha(audit_path) != receipt["input_audit_sha256"]
                    or run["stream_events"][0]["cache_tensor_artifacts"]
                    != receipt["tensor_artifacts"]
                    or run["stream_events"][0]["frame_sha256s"]
                    != [artifact["frame_sha256"]]
                    or audit["effective"]["pixel_values"]["sha256"]
                    != measured["cells"][cell]["pixel_values_sha256"]
                ):
                    raise ValueError("historical 3/4 anchor differs")
                historical[label][cell] = path
                historical_hashes[str(path)] = _sha(path)
                historical_audits[str(audit_path)] = _sha(audit_path)
    reference = _read(next(iter(historical.values()))["mm"])
    event = reference["stream_events"][0]
    if (
        reference["model_id"] != config["model_id"]
        or event["assistant_text"] != config["expected_source_response"]
        or [s["token_id"] for s in event["generation"]["steps"]]
        != config["expected_generation_step_ids"]
    ):
        raise ValueError("historical source response differs")
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
        raise ValueError("token partition differs")
    for cells in historical.values():
        for path in cells.values():
            run = _read(path)
            _validate_run(run, path, Path(run["manifest_path"]), reference)
            for contract in contracts.values():
                _validate_target_contract(run, path, contract)
    code_names = [
        "src/fractal_vlm_state_probe/cli/run_map_seed_cache_study.py",
        "src/fractal_vlm_state_probe/map_seed_cache.py",
        *study["frozen_specification"]["code_sha256"],
        *old_publication["frozen_specification"]["code_sha256"],
    ]
    signature = {
        "schema_version": 1,
        "analysis_kind": "map_seed_cache_frozen_specification",
        "config": config,
        "input_summary_path": str(args.input_summary.resolve()),
        "input_summary_sha256": _sha(args.input_summary),
        "published_input_sha256": _sha(args.published_input),
        "historical_summary_sha256": _sha(args.historical_root / "study_summary.json"),
        "historical_publication_sha256": _sha(args.published_anchor),
        "historical_root": str(args.historical_root.resolve()),
        "historical_run_sha256": historical_hashes,
        "historical_audit_sha256": historical_audits,
        "manifest_sha256": manifests,
        "hierarchy_metadata": metadata,
        "code_sha256": {name: _sha(repo / name) for name in code_names},
    }
    frozen_path = root / "frozen_specification.json"
    if root.exists():
        if (
            not args.resume
            or not frozen_path.exists()
            or _read(frozen_path) != signature
        ):
            raise ValueError(
                "use a fresh root or resume its exact frozen specification"
            )
    else:
        root.mkdir(parents=True)
        write_json(frozen_path, signature)
        write_json(
            root / "registration.json",
            {
                "date": config["registered_date"],
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
            },
        )
    args.output_owned = True
    _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
    if _read(root / "model_snapshot.json") != old_publication["model_snapshot"]:
        raise ValueError("model snapshot changed")
    if args.preflight_only:
        return {
            "status": "PASS",
            "planned_calibration_cells": 128,
            "planned_panel_cells": 512,
        }
    runtime = _load_mlx_runtime(config["model_id"])
    generator = importlib.import_module("mlx_vlm.generate")
    cls = type(runtime["processor"].image_processor)
    expected_processor = study["frozen_specification"]["processor_provenance"]
    if (
        f"{cls.__module__}.{cls.__name__}" != expected_processor["implementation"]
        or _sha(Path(inspect.getfile(cls)))
        != expected_processor["implementation_sha256"]
    ):
        raise ValueError("actual processor changed")
    modules = [
        generator,
        importlib.import_module("mlx_vlm.utils"),
        inspect.getmodule(type(runtime["model"])),
        inspect.getmodule(type(runtime["model"].language_model)),
    ]
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
    if identity != old_publication["runtime_identity"] or (
        (root / "runtime_identity.json").exists()
        and _read(root / "runtime_identity.json") != identity
    ):
        raise ValueError("actual model-path runtime changed")
    write_json(root / "runtime_identity.json", identity)
    registry, paths = {}, {key: {} for key in MAPS}
    saved_registry = (
        _read(root / "capture_registry.json")
        if (root / "capture_registry.json").exists()
        else {}
    )

    def capture(manifest, path, measured):
        audit_path = path.with_suffix(".input_audit.json")
        context = {
            "manifest_path": str(manifest),
            "manifest_sha256": _sha(manifest),
            "frozen_specification_sha256": _sha(frozen_path),
        }
        relative = str(path.relative_to(root))
        if relative in saved_registry and (
            _sha(path) != saved_registry[relative]["run_sha256"]
            or _sha(audit_path) != saved_registry[relative]["input_audit_sha256"]
        ):
            raise ValueError("resumed capture changed after registration")
        if not path.exists():
            if (path.parent / f"{path.stem}_cache_tensors").exists():
                raise ValueError("incomplete capture requires separate audit")
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
            raise ValueError("prepared-input receipt differs")
        _validate_run(run, path, manifest, reference)
        for contract in contracts.values():
            _validate_target_contract(run, path, contract)
        artifacts = run["stream_events"][0]["cache_tensor_artifacts"]
        if len(artifacts) != 3 or any(
            a["source_dtype"] != "mlx.core.bfloat16" for a in artifacts
        ):
            raise ValueError("target count or native dtype differs")
        registry[relative] = {
            "run_sha256": _sha(path),
            "input_audit_path": str(audit_path),
            "input_audit_sha256": _sha(audit_path),
            "tensor_artifacts": artifacts,
        }
        return path, audit["effective"]

    def compare(old, new, label, cell):
        return [
            {
                "pair_id": label,
                "cell": cell,
                "target": target,
                "before_path": str(old),
                "after_path": str(new),
                **tensor_difference(
                    _validate_target_contract(_read(old), old, contract),
                    _validate_target_contract(_read(new), new, contract),
                ),
            }
            for target, contract in contracts.items()
        ]

    calibration = {"status": "INCOMPLETE", "planned_cells": 128, "checks": []}
    for record in records:
        label, measured = record["pair_id"], record["maps"][ANCHOR]
        print(f"anchor calibration {label}", flush=True)
        for cell in ("mm", "jj", "mj", "jm"):
            path, inputs = capture(
                Path(measured["artifacts"][cell]["manifest_path"]),
                root / "calibration" / f"{label}_{cell}.json",
                measured["cells"][cell],
            )
            old = historical[label][cell]
            if inputs != _read(old.with_suffix(".input_audit.json"))["effective"]:
                raise ValueError("calibration prepared input differs")
            checks = compare(old, path, label, cell)
            calibration["checks"].extend(checks)
            write_json(root / "calibration.json", calibration)
            require_calibration(checks, expected_count=3)
        write_json(root / "capture_registry.json", registry)
    require_calibration(calibration["checks"], expected_count=384)
    calibration.update(status="PASS", completed_cells=128, tensor_sidecars=384)
    write_json(root / "calibration.json", calibration)
    print("anchor calibration PASS; beginning four registered maps", flush=True)
    anchor_checks = []
    for phase, replicates in (("reference", (1, 2)), ("test", (3, 4))):
        if phase == "test":
            refs = {
                k: v for k, v in registry.items() if k.startswith("panel/reference/")
            }
            path = root / "frozen_panel_references.json"
            if len(refs) != 256 or (path.exists() and _read(path) != refs):
                raise ValueError("frozen references differ")
            write_json(path, refs)
        for key in MAPS:
            for record in records:
                if record["replicate"] not in replicates:
                    continue
                label, measured = record["pair_id"], record["maps"][key]
                print(f"{phase} {key} {label}", flush=True)
                paths[key][label] = {}
                for cell in ("mm", "jj", "mj", "jm"):
                    path, inputs = capture(
                        Path(measured["artifacts"][cell]["manifest_path"]),
                        root / "panel" / phase / key / f"{label}_{cell}.json",
                        measured["cells"][cell],
                    )
                    old = historical[label][cell]
                    if changed_input_keys(
                        _read(old.with_suffix(".input_audit.json"))["effective"], inputs
                    ) != ([] if key == ANCHOR else ["pixel_values"]):
                        raise ValueError("map panel changed non-pixel model inputs")
                    if key == ANCHOR:
                        checks = compare(old, path, label, cell)
                        anchor_checks.extend(checks)
                        write_json(root / "panel_anchor_rechecks.json", anchor_checks)
                        require_calibration(checks, expected_count=3)
                    paths[key][label][cell] = path
                write_json(root / "capture_registry.json", registry)
    if len(registry) != 640:
        raise ValueError("capture denominator differs")
    require_calibration(anchor_checks, expected_count=384)
    result = analyze_map_panels(paths, metadata, captures, root)
    _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
    result.update(
        schema_version=1,
        analysis_kind="fixed_fraction_map_seed_cache_transfer",
        date=config["registered_date"],
        frozen_specification_sha256=_sha(frozen_path),
        calibration_summary_sha256=_sha(root / "calibration.json"),
        capture_registry_sha256=_sha(root / "capture_registry.json"),
        frozen_reference_registry_sha256=_sha(root / "frozen_panel_references.json"),
        panel_anchor_rechecks_sha256=_sha(root / "panel_anchor_rechecks.json"),
        input_frequency_context=published["input_context"],
        calibration_cells=128,
        calibration_tensors=384,
        panel_source_cells=512,
        panel_tensor_sidecars=1536,
        panel_factorial_analyses=384,
        previous_selected_counts={
            "source_cells": 1256,
            "tensors": 3624,
            "factorials": 906,
        },
        updated_selected_counts={
            "source_cells": 1768,
            "tensors": 5160,
            "factorials": 1290,
        },
        repeated_anchor_panel_counts={
            "source_cells": 128,
            "tensors": 384,
            "factorials": 96,
        },
        new_map_panel_counts={"source_cells": 384, "tensors": 1152, "factorials": 288},
        new_direct_probe_cells=0,
    )
    write_json(root / "study_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a 3/4 anchor and measure same-map versus cross-map cache correspondence."
    )
    for name in (
        "config",
        "input-summary",
        "published-input",
        "historical-root",
        "published-anchor",
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
            path = args.output_root / "execution_failures.json"
            failures = _read(path) if path.exists() else []
            write_json(
                path, [*failures, {"type": type(error).__name__, "message": str(error)}]
            )
        raise
    print(
        f"map seed cache study: {result.get('status', 'COMPLETE')}; {args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
