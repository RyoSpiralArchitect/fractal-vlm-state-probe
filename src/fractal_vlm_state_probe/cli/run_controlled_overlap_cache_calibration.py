from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from fractal_vlm_state_probe.cache_tensor_artifact import (
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cache_tensor_factorial import cache_tensor_regions
from fractal_vlm_state_probe.cli.capture_pairing_references import (
    _validate_target_contract,
)
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO, load_maps
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _read,
    _sha,
)
from fractal_vlm_state_probe.controlled_overlap_cache import (
    CELLS,
    TARGETS,
    planned_panel_blocks,
    require_complete_calibration,
    validate_capture,
    validate_config,
)
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.mlx_stream import _load_mlx_runtime
from fractal_vlm_state_probe.padding_cache import (
    prepared_input_audit,
    tensor_difference,
)
from fractal_vlm_state_probe.padding_policy import _checked_artifact, padding_gate
from fractal_vlm_state_probe.stimulus import write_json

CONFIG = REPO / "configs/controlled_overlap_cache_phase2_v1.json"
CODE = [
    "src/fractal_vlm_state_probe/controlled_overlap_cache.py",
    "src/fractal_vlm_state_probe/cli/run_controlled_overlap_cache_calibration.py",
]


def require_offline():
    if any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        raise ValueError("both offline environment flags must be set before execution")


def qualify():
    config = _read(CONFIG)
    validate_config(config)
    public_path, historical_path = (
        REPO / config[k] for k in ("phase1_publication", "historical_publication")
    )
    if (
        _sha(public_path) != config["phase1_publication_sha256"]
        or _sha(historical_path) != config["historical_publication_sha256"]
    ):
        raise ValueError("qualified publication hash changed")
    public, historical = _read(public_path), _read(historical_path)
    input_root, old_root = (
        REPO / config[k] for k in ("phase1_root", "historical_root")
    )
    study_path, old_study_path = (
        input_root / "input_summary.json",
        old_root / "study_summary.json",
    )
    study, previous = _read(study_path), _read(old_study_path)
    if (
        _sha(study_path) != public["input_summary_sha256"]
        or _sha(old_study_path) != historical["source_summary_sha256"]
        or historical["calibration_status"] != "PASS"
        or study["frozen_specification"] != public["frozen_specification"]
        or not study["cache_panel_eligible"]
        or not public["cache_panel_eligible"]
    ):
        raise ValueError("input or historical cache qualification differs")
    for name, receipt in public["ledgers"].items():
        if _sha(public_path.parent / name) != receipt["sha256"]:
            raise ValueError("published input ledger changed")
    frozen = study["frozen_specification"]
    if _sha(Path(frozen["map_report"])) != frozen["map_report_sha256"]:
        raise ValueError("coordinate report changed")
    report, maps = load_maps(Path(frozen["map_report"]))
    if report != public["map_construction"]:
        raise ValueError("coordinate construction publication differs")
    metadata = frozen["hierarchy_metadata"]
    if (
        len(planned_panel_blocks(metadata)) != 512
        or metadata != historical["frozen_specification"]["hierarchy_metadata"]
    ):
        raise ValueError("source split or planned panel differs")
    records = study["records"]
    if len(records) != 896 or {(r["map"], r["pair_id"]) for r in records} != {
        (m, label) for m in maps for label in metadata
    }:
        raise ValueError("input block coverage differs")
    manifests = {}
    for row in records:
        if (
            set(row["cells"]) != set(CELLS)
            or set(row["artifacts"]) != set(CELLS)
            or padding_gate(row["cells"]) != row["gate"]
            or not row["gate"]["accepted"]
            or any(row[k] != v for k, v in metadata[row["pair_id"]].items())
        ):
            raise ValueError("all-cell gate or hierarchy changed")
        for a in row["artifacts"].values():
            _checked_artifact(a)
            manifests[a["manifest_path"]] = a["manifest_sha256"]
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    if runtime != frozen["runtime"]:
        raise ValueError("qualified input runtime changed")
    registry_path = old_root / "capture_registry.json"
    if _sha(registry_path) != previous["capture_registry_sha256"]:
        raise ValueError("historical registry changed")
    registry = _read(registry_path)
    old_input_path = Path(historical["frozen_specification"]["input_summary_path"])
    if (
        _sha(old_input_path)
        != historical["frozen_specification"]["input_summary_sha256"]
    ):
        raise ValueError("historical input receipt changed")
    old_inputs = {
        r["pair_id"]: r["maps"][config["historical_map"]]
        for r in _read(old_input_path)["records"]
    }
    sources = {
        str(p): _sha(p)
        for p in (
            public_path,
            historical_path,
            study_path,
            old_study_path,
            registry_path,
            old_input_path,
        )
    }
    jobs = []
    for label, meta in sorted(metadata.items()):
        phase = "reference" if meta["replicate"] in (1, 2) else "test"
        for cell in CELLS:
            key = f"panel/{phase}/{config['historical_map']}/{label}_{cell}.json"
            path = old_root / key
            audit_path = path.with_suffix(".input_audit.json")
            run, audit, receipt = _read(path), _read(audit_path), registry[key]
            if (
                _sha(path) != receipt["run_sha256"]
                or _sha(audit_path) != receipt["input_audit_sha256"]
                or run["stream_events"][0]["cache_tensor_artifacts"]
                != receipt["tensor_artifacts"]
            ):
                raise ValueError("historical capture receipt changed")
            validate_capture(run, audit, run, audit["effective"], config)
            measured, artifact = (
                old_inputs[label]["cells"][cell],
                old_inputs[label]["artifacts"][cell],
            )
            _checked_artifact(artifact)
            if audit["effective"]["pixel_values"]["sha256"] != measured[
                "pixel_values_sha256"
            ] or run["stream_events"][0]["frame_sha256s"] != [artifact["frame_sha256"]]:
                raise ValueError("historical pixels differ from input audit")
            for target in TARGETS:
                _validate_target_contract(run, path, contract(run, target, config))
            sources.update({str(p): _sha(p) for p in (path, audit_path)})
            manifests[artifact["manifest_path"]] = artifact["manifest_sha256"]
            jobs.append(
                {
                    "pair_id": label,
                    "cell": cell,
                    "historical_run": str(path),
                    "manifest_path": artifact["manifest_path"],
                    "pixel_values_sha256": measured["pixel_values_sha256"],
                }
            )
    code = {}
    for prior in (frozen, historical["frozen_specification"]):
        for name, digest in prior["code_sha256"].items():
            if _sha(REPO / name) != digest:
                raise ValueError(f"qualified measurement code changed: {name}")
            code[name] = digest
    code.update({name: _sha(REPO / name) for name in CODE})
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    for name, digest in {**code, str(CONFIG.relative_to(REPO)): _sha(CONFIG)}.items():
        if (
            hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
            ).hexdigest()
            != digest
        ):
            raise ValueError("commit current specification/code before execution")
    signature = {
        "schema_version": 1,
        "analysis_kind": "controlled_overlap_cache_calibration_registration",
        "registered_design_commit": commit,
        "config": config,
        "code_sha256": code,
        "source_sha256": sources,
        "manifest_sha256": manifests,
        "hierarchy_metadata": metadata,
        "model_snapshot": historical["model_snapshot"],
        "runtime_identity": historical["runtime_identity"],
        "processor_provenance": frozen["processor_provenance"],
        "jobs": jobs,
    }
    return signature


def contract(run, target, config):
    capture = parse_cache_tensor_capture_spec(target)
    layout = run["stream_events"][0]["cache_token_layout"]
    regions = cache_tensor_regions(layout, sequence_length=397)
    for name, (start, stop) in config["region_positions"].items():
        if regions[name] != list(range(start, stop)):
            raise ValueError("qualified token region differs")
    return {
        "layer_index": capture.layer_index,
        "tensor": capture.tensor,
        "tensor_shape": config["expected_tensor_shape"],
        "cells": {"mm": {"cache_token_layout": layout}},
    }


def run(mode, root, preflight_root=None):
    require_offline()
    if root.exists():
        raise FileExistsError("use a fresh output root; no retry or resume")
    root.mkdir(parents=True)
    started = 0
    try:
        frozen = qualify()
        config = frozen["config"]
        free = shutil.disk_usage(root).free
        if free < config["calibration"]["minimum_free_disk_gib"] * 2**30:
            raise ValueError("insufficient free disk for registered calibration")
        write_json(root / "frozen_specification.json", frozen)
        _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
        if _read(root / "model_snapshot.json") != frozen["model_snapshot"]:
            raise ValueError("qualified model snapshot changed")
        if mode == "preflight":
            result = {
                "schema_version": 1,
                "analysis_kind": "controlled_overlap_cache_preflight",
                "status": "PASS",
                "frozen_specification_sha256": _sha(root / "frozen_specification.json"),
                "model_snapshot_sha256": _sha(root / "model_snapshot.json"),
                "historical_cells_verified": 128,
                "historical_tensors_verified": 384,
                "phase1_input_blocks_verified": 896,
                "phase1_input_cells_verified": 3584,
                "free_disk_bytes": free,
                "new_model_forwards": 0,
                "model_weights_loaded": False,
                "panel_started": False,
            }
            write_json(root / "preflight.json", result)
            return result
        if mode != "calibrate" or preflight_root is None:
            raise ValueError(
                "calibration requires a qualified preflight; no panel mode exists"
            )
        preflight = _read(preflight_root / "preflight.json")
        if (
            preflight["status"] != "PASS"
            or (preflight_root / "execution_failure.json").exists()
            or _read(preflight_root / "frozen_specification.json") != frozen
            or _sha(preflight_root / "frozen_specification.json")
            != preflight["frozen_specification_sha256"]
            or _sha(preflight_root / "model_snapshot.json")
            != preflight["model_snapshot_sha256"]
        ):
            raise ValueError("preflight changed or failed")
        write_json(
            root / "preflight_receipt.json",
            {
                "path": str(preflight_root.resolve()),
                "sha256": _sha(preflight_root / "preflight.json"),
            },
        )
        runtime = _load_mlx_runtime(config["model_id"])
        generator = importlib.import_module("mlx_vlm.generate")
        cls = type(runtime["processor"].image_processor)
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
        write_json(root / "runtime_identity.json", identity)
        if (
            identity != frozen["runtime_identity"]
            or f"{cls.__module__}.{cls.__name__}"
            != frozen["processor_provenance"]["implementation"]
        ):
            raise ValueError("actual model runtime or processor changed")
        captures = tuple(parse_cache_tensor_capture_spec(t) for t in TARGETS)
        checks, registry = [], {}
        for job in frozen["jobs"]:
            label, cell = job["pair_id"], job["cell"]
            old_path = Path(job["historical_run"])
            old, old_audit = (
                _read(old_path),
                _read(old_path.with_suffix(".input_audit.json")),
            )
            path = root / "calibration" / f"{label}_{cell}.json"
            audit_path, manifest = (
                path.with_suffix(".input_audit.json"),
                Path(job["manifest_path"]),
            )
            context = {
                "manifest_path": str(manifest),
                "manifest_sha256": _sha(manifest),
                "frozen_specification_sha256": _sha(root / "frozen_specification.json"),
            }
            write_json(
                root / "progress.json",
                {
                    "attempted_source_cells": started + 1,
                    "completed_source_cells": len(registry),
                    "current": job,
                    "panel_started": False,
                },
            )
            started += 1
            with prepared_input_audit(
                generator,
                audit_path=audit_path,
                context=context,
                expected_pixel_sha256=job["pixel_values_sha256"],
                expected_sizes=config["expected_image_sizes"],
            ):
                run_cumulative_replay_probe(
                    CumulativeReplayRunConfig(
                        manifest_path=manifest,
                        output_path=path,
                        model_id=config["model_id"],
                        max_frames=1,
                        max_tokens=config["source_max_tokens"],
                        seed=config["source_seed"],
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
            current, audit = _read(path), _read(audit_path)
            validate_capture(current, audit, old, old_audit["effective"], config)
            if audit["context"] != context:
                raise ValueError("prepared input context differs")
            registry[str(path.relative_to(root))] = {
                "run_sha256": _sha(path),
                "input_audit_sha256": _sha(audit_path),
                "historical_run": str(old_path),
                "tensor_artifacts": current["stream_events"][0][
                    "cache_tensor_artifacts"
                ],
            }
            for target in TARGETS:
                c = contract(old, target, config)
                before, after = (
                    _validate_target_contract(r, p, c)
                    for r, p in ((old, old_path), (current, path))
                )
                check = {
                    "pair_id": label,
                    "cell": cell,
                    "target": target,
                    "before_path": str(old_path),
                    "after_path": str(path),
                    "before_array_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                    "after_array_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
                    **tensor_difference(before, after),
                }
                checks.append(check)
            write_json(root / "capture_registry.json", registry)
            write_json(
                root / "calibration.json",
                {
                    "schema_version": 1,
                    "status": "INCOMPLETE",
                    "completed_cells": len(registry),
                    "checks": checks,
                },
            )
            if not all(c["bitwise_equal"] for c in checks[-3:]):
                raise ValueError("tensor mismatch; retained failure; no panel")
            print(
                f"historical calibration {len(registry)}/128 {label}/{cell}", flush=True
            )
        require_complete_calibration(checks, frozen["hierarchy_metadata"])
        _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
        for group in ("code_sha256", "source_sha256", "manifest_sha256"):
            for name, digest in frozen[group].items():
                if (
                    _sha(REPO / name if group == "code_sha256" else Path(name))
                    != digest
                ):
                    raise ValueError("frozen evidence changed during calibration")
        write_json(
            root / "calibration.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "completed_cells": 128,
                "checks": checks,
            },
        )
        result = {
            "schema_version": 1,
            "analysis_kind": "controlled_overlap_cache_calibration",
            "status": "PASS",
            "frozen_specification_sha256": _sha(root / "frozen_specification.json"),
            "calibration_sha256": _sha(root / "calibration.json"),
            "capture_registry_sha256": _sha(root / "capture_registry.json"),
            "runtime_identity_sha256": _sha(root / "runtime_identity.json"),
            "model_snapshot_sha256": _sha(root / "model_snapshot.json"),
            "preflight_receipt_sha256": _sha(root / "preflight_receipt.json"),
            "calibration_source_cells": 128,
            "calibration_tensor_comparisons": 384,
            "new_model_forwards": 128,
            "new_experimental_panel_cells": 0,
            "new_direct_probe_cells": 0,
            "panel_started": False,
            "panel_automatically_released": False,
            "confirmation_cohort": config["confirmation_cohort"],
        }
        write_json(root / "summary.json", result)
        return result
    except Exception as error:
        write_json(
            root / "execution_failure.json",
            {
                "schema_version": 1,
                "status": "FAILED",
                "type": type(error).__name__,
                "message": str(error),
                "attempted_source_cells": started,
                "panel_started": False,
                "retry_performed": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Offline historical calibration only; never launches the controlled-overlap panel."
    )
    parser.add_argument("mode", choices=("preflight", "calibrate"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--preflight-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.mode,
                args.output_root.resolve(),
                args.preflight_root.resolve() if args.preflight_root else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
