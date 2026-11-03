from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_tensor_artifact import (
    _to_float32_numpy,
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cli.capture_pairing_references import (
    _validate_target_contract,
)
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO
from fractal_vlm_state_probe.cli.run_controlled_overlap_cache_calibration import (
    contract,
    qualify as qualify_cache,
    require_offline,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _read,
    _sha,
)
from fractal_vlm_state_probe.controlled_overlap_cache import TARGETS, validate_capture
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.mlx_stream import _load_mlx_runtime
from fractal_vlm_state_probe.padding_cache import prepared_input_audit
from fractal_vlm_state_probe.representation_observation import (
    STAGES,
    array_receipts,
    observe_projector,
    require_complete,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json

CONFIG = REPO / "configs/representation_observation_calibration_v1.json"
CODE = (
    "configs/representation_observation_calibration_v1.json",
    "src/fractal_vlm_state_probe/representation_observation.py",
    "src/fractal_vlm_state_probe/cli/run_representation_observation_calibration.py",
    "src/fractal_vlm_state_probe/cli/summarize_representation_observation_calibration.py",
    "scripts/run_representation_observation_calibration.py",
    "scripts/summarize_representation_observation_calibration.py",
    "src/fractal_vlm_state_probe/cli/summarize_controlled_overlap_cache_calibration.py",
)


def assert_frozen(frozen):
    for group in ("code_sha256", "source_sha256", "manifest_sha256"):
        for name, digest in frozen[group].items():
            path = REPO / name if group == "code_sha256" else Path(name)
            if _sha(path) != digest:
                raise ValueError(f"frozen {group} changed: {name}")


def qualify():
    frozen, config = qualify_cache(), _read(CONFIG)
    validate_config(config)
    public_path = REPO / config["baseline_publication"]
    if _sha(public_path) != config["baseline_publication_sha256"]:
        raise ValueError("qualified baseline publication changed")
    public = _read(public_path)
    frozen["source_sha256"].update(public["frozen_specification"]["source_sha256"])
    frozen["source_sha256"][str(public_path)] = _sha(public_path)
    for name, receipt in public["ledgers"].items():
        frozen["source_sha256"][str(public_path.parent / name)] = receipt["sha256"]
    archive = public_path.parent / "interaction_vectors.npz"
    frozen["source_sha256"][str(archive)] = public["interaction_archive_sha256"]
    frozen["code_sha256"].update(public["frozen_specification"]["code_sha256"])
    installed = {}
    for name, digest in config["installed_sources"].items():
        path = Path(importlib.util.find_spec(name).origin)
        if _sha(path) != digest:
            raise ValueError("installed observation-site source changed")
        installed[name] = {"path": str(path), "sha256": digest}
        frozen["source_sha256"][str(path)] = digest
    commit = frozen["registered_design_commit"]
    for name in CODE:
        frozen["code_sha256"][name] = _sha(REPO / name)
    for name, digest in frozen["code_sha256"].items():
        stored = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
        if hashlib.sha256(stored).hexdigest() != digest:
            raise ValueError(
                "commit all observation/config/verifier code before replay"
            )
    subprocess.check_call(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            config["preceding_merge_commit"],
            commit,
        ],
        cwd=REPO,
    )
    frozen.update(
        analysis_kind="representation_observation_calibration_registration",
        observation_config=config,
        installed_observation_sources=installed,
    )
    assert_frozen(frozen)
    return frozen


def runtime_identity(runtime):
    generator = importlib.import_module("mlx_vlm.generate")
    cls = type(runtime["processor"].image_processor)
    modules = [
        generator,
        importlib.import_module("mlx_vlm.utils"),
        inspect.getmodule(type(runtime["model"])),
        inspect.getmodule(type(runtime["model"].language_model)),
    ]
    return {
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


def save_observation(root, relative, records, config):
    if len(records) != 1:
        raise ValueError("one observed projector call required")
    arrays = {}
    for site, value in zip(("encoder", "projector"), records[0]):
        if str(value.dtype) != config["source_dtype"]:
            raise ValueError("native representation dtype differs")
        arrays[site] = _to_float32_numpy(value)
    receipts = array_receipts(arrays, config)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"path": relative, "sha256": _sha(path), "arrays": receipts}


def run(mode, root, preflight_root=None):
    require_offline()
    if root.exists():
        raise FileExistsError("fresh output root required; no retry/resume")
    root.mkdir(parents=True)
    started, rows, registry = 0, [], {}
    try:
        frozen = qualify()
        config, original_config = frozen["observation_config"], frozen["config"]
        if shutil.disk_usage(root).free < config["minimum_free_disk_gib"] * 2**30:
            raise ValueError("insufficient calibration disk reserve")
        write_json(root / "frozen_specification.json", frozen)
        frozen_sha = _sha(root / "frozen_specification.json")
        _freeze_model_snapshot(
            original_config["model_id"], root / "model_snapshot.json"
        )
        if _read(root / "model_snapshot.json") != frozen["model_snapshot"]:
            raise ValueError("model snapshot changed")
        if mode == "preflight":
            result = {
                "schema_version": 1,
                "analysis_kind": "representation_observation_preflight",
                "status": "PASS",
                "frozen_specification_sha256": frozen_sha,
                "model_snapshot_sha256": _sha(root / "model_snapshot.json"),
                "new_model_forwards": 0,
                "model_weights_loaded": False,
                "observer_installed": False,
                "experimental_panel_released": False,
            }
            write_json(root / "preflight.json", result)
            return result
        if mode != "calibrate" or preflight_root is None:
            raise ValueError("explicit passing preflight required")
        preflight = _read(preflight_root / "preflight.json")
        if (
            preflight["status"] != "PASS"
            or preflight["new_model_forwards"] != 0
            or preflight["model_weights_loaded"]
            or preflight["observer_installed"]
            or (preflight_root / "execution_failure.json").exists()
            or _read(preflight_root / "frozen_specification.json") != frozen
            or _sha(preflight_root / "frozen_specification.json")
            != preflight["frozen_specification_sha256"]
            or _sha(preflight_root / "model_snapshot.json")
            != preflight["model_snapshot_sha256"]
        ):
            raise ValueError("preflight differs or failed")
        write_json(
            root / "preflight_receipt.json",
            {
                "path": str(preflight_root),
                "sha256": _sha(preflight_root / "preflight.json"),
            },
        )
        runtime = _load_mlx_runtime(original_config["model_id"])
        identity = runtime_identity(runtime)
        write_json(root / "runtime_identity.json", identity)
        if identity != frozen["runtime_identity"]:
            raise ValueError("original qualified runtime changed")
        projector = runtime["model"].mm_projector
        cls = type(projector)
        original_call = cls.__call__
        if (
            f"{cls.__module__}.{cls.__name__}" != config["projector_class"]
            or [type(m).__name__ for m in projector] != config["projector_components"]
        ):
            raise ValueError("actual projector observation site differs")
        generator = importlib.import_module("mlx_vlm.generate")
        captures = tuple(parse_cache_tensor_capture_spec(t) for t in TARGETS)
        for job in frozen["jobs"]:
            stage_tensors = {}
            old_path = Path(job["historical_run"])
            old = _read(old_path)
            old_inputs = _read(old_path.with_suffix(".input_audit.json"))["effective"]
            for stage in STAGES:
                if (
                    cls.__call__ is not original_call
                    or runtime["model"].mm_projector is not projector
                ):
                    raise ValueError("observation site was not restored")
                if (
                    shutil.disk_usage(root).free
                    < config["minimum_free_disk_gib"] * 2**30
                ):
                    raise ValueError("calibration disk reserve breached")
                relative = f"calibration/{job['pair_id']}_{job['cell']}_{stage}.json"
                path = root / relative
                audit_path = path.with_suffix(".input_audit.json")
                manifest = Path(job["manifest_path"])
                context = {
                    "manifest_path": str(manifest),
                    "manifest_sha256": _sha(manifest),
                    "frozen_specification_sha256": frozen_sha,
                }
                write_json(
                    root / "progress.json",
                    {
                        "attempted_source_contexts": started + 1,
                        "completed_source_contexts": len(rows),
                        "stage": stage,
                        "job": job,
                    },
                )
                started += 1
                observer = (
                    observe_projector(projector) if stage == "on" else nullcontext([])
                )
                with prepared_input_audit(
                    generator,
                    audit_path=audit_path,
                    context=context,
                    expected_pixel_sha256=job["pixel_values_sha256"],
                    expected_sizes=original_config["expected_image_sizes"],
                ):
                    with observer as observed:
                        run_cumulative_replay_probe(
                            CumulativeReplayRunConfig(
                                manifest_path=manifest,
                                output_path=path,
                                model_id=original_config["model_id"],
                                max_frames=1,
                                max_tokens=original_config["source_max_tokens"],
                                seed=original_config["source_seed"],
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
                if cls.__call__ is not original_call:
                    raise ValueError("observer restoration failed")
                current, audit = _read(path), _read(audit_path)
                validate_capture(current, audit, old, old_inputs, original_config)
                if audit["context"] != context:
                    raise ValueError("prepared-input context differs")
                hashes = {}
                stage_tensors[stage] = {}
                for target in TARGETS:
                    c = contract(old, target, original_config)
                    before, after = (
                        _validate_target_contract(r, p, c)
                        for r, p in ((old, old_path), (current, path))
                    )
                    if (
                        before.dtype != after.dtype
                        or before.shape != after.shape
                        or before.tobytes() != after.tobytes()
                    ):
                        raise ValueError("selected cache bytes changed")
                    hashes[target] = hashlib.sha256(after.tobytes()).hexdigest()
                    stage_tensors[stage][target] = after
                representation = (
                    save_observation(
                        root,
                        f"representations/{job['pair_id']}_{job['cell']}.npz",
                        observed,
                        config,
                    )
                    if stage == "on"
                    else None
                )
                receipt = {
                    "run_sha256": _sha(path),
                    "input_audit_sha256": _sha(audit_path),
                    "historical_run": str(old_path),
                    "tensor_artifacts": current["stream_events"][0][
                        "cache_tensor_artifacts"
                    ],
                }
                registry[relative] = receipt
                row = {
                    "pair_id": job["pair_id"],
                    "cell": job["cell"],
                    "stage": stage,
                    "ordinal": started,
                    "path": relative,
                    **receipt,
                    "historical_cache_exact": True,
                    "cache_array_sha256": hashes,
                    "observer_restored": True,
                    "observer_calls": len(observed),
                    "representation": representation,
                }
                rows.append(row)
                with (root / "cells.jsonl").open("a") as handle:
                    handle.write(
                        json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    )
                write_json(root / "capture_registry.json", registry)
                observed.clear()
            for off_stage in ("off_before", "off_after"):
                for target in TARGETS:
                    if (
                        stage_tensors[off_stage][target].tobytes()
                        != stage_tensors["on"][target].tobytes()
                    ):
                        raise ValueError("ON/OFF cache bytes changed")
            print(
                f"observation calibration {len(rows) // 3}/128 ({len(rows)}/384 contexts)",
                flush=True,
            )
        require_complete(rows, frozen["jobs"])
        assert_frozen(frozen)
        _freeze_model_snapshot(
            original_config["model_id"], root / "model_snapshot_after.json"
        )
        if _read(root / "model_snapshot_after.json") != frozen["model_snapshot"]:
            raise ValueError("model snapshot changed during calibration")
        result = {
            "schema_version": 1,
            "analysis_kind": "representation_observation_calibration",
            "status": "PASS",
            "registered_design_commit": frozen["registered_design_commit"],
            "calibration_source_contexts": 384,
            "historical_source_cells": 128,
            "historical_tensor_comparisons": 1152,
            "on_off_tensor_comparisons": 768,
            "observed_source_cells": 128,
            "representation_arrays": 256,
            "new_model_forwards": 384,
            "model_forward_unit": "fresh_source_context_not_internal_generation_steps",
            "new_experimental_panel_cells": 0,
            "new_direct_probe_cells": 0,
            "new_correspondence_scores": 0,
            "experimental_panel_released": False,
            "confirmation_cohort": "not_generated_not_sealed",
            "receipts": {
                name: _sha(root / name)
                for name in (
                    "frozen_specification.json",
                    "model_snapshot.json",
                    "model_snapshot_after.json",
                    "preflight_receipt.json",
                    "runtime_identity.json",
                    "cells.jsonl",
                    "capture_registry.json",
                )
            },
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
                "attempted_source_contexts": started,
                "completed_source_contexts": len(rows),
                "retry_performed": False,
                "experimental_panel_released": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Read-only observation calibration; no representation scores."
    )
    parser.add_argument("mode", choices=("preflight", "calibrate"))
    parser.add_argument("--output-root", type=Path, required=True)
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
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
