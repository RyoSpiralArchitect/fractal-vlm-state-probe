from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli import (
    run_representation_observation_calibration as cal,
)
from fractal_vlm_state_probe.cli import (
    summarize_representation_observation_calibration as cal_pub,
)
from fractal_vlm_state_probe.cli.analyze_nonlearned_processing_panel import read_jsonl
from fractal_vlm_state_probe.controlled_overlap_cache_panel import ordered_jobs
from fractal_vlm_state_probe.representation_observation import (
    array_receipts,
    observe_projector,
)
from fractal_vlm_state_probe.representation_panel import (
    SITES,
    array_path,
    codec_check,
    disk_budget,
    pack,
    run_path,
    seal,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json

REPO, read, sha = cal.REPO, cal._read, cal._sha
CONFIG = REPO / "configs/representation_correspondence_panel_v1.json"
CODE = [
    "configs/representation_correspondence_panel_v1.json",
    "src/fractal_vlm_state_probe/representation_panel.py",
    "src/fractal_vlm_state_probe/cli/run_representation_panel.py",
    "src/fractal_vlm_state_probe/cli/analyze_representation_panel.py",
    "src/fractal_vlm_state_probe/cli/summarize_representation_panel.py",
    "scripts/run_representation_panel.py",
    "scripts/analyze_representation_panel.py",
    "scripts/summarize_representation_panel.py",
    "src/fractal_vlm_state_probe/cli/analyze_controlled_overlap.py",
    "src/fractal_vlm_state_probe/cli/summarize_controlled_overlap.py",
]


def check_space(root, required):
    free = shutil.disk_usage(root).free
    if free < required:
        raise ValueError(f"disk gate: available {free}, required {required} bytes")
    return free


def pinned_file(frozen, path, expected):
    if sha(path) != expected:
        raise ValueError(f"qualified source changed: {path}")
    frozen["source_sha256"][str(path)] = expected


def specification(base):
    frozen, config = copy.deepcopy(base), read(CONFIG)
    validate_config(config)
    prior_path = REPO / config["observation_publication"]
    pinned_file(frozen, prior_path, config["observation_publication_sha256"])
    prior = read(prior_path)
    if prior["status"] != "PASS" or prior["new_correspondence_scores"] != 0:
        raise ValueError("observation qualification unavailable")
    for name, value in prior["frozen_specification"]["code_sha256"].items():
        if sha(REPO / name) != value:
            raise ValueError(f"previously frozen code changed: {name}")
    for name, receipt in prior["ledgers"].items():
        pinned_file(frozen, prior_path.parent / name, receipt["sha256"])
    old_obs_root = REPO / config["observation_root"]
    old_reps = read_jsonl(prior_path.parent / "representations.jsonl")
    if len(old_reps) != 128 or [(r["pair_id"], r["cell"]) for r in old_reps] != [
        (j["pair_id"], j["cell"]) for j in frozen["jobs"]
    ]:
        raise ValueError("historical representation grid differs")
    for row in old_reps:
        p = old_obs_root / row["path"]
        pinned_file(frozen, p, row["sha256"])
        with np.load(p, allow_pickle=False) as archive:
            arrays = {s: archive[s] for s in archive.files}
        if array_receipts(arrays, frozen["observation_config"]) != row["arrays"]:
            raise ValueError("historical representation receipt differs")
        for a in arrays.values():
            pack(a)
    prior_cache = REPO / config["cache_publication"]
    pinned_file(frozen, prior_cache, config["cache_publication_sha256"])
    old_root = REPO / config["prior_panel_root"]
    cache_pub = read(prior_cache)
    pinned_file(
        frozen, old_root / "capture_summary.json", cache_pub["capture_summary_sha256"]
    )
    old = read(old_root / "capture_summary.json")
    if old != cache_pub["capture_summary"] or old["status"] != "COMPLETE":
        raise ValueError("prior cache capture differs")
    pinned_file(
        frozen,
        old_root / "capture_registry.json",
        old["receipts"]["capture_registry.json"],
    )
    registry = read(old_root / "capture_registry.json")
    jobs = ordered_jobs(frozen["hierarchy_metadata"])
    inputs = read(REPO / frozen["config"]["phase1_root"] / "input_summary.json")
    accepted = {(b["map"], b["pair_id"]): b for b in inputs["records"]}
    cache_files = 0
    for job in jobs:
        rel = run_path(job)
        p, receipt = old_root / rel, registry[rel]
        if any(receipt[k] != v for k, v in job.items()) or receipt["path"] != rel:
            raise ValueError("prior cache job mapping differs")
        pinned_file(frozen, p, receipt["run_sha256"])
        pinned_file(
            frozen, p.with_suffix(".input_audit.json"), receipt["input_audit_sha256"]
        )
        anchor, audit = read(p), read(p.with_suffix(".input_audit.json"))
        block = accepted[(job["map"], job["pair_id"])]
        artifact, expected = (
            block["artifacts"][job["cell"]],
            block["cells"][job["cell"]],
        )
        if (
            not block["gate"]["accepted"]
            or anchor["manifest_path"] != artifact["manifest_path"]
            or audit["context"]["manifest_path"] != artifact["manifest_path"]
            or audit["context"]["manifest_sha256"] != artifact["manifest_sha256"]
            or anchor["stream_events"][0]["frame_sha256s"] != [artifact["frame_sha256"]]
            or audit["effective"]["pixel_values"]["sha256"]
            != expected["pixel_values_sha256"]
        ):
            raise ValueError("prior cache anchor differs from accepted input cell")
        cal.validate_capture(
            anchor, audit, anchor, audit["effective"], frozen["config"]
        )
        if (
            anchor["stream_events"][0]["cache_tensor_artifacts"]
            != receipt["tensor_artifacts"]
        ):
            raise ValueError("prior cache artifact registry differs")
        for a in receipt["tensor_artifacts"]:
            target = (p.parent / a["path"]).resolve()
            if not target.is_relative_to(p.parent):
                raise ValueError("invalid cache anchor path")
            pinned_file(frozen, target, a["sha256"])
            cache_files += 1
    if len(jobs) != 2048 or cache_files != 6144:
        raise ValueError("prior panel anchor coverage differs")
    commit = frozen["registered_design_commit"]
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
    for name in CODE:
        value = sha(REPO / name)
        if (
            hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
            ).hexdigest()
            != value
        ):
            raise ValueError("commit full panel implementation before measurement")
        frozen["code_sha256"][name] = value
    frozen.update(
        analysis_kind="representation_correspondence_registration",
        release=config,
        panel_jobs=jobs,
        accepted_panel_anchor_cells=len(jobs),
        prior_panel_cache_files=cache_files,
        map_recipes={
            k: v["recipe"]
            for k, v in read(REPO / frozen["config"]["phase1_publication"])[
                "map_construction"
            ]["maps"].items()
        },
        disk_budget=disk_budget(config),
        finite_codec_words_checked=codec_check(),
        historical_codec_arrays_checked=2 * len(old_reps),
    )
    return frozen


@contextmanager
def retain_loaded_runtime():
    # Let the immutable calibration perform its normal single load; retain its exact return.
    original, loaded = cal._load_mlx_runtime, []

    def load_once(*args, **kwargs):
        if loaded:
            raise ValueError("more than one model load")
        runtime = original(*args, **kwargs)
        loaded.append(runtime)
        return runtime

    cal._load_mlx_runtime = load_once
    try:
        yield loaded
        if len(loaded) != 1:
            raise ValueError("one real calibration model load required")
    finally:
        cal._load_mlx_runtime = original


def recalibration_parity(frozen, root):
    cfg = frozen["release"]
    prior_dir = (REPO / cfg["observation_publication"]).parent
    historic = read_jsonl(prior_dir / "representations.jsonl")
    fresh = read_jsonl(root / "recalibration_verified/representations.jsonl")
    if len(historic) != 128 or len(fresh) != 128:
        raise ValueError("cross-process calibration incomplete")
    for a, b in zip(historic, fresh):
        if any(a[k] != b[k] for k in ("pair_id", "cell", "path", "arrays")):
            raise ValueError("fresh observation differs from Note 0068")
        before, after = (
            REPO / cfg["observation_root"] / a["path"],
            root / "recalibration" / b["path"],
        )
        if sha(before) != a["sha256"] or sha(after) != b["sha256"]:
            raise ValueError("recalibration array sidecar changed")
        with (
            np.load(before, allow_pickle=False) as aa,
            np.load(after, allow_pickle=False) as bb,
        ):
            for site in SITES:
                if aa[site].tobytes() != bb[site].tobytes():
                    raise ValueError("historical representation bytes differ")
    return {
        "source_cells": 128,
        "representation_array_pairs_exact": 256,
        "verified_summary_sha256": sha(root / "recalibration_verified/summary.json"),
        "calibration_summary_sha256": sha(root / "recalibration/summary.json"),
    }


def packed_observation(root, job, records, obs_config):
    if len(records) != 1:
        raise ValueError("one registered projector call required")
    arrays = {}
    for site, value in zip(SITES, records[0]):
        if str(value.dtype) != obs_config["source_dtype"]:
            raise ValueError("native site dtype differs")
        arrays[site] = cal._to_float32_numpy(value)
    info = array_receipts(arrays, obs_config)
    words = {s: pack(a) for s, a in arrays.items()}
    path = root / array_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **words)
    return {
        "path": array_path(job),
        "sha256": sha(path),
        "arrays": info,
        "packed_dtype": "<u2",
    }


def capture_cell(root, job, runtime, projector, original, frozen, frozen_sha):
    rel, cfg = run_path(job), frozen["config"]
    path = root / rel
    old_path = REPO / frozen["release"]["prior_panel_root"] / rel
    old, old_audit = read(old_path), read(old_path.with_suffix(".input_audit.json"))
    manifest = Path(old_audit["context"]["manifest_path"])
    if (
        str(manifest) != old["manifest_path"]
        or sha(manifest) != frozen["manifest_sha256"][str(manifest)]
    ):
        raise ValueError("panel manifest differs")
    context = {
        "manifest_path": str(manifest),
        "manifest_sha256": sha(manifest),
        "frozen_specification_sha256": frozen_sha,
    }
    audit_path = path.with_suffix(".input_audit.json")
    if (
        type(projector).__call__ is not original
        or runtime["model"].mm_projector is not projector
    ):
        raise ValueError("qualified observed instance or method changed")
    with cal.prepared_input_audit(
        importlib.import_module("mlx_vlm.generate"),
        audit_path=audit_path,
        context=context,
        expected_pixel_sha256=old_audit["effective"]["pixel_values"]["sha256"],
        expected_sizes=cfg["expected_image_sizes"],
    ):
        with observe_projector(projector) as records:
            cal.run_cumulative_replay_probe(
                cal.CumulativeReplayRunConfig(
                    manifest_path=manifest,
                    output_path=path,
                    model_id=cfg["model_id"],
                    max_frames=1,
                    max_tokens=cfg["source_max_tokens"],
                    seed=cfg["source_seed"],
                    temperature=0,
                    probe_temperature=0,
                    probe_preset="default",
                    cache_summary_max_layers=None,
                    source_cache_only=True,
                    include_frame_artifacts=False,
                    cache_tensor_captures=tuple(
                        cal.parse_cache_tensor_capture_spec(t) for t in cal.TARGETS
                    ),
                ),
                mlx_runtime=runtime,
            )
    if type(projector).__call__ is not original:
        raise ValueError("observer restoration failed")
    current, audit = read(path), read(audit_path)
    cal.validate_capture(current, audit, old, old_audit["effective"], cfg)
    if audit["context"] != context:
        raise ValueError("prepared input context differs")
    hashes = {}
    for target in cal.TARGETS:
        c = cal.contract(old, target, cfg)
        a, b = (
            cal._validate_target_contract(r, p, c)
            for r, p in ((old, old_path), (current, path))
        )
        if a.shape != b.shape or a.dtype != b.dtype or a.tobytes() != b.tobytes():
            raise ValueError("Note 0060 selected-cache parity failed")
        hashes[target] = hashlib.sha256(b.tobytes()).hexdigest()
    rep = packed_observation(root, job, records, frozen["observation_config"])
    return {
        **job,
        "path": rel,
        "run_sha256": sha(path),
        "input_audit_sha256": sha(audit_path),
        "tensor_artifacts": current["stream_events"][0]["cache_tensor_artifacts"],
        "prior_run": str(old_path),
        "cache_array_sha256": hashes,
        "observer_calls": len(records),
        "observer_restored": True,
        "representation": rep,
        "pixel_values_sha256": audit["effective"]["pixel_values"]["sha256"],
    }


def run(mode, root, preflight=None):
    cal.require_offline()
    if root.exists():
        raise FileExistsError("fresh root required; no retry or resume")
    root.mkdir(parents=True)
    rows, attempted, stage = [], 0, "preflight"
    try:
        config = read(CONFIG)
        validate_config(config)
        free = check_space(root, disk_budget(config)["required_at_launch_bytes"])
        if mode == "preflight":
            base_root = root / "calibration_preflight"
            cal.run("preflight", base_root)
            base = read(base_root / "frozen_specification.json")
        elif mode == "run" and preflight:
            base = cal.qualify()
        else:
            raise ValueError("explicit run with committed-design preflight required")
        frozen = specification(base)
        write_json(root / "frozen_specification.json", frozen)
        frozen_sha = sha(root / "frozen_specification.json")
        if mode == "preflight":
            result = {
                "schema_version": 1,
                "analysis_kind": "representation_panel_preflight",
                "status": "PASS",
                "frozen_specification_sha256": frozen_sha,
                "available_bytes": free,
                "disk_budget": frozen["disk_budget"],
                "calibration_preflight_sha256": sha(base_root / "preflight.json"),
                "new_model_forwards": 0,
                "model_weights_loaded": False,
                "observer_installed": False,
            }
            write_json(root / "preflight.json", result)
            return result
        pre = read(preflight / "preflight.json")
        if (
            pre["status"] != "PASS"
            or pre["new_model_forwards"] != 0
            or pre["model_weights_loaded"]
            or pre["observer_installed"]
            or (preflight / "execution_failure.json").exists()
            or frozen != read(preflight / "frozen_specification.json")
            or pre["frozen_specification_sha256"] != frozen_sha
            or sha(preflight / "frozen_specification.json") != frozen_sha
            or pre["disk_budget"] != frozen["disk_budget"]
            or pre["available_bytes"]
            < frozen["disk_budget"]["required_at_launch_bytes"]
            or sha(preflight / "calibration_preflight/preflight.json")
            != pre["calibration_preflight_sha256"]
        ):
            raise ValueError("qualified panel preflight differs")
        write_json(
            root / "preflight_receipt.json",
            {"path": str(preflight), "sha256": sha(preflight / "preflight.json")},
        )
        stage = "recalibration"
        with retain_loaded_runtime() as loaded:
            cal.run(
                "calibrate", root / "recalibration", preflight / "calibration_preflight"
            )
        runtime = loaded[0]
        cal_pub.publish(root / "recalibration", root / "recalibration_verified")
        write_json(
            root / "recalibration_parity.json", recalibration_parity(frozen, root)
        )
        if cal.runtime_identity(runtime) != frozen["runtime_identity"]:
            raise ValueError("retained runtime identity differs")
        projector = runtime["model"].mm_projector
        original = type(projector).__call__
        write_json(
            root / "runtime_receipt.json",
            {
                "pid": os.getpid(),
                "model_loads": 1,
                "model_object_id": id(runtime["model"]),
                "processor_object_id": id(runtime["processor"]),
                "calibration_and_panel_use_exact_loader_return": True,
            },
        )
        print(
            "384 calibration contexts passed; 256 historical site arrays byte-exact",
            flush=True,
        )
        stage = "panel"
        for i, job in enumerate(frozen["panel_jobs"]):
            if i == 256:
                receipt = seal(rows, frozen["panel_jobs"])
                write_json(root / "reference_cells.json", rows)
                write_json(
                    root / "reference_seal.json",
                    {**receipt, "sha256": sha(root / "reference_cells.json")},
                )
                print("256 reference cells sealed; first test ordinal 641", flush=True)
            check_space(root, config["reserve_bytes"] + 8 * 2**20)
            attempted += 1
            write_json(
                root / "progress.json",
                {
                    "stage": stage,
                    "attempted_panel": attempted,
                    "completed_panel": len(rows),
                    "job": job,
                },
            )
            row = capture_cell(
                root, job, runtime, projector, original, frozen, frozen_sha
            )
            row["ordinal"] = 385 + i
            with (root / "cells.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
            rows.append(row)
            if len(rows) % 64 == 0:
                print(f"representation cells {len(rows)}/2048", flush=True)
        if len(rows) != 2048 or attempted != 2048:
            raise ValueError("panel coverage incomplete")
        cal.assert_frozen(frozen)
        cal._freeze_model_snapshot(
            frozen["config"]["model_id"], root / "model_snapshot_after.json"
        )
        if read(root / "model_snapshot_after.json") != frozen["model_snapshot"]:
            raise ValueError("model snapshot changed")
        result = {
            "schema_version": 1,
            "analysis_kind": "representation_panel_capture",
            "status": "COMPLETE",
            "calibration_contexts": 384,
            "panel_cells": 2048,
            "new_source_contexts": 2432,
            "site_arrays": 4096,
            "panel_cache_parity_pairs": 6144,
            "same_model_instance": True,
            "new_direct_probe_cells": 0,
            "receipts": {
                n: sha(root / n)
                for n in (
                    "frozen_specification.json",
                    "preflight_receipt.json",
                    "recalibration_parity.json",
                    "runtime_receipt.json",
                    "reference_cells.json",
                    "reference_seal.json",
                    "cells.jsonl",
                    "model_snapshot_after.json",
                )
            },
        }
        write_json(root / "capture_summary.json", result)
        return result
    except Exception as e:
        write_json(
            root / "execution_failure.json",
            {
                "schema_version": 1,
                "status": "FAILED",
                "stage": stage,
                "type": type(e).__name__,
                "message": str(e),
                "attempted_panel_cells": attempted,
                "completed_panel_cells": len(rows),
                "primary_results_withheld": True,
                "retry_performed": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path)
    a = parser.parse_args()
    print(
        json.dumps(
            run(
                a.mode,
                a.output_root.resolve(),
                a.preflight_root.resolve() if a.preflight_root else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
