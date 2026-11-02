from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cli.run_controlled_overlap import (
    BASELINE,
    REPO,
    load_maps,
    qualified_processor,
    qualified_sources,
    read,
)
from fractal_vlm_state_probe.nonlearned_processing import FEATURES, STARTS
from fractal_vlm_state_probe.nonlearned_processing_calibration import (
    synthetic_calibration,
)
from fractal_vlm_state_probe.padding_policy import _checked_artifact, padding_gate
from fractal_vlm_state_probe.stimulus import sha256_file, write_json

CONFIG = REPO / "configs/nonlearned_processing_phase3_v1.json"
CODE = (
    "configs/nonlearned_processing_phase3_v1.json",
    "src/fractal_vlm_state_probe/nonlearned_processing.py",
    "src/fractal_vlm_state_probe/nonlearned_processing_calibration.py",
    "src/fractal_vlm_state_probe/cli/run_nonlearned_processing_calibration.py",
    "scripts/run_nonlearned_processing_calibration.py",
)


def assert_hashes(frozen):
    for group in ("code_sha256", "source_sha256"):
        for name, digest in frozen[group].items():
            path = REPO / name if group == "code_sha256" else Path(name)
            if sha256_file(path) != digest:
                raise ValueError(f"frozen {group} changed: {name}")


def specification():
    config = read(CONFIG)
    if config["features"] != list(FEATURES) or config["starts"] != list(STARTS):
        raise ValueError("feature registration differs")
    publications, publication_sources = {}, {}
    for phase in ("phase1", "phase2"):
        path = REPO / config[f"{phase}_publication"]
        if sha256_file(path) != config[f"{phase}_publication_sha256"]:
            raise ValueError("qualified publication changed")
        public = read(path)
        publication_sources[str(path)] = sha256_file(path)
        for name, receipt in public["ledgers"].items():
            if sha256_file(path.parent / name) != receipt["sha256"]:
                raise ValueError("qualified publication ledger changed")
            publication_sources[str(path.parent / name)] = receipt["sha256"]
        publications[phase] = public
    phase1, phase2 = publications["phase1"], publications["phase2"]
    study_path = REPO / config["phase1_root"] / "input_summary.json"
    study = read(study_path)
    if (
        sha256_file(study_path) != phase1["input_summary_sha256"]
        or study["frozen_specification"] != phase1["frozen_specification"]
    ):
        raise ValueError("qualified input receipt changed")
    if not study["cache_panel_eligible"] or phase2["status"] != "COMPLETE":
        raise ValueError("upstream input/cache stage incomplete")
    frozen = study["frozen_specification"]
    if sha256_file(Path(frozen["map_report"])) != frozen["map_report_sha256"]:
        raise ValueError("coordinate report changed")
    report, maps = load_maps(Path(frozen["map_report"]))
    baseline, prior, _, metadata, sources = qualified_sources()
    sources.update(publication_sources)
    if (
        metadata != frozen["hierarchy_metadata"]
        or metadata != phase2["frozen_specification"]["hierarchy_metadata"]
    ):
        raise ValueError("registered reference/test hierarchy changed")
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    if runtime != frozen["runtime"]:
        raise ValueError("qualified processor runtime changed")
    rows = study["records"]
    if len(rows) != 896 or {(r["map"], r["pair_id"]) for r in rows} != {
        (key, label) for key in maps for label in metadata
    }:
        raise ValueError("all-cell input coverage differs")
    for row in rows:
        if (
            set(row["cells"]) != {"mm", "jj", "mj", "jm"}
            or set(row["artifacts"]) != set(row["cells"])
            or not row["gate"]["accepted"]
            or padding_gate(row["cells"]) != row["gate"]
        ):
            raise ValueError("all-cell input eligibility changed")
        for c, artifact in row["artifacts"].items():
            rgb, manifest = _checked_artifact(artifact)
            if list(rgb.shape) != config["raw_shape_hwc"] or any(
                row[k] != v for k, v in metadata[row["pair_id"]].items()
            ):
                raise ValueError("source geometry or metadata changed")
            m = Path(artifact["manifest_path"])
            sources[str(m)] = sha256_file(m)
            sources[str((m.parent / manifest["frames"][0]["path"]).resolve())] = (
                artifact["frame_sha256"]
            )
    jobs = []
    for record in sorted(prior["records"], key=lambda r: r["pair_id"]):
        anchor = record["maps"]["historical_3q"]
        for cell in ("mm", "jj", "mj", "jm"):
            artifact, measured = anchor["artifacts"][cell], anchor["cells"][cell]
            _, manifest = _checked_artifact(artifact)
            m = Path(artifact["manifest_path"])
            sources[str(m)] = sha256_file(m)
            sources[str((m.parent / manifest["frames"][0]["path"]).resolve())] = (
                artifact["frame_sha256"]
            )
            jobs.append(
                {
                    "pair_id": record["pair_id"],
                    "cell": cell,
                    "artifact": artifact,
                    "pixel_values_sha256": measured["pixel_values_sha256"],
                    "processor_non_pixel_payload": measured[
                        "processor_non_pixel_payload"
                    ],
                }
            )
    if len(jobs) != config["calibration"]["historical_processor_cells"]:
        raise ValueError("historical processor calibration coverage differs")
    for path in (
        study_path,
        Path(frozen["map_report"]),
        REPO / config["phase1_publication"],
        REPO / config["phase2_publication"],
        BASELINE,
    ):
        sources[str(path)] = sha256_file(path)
    code = dict(phase2["frozen_specification"]["code_sha256"])
    code.update({name: sha256_file(REPO / name) for name in CODE})
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    for name, digest in code.items():
        stored = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
        if hashlib.sha256(stored).hexdigest() != digest:
            raise ValueError("commit current measurement/config before calibration")
    result = {
        "schema_version": 1,
        "analysis_kind": "nonlearned_processing_registration",
        "config": config,
        "registered_design_commit": commit,
        "code_sha256": code,
        "source_sha256": sources,
        "hierarchy_metadata": metadata,
        "runtime": runtime,
        "processor_provenance": baseline["frozen_specification"][
            "processor_provenance"
        ],
        "map_report_sha256": frozen["map_report_sha256"],
        "maps_verified": len(report["maps"]),
        "input_blocks_verified": len(rows),
        "processor_jobs": jobs,
    }
    assert_hashes(result)
    return result


def run(mode, root, preflight_root=None):
    if any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        raise ValueError("offline flags required before execution")
    if root.exists():
        raise FileExistsError("fresh output root required; no retry or resume")
    root.mkdir(parents=True)
    calls, checks = 0, []
    try:
        frozen = specification()
        if (
            shutil.disk_usage(root).free
            < frozen["config"]["calibration"]["minimum_free_disk_gib"] * 2**30
        ):
            raise ValueError("insufficient calibration disk reserve")
        write_json(root / "frozen_specification.json", frozen)
        frozen_sha = sha256_file(root / "frozen_specification.json")
        if mode == "preflight":
            result = {
                "schema_version": 1,
                "analysis_kind": "nonlearned_processing_preflight",
                "status": "PASS",
                "frozen_specification_sha256": frozen_sha,
                "input_blocks_verified": 896,
                "input_cells_verified": 3584,
                "new_processor_calls": 0,
                "new_model_forwards": 0,
                "processor_instantiated": False,
                "model_weights_loaded": False,
                "experimental_panel_released": False,
            }
            write_json(root / "preflight.json", result)
            return result
        if mode != "calibrate" or preflight_root is None:
            raise ValueError("calibration requires explicit passing preflight")
        preflight = read(preflight_root / "preflight.json")
        if (
            preflight["status"] != "PASS"
            or (preflight_root / "execution_failure.json").exists()
            or read(preflight_root / "frozen_specification.json") != frozen
            or sha256_file(preflight_root / "frozen_specification.json")
            != preflight["frozen_specification_sha256"]
        ):
            raise ValueError("preflight differs from frozen calibration")
        write_json(
            root / "preflight_receipt.json",
            {
                "path": str(preflight_root),
                "sha256": sha256_file(preflight_root / "preflight.json"),
            },
        )
        synthetic, arrays = synthetic_calibration(frozen["config"])
        np.savez(root / "synthetic_features.npz", **arrays)
        write_json(root / "synthetic_calibration.json", synthetic)
        baseline, _, _, _, _ = qualified_sources()
        processor, _ = qualified_processor(baseline)
        for job in frozen["processor_jobs"]:
            calls += 1
            write_json(
                root / "progress.json",
                {
                    "attempted_processor_calls": calls,
                    "completed_processor_checks": len(checks),
                    "current_job": job,
                },
            )
            rgb, _ = _checked_artifact(job["artifact"])
            payload = processor(
                images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
            )
            pixels = np.asarray(payload["pixel_values"])
            non_pixel = {
                k: np.asarray(v).tolist()
                for k, v in payload.items()
                if k != "pixel_values"
            }
            if (
                pixels.dtype != np.float32
                or list(pixels.shape) != frozen["config"]["processor_shape_nchw"]
                or not np.isfinite(pixels).all()
                or sha256_array(pixels) != job["pixel_values_sha256"]
                or non_pixel != job["processor_non_pixel_payload"]
            ):
                raise ValueError(
                    "historical processor byte/metadata calibration differs"
                )
            checks.append(
                {
                    "pair_id": job["pair_id"],
                    "cell": job["cell"],
                    "pixel_values_sha256": sha256_array(pixels),
                    "shape": list(pixels.shape),
                    "dtype": str(pixels.dtype),
                    "processor_non_pixel_payload": non_pixel,
                    "manifest_path": job["artifact"]["manifest_path"],
                    "manifest_sha256": job["artifact"]["manifest_sha256"],
                    "exact": True,
                }
            )
            with (root / "processor_checks.jsonl").open("a") as handle:
                handle.write(json.dumps(checks[-1], separators=(",", ":")) + "\n")
            if calls % 16 == 0:
                print(f"historical processor calibration {calls}/128", flush=True)
        if len(checks) != 128 or calls != 128:
            raise ValueError("processor calibration denominator differs")
        assert_hashes(frozen)
        result = {
            "schema_version": 1,
            "analysis_kind": "nonlearned_processing_calibration",
            "status": "PASS",
            "registered_design_commit": frozen["registered_design_commit"],
            "synthetic_feature_oracle_pairs": len(synthetic["feature_checks"]),
            "maximum_feature_oracle_abs_error": synthetic[
                "maximum_feature_oracle_abs_error"
            ],
            "historical_processor_checks": len(checks),
            "new_processor_calls": calls,
            "new_model_forwards": 0,
            "model_weights_loaded": False,
            "experimental_source_cells_scored": 0,
            "experimental_panel_released": False,
            "encoder_or_projector_released": False,
            "confirmation_cohort": "not_generated_not_sealed",
            "receipts": {
                p: sha256_file(root / p)
                for p in (
                    "frozen_specification.json",
                    "preflight_receipt.json",
                    "synthetic_features.npz",
                    "synthetic_calibration.json",
                    "processor_checks.jsonl",
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
                "type": type(error).__name__,
                "message": str(error),
                "attempted_processor_calls": calls,
                "completed_processor_checks": len(checks),
                "experimental_results_withheld": True,
            },
        )
        raise


def sha256_array(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Registered non-learned baseline calibration; no image scoring or model forwards."
    )
    parser.add_argument("mode", choices=("preflight", "calibrate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path)
    args = parser.parse_args()
    result = run(
        args.mode,
        args.output_root.resolve(),
        args.preflight_root.resolve() if args.preflight_root else None,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
