from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path

from fractal_vlm_state_probe.cache_tensor_artifact import (
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cli.capture_pairing_references import (
    _validate_target_contract,
)
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO
from fractal_vlm_state_probe.cli.run_controlled_overlap_cache_calibration import (
    CONFIG,
    contract,
    qualify,
    require_offline,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _read,
    _sha,
)
from fractal_vlm_state_probe.controlled_overlap_cache import (
    TARGETS,
    require_complete_calibration,
    validate_capture,
)
from fractal_vlm_state_probe.controlled_overlap_cache_panel import (
    ordered_jobs,
    require_reference_seal,
    validate_panel_capture,
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
from fractal_vlm_state_probe.stimulus import write_json

RELEASE = REPO / "configs/controlled_overlap_cache_release_v1.json"
CODE = [
    "src/fractal_vlm_state_probe/controlled_overlap_cache_panel.py",
    "src/fractal_vlm_state_probe/cli/run_controlled_overlap_cache_panel.py",
    "src/fractal_vlm_state_probe/cli/analyze_controlled_overlap_cache_panel.py",
    "scripts/run_controlled_overlap_cache_panel.py",
    "scripts/analyze_controlled_overlap_cache_panel.py",
]


def specification():
    frozen, release = qualify(), _read(RELEASE)
    if (
        _sha(CONFIG) != release["phase2_config_sha256"]
        or release["release_id"] != "controlled_overlap_cache_panel_v1"
        or release["resume_or_retry"]
        or not release["same_model_instance_for_calibration_and_panel"]
        or release["primary_required_q_sixteenths"] != [0, 4]
        or release["direction_pairs"] != [[0, 1], [0, 4]]
    ):
        raise ValueError("panel release differs from registered design")
    prior_path = REPO / release["prior_calibration_publication"]
    prior = _read(prior_path)
    if (
        _sha(prior_path) != release["prior_calibration_publication_sha256"]
        or prior["status"] != "PASS"
    ):
        raise ValueError("prior calibration publication changed")
    for name, digest in prior["frozen_specification"]["code_sha256"].items():
        if _sha(REPO / name) != digest:
            raise ValueError("qualified calibration code changed")
    frozen["source_sha256"][str(prior_path)] = _sha(prior_path)
    for name in [*CODE, str(RELEASE.relative_to(REPO))]:
        digest = _sha(REPO / name)
        if (
            hashlib.sha256(
                subprocess.check_output(
                    ["git", "show", f"{frozen['registered_design_commit']}:{name}"],
                    cwd=REPO,
                )
            ).hexdigest()
            != digest
        ):
            raise ValueError("commit panel implementation before execution")
        frozen["code_sha256"][name] = digest
    frozen.update(
        analysis_kind="controlled_overlap_cache_panel_registration",
        release=release,
        panel_jobs=ordered_jobs(frozen["hierarchy_metadata"]),
    )
    return frozen


def run(mode, root, preflight_root=None):
    require_offline()
    if root.exists():
        raise FileExistsError("fresh root required; no retry/resume")
    root.mkdir(parents=True)
    registry, attempted, checks = {}, {"calibration": 0, "panel": 0}, []
    stage = "preflight"
    try:
        frozen = specification()
        config = frozen["config"]
        if (
            shutil.disk_usage(root).free
            < config["panel"]["minimum_free_disk_gib"] * 2**30
        ):
            raise ValueError("insufficient disk for registered panel")
        write_json(root / "frozen_specification.json", frozen)
        frozen_sha = _sha(root / "frozen_specification.json")
        _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
        if _read(root / "model_snapshot.json") != frozen["model_snapshot"]:
            raise ValueError("model fingerprint differs")
        if mode == "preflight":
            result = {
                "schema_version": 1,
                "status": "PASS",
                "frozen_specification_sha256": frozen_sha,
                "model_snapshot_sha256": _sha(root / "model_snapshot.json"),
                "new_model_forwards": 0,
                "model_weights_loaded": False,
                "calibration_cells_planned": 128,
                "panel_cells_planned": 2048,
            }
            write_json(root / "preflight.json", result)
            return result
        if mode != "run" or preflight_root is None:
            raise ValueError("explicit run and qualified preflight required")
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
            raise ValueError("preflight identity differs")
        write_json(
            root / "preflight_receipt.json",
            {
                "path": str(preflight_root),
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
            raise ValueError("qualified runtime/processor changed")
        captures = tuple(parse_cache_tensor_capture_spec(t) for t in TARGETS)
        historical = {(j["pair_id"], j["cell"]): j for j in frozen["jobs"]}
        measured = {
            (r["map"], r["pair_id"]): r
            for r in _read(REPO / config["phase1_root"] / "input_summary.json")[
                "records"
            ]
        }

        def capture(job, kind):
            label, cell = job["pair_id"], job["cell"]
            h = historical[(label, cell)]
            old_path = Path(h["historical_run"])
            old, old_inputs = (
                _read(old_path),
                _read(old_path.with_suffix(".input_audit.json"))["effective"],
            )
            if kind == "calibration":
                manifest, pixel = Path(h["manifest_path"]), h["pixel_values_sha256"]
                relative = f"calibration/{label}_{cell}.json"
            else:
                record = measured[(job["map"], label)]
                artifact, cell_input = record["artifacts"][cell], record["cells"][cell]
                manifest, pixel = (
                    Path(artifact["manifest_path"]),
                    cell_input["pixel_values_sha256"],
                )
                relative = f"panel/{job['phase']}/{job['map']}/{label}_{cell}.json"
            if _sha(manifest) != frozen["manifest_sha256"][str(manifest)]:
                raise ValueError("capture manifest changed")
            path, audit_path = (
                root / relative,
                (root / relative).with_suffix(".input_audit.json"),
            )
            context = {
                "manifest_path": str(manifest),
                "manifest_sha256": _sha(manifest),
                "frozen_specification_sha256": frozen_sha,
            }
            attempted[kind] += 1
            write_json(
                root / "progress.json",
                {
                    "stage": kind,
                    "attempted": attempted,
                    "completed_cells": len(registry),
                    "current_path": str(path),
                    "current_job": job,
                },
            )
            with prepared_input_audit(
                generator,
                audit_path=audit_path,
                context=context,
                expected_pixel_sha256=pixel,
                expected_sizes=config["expected_image_sizes"],
            ):
                run_cumulative_replay_probe(
                    CumulativeReplayRunConfig(
                        manifest_path=manifest,
                        output_path=path,
                        model_id=config["model_id"],
                        max_frames=1,
                        max_tokens=2,
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
            if audit["context"] != context:
                raise ValueError("prepared-input provenance differs")
            if kind == "calibration":
                validate_capture(current, audit, old, old_inputs, config)
            else:
                validate_panel_capture(
                    current, audit, old, old_inputs, cell_input, artifact, config
                )
            for target in TARGETS:
                c = contract(old, target, config)
                after = _validate_target_contract(current, path, c)
                if kind == "calibration":
                    before = _validate_target_contract(old, old_path, c)
                    checks.append(
                        {
                            "pair_id": label,
                            "cell": cell,
                            "target": target,
                            "before_path": str(old_path),
                            "after_path": str(path),
                            "before_array_sha256": hashlib.sha256(
                                before.tobytes()
                            ).hexdigest(),
                            "after_array_sha256": hashlib.sha256(
                                after.tobytes()
                            ).hexdigest(),
                            **tensor_difference(before, after),
                        }
                    )
            if kind == "calibration" and not all(
                c["bitwise_equal"] for c in checks[-3:]
            ):
                write_json(
                    root / "calibration.json", {"status": "FAILED", "checks": checks}
                )
                raise ValueError("calibration mismatch; panel withheld")
            receipt = {
                "ordinal": len(registry) + 1,
                "kind": kind,
                **job,
                "path": relative,
                "run_sha256": _sha(path),
                "input_audit_sha256": _sha(audit_path),
                "historical_run": str(old_path),
                "tensor_artifacts": current["stream_events"][0][
                    "cache_tensor_artifacts"
                ],
            }
            registry[relative] = receipt
            with (root / "captures.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt, separators=(",", ":")) + "\n")
            return receipt

        stage = "calibration"
        for job in frozen["jobs"]:
            capture(job, "calibration")
            if len(registry) % 4 == 0:
                print(f"calibration {len(registry)}/128", flush=True)
        require_complete_calibration(checks, frozen["hierarchy_metadata"])
        write_json(
            root / "calibration.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "completed_cells": 128,
                "checks": checks,
            },
        )
        print("calibration PASS; same runtime entering registered panel", flush=True)
        stage, sealed = "panel", False
        for job in frozen["panel_jobs"]:
            if job["phase"] == "test" and not sealed:
                refs = require_reference_seal(registry, frozen["hierarchy_metadata"])
                write_json(root / "frozen_panel_references.json", refs)
                write_json(
                    root / "reference_seal.json",
                    {
                        "sha256": _sha(root / "frozen_panel_references.json"),
                        "reference_cells": 256,
                        "completed_cells_at_seal": len(registry),
                        "first_test_ordinal": len(registry) + 1,
                    },
                )
                sealed = True
                print("all 256 reference cells sealed before test capture", flush=True)
            capture(job, "panel")
            if attempted["panel"] % 4 == 0:
                print(
                    f"panel {attempted['panel']}/2048 {job['phase']} {job['map']} {job['pair_id']}",
                    flush=True,
                )
        if (
            len(registry) != 2176
            or attempted != {"calibration": 128, "panel": 2048}
            or not sealed
        ):
            raise ValueError("capture denominator differs")
        write_json(root / "capture_registry.json", registry)
        _freeze_model_snapshot(config["model_id"], root / "model_snapshot.json")
        for group in ("code_sha256", "source_sha256", "manifest_sha256"):
            for name, digest in frozen[group].items():
                if (
                    _sha(REPO / name if group == "code_sha256" else Path(name))
                    != digest
                ):
                    raise ValueError("frozen evidence changed during capture")
        if (
            _sha(root / "frozen_panel_references.json")
            != _read(root / "reference_seal.json")["sha256"]
        ):
            raise ValueError("frozen reference registry changed")
        result = {
            "schema_version": 1,
            "analysis_kind": "controlled_overlap_cache_panel_capture",
            "status": "COMPLETE",
            "calibration_source_cells": 128,
            "calibration_tensor_comparisons": 384,
            "panel_source_cells": 2048,
            "panel_tensor_sidecars": 6144,
            "panel_factorial_blocks": 512,
            "new_direct_probe_cells": 0,
            "same_model_instance_for_calibration_and_panel": True,
            "pid": os.getpid(),
            "confirmation_cohort": "not_generated_not_sealed",
            "receipts": {
                n: _sha(root / n)
                for n in (
                    "frozen_specification.json",
                    "model_snapshot.json",
                    "runtime_identity.json",
                    "calibration.json",
                    "capture_registry.json",
                    "captures.jsonl",
                    "frozen_panel_references.json",
                    "reference_seal.json",
                    "preflight_receipt.json",
                )
            },
        }
        write_json(root / "capture_summary.json", result)
        return result
    except Exception as error:
        write_json(root / "capture_registry.json", registry)
        write_json(
            root / "execution_failure.json",
            {
                "schema_version": 1,
                "status": "FAILED",
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "attempted": attempted,
                "completed_cells": len(registry),
                "retry_performed": False,
                "primary_results_withheld": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
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
        )
    )


if __name__ == "__main__":
    main()
