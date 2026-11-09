"""Offline preflight and explicit release gates; never import a model runtime."""

import hashlib
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

from .qwen_calibration_contract import PLAN, storage_bound
from .stat_matched_source_storage import inside, read, sha, write

PLAN_SHA = "0b009c3c7496dd995e0de1cf25b75a793a8ae8ea93b2abacbb3516b3e269c189"
PREFLIGHT = "runs/qwen_observation_preflight_v1"
RELEASE = "examples/qwen_native_execution_release_v1.json"
JOBS = "examples/research_notes/0102_qwen_calibration_registration/jobs.json"


def registered(repo):
    if sha(inside(repo, PLAN)) != PLAN_SHA:
        raise ValueError("immutable calibration registration differs")
    plan = read(repo / PLAN)
    for path, digest in plan["sources"].items():
        if sha(inside(repo, path)) != digest:
            raise ValueError("registered source file differs")
    receipt = read(repo / "examples/research_notes/0102_qwen_calibration_registration/preparation_receipt.json")
    for path, digest in receipt["preparation_code_sha256"].items():
        if sha(inside(repo, path)) != digest:
            raise ValueError("preparation software differs")
    jobs = read(inside(repo, JOBS))["jobs"]
    if sha(repo / JOBS) != read(repo / "examples/research_notes/0102_qwen_calibration_registration/summary.json")["output_sha256"]["jobs.json"]:
        raise ValueError("registered job metadata differs")
    for job in jobs:
        if sha(inside(repo / plan["capture_root"], job["raw"]["path"])) != job["raw"]["sha256"]:
            raise ValueError("existing development source differs")
    return plan, jobs


def software(repo):
    paths = subprocess.check_output(["git", "ls-files", "src/fractal_vlm_state_probe/*.py",
                                     "scripts/run_qwen_observation_calibration.py"], cwd=repo, text=True).splitlines()
    required = {f"src/fractal_vlm_state_probe/qwen_calibration_{name}.py"
                for name in ("capture", "native", "verify", "admission")}
    required.add("scripts/run_qwen_observation_calibration.py")
    if not required.issubset(paths):
        raise ValueError("capture/verifier software must be committed before preflight")
    result = {}
    for path in paths:
        digest = sha(inside(repo, path))
        committed = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=repo)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError("uncommitted observation software")
        result[path] = digest
    return result


def environment(plan):
    from huggingface_hub import try_to_load_from_cache

    config = try_to_load_from_cache(plan["model_id"], "config.json", revision=plan["snapshot_revision"])
    if not isinstance(config, str):
        raise ValueError("pinned model snapshot not present offline")
    snapshot = Path(config).parent
    if snapshot.name != plan["snapshot_revision"]:
        raise ValueError("model snapshot revision differs")
    hub = snapshot.parent.parent.resolve()
    files = {}
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
        if not path.resolve().is_relative_to(hub):
            raise ValueError("snapshot link escapes pinned model cache")
        files[str(path.relative_to(snapshot))] = sha(path)
    if not any(p.endswith(".safetensors") for p in files):
        raise ValueError("offline weights unavailable")
    packages = {}
    for module, distribution in (("mlx_vlm", "mlx-vlm"), ("mlx_lm", "mlx-lm"),
                                 ("transformers", "transformers"), ("mlx", "mlx"),
                                 ("huggingface_hub", "huggingface-hub"), ("PIL", "Pillow"), ("numpy", "numpy")):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.submodule_search_locations:
            raise ValueError("required installed package unavailable")
        root = Path(next(iter(spec.submodule_search_locations)))
        hashes = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
                  if p.is_file() and p.suffix in (".py", ".so", ".dylib")}
        packages[module] = {"version": importlib.metadata.version(distribution), "root": str(root), "files": hashes}
    if packages["mlx_vlm"]["files"].get("generate.py") != plan["installed_generation_source"]["sha256"]:
        raise ValueError("native first-yield source differs")
    return {"snapshot": str(snapshot), "snapshot_files": files, "packages": packages}


def preflight(repo):
    plan, _ = registered(repo)
    root = inside(repo, PREFLIGHT)
    root.mkdir(parents=True, exist_ok=False)
    try:
        code = software(repo)
        env = environment(plan)
        free = shutil.disk_usage(repo).free
        storage_bound(plan)
        if free < plan["storage"]["required_launch_free_bytes"]:
            raise ValueError("calibration launch disk gate not met")
        value = {"status": "OFFLINE_PREFLIGHT_ONLY", "plan_sha256": PLAN_SHA,
                 "software": code, "environment": env, "free_bytes": free,
                 "model_contexts": 0, "processor_calls": 0, "native_execution_released": False}
        write(root / "preflight.json", value)
        return value
    except Exception as error:
        write(root / "failure.json", {"error": str(error), "retry_authorized": False})
        raise


def admit(repo, *, for_capture=True):
    # Check explicit release before importing even the offline snapshot locator.
    release = read(inside(repo, RELEASE))
    if (release.get("approved_for_native_execution") is not True
            or release.get("cohort_generation_authorized") is not False
            or release.get("scope") != "qwen_observation_calibration_v1_only"):
        raise ValueError("separate calibration-only execution release required")
    plan, jobs = registered(repo)
    root = inside(repo, PREFLIGHT)
    if (root / "failure.json").exists():
        raise ValueError("failed preflight cannot release acquisition")
    pf = read(root / "preflight.json")
    if (release["plan_sha256"] != PLAN_SHA or release["preflight_sha256"] != sha(root / "preflight.json")
            or release["software"] != pf["software"] or software(repo) != pf["software"]
            or pf["status"] != "OFFLINE_PREFLIGHT_ONLY" or pf["plan_sha256"] != PLAN_SHA):
        raise ValueError("release/preflight/software/runtime snapshot differs")
    if for_capture and pf["environment"] != environment(plan):
        raise ValueError("installed runtime or model snapshot changed")
    if for_capture and shutil.disk_usage(repo).free < plan["storage"]["required_launch_free_bytes"]:
        raise ValueError("launch disk gate no longer met")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return plan, jobs, {"release_sha256": sha(repo / RELEASE),
                        "preflight_sha256": sha(root / "preflight.json"), **pf}
