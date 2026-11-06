"""File-only audit of the zero-forward disk gate; never reads calibration score vectors."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/first_token_prompt_panel_v1.json"


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def audit(destination, replay_snapshot=None):
    cfg = read(ROOT / CONFIG)
    pre = ROOT / cfg["preflight_root"]
    failure = read(pre / "failure.json")
    expected_failure = (
        "41425d1837b6fbe219dba834a6b0d8e8571813b183313ca855645f55107b051e"
    )
    if sha(pre / "failure.json") != expected_failure:
        raise ValueError("original preflight failure changed")
    if (
        failure["mode"] != "preflight"
        or failure["attempted"]
        or failure["completed"]
        or failure["calibrated_rows"]
        or failure["panel_contexts"]
        or failure["retry"]
        or failure["type"] != "ValueError"
        or failure["message"] != "insufficient registered disk reserve"
        or sorted(p.name for p in pre.iterdir()) != ["failure.json"]
        or (ROOT / cfg["capture_root"]).exists()
    ):
        raise ValueError("not the unstarted storage-gate failure")
    previous_path = ROOT / cfg["qualification_root"] / "summary.json"
    if sha(previous_path) != cfg["qualification_summary_sha256"]:
        raise ValueError("preceding calibration publication changed")
    previous = read(previous_path)
    newfiles = (
        CONFIG,
        "src/fractal_vlm_state_probe/first_token_panel.py",
        "src/fractal_vlm_state_probe/cli/run_first_token_prompt_panel.py",
        "scripts/run_first_token_prompt_panel.py",
        "tests/test_first_token_prompt_panel.py",
        "docs/research_notes/0081_first_token_prompt_panel_registration.md",
    )
    commit = (
        subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", *newfiles], cwd=ROOT
        )
        .decode()
        .strip()
    )
    hashes = {
        **previous["code_sha256"],
        **{name: sha(ROOT / name) for name in newfiles},
    }
    for name, digest in hashes.items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if sha(ROOT / name) != digest or hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("frozen registration/code changed after disk rejection")
    need = cfg["reserve_bytes"] + cfg["launch_allowance_bytes"]
    if replay_snapshot is None:
        free = shutil.disk_usage(ROOT).free
        snapshot = {
            "schema_version": 1,
            "analysis_kind": "post_preflight_storage_observation",
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "observed_after_failure": True,
            "free_bytes": free,
            "required_bytes": need,
            "shortfall_bytes": max(0, need - free),
            "free_at_failure_bytes": None,
            "note": "The original failure did not record free bytes; this is a later observation.",
        }
    else:
        snapshot = read(replay_snapshot)
        if (
            snapshot["required_bytes"] != need
            or not snapshot["observed_after_failure"]
            or snapshot["free_at_failure_bytes"] is not None
            or snapshot["shortfall_bytes"] != max(0, need - snapshot["free_bytes"])
        ):
            raise ValueError("storage snapshot does not describe this registered gate")
    write(destination / "storage_snapshot.json", snapshot)
    result = {
        "schema_version": 1,
        "analysis_kind": "first_token_prompt_panel_preflight_failure",
        "status": "NOT_RUN",
        "registered_commit": commit,
        "code_sha256": hashes,
        "failure_sha256": expected_failure,
        "failure": failure,
        "storage_snapshot_sha256": sha(destination / "storage_snapshot.json"),
        "storage_snapshot": snapshot,
        "budget_bytes": {
            key: cfg[key]
            for key in (
                "calibration_payload_bytes",
                "panel_payload_bytes",
                "overhead_allowance_bytes",
                "reserve_bytes",
                "launch_allowance_bytes",
            )
        },
        "required_free_bytes": need,
        "model_weights_loaded": False,
        "new_contexts": 0,
        "new_calibration_contexts": 0,
        "new_panel_contexts": 0,
        "planned_contexts": 352,
        "planned_pairs": 32,
        "completed_pairs": 0,
        "effect_estimate": None,
        "portable_score_archive_created": False,
        "audit_model_calls": 0,
        "audit_score_vectors_read": 0,
        "retry": False,
        "confirmation_cohort": "not_generated_not_sealed",
    }
    write(destination / "summary.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--replay-snapshot", type=Path)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    result = audit(args.output_root, args.replay_snapshot)
    print(result["status"], args.output_root)
