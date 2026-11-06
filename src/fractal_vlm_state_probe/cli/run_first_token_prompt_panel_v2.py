"""New acquisition namespace after the retained v1 zero-forward disk rejection."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

from fractal_vlm_state_probe.cli import run_first_token_prompt_panel as v1
from fractal_vlm_state_probe.native_suffix_replay import replaced

CFG = "configs/first_token_prompt_panel_v2.json"
V1_CFG, V1_CODE, V1_QUALIFY, V1_REQUIRE = v1.CFG, v1.CODE, v1.qualify, v1.require_config
CODE = (
    *V1_CODE,
    CFG,
    "src/fractal_vlm_state_probe/cli/run_first_token_prompt_panel_v2.py",
    "scripts/run_first_token_prompt_panel_v2.py",
    "tests/test_first_token_prompt_panel_v2.py",
    "docs/research_notes/0083_first_token_prompt_panel_v2_registration.md",
)


def same_design(original, current):
    allowed = {
        "study_id",
        "preflight_root",
        "capture_root",
        "parent_commit",
        "retained_v1_failure_sha256",
        "retained_v1_audit_sha256",
        "retained_v1_commit",
        "storage_recovery_receipt",
        "storage_recovery_sha256",
    }
    if (
        {k: v for k, v in original.items() if k not in allowed}
        != {k: v for k, v in current.items() if k not in allowed}
        or current["study_id"] != "first_token_prompt_panel_v2"
        or current["preflight_root"] != "runs/first_token_prompt_preflight_v2"
        or current["capture_root"] != "runs/first_token_prompt_panel_v2"
        or current["parent_commit"] != "bdebbd6e63831b8588d9706e08dc18bc227730ae"
        or current["retained_v1_commit"] != current["parent_commit"]
    ):
        raise ValueError(
            "v2 changed a scientific condition, storage budget or fixed release namespace"
        )
    V1_REQUIRE(original)


def require_config(cfg):
    same_design(v1.base._read(v1.REPO / V1_CFG), cfg)


def qualify(root):
    cfg = v1.base._read(v1.REPO / CFG)
    original = v1.base._read(v1.REPO / V1_CFG)
    same_design(original, cfg)
    failed = v1.REPO / original["preflight_root"] / "failure.json"
    audit = (
        v1.REPO
        / "examples/research_notes/0082_first_token_prompt_preflight_stop/summary.json"
    )
    v1.prior.checked(failed, cfg["retained_v1_failure_sha256"])
    recovery = v1.base._read(
        v1.prior.checked(
            v1.REPO / cfg["storage_recovery_receipt"], cfg["storage_recovery_sha256"]
        )
    )
    if (
        recovery["status"] != "COMPLETE"
        or recovery["approved_scope"] != ["pip/http-v2", "pip/http"]
        or any(v["exists"] for v in recovery["after_http_caches"].values())
        or recovery["locally_built_wheels_preserved"] != 13
        or any(
            recovery[k]
            for k in (
                "installed_packages_modified",
                "model_cache_modified",
                "research_runs_modified",
            )
        )
    ):
        raise ValueError("approved HTTP-only storage recovery receipt differs")
    retained = v1.base._read(v1.prior.checked(audit, cfg["retained_v1_audit_sha256"]))
    if (
        retained["effect_estimate"] is not None
        or retained["new_contexts"] != 0
        or retained["failure"] != v1.base._read(failed)
        or (v1.REPO / original["capture_root"]).exists()
    ):
        raise ValueError(
            "v1 acquired data or no longer matches its retained zero-forward stop"
        )
    spec, old, anchors, known, checks = V1_QUALIFY(root)
    if any(spec["code_sha256"].get(k) != h for k, h in retained["code_sha256"].items()):
        raise ValueError("a v1 frozen file changed")
    v1.prior.git(
        "merge-base",
        "--is-ancestor",
        cfg["retained_v1_commit"],
        spec["registered_commit"],
    )
    spec["retained_v1_failure_sha256"] = cfg["retained_v1_failure_sha256"]
    spec["retained_v1_audit_sha256"] = cfg["retained_v1_audit_sha256"]
    return spec, old, anchors, known, checks


def run(mode, root):
    # Only release/config binding changes; all capture, effect and verification code is v1.
    with ExitStack() as stack:
        for key, value in (
            ("CFG", CFG),
            ("CODE", CODE),
            ("require_config", require_config),
            ("qualify", qualify),
        ):
            stack.enter_context(replaced(v1, key, value))
        return v1.run(mode, root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preflight", "capture", "verify"))
    p.add_argument("--output-root", required=True, type=Path)
    args = p.parse_args()
    result = run(args.mode, args.output_root)
    print(f"{args.mode}: {result['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
