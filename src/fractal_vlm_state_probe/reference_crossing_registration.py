"""Metadata-only preparation of a post-hoc crossing plan; never loads arrays."""

from pathlib import Path

from .stat_matched_source_storage import inside, ledger, read, sha, write

PUBLIC = "examples/research_notes/0096_reference_crossing_registration"
PLAN = PUBLIC + "/plan.json"
PLAN_SHA256 = "37d2e3554c5c43548b8a1598186464ae4f3f46ab8698f8dd8fcd62b2deb54088"
PARENT = "examples/research_notes/0093_stat_matched_source_results/"
CAPTURE = "runs/stat_matched_source_capture_v1"
VIEWS = {
    "encoder": ([1, 256, 3072], "uint16"),
    "projector": ([1, 256, 896], "uint16"),
    "full_raw_rgb": ([320, 320, 3], "uint8"),
    "full_processor_pixel_values": ([1, 3, 1024, 1024], "float32"),
}
IMPLEMENTATION = [
    PLAN,
    "src/fractal_vlm_state_probe/reference_crossing_registration.py",
    "src/fractal_vlm_state_probe/cli/prepare_reference_crossing.py",
    "scripts/prepare_reference_crossing.py",
    "tests/test_reference_crossing_registration.py",
]


def validate_plan(plan):
    """Require the complete registered contract, including false/null types."""
    path = Path(__file__).resolve().parents[2] / PLAN
    if sha(path) != PLAN_SHA256:
        raise ValueError("registered plan bytes differ")
    expected = read(path)
    # The caller may not weaken a plan at runtime via a replacement object.
    if plan != expected:
        raise ValueError("registered plan differs")
    fixed = {
        "study_id": "reference_crossing_v1",
        "status": "REGISTERED_PREPARATION_ONLY_NOT_RELEASED",
        "prior_outcomes_known": True,
        "not_preregistered_before_note0093": True,
        "approved_for_score_execution": False,
        "capture_root": CAPTURE,
        "views": list(VIEWS),
        "relative_floors": [1e-8, 1e-6, 1e-4],
        "primary_relative_floor": 1e-6,
        "profiles": 12,
        "reference_pairs": 16,
        "test_pairs": 16,
        "families": 8,
        "cells": ["mm", "mj", "jm", "jj"],
        "reference_shifts": [0, 1, 2, 3],
        "test_shifts": [0, 1, 2, 3],
        "selected_arrays": 2048,
        "reference_cancellation_threshold": 1e-12,
        "both_reference_replicates_required": True,
        "absolute_check_tolerance": 1e-12,
        "relative_check_tolerance": 0,
        "score_shape_per_profile": [4, 4, 16, 8],
        "new_hypothesis_tests": False,
        "p_values": None,
        "confidence_intervals": None,
        "equivalence_tested": False,
        "source_generation": 0,
        "processor_calls": 0,
        "model_contexts": 0,
        "preparation_payload_reads": 0,
        "implementation_files": IMPLEMENTATION,
        "preparation_root": "runs/reference_crossing_preparation_v1",
    }
    for key, value in fixed.items():
        if type(plan.get(key)) is not type(value) or plan[key] != value:
            raise ValueError(f"registered boundary differs: {key}")
    sources = {CAPTURE + "/" + n for n in (
        "capture.json", "frozen.json", "models.jsonl", "candidates.jsonl"
    )} | {PARENT + n for n in (
        "summary.json", "profiles.jsonl", "verification_receipts.json"
    )} | {"configs/stat_matched_source_panel_v1.json",
         "configs/stat_matched_source_acquisition_release_v1.json"}
    if set(plan["sources"]) != sources:
        raise ValueError("complete pinned metadata inventory required")
    return plan


def selected_inventory(repo, spec, capture, models, candidates):
    """Bind all 2,048 receipts without reading or hashing array payloads."""
    pairs = spec["sources"]["pairs"]
    expected = {(p["pair_id"], s, c): p["split"]
                for p in pairs for s in range(4) for c in ("mm", "mj", "jm", "jj")}
    if len(expected) != 512:
        raise ValueError("complete 32-pair source design required")
    result, used_paths = [], set()
    for kind, rows in (("model", models), ("candidate", candidates)):
        seen = set()
        for row in rows:
            if kind == "model" and row["phase"] == "sham":
                continue
            key = (row["pair_id"], row["shift"], row["cell"])
            if (key not in expected or key in seen
                    or row["phase" if kind == "model" else "split"] != expected[key]):
                raise ValueError("duplicate, missing or wrong-split cell identity")
            seen.add(key)
            receipts = row["sites"] if kind == "model" else {
                "full_raw_rgb": row["raw"],
                "full_processor_pixel_values": row["pixels"],
            }
            names = {"encoder", "projector"} if kind == "model" else {
                "full_raw_rgb", "full_processor_pixel_values"}
            if set(receipts) != names:
                raise ValueError("selected view coverage differs")
            for view in sorted(receipts):
                receipt = receipts[view]
                path = inside(repo / CAPTURE, receipt["path"])
                if (receipt["path"] in used_paths or not path.is_file()
                        or (receipt["shape"], receipt["dtype"]) != VIEWS[view]
                        or capture["receipts"].get(receipt["path"]) != receipt["sha256"]):
                    raise ValueError("array receipt shape/path/hash coverage differs")
                for name in ("sha256", "word_sha256"):
                    digest = receipt[name]
                    if (not isinstance(digest, str) or len(digest) != 64
                            or any(c not in "0123456789abcdef" for c in digest)):
                        raise ValueError("array receipt needs SHA256 digests")
                used_paths.add(receipt["path"])
                result.append({"pair_id": key[0], "shift": key[1], "cell": key[2],
                               "split": expected[key], "view": view, "receipt": receipt})
        if seen != set(expected):
            raise ValueError("complete cell inventory required")
    if len(result) != 2048:
        raise ValueError("all selected arrays required")
    return sorted(result, key=lambda r: (r["pair_id"], r["shift"], r["cell"], r["view"]))


def qualify(repo):
    repo = Path(repo)
    plan = validate_plan(read(repo / PLAN))
    for path, digest in plan["sources"].items():
        if sha(inside(repo, path)) != digest:
            raise ValueError(f"pinned source differs: {path}")
    release = read(repo / "configs/stat_matched_source_acquisition_release_v1.json")
    if len(release["code_sha256"]) != 135:
        raise ValueError("complete inherited freeze required")
    for path, digest in release["code_sha256"].items():
        if sha(inside(repo, path)) != digest:
            raise ValueError(f"inherited implementation differs: {path}")
    parent = read(repo / (PARENT + "summary.json"))
    receipts = read(repo / (PARENT + "verification_receipts.json"))
    capture = read(repo / CAPTURE / "capture.json")
    if (parent["status"] != "DESCRIPTIVE_PROFILES_VERIFIED"
            or parent["capture_sha256"] != sha(repo / CAPTURE / "capture.json")
            or receipts["first_summary"] != receipts["second_summary"]
            or receipts["first_summary"]["status"] != "VERIFIED"
            or receipts["first_summary"]["capture_sha256"] != parent["capture_sha256"]
            or receipts["first_tree_sha256"] != receipts["second_tree_sha256"]
            or capture["status"] != "COMPLETE_UNVERIFIED"):
        raise ValueError("qualified parent evidence differs")
    for name in ("frozen.json", "models.jsonl", "candidates.jsonl"):
        if capture["receipts"][name] != sha(repo / CAPTURE / name):
            raise ValueError("capture metadata receipt differs")
    spec = read(repo / "configs/stat_matched_source_panel_v1.json")
    inventory = selected_inventory(repo, spec, capture,
                                   ledger(repo / CAPTURE, "models.jsonl"),
                                   ledger(repo / CAPTURE, "candidates.jsonl"))
    return plan, inventory


def prepare(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    plan = validate_plan(read(repo / PLAN))
    if output != (repo / plan["preparation_root"]).resolve():
        raise ValueError("canonical metadata preparation root required")
    inside(repo, plan["preparation_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        plan, inventory = qualify(repo)
        write(output / "plan.json", plan)
        write(output / "selected_inventory.json", {
            "schema_version": 1, "analysis_kind": "reference_crossing_input_inventory",
            "arrays": inventory,
        })
        result = {
            "schema_version": 1,
            "analysis_kind": "reference_crossing_metadata_preparation",
            "status": "METADATA_PREPARED_NOT_SCORE_RELEASED",
            "plan_sha256": sha(repo / PLAN),
            "implementation_sha256": {p: sha(repo / p) for p in IMPLEMENTATION},
            "inventory_sha256": sha(output / "selected_inventory.json"),
            "selected_arrays": len(inventory),
            "inherited_files_exact": 135,
            "array_payloads_read": 0,
            "array_payload_hashes_verified_now": False,
            "new_scores": 0, "source_generation": 0, "processor_calls": 0,
            "model_contexts": 0, "score_execution_released": False,
            "committed_implementation_checked": False,
        }
        write(output / "summary.json", result)
        return result
    except Exception as error:
        write(output / "failure.json", {"error": str(error), "retry_authorized": False})
        raise
