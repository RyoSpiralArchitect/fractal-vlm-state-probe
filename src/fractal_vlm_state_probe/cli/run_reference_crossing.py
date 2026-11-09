"""One committed saved-array run and two independent native-content verifications."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe import reference_crossing as core
from fractal_vlm_state_probe import reference_crossing_oracle as oracle
from fractal_vlm_state_probe import reference_crossing_registration as registration
from fractal_vlm_state_probe import stat_matched_source as parent_core
from fractal_vlm_state_probe.stat_matched_source_storage import (
    append,
    inside,
    ledger,
    load_array,
    read,
    sha,
    write,
)

REPO = Path(__file__).resolve().parents[3]
RELEASE = "examples/research_notes/0097_reference_crossing_execution/release.json"


def tree(root):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("symlinks are not execution artifacts")
    return {
        str(p.relative_to(root)): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def committed(repo, release):
    names = release["implementation_files"]
    expected = [
        registration.PLAN,
        RELEASE,
        "src/fractal_vlm_state_probe/reference_crossing_registration.py",
        "src/fractal_vlm_state_probe/reference_crossing.py",
        "src/fractal_vlm_state_probe/reference_crossing_oracle.py",
        "src/fractal_vlm_state_probe/cli/run_reference_crossing.py",
        "scripts/run_reference_crossing.py",
        "tests/test_reference_crossing.py",
    ]
    if names != expected:
        raise ValueError("complete execution implementation inventory required")
    commit = (
        subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", *names], cwd=repo
        )
        .decode()
        .strip()
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", release["parent_commit"], commit],
        cwd=repo,
        check=True,
    )
    hashes = {}
    for name in names:
        digest = sha(repo / name)
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("commit the complete execution code before scoring")
        hashes[name] = digest
    return {"implementation_commit": commit, "implementation_sha256": hashes}


def setup(repo):
    plan, inventory = registration.qualify(repo)
    release = read(repo / RELEASE)
    if (
        release["study_id"] != plan["study_id"]
        or release["approved_for_one_saved_array_run"] is not True
        or release["new_acquisition_authorized"] is not False
        or release["operator_authorization"]
        != "explicit_implementation_execution_observation_summary_integration_2026_11_09"
        or release["plan_sha256"] != sha(repo / registration.PLAN)
        or any(
            type(release[k]) is not int or release[k] != 0
            for k in ("source_generation", "processor_calls", "model_contexts")
        )
        or release["score_root"] != plan["future_output_root"]
        or release["verification_roots"] != plan["future_verification_roots"]
    ):
        raise ValueError("separate unchanged bounded execution approval required")
    prep = repo / plan["preparation_root"]
    if (
        sha(prep / "summary.json") != release["metadata_preparation_summary_sha256"]
        or sha(prep / "selected_inventory.json") != release["selected_inventory_sha256"]
        or read(prep / "selected_inventory.json")["arrays"] != inventory
    ):
        raise ValueError("selected preparation inventory differs")
    provenance = committed(repo, release)
    provenance.update(
        {
            "plan_sha256": sha(repo / registration.PLAN),
            "release_sha256": sha(repo / RELEASE),
            "sources": plan["sources"],
        }
    )
    spec = read(repo / "configs/stat_matched_source_panel_v1.json")
    original = {
        (p["view"], p["relative_floor"]): p
        for p in ledger(repo / registration.PARENT, "profiles.jsonl")
    }
    return plan, inventory, provenance, spec, original


def loader(repo, inventory, independent=False):
    rows = {
        (r["pair_id"], r["shift"], r["cell"], r["view"]): r["receipt"]
        for r in inventory
    }

    def cells(pair, shift, view):
        values = {}
        for cell in ("mm", "mj", "jm", "jj"):
            receipt = rows[pair, shift, cell, view]
            if independent:
                a = oracle.load(repo / registration.CAPTURE, receipt, view)
            else:
                a = load_array(repo / registration.CAPTURE, receipt)
                if view in ("encoder", "projector"):
                    a = (a.astype(np.uint32) << 16).view(np.float32)
                    if (
                        hashlib.sha256(a.tobytes(order="C")).hexdigest()
                        != receipt["expanded_word_sha256"]
                    ):
                        raise ValueError("expanded native word hash differs")
            values[cell] = a
        return values

    return cells


def run(mode, output):
    repo, output = REPO, Path(output).absolute()
    plan = registration.validate_plan(read(repo / registration.PLAN))
    roots = (
        [plan["future_output_root"]]
        if mode == "run"
        else plan["future_verification_roots"]
    )
    if mode not in ("run", "verify") or output not in [repo / p for p in roots]:
        raise ValueError("canonical fresh run/verification root required")
    inside(repo, str(output.relative_to(repo)))
    source = repo / plan["future_output_root"]
    verification_roots = [repo / p for p in plan["future_verification_roots"]]
    for root in [source, *verification_roots]:
        inside(repo, str(root.relative_to(repo)))
    if any((p / "failure.json").exists() for p in [source, *verification_roots]):
        raise ValueError("a retained failure blocks all further execution")
    if mode == "verify" and output == verification_roots[1]:
        first_summary = verification_roots[0] / "summary.json"
        if (
            not first_summary.is_file()
            or read(first_summary)["status"] != "VERIFIED_NATIVE_CROSSINGS"
            or read(first_summary)["source_summary_sha256"]
            != sha(source / "summary.json")
        ):
            raise ValueError("successful first verification required before replay")
    if shutil.disk_usage(repo).free < 2 * 2**30:
        raise ValueError("two GiB working reserve required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        plan, inventory, provenance, spec, original = setup(repo)
        if mode == "run":
            cells = loader(repo, inventory)
            for view in ("encoder", "projector"):
                for floor in parent_core.FLOORS:
                    p = parent_core.analyze_view(
                        cells, spec, view, relative_floor=floor
                    )
                    old = original[view, floor]
                    for key in (
                        "scores",
                        "margins",
                        "paired_margin_change",
                        "available_test_pairs",
                    ):
                        core.compare(p[key], old[key])
                    append(
                        output,
                        "parent_parity.jsonl",
                        {"view": view, "relative_floor": floor, "status": "PASS"},
                    )
                print(json.dumps({"parent_parity": view, "status": "PASS"}), flush=True)
            write(output / "plan.json", plan)
            write(output / "provenance.json", provenance)
            for view in plan["views"]:
                for p in core.profiles(cells, spec, view):
                    if view in ("encoder", "projector"):
                        core.primary_parity(p, original[view, p["relative_floor"]])
                    append(output, "profiles.jsonl", p)
                print(json.dumps({"scored": view, "status": "UNVERIFIED"}), flush=True)
            status = "SCORED_UNVERIFIED"
        else:
            source = repo / plan["future_output_root"]
            summary = read(source / "summary.json")
            actual_tree = tree(source)
            actual_tree.pop("summary.json", None)
            if (
                summary["status"] != "SCORED_UNVERIFIED"
                or summary["output_sha256"] != actual_tree
                or read(source / "provenance.json") != provenance
                or read(source / "plan.json") != plan
                or (source / "failure.json").exists()
                or set(actual_tree)
                != {
                    "plan.json",
                    "provenance.json",
                    "parent_parity.jsonl",
                    "profiles.jsonl",
                }
            ):
                raise ValueError("source execution tree/provenance differs")
            stored = ledger(source, "profiles.jsonl")
            expected_keys = [
                (v, f) for v in plan["views"] for f in plan["relative_floors"]
            ]
            if [(p["view"], p["relative_floor"]) for p in stored] != expected_keys:
                raise ValueError("complete ordered twelve-profile coverage required")
            saved = {(p["view"], p["relative_floor"]): p for p in stored}
            cells = loader(repo, inventory, independent=True)
            for view in plan["views"]:
                for p in oracle.profiles(cells, spec, view):
                    error = core.compare(saved[view, p["relative_floor"]], p)
                    if view in ("encoder", "projector"):
                        core.primary_parity(p, original[view, p["relative_floor"]])
                    append(output, "profiles.jsonl", p)
                    append(
                        output,
                        "checks.jsonl",
                        {
                            "view": view,
                            "relative_floor": p["relative_floor"],
                            "maximum_absolute_difference": error,
                            "status": "PASS",
                        },
                    )
                print(
                    json.dumps({"verified_native_view": view, "status": "PASS"}),
                    flush=True,
                )
            status = "VERIFIED_NATIVE_CROSSINGS"
        result = {
            "schema_version": 1,
            "analysis_kind": "reference_crossing_" + mode,
            "status": status,
            "posthoc": True,
            "profiles": 12,
            "selected_arrays": 2048,
            "source_generation": 0,
            "processor_calls": 0,
            "model_contexts": 0,
            "plan_sha256": sha(repo / registration.PLAN),
            "provenance": provenance,
            "output_sha256": tree(output),
        }
        if mode == "verify":
            result["source_summary_sha256"] = sha(source / "summary.json")
        if mode == "verify" and output == verification_roots[1]:
            canonical_summary = (
                json.dumps(
                    result, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
            prospective = {
                **tree(output),
                "summary.json": hashlib.sha256(canonical_summary.encode()).hexdigest(),
            }
            if tree(verification_roots[0]) != prospective:
                raise ValueError(
                    "complete verification trees differ; no replacement allowed"
                )
        write(output / "summary.json", result)
        return result
    except Exception as error:
        write(output / "failure.json", {"error": str(error), "retry_authorized": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.mode, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
