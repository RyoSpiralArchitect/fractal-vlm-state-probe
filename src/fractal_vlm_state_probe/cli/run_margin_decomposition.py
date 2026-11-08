"""Committed, score-only post-hoc accounting and exact-rational file replay."""

import argparse
import hashlib
import subprocess
from pathlib import Path

from fractal_vlm_state_probe import margin_decomposition as accounting
from fractal_vlm_state_probe import margin_decomposition_oracle as oracle
from fractal_vlm_state_probe.stat_matched_source_storage import (
    append,
    ledger,
    read,
    sha,
    write,
)

REPO = Path(__file__).resolve().parents[3]
PUBLIC = "examples/research_notes/0094_margin_decomposition"
PLAN = PUBLIC + "/plan.json"
PARENT = "examples/research_notes/0093_stat_matched_source_results/"
IMPLEMENTATION = [
    PLAN,
    "src/fractal_vlm_state_probe/margin_decomposition.py",
    "src/fractal_vlm_state_probe/margin_decomposition_oracle.py",
    "src/fractal_vlm_state_probe/cli/run_margin_decomposition.py",
    "scripts/run_margin_decomposition.py",
    "tests/test_margin_decomposition.py",
]


def files(root):
    paths = sorted(root.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("symlinks are not score-analysis artifacts")
    return {str(p.relative_to(root)): sha(p) for p in paths if p.is_file()}


def run_summary(parent, outputs):
    return {
        "schema_version": 1,
        "analysis_kind": "matched_margin_decomposition",
        "status": "DECOMPOSED_UNVERIFIED",
        "posthoc": True,
        "profiles": 18,
        "parent_capture_sha256": parent["capture_sha256"],
        "primary_results_withheld": True,
        "source_generation": 0,
        "processor_calls": 0,
        "model_contexts": 0,
        "output_sha256": outputs,
    }


def qualify(repo):
    plan = read(repo / PLAN)
    required_sources = {
        PARENT + n
        for n in ("profiles.jsonl", "summary.json", "verification_receipts.json")
    } | {
        "configs/stat_matched_source_panel_v1.json",
        "configs/stat_matched_source_acquisition_release_v1.json",
    }
    if (
        set(plan["sources"]) != required_sources
        or plan["implementation_files"] != IMPLEMENTATION
    ):
        raise ValueError("complete parent and implementation hash coverage required")
    if any(
        plan[k] != v
        for k, v in {
            "profiles": 18,
            "test_pairs": 16,
            "families": 8,
            "shifts": [0, 1, 2, 3],
            "p_values": None,
            "confidence_intervals": None,
            "equivalence_tested": False,
        }.items()
    ):
        raise ValueError("fixed plan denominators or inference boundary differ")
    if (
        plan["study_id"] != "matched_margin_decomposition_v1"
        or plan["scope"] != "posthoc_saved_score_algebra_only_no_new_acquisition"
        or not plan["prior_outcomes_known"]
        or not plan["not_preregistered_before_note0093"]
    ):
        raise ValueError("post-hoc score-only authority differs")
    for name, digest in plan["sources"].items():
        if sha(repo / name) != digest:
            raise ValueError("frozen parent score/source hash differs")
    if plan["closure_absolute_tolerance"] != accounting.TOLERANCE or any(
        plan[k]
        for k in (
            "ratio_attribution",
            "reference_reselection",
            "new_reference_fits",
            "new_hypothesis_tests",
            "source_generation",
            "processor_calls",
            "model_contexts",
            "native_array_reads_required",
        )
    ):
        raise ValueError("fixed analysis boundary differs")
    release = read(repo / "configs/stat_matched_source_acquisition_release_v1.json")
    if len(release["code_sha256"]) != 135:
        raise ValueError("complete preceding 135-file freeze required")
    for name, digest in release["code_sha256"].items():
        if sha(repo / name) != digest:
            raise ValueError("preceding frozen implementation changed")
    parent = read(repo / (PARENT + "summary.json"))
    receipts = read(repo / (PARENT + "verification_receipts.json"))
    if (
        parent["status"] != "DESCRIPTIVE_PROFILES_VERIFIED"
        or parent["output_sha256"]["profiles.jsonl"]
        != sha(repo / (PARENT + "profiles.jsonl"))
        or receipts["first_summary"] != receipts["second_summary"]
        or receipts["first_summary"]["status"] != "VERIFIED"
        or receipts["first_summary"]["capture_sha256"] != parent["capture_sha256"]
        or receipts["first_tree_sha256"] != receipts["second_tree_sha256"]
    ):
        raise ValueError("qualified parent evidence differs")
    profiles = ledger(repo / PARENT, "profiles.jsonl")
    spec = read(repo / "configs/stat_matched_source_panel_v1.json")
    accounting.validate_profiles(profiles, spec)
    return plan, profiles, spec, parent


def committed_code(repo, plan):
    names = plan["implementation_files"]
    commit = (
        subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", *names], cwd=repo
        )
        .decode()
        .strip()
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["parent_commit"], commit],
        cwd=repo,
        check=True,
    )
    hashes = {}
    for name in names:
        digest = sha(repo / name)
        if (
            hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
            ).hexdigest()
            != digest
        ):
            raise ValueError("commit the analysis plan/code before score computation")
        hashes[name] = digest
    return {
        "implementation_commit": commit,
        "code_sha256": hashes,
        "sources": plan["sources"],
        "plan_sha256": sha(repo / PLAN),
    }


def run(mode, output, *, capture=None):
    repo, output = REPO, Path(output).resolve()
    plan = read(repo / PLAN)
    expected = [plan["output_root"]] if mode == "run" else plan["verification_roots"]
    if mode not in ("run", "verify") or output not in [
        (repo / name).resolve() for name in expected
    ]:
        raise ValueError("registered canonical output root required")
    if mode == "verify" and (
        capture is None
        or Path(capture).resolve() != (repo / plan["output_root"]).resolve()
    ):
        raise ValueError("original canonical score-only run required")
    output.mkdir(parents=True, exist_ok=False)
    try:
        plan, profiles, spec, parent = qualify(repo)
        provenance = committed_code(repo, plan)
        if mode == "run":
            write(output / "plan.json", plan)
            write(output / "provenance.json", provenance)
            for p in profiles:
                append(output, "decomposition.jsonl", accounting.decompose(p, spec))
            result = run_summary(parent, files(output))
        else:
            source = Path(capture).resolve()
            if (source / "failure.json").exists():
                raise ValueError("failed score run cannot be promoted")
            summary = read(source / "summary.json")
            actual_files = files(source)
            actual_files.pop("summary.json", None)
            if (
                summary != run_summary(parent, actual_files)
                or set(actual_files)
                != {"plan.json", "provenance.json", "decomposition.jsonl"}
                or read(source / "plan.json") != plan
                or read(source / "provenance.json") != provenance
            ):
                raise ValueError("score-only run provenance/inventory differs")
            actual = ledger(source, "decomposition.jsonl")
            if len(actual) != 18:
                raise ValueError("all eighteen decomposition profiles required")
            for p, stored in zip(profiles, actual, strict=True):
                expected_row = oracle.decompose(p, spec)
                error = accounting.compare(stored, expected_row)
                append(
                    output,
                    "checks.jsonl",
                    {
                        "view": p["view"],
                        "relative_floor": p["relative_floor"],
                        "maximum_absolute_difference": error,
                        "status": "PASS",
                    },
                )
            result = {
                "schema_version": 1,
                "analysis_kind": "matched_margin_decomposition_verification",
                "status": "VERIFIED_SCORE_DECOMPOSITION",
                "posthoc": True,
                "profiles": 18,
                "run_summary_sha256": sha(source / "summary.json"),
                "parent_capture_sha256": parent["capture_sha256"],
                "source_generation": 0,
                "processor_calls": 0,
                "model_contexts": 0,
                "checks_sha256": sha(output / "checks.jsonl"),
            }
        write(output / "summary.json", result)
        return result
    except BaseException as error:
        write(
            output / "failure.json",
            {
                "schema_version": 1,
                "analysis_kind": "matched_margin_decomposition_failure",
                "status": "FAILED_SCORE_ANALYSIS_RETAINED",
                "type": type(error).__name__,
                "message": str(error),
                "source_generation": 0,
                "processor_calls": 0,
                "model_contexts": 0,
                "primary_results_withheld": True,
            },
        )
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("run", "verify"))
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--capture-root", type=Path)
    args = p.parse_args()
    print(run(args.mode, args.output_root, capture=args.capture_root)["status"])


if __name__ == "__main__":
    main()
