"""Package only a fully qualified crossing run; never infer or rescore arrays."""

import shutil
from pathlib import Path

from fractal_vlm_state_probe.cli.run_reference_crossing import tree
from fractal_vlm_state_probe.reference_crossing_registration import PLAN
from fractal_vlm_state_probe.stat_matched_source_storage import (
    inside,
    ledger,
    read,
    sha,
    write,
)

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / "examples/research_notes/0098_reference_crossing_results"


def main():
    plan = read(REPO / PLAN)
    source = REPO / plan["future_output_root"]
    first, second = [REPO / p for p in plan["future_verification_roots"]]
    for root in (source, first, second):
        inside(REPO, str(root.relative_to(REPO)))
    if any((p / "failure.json").exists() for p in (source, first, second)):
        raise ValueError("retained failure blocks publication")
    a, b = read(first / "summary.json"), read(second / "summary.json")
    for path, digest in a["provenance"]["implementation_sha256"].items():
        if sha(inside(REPO, path)) != digest:
            raise ValueError("qualified execution implementation changed")
    if (
        a != b
        or a["status"] != "VERIFIED_NATIVE_CROSSINGS"
        or tree(first) != tree(second)
        or a["source_summary_sha256"] != sha(source / "summary.json")
    ):
        raise ValueError("two identical native verifications required")
    for root, summary in (
        (source, read(source / "summary.json")),
        (first, a),
        (second, b),
    ):
        files = tree(root)
        files.pop("summary.json")
        if files != summary["output_sha256"]:
            raise ValueError("qualified output tree changed")
    profiles = ledger(source, "profiles.jsonl")
    if [(p["view"], p["relative_floor"]) for p in profiles] != [
        (v, f) for v in plan["views"] for f in plan["relative_floors"]
    ]:
        raise ValueError("complete registered profile order required")
    PUBLIC.mkdir(parents=True, exist_ok=False)
    for name, original in (
        ("profiles.jsonl", source / "profiles.jsonl"),
        ("execution_summary.json", source / "summary.json"),
        ("oracle_checks.jsonl", first / "checks.jsonl"),
    ):
        shutil.copyfile(original, PUBLIC / name)
    receipts = {
        "schema_version": 1,
        "analysis_kind": "reference_crossing_receipts",
        "first_summary": a,
        "second_summary": b,
        "first_tree_sha256": tree(first),
        "second_tree_sha256": tree(second),
        "byte_identical": True,
        "new_model_contexts": 0,
        "new_processor_calls": 0,
    }
    write(PUBLIC / "verification_receipts.json", receipts)
    primary = [
        {
            "view": p["view"],
            "relative_floor": p["relative_floor"],
            "available_test_pairs": p["available_test_pairs"],
            "overall": p["overall"],
            "family_results": p["family_results"],
        }
        for p in profiles
        if p["relative_floor"] == plan["primary_relative_floor"]
    ]
    write(
        PUBLIC / "summary.json",
        {
            "schema_version": 1,
            "analysis_kind": "reference_crossing_results",
            "status": "VERIFIED_POSTHOC_REFERENCE_CROSSINGS",
            "date": "2026-11-09",
            "posthoc": True,
            "not_preregistered_before_note0093": True,
            "implementation_commit": a["provenance"]["implementation_commit"],
            "plan_sha256": a["plan_sha256"],
            "profiles": 12,
            "test_pairs": 16,
            "known_families": 8,
            "selected_arrays": 2048,
            "verification_runs": 2,
            "byte_identical_verifications": True,
            "primary_profiles": primary,
            "all_floor_endpoints_identical": all(
                all(
                    p["overall"]
                    == next(
                        q["overall"]
                        for q in profiles
                        if q["view"] == view and q["relative_floor"] == 1e-6
                    )
                    for p in profiles
                    if p["view"] == view
                )
                for view in plan["views"]
            ),
            "maximum_absolute_oracle_difference": max(
                p["maximum_absolute_difference"] for p in ledger(first, "checks.jsonl")
            ),
            "source_generation": 0,
            "processor_calls": 0,
            "model_contexts": 0,
            "p_values": None,
            "confidence_intervals": None,
            "equivalence_tested": False,
            "confirmation_cohort": "not_generated_not_sealed",
            "full_native_payloads_published": False,
            "public_replay_scope": "saved_scores_aggregate_and_missingness_not_native_word_reverification",
            "output_sha256": {p.name: sha(p) for p in PUBLIC.iterdir() if p.is_file()},
        },
    )
    print(PUBLIC)


if __name__ == "__main__":
    main()
