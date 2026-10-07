from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _read,
    _validate_panel_split,
)
from fractal_vlm_state_probe.frequency_control import ARMS, frequency_gate
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish input feasibility without promoting unmatched cache claims."
    )
    parser.add_argument("--input-acceptance", required=True, type=Path)
    parser.add_argument("--padding-audit", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    receipt, padding = _read(args.input_acceptance), _read(args.padding_audit)
    if padding["input_acceptance_sha256"] != sha256_file(args.input_acceptance):
        raise ValueError("padding audit belongs to a different input receipt")
    root = args.input_acceptance.parent
    statuses = {}
    for arm in ARMS:
        rejected = None
        try:
            _validate_panel_split(
                *[
                    _read(root / arm / f"{role}_panel.json")
                    for role in ("reference", "test")
                ]
            )
        except ValueError as exc:
            rejected = str(exc)
        accepted = [r for r in receipt["records"] if r["arms"][arm]["gate"]["accepted"]]
        families = {r["pairing_family"] for r in receipt["records"]}
        complete = [
            f
            for f in families
            if {r["replicate"] for r in accepted if r["pairing_family"] == f}
            == {1, 2, 3, 4}
        ]
        statuses[arm] = {
            **receipt["arms"][arm],
            "accepted_reference_blocks": sum(
                r["replicate"] in (1, 2) for r in accepted
            ),
            "accepted_test_blocks": sum(r["replicate"] in (3, 4) for r in accepted),
            "complete_reference_test_families": sorted(complete),
            "actual_capture_gate_rejection": rejected,
            "accepted_pair_ids": [r["pair_id"] for r in accepted],
        }
    raw_passes = 0
    records = []
    for record in receipt["records"]:
        raw_stats = {
            c: {**s, "processor": s["raw"], "pixel_values_shape": [3, 240, 320]}
            for c, s in record["arms"]["shared_pixel_permutation"]["stats"].items()
        }
        raw_gate = frequency_gate(raw_stats)
        raw_passes += raw_gate["accepted"]
        candidates = [
            {k: candidate[k] for k in ("cutoff_a", "cutoff_b")}
            | {
                "accepted": candidate["gate"]["accepted"],
                "common_centroid_target": candidate["gate"]["common_centroid_target"],
                "max_centroid_relative_error": candidate["gate"][
                    "max_centroid_relative_error"
                ],
                "hf_max_pairwise_absolute_difference": candidate["gate"][
                    "hf_max_pairwise_absolute_difference"
                ],
            }
            for candidate in record["low_pass_candidate_gates"]
        ]
        records.append(
            {
                k: v
                for k, v in record.items()
                if k not in ("low_pass_candidate_gates", "low_pass_spatial_stats")
            }
            | {
                "low_pass_candidate_gates": candidates,
                "raw_permutation_diagnostic_gate": raw_gate,
            }
        )
    fractions = [
        v["variance_decomposition"]["between_region_variance_fraction"]
        for r in padding["records"]
        for v in r["cells"].values()
    ]
    code_paths = [
        "src/fractal_vlm_state_probe/frequency_control.py",
        "src/fractal_vlm_state_probe/cli/prepare_frequency_control_panel.py",
        "src/fractal_vlm_state_probe/cli/analyze_frequency_padding_audit.py",
        "src/fractal_vlm_state_probe/cli/summarize_frequency_control_feasibility.py",
        "src/fractal_vlm_state_probe/cli/run_pairing_seed_validation.py",
        "src/fractal_vlm_state_probe/image_stats.py",
        "src/fractal_vlm_state_probe/processor_image_stats.py",
        "configs/frequency_control_fastvlm_v1.json",
    ]
    write_json(
        args.output_json,
        {
            "schema_version": 1,
            "analysis_kind": "all_cell_frequency_control_feasibility_snapshot",
            "date": "2026-10-07",
            "design_registered_commit": "da877cd",
            "qualified_loader_commit_before_input_measurement": "eb7c0d9",
            "status": "NO_COMPLETE_ELIGIBLE_ARM"
            if not any(s["all_blocks_accepted"] for s in statuses.values())
            else "INPUT_ACCEPTED",
            "cache_measurement_included": False,
            "new_cache_forwards": 0,
            "new_cache_tensors": 0,
            "new_held_out_cache_tests": 0,
            "previous_selected_vector_counts_unchanged": {
                "cells": 616,
                "tensors": 1704,
                "factorials": 426,
            },
            "config": receipt["config"],
            "processor_provenance": receipt["processor_provenance"],
            "input_receipt_sha256": sha256_file(args.input_acceptance),
            "input_specification_sha256": receipt["input_specification_sha256"],
            "permutation_sha256": receipt["permutation_sha256"],
            "code_and_config_sha256": {p: sha256_file(Path(p)) for p in code_paths},
            "source_pair_blocks": len(records),
            "original_accepted_blocks": sum(
                r["original_gate"]["accepted"] for r in records
            ),
            "low_pass_candidate_cutoff_pairs": sum(
                len(r["low_pass_candidate_gates"]) for r in records
            ),
            "joint_rgb_multiset_checks_passed": sum(
                s["rgb_multiset_preserved"]
                for r in records
                for a in ARMS
                for s in r["arms"][a]["stats"].values()
            ),
            "decoded_pixel_sham_checks_passed": sum(
                v for r in records for v in r["sham_pixels_equal"].values()
            ),
            "raw_permutation_diagnostic_passes": raw_passes,
            "arms": statuses,
            "blocks_by_family": dict(Counter(r["pairing_family"] for r in records)),
            "records": records,
            "padding_audit": {
                **padding,
                "artifact_sha256": sha256_file(args.padding_audit),
                "between_region_variance_fraction_min": min(fractions),
                "between_region_variance_fraction_max": max(fractions),
            },
            "claim_boundaries": receipt["claim_boundaries"]
            + padding["claim_boundaries"]
            + [
                "Bounded-grid failure is not a proof that all valid frequency-matching transforms are impossible.",
                "No cache correspondence after accepted matching, primary p-value, or semantic/causal attribution is measured.",
                "Whole-processor acceptance remains unchanged; support-only passes never promote a block.",
            ],
        },
    )
    print(f"wrote frequency feasibility snapshot to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
