"""No confirmation images, maps, processor calls or model contexts."""

import copy
from pathlib import Path

import pytest

from fractal_vlm_state_probe import confirmation_preparation as core
from fractal_vlm_state_probe.stat_matched_source_storage import read

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    plan = read(ROOT / core.PLAN)
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    tests = [p for p in spec["sources"]["pairs"] if p["split"] == "test"]
    families = list(dict.fromkeys(p["pairing_family"] for p in tests))
    profiles = []
    for view, scale in (("encoder", 1), ("projector", 2)):
        rows = []
        for p in tests:
            f = families.index(p["pairing_family"])
            rows.append(
                {
                    "pair_id": p["pair_id"],
                    "family": p["pairing_family"],
                    "reference_difference": float(
                        f + scale * (1 if p["replicate"] == 7 else -1) / 8
                    ),
                }
            )
        profiles.append(
            {
                "view": view,
                "relative_floor": 1e-6,
                "test_pair_ids": [p["pair_id"] for p in tests],
                "pairs": rows,
            }
        )
    return plan, spec, profiles


def test_seed_allocation_is_repeatable_complete_and_not_generation():
    plan, spec, _ = fixture()
    previous = copy.deepcopy(spec)
    a = core.allocated_pairs(spec, plan)
    assert a == core.allocated_pairs(spec, plan)
    assert spec == previous
    assert len(a) == 80
    assert sum(p["split"] == "reference" for p in a) == 16
    assert sum(p["split"] == "test" for p in a) == 64
    assert len({p[role]["seed"] for p in a for role in ("source_a", "source_b")}) == 160
    for p in a:
        assert p["replicate"] in (
            plan["candidate"]["reference_replicates"]
            if p["split"] == "reference"
            else plan["candidate"]["test_replicates"]
        )
        assert p["source_a"]["kind"] and p["source_b"]["kind"]
    audit = core.seed_audit(a, spec["sources"]["defaults"], [])
    assert audit["collisions"] == []
    assert audit["decoded_content_overlap_checked"] is False


def test_historical_rng_collision_stops_without_replacement():
    plan, spec, _ = fixture()
    pairs = core.allocated_pairs(spec, plan)
    history = [
        {
            "source": pairs[0]["source_a"],
            "defaults": spec["sources"]["defaults"],
            "identity": "synthetic_existing_source",
        }
    ]
    with pytest.raises(ValueError, match="initializer collision"):
        core.seed_audit(pairs, spec["sources"]["defaults"], history)


def test_precision_keeps_sites_paired_and_does_not_use_family_means():
    plan, spec, profiles = fixture()
    settings = {**plan["development_precision"], "simulation_trials": 200}
    a = core.precision_plan(profiles, spec, settings)
    shifted = copy.deepcopy(profiles)
    for p in shifted:
        for row in p["pairs"]:
            row["reference_difference"] += 1024
    assert a == core.precision_plan(shifted, spec, settings)
    assert a == core.precision_plan(profiles, spec, settings)
    for encoder, projector in zip(a["rows"][::2], a["rows"][1::2], strict=True):
        for key in ("half_width_median", "half_width_p90", "half_width_p95"):
            assert projector[key] == pytest.approx(2 * encoder[key], abs=1e-12)
    assert not a["sample_size_selected_by_observed_effect_or_significance"]
    assert not a["new_map_or_reference_variance_estimated"]


def test_zero_residuals_are_not_claimed_real_precision_or_equivalence():
    plan, spec, profiles = fixture()
    for p in profiles:
        for row in p["pairs"]:
            row["reference_difference"] = 1.0
    result = core.precision_plan(
        profiles, spec, {**plan["development_precision"], "simulation_trials": 100}
    )
    assert all(r["half_width_p95"] == 0 for r in result["rows"])
    assert (
        result["status"] == "DEVELOPMENT_ONLY_NOT_CONFIRMATION_OR_COVERAGE_VALIDATION"
    )
    assert not result["exact_bootstrap_coverage_established"]


@pytest.mark.parametrize(
    "damage", ["null", "nan", "bool", "duplicate", "order", "pair_order", "family"]
)
def test_missing_or_incomplete_development_data_rejected(damage):
    plan, spec, profiles = fixture()
    if damage in ("null", "nan", "bool"):
        profiles[0]["pairs"][0]["reference_difference"] = {
            "null": None,
            "nan": float("nan"),
            "bool": True,
        }[damage]
    elif damage == "duplicate":
        profiles.append(copy.deepcopy(profiles[0]))
    elif damage == "order":
        profiles[0]["test_pair_ids"].reverse()
    elif damage == "pair_order":
        profiles[0]["pairs"].reverse()
    else:
        profiles[0]["pairs"][0]["family"] = "not_in_registered_inventory"
    with pytest.raises(ValueError):
        core.residuals(profiles, spec, plan["development_precision"])


def test_planned_budget_and_dependence_are_explicit():
    plan, _, _ = fixture()
    c, b = plan["candidate"], plan["budget_candidates"]
    assert (
        c["known_families"]
        * (c["reference_pairs_per_family"] + c["test_pairs_per_family"])
        * 16
        == c["candidate_cells"]
        == 1280
    )
    assert (
        b["historical_model_contexts"]
        + b["sham_model_contexts"]
        + b["experimental_model_contexts"]
        == b["model_source_contexts_upper"]
        == 1688
    )
    assert (
        b["combined_processor_only_upper"]
        == b["map_calibration_processor_only_upper"]
        + b["acquisition_processor_only_upper"]
    )
    assert (
        b["required_launch_free_bytes"]
        == b["payload_allowance_bytes"]
        + b["reserve_bytes"]
        + b["overhead_allowance_bytes"]
    )
    assert plan["new_map_candidate"]["powers_are_independent_maps"] is False
    assert plan["new_map_candidate"]["independent_map_orbits"] == 1
    inference = plan["candidate_inference"]
    assert len(inference["primary_endpoints"]) == 2
    assert inference["quantiles"] == [0.0125, 0.9875]
    assert inference["coverage_is_exact_or_guaranteed"] is False
    assert inference["equivalence_tested"] is False
    assert all(
        v is False
        for v in [
            plan["confirmation_generation_authorized"],
            plan["processor_execution_authorized"],
            plan["model_execution_authorized"],
        ]
    )


def test_preparation_refuses_noncanonical_output_without_generation(tmp_path):
    with pytest.raises(ValueError, match="canonical"):
        core.prepare(ROOT, tmp_path)
