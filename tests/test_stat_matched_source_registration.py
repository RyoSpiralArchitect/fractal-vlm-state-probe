"""Metadata contracts only; no source, map, feature or model generation."""

import hashlib
import json
from collections import Counter
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/stat_matched_source_panel_v1.json"


def read(path):
    return json.loads(path.read_bytes())


def test_design_is_not_acquisition_or_confirmation_release():
    s = read(CFG)
    assert s["registered_date"] == "2026-11-08"
    assert s["scope"] == "fresh_seed_known_family_development_design_only"
    assert s["status_at_registration"] == "not_implemented_not_released"
    assert not s["execution_authorized_by_this_file"]
    assert s["execution_runner"] is None
    assert set(s["new_observations_at_registration"].values()) == {0}
    assert s["cohort_status"] == "identities_allocated_not_generated_not_sealed"
    assert s["confirmation"] == {
        "status": "not_generated_not_sealed",
        "cohort_manifest": None,
        "sample_size": None,
    }
    assert s["release"]["implementation_commit"] is None
    assert s["release"]["public_registration_receipt"] is None
    assert s["release"]["public_freeze_required_before_acquisition"]
    assert "explicit_execution_authorization" in s["release"]["conditions"]
    ceiling = s["claim_ceiling"].copy()
    assert ceiling.pop("within_known_generator_families_only")
    assert set(ceiling.values()) == {False}


def test_all_118_inherited_files_and_added_pins_remain_exact():
    s = read(CFG)
    for path, digest in s["pinned_artifacts"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    previous = read(ROOT / s["preserve_inherited_frozen_hashes_from"])
    assert previous["status"] == "PASS"
    assert len(previous["code_sha256"]) == s["inherited_frozen_files"] == 118
    for path, digest in previous["code_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_exact_new_source_allocations_and_inherited_parameters():
    s = read(CFG)["sources"]
    old = read(ROOT / "configs/generator_pairing_seed_holdout_v1.json")
    older = read(ROOT / "configs/generator_pairing_transfer_v1.json")
    assert s["defaults"] == old["source_defaults"]
    templates = [p for p in old["pairs"] if p["replicate"] == 3]
    pairs = s["pairs"]
    assert len(pairs) == len({p["pair_id"] for p in pairs}) == 32
    assert not {p["pair_id"] for p in pairs} & {
        p["pair_id"] for p in old["pairs"] + older["pairs"]
    }
    assert Counter(p["split"] for p in pairs) == {"reference": 16, "test": 16}
    assert Counter(p["broad_class"] for p in pairs) == {
        "geometry": 16,
        "stochastic": 16,
    }
    seeds = []
    for i, template in enumerate(templates):
        for rep in range(5, 9):
            pair = pairs[4 * i + rep - 5]
            assert pair["pair_id"] == template["pair_id"].replace("_r3", f"_r{rep}")
            assert pair["pairing_family"] == template["pairing_family"]
            assert pair["replicate"] == rep
            assert pair["split"] == ("reference" if rep <= 6 else "test")
            for j, role in enumerate(("source_a", "source_b")):
                source, expected = pair[role].copy(), template[role].copy()
                seed = source.pop("seed")
                expected.pop("seed")
                assert source == expected
                assert seed == 2026110801 + 1009 * (8 * i + 2 * (rep - 5) + j)
                seeds.append(seed)
    assert s["seed_base"] == 2026110801 and s["seed_stride"] == 1009
    assert seeds == [s["seed_base"] + s["seed_stride"] * i for i in range(64)]
    assert s["seed_range_inclusive"] == [seeds[0], seeds[-1]]
    assert not set(seeds) & {
        p[role]["seed"]
        for p in old["pairs"] + older["pairs"]
        for role in ("source_a", "source_b")
    }
    assert s["reference_replicates"] == [5, 6] and s["test_replicates"] == [7, 8]
    assert s["historical_inventory_required_before_generation"]
    assert "stop_without_replacement" in s["collision_policy"]


def test_payload_first_construction_reuses_qualified_geometry():
    c = read(CFG)["construction"]
    old = read(ROOT / c["inherited_geometry"])["construction"]
    assert c["raw_shape_hwc"] == old["raw_shape_hwc"] == [320, 320, 3]
    assert c["payload_source_shape_hwc"] == [240, 320, 3]
    assert c["background_rgb"] == [128, 128, 128]
    assert c["payload_pixels"] == 6 * 8 * 4 * 6 * 6 == 6912
    assert prod(c["packed_shape_hwc"]) == c["payload_pixels"] * 3
    assert c["packed_shape_hwc"] == [72, 96, 3]
    assert c["payload_order"] == ["cell_row", "cell_col", "tile", "u", "v", "channel"]
    assert (
        c["palette_order"]
        == "sample_source_payloads_then_pack_C_order_then_rank_match_then_embed_then_shift"
    )
    assert c["payload_cells"] == {
        "mm": "A",
        "mj": "luminance_quantile_match_rgb(A,reference=B)",
        "jm": "luminance_quantile_match_rgb(B,reference=A)",
        "jj": "B",
    }
    assert (
        not c["recompute_palette_after_shift"] and not c["search_or_rejection_sampling"]
    )
    assert c["shifts"] == old["arrangement_shifts"] == [0, 1, 2, 3]
    assert c["exact_raw_palette_pairs"] == [["mm", "jm"], ["mj", "jj"]]
    assert (ROOT / c["map_reused_from"]).is_file()


def test_derived_rng_streams_are_disjoint_across_split_and_known_history():
    s = read(CFG)["sources"]
    rules = s["derived_rng_rules"]
    assert rules == {
        "voronoi": {"direct_offsets": [0, 3], "frame_salts": []},
        "quasicrystal": {"direct_offsets": [5], "frame_salts": []},
        "checkerboard": {"direct_offsets": [0], "frame_salts": []},
        "square_tiling": {"direct_offsets": [0, 17], "frame_salts": []},
        "triangle_tiling": {"direct_offsets": [1, 17], "frame_salts": []},
        "hex_tiling": {"direct_offsets": [2, 17], "frame_salts": []},
        "white_noise": {"direct_offsets": [], "frame_salts": [101]},
        "blue_noise": {"direct_offsets": [], "frame_salts": [211]},
        "random_dots": {"direct_offsets": [0], "frame_salts": [307]},
    }
    assert s["derived_rng_frame_index"] == 0
    assert s["derived_rng_frame_formula"] == (
        "(seed*1000003+frame_index*9176+salt)%4294967295"
    )

    def streams(source):
        seed = source["seed"]
        rule = rules[source["kind"]]
        return [seed + x for x in rule["direct_offsets"]] + [
            (seed * 1000003 + salt) % 4294967295 for salt in rule["frame_salts"]
        ]

    # Reproduce the rejected draft's component reuse using integers, not RNG.
    assert set(streams({"kind": "triangle_tiling", "seed": 2026110820})) & set(
        streams({"kind": "square_tiling", "seed": 2026110821})
    ) == {2026110821}
    allocated = [
        x
        for p in s["pairs"]
        for role in ("source_a", "source_b")
        for x in streams(p[role])
    ]
    assert len(allocated) == len(set(allocated)) == 104
    historical = {
        x
        for config in (
            "generator_pairing_transfer_v1.json",
            "generator_pairing_seed_holdout_v1.json",
        )
        for p in read(ROOT / "configs" / config)["pairs"]
        for role in ("source_a", "source_b")
        for x in streams(p[role])
    }
    assert not set(allocated) & historical
    assert "derived_rng_seed" in s["collision_policy"]


def test_denominators_processor_gates_and_capacity_arithmetic():
    s = read(CFG)
    b, g = s["budget"], s["input_gates"]
    assert b["raw_sources"] == b["raw_source_frames"] == 2 * b["source_pairs"] == 64
    assert b["candidate_cells"] == b["candidate_processor_calls"] == 32 * 4 * 4 == 512
    assert b["reference_cells"] == b["test_cells"] == 16 * 4 * 4 == 256
    assert g["matched_vector_comparisons"] == 32 * 4 * 3 * 12 == 4608
    assert g["joint_word_multiset_comparisons"] == 32 * 4 * 3 * 2 * 65 == 49920
    assert g["oracle_vector_pairs"] == 512 * 12 == 6144
    assert g["raw_palette_multiset_comparisons"] == 32 * 2 * 4 == 256
    assert (
        g["feature_absolute_tolerance"] == 1e-10
        and g["feature_relative_tolerance"] == 0
    )
    assert (
        b["processor_only_calls"]
        == (
            b["historical_processor_calls"]
            + b["synthetic_processor_calls"]
            + b["candidate_processor_calls"]
        )
        == 128 + 25 + 512
        == 665
    )
    assert (
        b["total_model_contexts"]
        == (
            b["historical_model_contexts"]
            + b["sham_model_contexts"]
            + b["experimental_model_contexts"]
        )
        == 384 + 24 + 512
        == 920
    )
    assert b["experimental_representation_arrays"] == 512 * 2 == 1024
    assert b["experimental_interaction_vectors"] == 32 * 4 * 2 == 256
    assert b["new_coordinate_maps"] == 0 and b["coordinate_maps_reused"] == 4
    assert (
        b["required_launch_free_bytes"]
        == (
            b["reserve_bytes"]
            + b["payload_allowance_bytes"]
            + b["overhead_allowance_bytes"]
        )
        == 28 * 2**30
    )
    assert b["reserve_bytes"] == 8 * 2**30 and b["scratch_vector_limit"] == 64
    assert not b["compression_savings_assumed"]


def test_reference_seal_observation_and_shams_precede_test_generation():
    s = read(CFG)
    o, r = s["observation"], s["references"]
    assert r["shifts_used"] == [0] and r["replicates"] == [5, 6]
    assert r["seal_before_test_source_generation"]
    assert r["reference_context_last_ordinal"] == 384 + 24 + 256 == 664
    assert r["first_test_context_ordinal"] == 665
    assert o["historical_calibration_contexts"] == 384
    assert o["sham_contexts"] == o["sham_inputs"] * 3 == 24
    assert o["same_process_calibration_and_single_model_load"]
    assert o["source_max_tokens"] == 2 and o["temperature"] == 0
    assert "no_forced_The_image" in o["new_source_suffix_policy"]
    assert o["sham_cache_targets"] == ["1:keys", "12:keys", "23:values"]
    old = read(ROOT / o["inherited_observer_config"])
    assert o["sites"] == {k: v["shape"] for k, v in old["sites"].items()}
    steps = s["sequence"]
    assert len(steps) == 9
    assert steps[5].startswith("capture_all_256_reference_cells_then_seal")
    assert steps[6].startswith("only_after_reference_seal_generate_test_sources")
    assert steps[7] == "capture_256_test_cells_without_scores"
    assert steps[8].startswith("independent_file_only_verification_twice")


def test_estimands_keep_all_views_low_norms_and_missing_denominators():
    s = read(CFG)
    a, n = s["analysis"], s["low_norm"]
    old = read(ROOT / "configs/nonlearned_processing_phase3_v1.json")
    assert a["views"] == ["encoder", "projector"] + [
        f"{start}/{feature}" for start in old["starts"] for feature in old["features"]
    ]
    assert a["primary_endpoints"] == [
        "encoder_mean_s1_s2_s3_margin_minus_s0_margin",
        "projector_mean_s1_s2_s3_margin_minus_s0_margin",
    ]
    assert a["score_matrices"] == len(a["views"]) * 4 == 56
    assert a["primary_test_pairs"] == 16 and a["primary_families"] == 8
    assert a["p_values"] == a["confidence_intervals"] == 0
    assert not a["equivalence_tested"]
    assert (
        "endpoint_unavailable" in a["missingness"]
        and "no_complete_case_promotion" in a["missingness"]
    )
    assert (
        n["baseline_contrast_error_envelope"]
        == "4*1e-10*sqrt(d)_from_four_coordinate_error_budgets"
    )
    assert n["baseline_direction_minimum"] == "norm(z)>100*4*1e-10*sqrt(d)"
    assert n["learned_primary_minimum"] == "norm(z)>1e-6*sum_cell_L2_norms"
    assert n["learned_sensitivity_relative_floors"] == [1e-8, 1e-6, 1e-4]
    assert n["reference_requires_both_replicates"]
    assert "not_a_failed_input_gate" in n["expected_structural_unavailability"]
    assert s["references"]["cancellation_threshold"] == 1e-12
    assert "no_winner" in n["report"]
