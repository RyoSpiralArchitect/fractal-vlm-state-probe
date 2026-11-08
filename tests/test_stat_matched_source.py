"""Small synthetic software fixtures only, never the allocated source cohort."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe import stat_matched_input as matching
from fractal_vlm_state_probe import stat_matched_source as m
from fractal_vlm_state_probe import stat_matched_source_oracle as oracle
from fractal_vlm_state_probe.nonlearned_processing import (
    extract_features,
    factorial_interaction,
)

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_bytes())


@pytest.fixture
def toy():
    s = read("configs/stat_matched_source_panel_v1.json")
    g = read(s["construction"]["inherited_geometry"])
    g["construction"].update(
        raw_shape_hwc=[80, 80, 3],
        map_shape=[80, 80],
        grid=[2, 2],
        content_rows=[0, 80],
        active_grid_rows=[0, 2],
        active_grid_cols=[0, 2],
    )
    s["construction"].update(
        raw_shape_hwc=[80, 80, 3],
        payload_source_shape_hwc=[80, 80, 3],
        packed_shape_hwc=[24, 24, 3],
        payload_pixels=576,
    )
    s["observation"]["sites"] = {"encoder": [3], "projector": [3]}
    y, x = np.indices((80, 80))
    a = np.stack(
        ((x * 11 + y * 7) % 256, (x * 3 + y) % 256, (x + y * 9) % 256), axis=-1
    ).astype(np.uint8)
    b = np.stack(
        ((y * 13 + x * 2) % 256, (y * 5 + x * 7) % 256, (y + x * 17) % 256), axis=-1
    ).astype(np.uint8)
    return s, g, a, b


@pytest.mark.parametrize("tied", [False, True])
def test_payload_rank_and_all_arrangements_match_independent_scalar_oracle(toy, tied):
    s, g, a, b = toy
    if tied:
        a[:] = 90
    before = (a.tobytes(), b.tobytes())
    identity = m.identity_quartet(a, b, s, g)
    assert m.palette_audit(identity)["matches"] == [True, True]
    for shift in range(4):
        expected = oracle.quartet(a, b, s, g, shift)
        mapping = matching.coordinate_map(g, shift)
        actual = {k: m.rearrange(v, mapping) for k, v in identity.items()}
        assert m.palette_audit(actual)["matches"] == [True, True]
        for cell in m.CELLS:
            assert actual[cell].tobytes() == expected[cell].tobytes()
            assert matching.joint_multisets(
                actual[cell], "raw_rgb", [2, 2]
            ) == matching.joint_multisets(identity[cell], "raw_rgb", [2, 2])
    assert (a.tobytes(), b.tobytes()) == before


def test_all_twelve_toy_feature_paths_and_expected_global_zero(toy):
    s, g, a, b = toy
    f = read("configs/nonlearned_processing_phase3_v1.json")
    f["grid"] = [2, 2]
    f["windows"]["processor_pixel_values"] = copy.deepcopy(f["windows"]["raw_rgb"])
    identity = m.identity_quartet(a, b, s, g)
    globals_by_shift = []
    for shift in range(4):
        global_cells = {}
        for cell in m.CELLS:
            actual = m.rearrange(identity[cell], matching.coordinate_map(g, shift))
            for start in ("raw_rgb", "processor_pixel_values"):

                def values(raw):
                    return (
                        raw
                        if start == "raw_rgb"
                        else np.ascontiguousarray(
                            np.moveaxis(
                                raw.astype(np.float32) / np.float32(255), -1, 0
                            )[None]
                        )
                    )

                left = extract_features(values(actual), start, f)
                right = extract_features(values(identity[cell]), start, f)
                for name in f["features"]:
                    np.testing.assert_allclose(
                        left[name], right[name], atol=1e-10, rtol=0
                    )
                if start == "raw_rgb":
                    global_cells[cell] = left["global_moments"]
        globals_by_shift.append(factorial_interaction(global_cells))
    assert all(np.count_nonzero(z) == 0 for z in globals_by_shift)


def test_invalid_payload_and_map_fail_closed(toy):
    s, g, a, b = toy
    for bad in (a.astype(float), a[:40], a[:, :, :2]):
        with pytest.raises(ValueError):
            m.identity_quartet(bad, b, s, g)
    with pytest.raises(ValueError):
        m.embed_payload(np.zeros((1, 1, 3), dtype=np.uint8), s, g)
    mapping = matching.coordinate_map(g, 1)
    mapping[0, 0] = mapping[0, 1]
    with pytest.raises(ValueError):
        m.rearrange(a, mapping)
    with pytest.raises(ValueError):
        m.rearrange(a, mapping.astype(float))


def history():
    result = []
    for name in (
        "generator_pairing_transfer_v1.json",
        "generator_pairing_seed_holdout_v1.json",
    ):
        old = read("configs/" + name)
        for p in old["pairs"]:
            for role in ("source_a", "source_b"):
                result.append(
                    {
                        "identity": p["pair_id"] + "/" + role,
                        "source": p[role],
                        "defaults": old["source_defaults"],
                    }
                )
    return result


def test_metadata_rng_inventory_catches_component_reuse_without_drawing_rng():
    s = read("configs/stat_matched_source_panel_v1.json")
    audit = m.source_spec_audit(s, history())
    assert audit["derived_initializers"] == audit["unique_initializers"] == 104
    assert not audit["collisions"]
    assert audit["supplied_historical_sources"] == 64
    assert (
        not audit["historical_inventory_complete"] and not audit["acquisition_released"]
    )
    s["sources"]["pairs"][10]["source_a"]["seed"] = (
        s["sources"]["pairs"][9]["source_b"]["seed"] + 1
    )
    assert m.source_spec_audit(s, history())["collisions"]
    with pytest.raises(ValueError):
        m.rng_seeds({"kind": "unknown", "seed": 1}, {"total_frames": 1})
    with pytest.raises(ValueError):
        m.rng_seeds({"kind": "voronoi", "seed": 1}, {"total_frames": 2})


def test_ordered_jobs_and_metadata_seal_require_every_reference_cell():
    s = read("configs/stat_matched_source_panel_v1.json")
    jobs = m.ordered_jobs(s)
    assert len(jobs) == 512 and jobs[0]["ordinal"] == 409
    assert jobs[255]["ordinal"] == 664 and jobs[256]["ordinal"] == 665
    assert all(j["split"] == "reference" for j in jobs[:256])
    rows = [
        {**j, "representation_sha256": "a" * 64, "prepared_pixel_sha256": "b" * 64}
        for j in jobs[:256]
    ]
    seal = m.seal_reference_metadata(rows, s)
    assert seal["cells"] == 256 and not seal["artifact_bytes_independently_verified"]
    assert m.seal_reference_metadata(copy.deepcopy(rows), s) == seal
    for bad in (
        rows[:-1],
        rows[::-1],
        [{**rows[0], "shift": 3}, *rows[1:]],
        [{**rows[0], "representation_sha256": "a"}, *rows[1:]],
    ):
        with pytest.raises(ValueError):
            m.seal_reference_metadata(bad, s)
    s["sources"]["pairs"][0]["split"] = "test"
    with pytest.raises(ValueError):
        m.ordered_jobs(s)


def cells(vector, scale=0):
    zero = np.full(len(vector), scale, dtype=float)
    return {
        "mm": zero.copy(),
        "mj": zero.copy(),
        "jm": zero.copy(),
        "jj": zero + vector,
    }


@pytest.mark.parametrize("learned", [False, True])
def test_direction_exact_zero_guards_and_cancelled_references(learned):
    _, unit, row = m.contrast_direction(cells([0, 0, 0]), learned=learned)
    assert unit is None and row["reason"] == "exact_zero"
    _, unit, row = m.contrast_direction(cells([1e-11, 0, 0], 1), learned=learned)
    assert unit is None and row["reason"] == "below_amplitude_guard"
    _, unit, row = m.contrast_direction(cells([1, 0, 0]), learned=learned)
    assert row["available"] and unit.tolist() == [1, 0, 0]
    assert m.family_reference([unit, -unit])[0] is None
    assert m.family_reference([unit, None])[0] is None
    assert m.family_reference([unit, unit])[0].tolist() == [1, 0, 0]


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e308, 1e-300])
def test_bad_numeric_scales_fail_closed(value):
    with pytest.raises(ValueError):
        m.contrast_direction(cells([value, 0, 0]), learned=True)


def loader_for(s):
    families = list(dict.fromkeys(p["pairing_family"] for p in s["sources"]["pairs"]))
    meta = {p["pair_id"]: p for p in s["sources"]["pairs"]}

    def load(pair_id, shift, view):
        p = meta[pair_id]
        f = families.index(p["pairing_family"])
        vector = np.array([1 + f, 2 + (f % 3), 1 + f * f], dtype=float)
        if p["split"] == "test":
            vector += shift * np.array([0.2, -0.1, 0.3]) + (p["replicate"] - 7) * 0.01
        return cells(vector)

    return load


@pytest.mark.parametrize("floor", m.FLOORS)
def test_complete_profile_matches_independent_scalar_oracle(toy, floor):
    s, _, _, _ = toy
    load = loader_for(s)
    actual = m.analyze_view(load, s, "encoder", relative_floor=floor)
    expected = oracle.profile(load, s, "encoder", floor)
    assert actual["available"] and len(actual["interactions"]) == 128
    assert len(actual["diagnostics"]) == 48 and len(actual["score_matrices"]) == 4
    assert actual["paired_margin_change"] == pytest.approx(
        expected["paired_margin_change"], abs=1e-12
    )
    np.testing.assert_allclose(actual["scores"], expected["scores"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        actual["margins"], expected["margins"], atol=1e-12, rtol=0
    )
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("phase", ["reference", "test"])
def test_unavailable_rows_withhold_primary_without_complete_case_promotion(toy, phase):
    s, _, _, _ = toy
    base = loader_for(s)
    target = next(p["pair_id"] for p in s["sources"]["pairs"] if p["split"] == phase)

    def load(pair_id, shift, view):
        return (
            cells([0, 0, 0])
            if pair_id == target and shift == 0
            else base(pair_id, shift, view)
        )

    actual = m.analyze_view(load, s, "encoder")
    expected = oracle.profile(load, s, "encoder")
    assert not actual["available"] and actual["paired_margin_change"] is None
    assert actual["fixed_test_pairs"] == 16 and actual["available_test_pairs"] < 16
    assert expected["paired_margin_change"] is None
    assert len(actual["interactions"]) == 128
    json.dumps(actual, allow_nan=False)


def test_raw_global_zero_is_retained_and_sensitivity_cannot_select_threshold(toy):
    s, _, _, _ = toy
    actual = m.analyze_view(lambda *args: cells([0] * 6), s, "raw_rgb/global_moments")
    assert not actual["available"] and actual["available_test_pairs"] == 0
    assert len(actual["references"]) == 8 and len(actual["interactions"]) == 128
    for view, floor in (
        ("encoder", 1e-7),
        ("raw_rgb/global_moments", 1e-8),
        ("other", 1e-6),
    ):
        with pytest.raises(ValueError):
            m.analyze_view(loader_for(s), s, view, relative_floor=floor)


def test_planned_uncompressed_payload_fits_without_compression():
    s = read("configs/stat_matched_source_panel_v1.json")
    b = m.disk_budget(s)
    assert sum(b["parts"].values()) == b["enumerated_payload_bytes"]
    assert b["enumerated_payload_bytes"] < 16 * 2**30
    assert b["required_launch_free_bytes"] == 28 * 2**30
    assert not b["acquisition_released"]
    s["budget"]["payload_allowance_bytes"] = 1
    with pytest.raises(ValueError):
        m.disk_budget(s)


@pytest.mark.parametrize("value", [0, 117])
def test_toy_constant_payload_shams_are_word_identical(toy, value):
    s, g, a, b = toy
    a[:], b[:] = value, value
    quartet = m.identity_quartet(a, b, s, g)
    for shift in range(4):
        mapping = matching.coordinate_map(g, shift)
        for raw in quartet.values():
            assert m.rearrange(raw, mapping).tobytes() == raw.tobytes()


def test_palette_sentinel_is_reported_instead_of_dropped(toy):
    s, g, a, b = toy
    quartet = m.identity_quartet(a, b, s, g)
    quartet["jm"][0, 0] = [1, 2, 3]
    assert m.palette_audit(quartet)["matches"] == [False, True]


def test_file_contract_qualification_and_registration_tamper_fail_closed(tmp_path):
    profile = read("configs/stat_matched_source_core_v1.json")
    assert len(profile["implementation_files"]) == 5
    assert all((ROOT / p).is_file() for p in profile["implementation_files"])
    actual = m.qualify_core(ROOT)
    assert actual["inherited_files"] == 118
    assert not actual["acquisition_released"] and not actual["runtime_qualified"]
    spec = read(profile["design_config"])
    previous = read(spec["preserve_inherited_frozen_hashes_from"])
    names = {
        "configs/stat_matched_source_core_v1.json",
        spec["preserve_inherited_frozen_hashes_from"],
        *profile["registration_sha256"],
        *spec["pinned_artifacts"],
        *previous["code_sha256"],
    }
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    assert m.qualify_core(tmp_path) == actual
    changed = tmp_path / profile["design_config"]
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen file differs"):
        m.qualify_core(tmp_path)


def test_decoded_rgb_collisions_remain_visible_for_new_and_prior_sources():
    old = [{"identity": "old", "decoded_rgb_sha256": "a" * 64}]
    new = [
        {"identity": "new1", "decoded_rgb_sha256": "a" * 64},
        {"identity": "new2", "decoded_rgb_sha256": "a" * 64},
    ]
    result = m.decoded_hash_audit(new, old)
    assert len(result["collisions"]) == 2
    assert result["collisions"][1]["new_conflict"] == "new1"
    assert result["collisions"][0]["historical_conflicts"] == ["old"]
    assert not result["historical_inventory_complete"]
    with pytest.raises(ValueError):
        m.decoded_hash_audit([{"identity": "bad", "decoded_rgb_sha256": "bad"}], [])


def test_low_norm_boundary_and_all_fixed_floor_outcomes_are_retained():
    _, _, row = m.contrast_direction(cells([0, 0, 0]), learned=False)
    threshold = row["threshold"]
    assert not m.contrast_direction(cells([threshold, 0, 0]), learned=False)[2][
        "available"
    ]
    assert m.contrast_direction(
        cells([np.nextafter(threshold, np.inf), 0, 0]), learned=False
    )[2]["available"]
    results = [
        m.contrast_direction(cells([1e-5, 0, 0], 1), learned=True, relative_floor=f)[2][
            "available"
        ]
        for f in m.FLOORS
    ]
    assert results == [True, True, False]


def test_baseline_profile_matches_scalar_oracle_and_bad_site_shapes_fail(toy):
    s, _, _, _ = toy
    load = loader_for(s)
    actual = m.analyze_view(load, s, "raw_rgb/grid_mean")
    expected = oracle.profile(load, s, "raw_rgb/grid_mean")
    assert actual["paired_margin_change"] == pytest.approx(
        expected["paired_margin_change"], abs=1e-12
    )
    assert actual["relative_floor"] is None and not actual["primary_learned_profile"]
    s["observation"]["sites"]["encoder"] = [4]
    with pytest.raises(ValueError, match="full learned site shape"):
        m.analyze_view(load, s, "encoder")
