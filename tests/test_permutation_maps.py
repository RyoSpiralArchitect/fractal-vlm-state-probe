import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.run_map_seed_input_study import validate_config
from fractal_vlm_state_probe.cli.summarize_graded_permutation_study import cycle_oracle
from fractal_vlm_state_probe.permutation_maps import (
    MAPS,
    NEW_SEEDS,
    map_overlap,
    seeded_map,
    summarize_input_maps,
)


def test_seeded_maps_reproduce_independent_cycle_oracle():
    results = []
    for seed in NEW_SEEDS:
        full, selection, mapping = seeded_map(seed, 96)
        assert np.array_equal(mapping, cycle_oracle(full, selection, 72))
        assert all(
            np.array_equal(x, y)
            for x, y in zip((full, selection, mapping), seeded_map(seed, 96))
        )
        assert np.array_equal(np.sort(mapping), np.arange(96))
        assert set(np.flatnonzero(mapping != np.arange(96))) <= set(selection[:72])
        results.append(mapping.tobytes())
    assert len(set(results)) == 3


@pytest.mark.parametrize("seed,size", [(True, 8), (-1, 8), (1.0, 8), (1, 7), (1, 0)])
def test_seeded_map_rejects_invalid_specification(seed, size):
    with pytest.raises(ValueError):
        seeded_map(seed, size)


def test_map_overlap_keeps_undefined_empty_support():
    identity = np.arange(8)
    assert map_overlap(identity, identity) == {
        "equal_destination_fraction": 1.0,
        "moved_support_jaccard": None,
    }
    moved = np.roll(identity, 1)
    assert map_overlap(identity, moved) == {
        "equal_destination_fraction": 0.0,
        "moved_support_jaccard": 0.0,
    }
    with pytest.raises(ValueError):
        map_overlap(identity, np.arange(16))


def complete_records():
    return [
        {
            "pair_id": f"f{f}_r{r}",
            "pairing_family": f"f{f}",
            "replicate": r,
            "maps": {m: {"gate": {"accepted": True}} for m in MAPS},
        }
        for f in range(8)
        for r in range(1, 5)
    ]


def test_all_maps_and_failed_block_denominators_retained():
    records = complete_records()
    assert all(
        r["accepted_blocks"] == 32 for r in summarize_input_maps(records).values()
    )
    records[0]["maps"][MAPS[1]]["gate"]["accepted"] = False
    result = summarize_input_maps(records)[MAPS[1]]
    assert result["accepted_blocks"] == 31
    assert result["total_blocks"] == 32
    assert len(result["complete_four_seed_families"]) == 7
    assert result["rejected_pair_ids"] == ["f0_r1"]


@pytest.mark.parametrize("corruption", ["block", "map", "replicate", "duplicate"])
def test_partial_panels_cannot_be_relabelled_complete(corruption):
    records = complete_records()
    if corruption == "block":
        records.pop()
    elif corruption == "map":
        records[0]["maps"].pop(MAPS[-1])
    elif corruption == "replicate":
        records[0]["replicate"] = 2
    else:
        records[-1] = copy.deepcopy(records[0])
    with pytest.raises(ValueError):
        summarize_input_maps(records)


def test_registered_config_rejects_seed_replacement_or_gate_relaxation():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/map_seed_input_fastvlm_v1.json").read_text())
    validate_config(config)
    for key, value in (
        ("new_map_seeds", [1, 2, 3]),
        ("selected_numerator", 7),
        ("centroid_relative_tolerance", 0.06),
        ("new_cache_forwards", 1),
    ):
        with pytest.raises(ValueError):
            validate_config({**config, key: value})
