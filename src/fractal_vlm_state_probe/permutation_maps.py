from __future__ import annotations

import numpy as np

from .graded_permutation import restricted_permutation, validate_permutation

ANCHOR = "historical_3q"
NEW_SEEDS = (20261016, 20261017, 20261018)
NEW_MAPS = tuple(f"seed_{seed}" for seed in NEW_SEEDS)
MAPS = (ANCHOR, *NEW_MAPS)


def seeded_map(seed: int, size: int):
    if type(seed) is not int or seed < 0 or size <= 0 or size % 8:
        raise ValueError(
            "map requires a nonnegative integer seed and size divisible by 8"
        )
    selection_seed, full_seed = np.random.SeedSequence(seed).spawn(2)
    selection = np.random.Generator(np.random.PCG64(selection_seed)).permutation(size)
    full = np.random.Generator(np.random.PCG64(full_seed)).permutation(size)
    return full, selection, restricted_permutation(full, selection, size * 6 // 8)


def map_overlap(left, right):
    validate_permutation(left)
    validate_permutation(right)
    if left.shape != right.shape:
        raise ValueError("map sizes differ")
    identity = np.arange(len(left))
    a, b = left != identity, right != identity
    union = int(np.count_nonzero(a | b))
    return {
        "equal_destination_fraction": float(np.mean(left == right)),
        "moved_support_jaccard": float(np.count_nonzero(a & b) / union)
        if union
        else None,
    }


def summarize_input_maps(records):
    if len(records) != 32 or len({r["pair_id"] for r in records}) != 32:
        raise ValueError("map audit requires all 32 unique blocks")
    families = {}
    for record in records:
        if set(record["maps"]) != set(MAPS):
            raise ValueError("map audit denominator differs")
        families.setdefault(record["pairing_family"], []).append(record["replicate"])
    if len(families) != 8 or any(sorted(v) != [1, 2, 3, 4] for v in families.values()):
        raise ValueError("map audit requires eight complete four-seed families")
    result = {}
    for key in MAPS:
        accepted = [r for r in records if r["maps"][key]["gate"]["accepted"]]
        result[key] = {
            "accepted_blocks": len(accepted),
            "total_blocks": 32,
            "accepted_pair_ids": [r["pair_id"] for r in accepted],
            "rejected_pair_ids": [
                r["pair_id"] for r in records if not r["maps"][key]["gate"]["accepted"]
            ],
            "complete_four_seed_families": sorted(
                f
                for f in families
                if {r["replicate"] for r in accepted if r["pairing_family"] == f}
                == {1, 2, 3, 4}
            ),
        }
    return result
