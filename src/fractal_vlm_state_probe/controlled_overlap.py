from __future__ import annotations

import numpy as np

from .graded_permutation import validate_permutation
from .source_coordinate_overlap import linear_summary


def rng(*keys):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(keys)))


def coordinate_counts(left, right):
    validate_permutation(left)
    validate_permutation(right)
    if left.shape != right.shape:
        raise ValueError("map dimensions differ")
    sites = np.arange(left.size)
    shared = left == right
    return {
        "left_fixed": int(np.sum(left == sites)),
        "right_fixed": int(np.sum(right == sites)),
        "shared_fixed": int(np.sum(shared & (left == sites))),
        "shared_moved": int(np.sum(shared & (left != sites))),
        "noncoincident": int(np.sum(~shared)),
        "content_pixels": left.size,
    }


def single_cycle(fixed, generator):
    result = np.arange(len(fixed))
    order = generator.permutation(np.flatnonzero(~fixed))
    if len(order) < 2:
        raise ValueError("at least two selected sites required")
    result[order] = np.roll(order, -1)
    return result


def construct_maps(config, *, size=76800):
    if size < 16 or size % 16:
        raise ValueError("content size must be a positive multiple of sixteen")
    maps, recipes = {}, {}
    for m, seed in enumerate(config["mask_seeds"]):
        order = rng(seed).permutation(size)
        fixed_sites, outside = order[: size // 4], order[size // 4 :]
        fixed = np.zeros(size, dtype=bool)
        fixed[fixed_sites] = True
        reference = f"m{m}_reference"
        maps[reference] = single_cycle(fixed, rng(config["reference_seed"], seed))
        recipes[reference] = {
            "mask": m,
            "q": None,
            "destination": None,
            "reference": reference,
            "attempt_shared_moved": [],
        }
        for q in config["overlap_sixteenths"]:
            count = size * q // 16
            test_fixed = np.zeros(size, dtype=bool)
            test_fixed[fixed_sites[:count]] = True
            test_fixed[outside[: size // 4 - count]] = True
            for d, destination_seed in enumerate(config["destination_seeds"]):
                key = f"m{m}_q{q}_d{d}"
                generator = rng(destination_seed, seed, q)
                attempts = []
                for _ in range(config["maximum_attempts_per_test_map"]):
                    candidate = single_cycle(test_fixed, generator)
                    audit = coordinate_counts(maps[reference], candidate)
                    attempts.append(audit["shared_moved"])
                    if audit["shared_moved"] == 0:
                        break
                else:
                    raise ValueError(
                        f"coordinate construction exhausted: {key}: {attempts}"
                    )
                if audit["shared_fixed"] != count or audit["right_fixed"] != size // 4:
                    raise ValueError("construction violated fixed-set counts")
                maps[key] = candidate
                recipes[key] = {
                    "mask": m,
                    "q": q,
                    "destination": d,
                    "reference": reference,
                    "attempt_shared_moved": attempts,
                }
    if len({v.tobytes() for v in maps.values()}) != len(maps):
        raise ValueError("duplicate constructed maps")
    return maps, recipes


def oracle_maps(config, *, size=76800):
    """Rebuild successors with explicit site assignment, independently of single_cycle."""

    def cycle(fixed_set, generator):
        selected = [u for u in range(size) if u not in fixed_set]
        sequence = generator.permutation(selected).tolist()
        result = list(range(size))
        for index, site in enumerate(sequence):
            result[site] = sequence[(index + 1) % len(sequence)]
        return np.asarray(result, dtype=np.int64)

    maps, attempts = {}, {}
    for m, seed in enumerate(config["mask_seeds"]):
        generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed])))
        order = generator.permutation(size).tolist()
        fixed, outside = order[: size // 4], order[size // 4 :]
        key = f"m{m}_reference"
        maps[key] = cycle(
            set(fixed),
            np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([config["reference_seed"], seed])
                )
            ),
        )
        for q in config["overlap_sixteenths"]:
            overlap = size * q // 16
            selected_fixed = set(fixed[:overlap] + outside[: size // 4 - overlap])
            for d, ds in enumerate(config["destination_seeds"]):
                name = f"m{m}_q{q}_d{d}"
                generator = np.random.Generator(
                    np.random.PCG64(np.random.SeedSequence([ds, seed, q]))
                )
                rejected = []
                for _ in range(config["maximum_attempts_per_test_map"]):
                    candidate = cycle(selected_fixed, generator)
                    shared = sum(
                        a == b and a != u
                        for u, (a, b) in enumerate(zip(maps[key], candidate))
                    )
                    rejected.append(shared)
                    if not shared:
                        break
                else:
                    raise ValueError("oracle exhausted coordinate attempts")
                maps[name], attempts[name] = candidate, rejected
    return maps, attempts


def verify_coordinate_recipe(config, maps, recipes):
    expected, attempts = oracle_maps(config, size=len(next(iter(maps.values()))))
    if set(maps) != set(expected) or set(recipes) != set(expected):
        raise ValueError("map recipe coverage differs")
    for key in maps:
        if not np.array_equal(maps[key], expected[key]):
            raise ValueError(f"independent map reconstruction differs: {key}")
        segments = key.split("_")
        mask = int(segments[0][1:])
        expected_recipe = {
            "mask": mask,
            "q": None if segments[1] == "reference" else int(segments[1][1:]),
            "destination": None if segments[1] == "reference" else int(segments[2][1:]),
            "reference": f"m{mask}_reference",
            "attempt_shared_moved": attempts.get(key, []),
        }
        if recipes[key]["attempt_shared_moved"] != attempts.get(key, []):
            raise ValueError("coordinate retry receipt differs")
        if recipes[key] != expected_recipe:
            raise ValueError("coordinate recipe metadata differs")
    return {"maps_verified": len(maps), "source_images_used": 0}


def score_summary(scores, labels, families, metadata):
    saved = linear_summary(
        {
            label: dict(zip(families, row.tolist()))
            for label, row in zip(labels, scores)
        },
        metadata,
    )
    winners = []
    for label, row in zip(labels, scores):
        maximum = np.max(row)
        best = np.flatnonzero(np.abs(row - maximum) <= 1e-12)
        winners.append(families[best[0]] if len(best) == 1 else None)
    saved["retrieval_correct_count"] = sum(
        f == metadata[k]["pairing_family"] for k, f in zip(labels, winners)
    )
    saved["retrieval_tied_count"] = winners.count(None)
    return saved


def margin_cube(score_cube, labels, families, metadata):
    # Input axes: view, mask, destination, q, test seed, reference family.
    result = np.empty((*score_cube.shape[:4], len(families), 2))
    broad = {
        f: next(m["broad_class"] for m in metadata.values() if m["pairing_family"] == f)
        for f in families
    }
    for i, family in enumerate(families):
        other = [
            j
            for j, f in enumerate(families)
            if f != family and broad[f] == broad[family]
        ]
        own_labels = [
            labels.index(k) for k in labels if metadata[k]["pairing_family"] == family
        ]
        if len(own_labels) != 2 or len(other) != 3:
            raise ValueError("registered hierarchy differs")
        for r, index in enumerate(own_labels):
            selected = score_cube[..., index, :]
            result[..., i, r] = selected[..., i] - np.mean(
                selected[..., other], axis=-1
            )
    return result


def paired_bootstrap(margins, config):
    if margins.shape != (2, 4, 2, 3, 8, 2) or not np.isfinite(margins).all():
        raise ValueError("registered paired bootstrap shape differs")
    spec = config["bootstrap"]
    generator = rng(spec["seed"])
    draws = np.empty((spec["draws"], 2, 2))
    for b in range(spec["draws"]):
        masks = generator.integers(0, 4, 4)
        families = generator.integers(0, 8, 8)
        sources = generator.integers(0, 2, (8, 2))
        value = np.zeros((2, 3))
        for mask in masks:
            destinations = generator.integers(0, 2, 2)
            block = margins[:, mask, destinations].mean(axis=1)
            selected = np.stack(
                [
                    block[:, :, f, sources[i]].mean(axis=-1)
                    for i, f in enumerate(families)
                ],
                axis=-1,
            )
            value += selected.mean(axis=-1) / 4
        draws[b, :, 0] = value[:, 0]
        draws[b, :, 1] = value[:, 2] - value[:, 0]
    return {
        "draws": spec["draws"],
        "quantiles": spec["quantiles"],
        "ranges": np.quantile(draws, spec["quantiles"], axis=0)
        .transpose(1, 2, 0)
        .tolist(),
        "interpretation": spec["interpretation"],
    }
