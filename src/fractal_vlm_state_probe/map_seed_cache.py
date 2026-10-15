from __future__ import annotations

from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np

from .cache_direction_holdout import (
    _load_interaction,
    evaluate_pairing_holdout,
    holm_adjust,
    validate_pairing_holdout_metadata,
)
from .cache_tensor_factorial import analyze_cache_tensor_factorial
from .cli.run_pairing_seed_validation import _read
from .graded_cache import directional_decomposition, endpoint_reference_vectors
from .permutation_maps import MAPS, NEW_MAPS
from .stimulus import write_json

REGIONS = ("image_tokens", "post_image")


def aggregate_map_scores(rows, metadata, regime):
    validate_pairing_holdout_metadata(metadata)
    expected_pairs = {
        (a, b) for a in NEW_MAPS for b in NEW_MAPS if (a == b) == (regime == "same_map")
    }
    if regime not in ("same_map", "cross_map"):
        raise ValueError("unknown map regime")
    if (
        len(rows) != len(expected_pairs)
        or {(r["reference_map"], r["test_map"]) for r in rows} != expected_pairs
    ):
        raise ValueError("aggregate must include every registered ordered map pair")
    if any(not r["available"] for r in rows):
        return {
            "available": False,
            "reason": "at least one required map pair is unavailable",
        }
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    broad = {
        f: next(m["broad_class"] for m in metadata.values() if m["pairing_family"] == f)
        for f in families
    }
    lookups = []
    for row in rows:
        scores = {r["label"]: r for r in row["test_seed_scores"]}
        if len(row["test_seed_scores"]) != len(labels) or set(scores) != set(labels):
            raise ValueError("aggregate test labels differ")
        if any(set(s["reference_cosines"]) != set(families) for s in scores.values()):
            raise ValueError("aggregate reference families differ")
        lookups.append(scores)
    # Average scores before assignment; the same family permutation acts on every map.
    scores = []
    for label in labels:
        family = metadata[label]["pairing_family"]
        values = {
            f: float(np.mean([r[label]["reference_cosines"][f] for r in lookups]))
            for f in families
        }
        if not np.isfinite(list(values.values())).all():
            raise ValueError("nonfinite map scores")
        alternatives = [
            f for f in families if f != family and broad[f] == broad[family]
        ]
        other = float(np.mean([values[f] for f in alternatives]))
        winners = [
            f for f in families if abs(values[f] - max(values.values())) <= 1e-12
        ]
        scores.append(
            {
                "label": label,
                **metadata[label],
                "reference_cosines": values,
                "own_family_cosine": values[family],
                "same_broad_other_family_cosine": other,
                "margin": values[family] - other,
                "retrieved_family": winners[0] if len(winners) == 1 else None,
                "retrieval_correct": len(winners) == 1 and winners[0] == family,
            }
        )
    matrix = np.array(
        [
            [
                np.mean(
                    [
                        r["reference_cosines"][f]
                        for r in scores
                        if r["pairing_family"] == family
                    ]
                )
                for f in families
            ]
            for family in families
        ]
    )
    margins = {
        f: float(np.mean([r["margin"] for r in scores if r["pairing_family"] == f]))
        for f in families
    }
    observed = float(np.mean(list(margins.values())))
    groups = [
        [i for i, f in enumerate(families) if broad[f] == b]
        for b in sorted(set(broad.values()))
    ]
    null = []
    for assignments in product(*(permutations(g) for g in groups)):
        null.append(
            float(
                np.mean(
                    [
                        matrix[i, j] - np.mean([matrix[i, k] for k in group if k != j])
                        for group, assignment in zip(groups, assignments)
                        for i, j in zip(group, assignment)
                    ]
                )
            )
        )
    extreme = sum(v >= observed - 1e-12 for v in null)
    return {
        "available": True,
        "test_seed_count": len(labels),
        "reference_labels": sorted(set(metadata) - set(labels)),
        "test_labels": labels,
        "ordered_map_pairs": [list(p) for p in sorted(expected_pairs)],
        "map_pair_count": len(rows),
        "own_family_cosine_mean": float(
            np.mean([r["own_family_cosine"] for r in scores])
        ),
        "other_family_cosine_mean": float(
            np.mean([r["same_broad_other_family_cosine"] for r in scores])
        ),
        "mean_family_margin": observed,
        "family_margins": margins,
        "positive_family_margin_count": sum(v > 0 for v in margins.values()),
        "retrieval_correct_count": sum(r["retrieval_correct"] for r in scores),
        "retrieval_tied_count": sum(r["retrieved_family"] is None for r in scores),
        "test_seed_scores": scores,
        "exact_block_test": {
            "assignment_count": len(null),
            "extreme_count": extreme,
            "p_greater": extreme / len(null),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "permutation_unit": "complete two-test-seed pairing-family block moved synchronously across maps",
            "multiplicity_adjusted": False,
        },
    }


def analyze_map_panels(paths, metadata, captures, root: Path):
    if tuple(paths) != MAPS:
        raise ValueError("all four registered maps are required")
    matrix_rows, primary, decompositions, factorial_paths, gaps = [], [], [], {}, []
    for target in captures:
        vectors, regions = {}, None
        for key in MAPS:
            vectors[key], factorial_paths[f"{key}/{target.identifier}"] = {}, {}
            for label, cells in paths[key].items():
                analysis = analyze_cache_tensor_factorial(
                    runs={c: _read(p) for c, p in cells.items()},
                    run_paths=cells,
                    layer_index=target.layer_index,
                    tensor=target.tensor,
                )
                path = root / "factorials" / key / label / f"{target.identifier}.json"
                write_json(path, analysis)
                factorial_paths[f"{key}/{target.identifier}"][label] = str(path)
                vector, current = _load_interaction(analysis)
                if regions is not None and current != regions:
                    raise ValueError("map token regions differ")
                regions = current
                vectors[key][label] = vector
        for view in REGIONS:
            regional = {
                k: {
                    label: np.take(v, regions[view], axis=-2)
                    for label, v in values.items()
                }
                for k, values in vectors.items()
            }
            block_rows = []
            for reference_map in MAPS:
                for test_map in MAPS:
                    mixed = endpoint_reference_vectors(
                        regional[reference_map], regional[test_map], metadata
                    )
                    result = evaluate_pairing_holdout(mixed, metadata)
                    result.pop("exact_block_test", None)
                    row = {
                        "reference_map": reference_map,
                        "test_map": test_map,
                        "target": target.identifier,
                        "view": view,
                        "primary": False,
                        **result,
                    }
                    block_rows.append(row)
            matrix_rows.extend(block_rows)
            for regime in ("same_map", "cross_map"):
                selected = [
                    r
                    for r in block_rows
                    if r["reference_map"] in NEW_MAPS
                    and r["test_map"] in NEW_MAPS
                    and (r["reference_map"] == r["test_map"]) == (regime == "same_map")
                ]
                primary.append(
                    {
                        "regime": regime,
                        "target": target.identifier,
                        "view": view,
                        "primary": True,
                        **aggregate_map_scores(selected, metadata, regime),
                    }
                )
            same, cross = primary[-2:]
            gaps.append(
                {
                    "target": target.identifier,
                    "view": view,
                    "same_minus_cross_mean_margin": same["mean_family_margin"]
                    - cross["mean_family_margin"]
                    if same["available"] and cross["available"]
                    else None,
                    "family_margin_differences": {
                        f: same["family_margins"][f] - cross["family_margins"][f]
                        for f in same["family_margins"]
                    }
                    if same["available"] and cross["available"]
                    else None,
                    "interpretation": "descriptive gap, not a between-regime significance test",
                }
            )
            for reference_map, compared_map in combinations(MAPS, 2):
                for label, meta in metadata.items():
                    decompositions.append(
                        {
                            "pair_id": label,
                            **meta,
                            "phase": "reference"
                            if meta["replicate"] in (1, 2)
                            else "test",
                            "reference_condition": reference_map,
                            "compared_condition": compared_map,
                            "target": target.identifier,
                            "view": view,
                            **directional_decomposition(
                                regional[reference_map][label],
                                regional[compared_map][label],
                            ),
                        }
                    )
    if len(primary) != 12 or len(matrix_rows) != 96 or len(decompositions) != 1152:
        raise ValueError("map analysis denominator differs")
    adjusted = holm_adjust(
        [r["exact_block_test"]["p_greater"] if r["available"] else 1 for r in primary]
    )
    for row, p in zip(primary, adjusted):
        row.update(holm_p_greater=p, holm_family_size=12)
    return {
        "primary_tests": primary,
        "map_matrix": matrix_rows,
        "directional_decompositions": decompositions,
        "same_cross_gaps": gaps,
        "factorial_paths": factorial_paths,
    }
