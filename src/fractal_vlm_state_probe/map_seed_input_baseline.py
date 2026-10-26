from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .cache_direction_holdout import evaluate_pairing_holdout, holm_adjust
from .graded_cache import directional_decomposition, endpoint_reference_vectors
from .map_seed_cache import aggregate_map_scores
from .permutation_maps import ANCHOR, MAPS, NEW_MAPS

VIEWS = ("raw_rgb", "processor_pixel_values")


def validate_config(config):
    expected = {
        "schema_version": 1,
        "study_id": "map_seed_input_baseline_fastvlm_v1",
        "maps": list(MAPS),
        "views": list(VIEWS),
        "raw_coordinates": "full_320_square_rgb_float64_div_255_including_padding",
        "processor_coordinates": "full_native_pixel_values_promoted_to_float64",
        "interaction": "jj_minus_jm_minus_mj_plus_mm",
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "primary_map_scope": "three_new_maps_only",
        "primary_regimes": ["same_map", "cross_map"],
        "primary_test_family_size": 4,
        "descriptive_matrix_views": 32,
        "directional_records": 384,
        "expected_cells": 512,
        "expected_interaction_vectors": 256,
        "raw_equivariance_tolerance": 1e-12,
        "new_cache_forwards": 0,
        "model_weights_loaded": False,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("input baseline differs from registration")


def canonical_raw(vector, mapping):
    if vector.shape != (320, 320, 3) or mapping.shape != (76800,):
        raise ValueError("raw canonicalization shape differs")
    if not np.array_equal(np.sort(mapping), np.arange(76800)):
        raise ValueError("raw canonicalization requires a permutation")
    result = vector.copy()
    result[40:280] = (
        vector[40:280].reshape(-1, 3)[np.argsort(mapping)].reshape(240, 320, 3)
    )
    return result


def raw_equivariance(vectors, maps, matrix):
    rows = []
    for label in vectors[ANCHOR]:
        anchor = canonical_raw(vectors[ANCHOR][label], maps[ANCHOR])
        for key in NEW_MAPS:
            value = canonical_raw(vectors[key][label], maps[key])
            equal = np.array_equal(value, anchor)
            if not equal:
                raise ValueError(
                    "raw interaction is not a common coordinate permutation"
                )
            rows.append({"pair_id": label, "map": key, "exact_inverse_equal": equal})
    diagonals = {
        r["reference_map"]: r for r in matrix if r["reference_map"] == r["test_map"]
    }
    if set(diagonals) != set(MAPS):
        raise ValueError("raw diagonal coverage differs")
    if any(not r["available"] for r in diagonals.values()):
        error = None
    else:
        lookups = {
            k: {r["label"]: r["reference_cosines"] for r in v["test_seed_scores"]}
            for k, v in diagonals.items()
        }
        error = max(
            abs(value - lookups[k][label][family])
            for k in NEW_MAPS
            for label, values in lookups[ANCHOR].items()
            for family, value in values.items()
        )
        if error > 1e-12:
            raise ValueError("same-map raw cosines violate permutation isometry")
    return {"inverse_checks": rows, "diagonal_cosine_max_abs_error": error}


def analyze_vectors(paths, metadata, maps):
    if set(paths) != set(VIEWS):
        raise ValueError("both input coordinate systems are required")
    matrix, primary, directions, gaps = [], [], [], []
    equivariance = None
    for view in VIEWS:
        if set(paths[view]) != set(MAPS) or any(
            set(p) != set(metadata) for p in paths[view].values()
        ):
            raise ValueError("input vector map/block coverage differs")
        vectors = {
            k: {
                label: np.load(Path(p), mmap_mode="r", allow_pickle=False)
                for label, p in blocks.items()
            }
            for k, blocks in paths[view].items()
        }
        block_rows = []
        for reference in MAPS:
            for test in MAPS:
                result = evaluate_pairing_holdout(
                    endpoint_reference_vectors(
                        vectors[reference], vectors[test], metadata
                    ),
                    metadata,
                )
                result.pop("exact_block_test", None)
                block_rows.append(
                    {
                        "reference_map": reference,
                        "test_map": test,
                        "view": view,
                        "primary": False,
                        **result,
                    }
                )
        matrix.extend(block_rows)
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
                    "view": view,
                    "regime": regime,
                    "primary": True,
                    **aggregate_map_scores(selected, metadata, regime),
                }
            )
        same, cross = primary[-2:]
        available = same["available"] and cross["available"]
        gaps.append(
            {
                "view": view,
                "same_minus_cross_mean_margin": same["mean_family_margin"]
                - cross["mean_family_margin"]
                if available
                else None,
                "family_margin_differences": {
                    f: same["family_margins"][f] - cross["family_margins"][f]
                    for f in same["family_margins"]
                }
                if available
                else None,
            }
        )
        for a, b in combinations(MAPS, 2):
            for label, meta in metadata.items():
                directions.append(
                    {
                        "pair_id": label,
                        **meta,
                        "phase": "reference" if meta["replicate"] in (1, 2) else "test",
                        "reference_condition": a,
                        "compared_condition": b,
                        "target": "input",
                        "view": view,
                        **directional_decomposition(
                            vectors[a][label], vectors[b][label]
                        ),
                    }
                )
        if view == "raw_rgb":
            equivariance = raw_equivariance(vectors, maps, block_rows)
    adjusted = holm_adjust(
        [r["exact_block_test"]["p_greater"] if r["available"] else 1 for r in primary]
    )
    for row, p in zip(primary, adjusted):
        row.update(holm_p_greater=p, holm_family_size=4)
    if len(matrix) != 32 or len(primary) != 4 or len(directions) != 384:
        raise ValueError("input analysis denominator differs")
    return {
        "primary_tests": primary,
        "map_matrix": matrix,
        "same_cross_gaps": gaps,
        "directional_decompositions": directions,
        "raw_equivariance": equivariance,
    }
