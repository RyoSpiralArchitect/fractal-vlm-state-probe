from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .cache_direction_holdout import (
    _load_interaction,
    analyze_pairing_seed_holdout,
    evaluate_pairing_holdout,
    holm_adjust,
)
from .cache_tensor_factorial import analyze_cache_tensor_factorial
from .cli.run_pairing_seed_validation import _read
from .stimulus import write_json

LEVELS = ("selected_06_of_08", "selected_07_of_08", "selected_08_of_08")
REGIONS = ("image_tokens", "post_image")


def directional_decomposition(reference, compared):
    a, b = (
        np.asarray(reference, dtype=np.float64),
        np.asarray(compared, dtype=np.float64),
    )
    if (
        a.shape != b.shape
        or a.size == 0
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError(
            "direction decomposition requires aligned finite nonempty vectors"
        )
    a, b = a.ravel(), b.ravel()
    aa, bb = float(np.dot(a, a)), float(np.dot(b, b))
    if aa == 0:
        return {"available": False, "reason": "zero reference vector"}
    alpha = float(np.dot(a, b) / aa)
    parallel = (alpha - 1) * a
    perpendicular = b - alpha * a
    delta = b - a
    delta_energy = float(np.dot(delta, delta))
    parallel_energy = float(np.dot(parallel, parallel))
    perpendicular_energy = float(np.dot(perpendicular, perpendicular))
    return {
        "available": True,
        "cosine": float(np.dot(a, b) / np.sqrt(aa * bb)) if bb else None,
        "norm_ratio": float(np.sqrt(bb / aa)),
        "reference_rms": float(np.sqrt(aa / a.size)),
        "compared_rms": float(np.sqrt(bb / b.size)),
        "projection_coefficient": alpha,
        "delta_rms": float(np.sqrt(delta_energy / a.size)),
        "parallel_delta_rms": float(np.sqrt(parallel_energy / a.size)),
        "perpendicular_delta_rms": float(np.sqrt(perpendicular_energy / a.size)),
        "perpendicular_delta_energy_fraction": perpendicular_energy / delta_energy
        if delta_energy
        else None,
        "relative_delta_l2": float(np.sqrt(delta_energy / aa)),
        "relative_pythagorean_residual": abs(
            delta_energy - parallel_energy - perpendicular_energy
        )
        / delta_energy
        if delta_energy
        else 0.0,
    }


def endpoint_reference_vectors(endpoint, compared, metadata):
    if set(endpoint) != set(compared) or set(endpoint) != set(metadata):
        raise ValueError("endpoint, compared and metadata labels differ")
    if any(m["replicate"] not in (1, 2, 3, 4) for m in metadata.values()):
        raise ValueError("unexpected reference/test replicate")
    return {
        k: endpoint[k] if metadata[k]["replicate"] in (1, 2) else compared[k]
        for k in metadata
    }


def input_frequency_context(study):
    result = []
    for record in study["records"]:
        endpoint = record["levels"][LEVELS[-1]]["gate"]
        for level in LEVELS:
            measured = record["levels"][level]
            gate = measured["gate"]
            result.append(
                {
                    "pair_id": record["pair_id"],
                    "level": level,
                    "within_level_accepted": gate["accepted"],
                    "common_centroid_target": gate["common_centroid_target"],
                    "common_hf_target": gate["common_hf_target"],
                    "centroid_target_relative_change_from_endpoint": gate[
                        "common_centroid_target"
                    ]
                    / endpoint["common_centroid_target"]
                    - 1,
                    "hf_target_change_from_endpoint": gate["common_hf_target"]
                    - endpoint["common_hf_target"],
                    "processor_centroid_by_cell": {
                        c: r["processor"]["spectral_centroid"]
                        for c, r in measured["cells"].items()
                    },
                    "processor_hf_by_cell": {
                        c: r["processor"]["high_frequency_energy_ratio"]
                        for c, r in measured["cells"].items()
                    },
                }
            )
    return result


def analyze_graded_panels(panel_paths, metadata, captures, root: Path):
    if tuple(panel_paths) != LEVELS:
        raise ValueError("graded cache analysis requires the three registered levels")
    primary, exploratory, transfer, decompositions, contrasts = [], [], [], [], []
    factorial_paths, holdout_paths = {}, []
    for target in captures:
        by_level, region_map = {}, None
        for level, blocks in panel_paths.items():
            analyses, paths, vectors = {}, {}, {}
            for label, cells in blocks.items():
                analysis = analyze_cache_tensor_factorial(
                    runs={c: _read(p) for c, p in cells.items()},
                    run_paths=cells,
                    layer_index=target.layer_index,
                    tensor=target.tensor,
                )
                path = root / "factorials" / level / label / f"{target.identifier}.json"
                write_json(path, analysis)
                analyses[label], paths[label] = analysis, path
                vectors[label], regions = _load_interaction(analysis)
                if region_map is not None and regions != region_map:
                    raise ValueError("graded cache token regions differ")
                region_map = regions
            holdout = analyze_pairing_seed_holdout(
                analyses, analysis_paths=paths, metadata=metadata
            )
            path = root / "holdout" / level / f"{target.identifier}.json"
            write_json(path, holdout)
            holdout_paths.append(str(path))
            factorial_paths[f"{level}/{target.identifier}"] = {
                k: str(v) for k, v in paths.items()
            }
            for view in holdout["views"]:
                row = {"condition": level, "target": target.identifier, **view}
                (primary if view["primary"] else exploratory).append(row)
            by_level[level] = vectors
        for region in REGIONS:
            regional = {
                level: {
                    k: np.take(v, region_map[region], axis=-2)
                    for k, v in vectors.items()
                }
                for level, vectors in by_level.items()
            }
            for level in LEVELS[:-1]:
                mixed = endpoint_reference_vectors(
                    regional[LEVELS[-1]], regional[level], metadata
                )
                result = evaluate_pairing_holdout(mixed, metadata)
                # Transfer is descriptive: omit a second inferential test family.
                result.pop("exact_block_test", None)
                native = next(
                    r
                    for r in primary
                    if (r["condition"], r["target"], r["view"])
                    == (level, target.identifier, region)
                )
                transfer.append(
                    {
                        "reference_condition": LEVELS[-1],
                        "test_condition": level,
                        "target": target.identifier,
                        "view": region,
                        "primary": False,
                        **result,
                        "endpoint_minus_native_reference_margin": result[
                            "mean_family_margin"
                        ]
                        - native["mean_family_margin"]
                        if result["available"] and native["available"]
                        else None,
                        "interpretation": "descriptive reference-frame transfer on the same test vectors; not a new hypothesis test",
                    }
                )
            for compared_level, reference_level in combinations(LEVELS, 2):
                for label in metadata:
                    decompositions.append(
                        {
                            "pair_id": label,
                            **metadata[label],
                            "phase": "reference"
                            if metadata[label]["replicate"] in (1, 2)
                            else "test",
                            "reference_condition": reference_level,
                            "compared_condition": compared_level,
                            "target": target.identifier,
                            "view": region,
                            **directional_decomposition(
                                regional[reference_level][label],
                                regional[compared_level][label],
                            ),
                        }
                    )
                left = next(
                    r
                    for r in primary
                    if (r["condition"], r["target"], r["view"])
                    == (compared_level, target.identifier, region)
                )
                right = next(
                    r
                    for r in primary
                    if (r["condition"], r["target"], r["view"])
                    == (reference_level, target.identifier, region)
                )
                available = left["available"] and right["available"]
                contrasts.append(
                    {
                        "compared_condition": compared_level,
                        "reference_condition": reference_level,
                        "target": target.identifier,
                        "view": region,
                        "available": available,
                        "compared_minus_reference_margin": left["mean_family_margin"]
                        - right["mean_family_margin"]
                        if available
                        else None,
                        "family_margin_differences": {
                            f: left["family_margins"][f] - right["family_margins"][f]
                            for f in left["family_margins"]
                        }
                        if available
                        else None,
                        "interpretation": "descriptive paired differences; significance labels are not a test of level differences",
                    }
                )
    if (
        len(primary),
        len(exploratory),
        len(transfer),
        len(decompositions),
        len(contrasts),
    ) != (18, 72, 12, 576, 18):
        raise ValueError("registered analysis denominator differs")
    for row, corrected in zip(
        primary,
        holm_adjust(
            [
                r["exact_block_test"]["p_greater"] if r["available"] else 1
                for r in primary
            ]
        ),
    ):
        row.update(holm_p_greater=corrected, holm_family_size=18)
    return {
        "primary_tests": primary,
        "primary_test_count": 18,
        "exploratory_views": exploratory,
        "exploratory_view_count": 72,
        "endpoint_reference_transfer": transfer,
        "directional_decompositions": decompositions,
        "paired_margin_differences": contrasts,
        "factorial_paths": factorial_paths,
        "holdout_paths": holdout_paths,
    }
