from __future__ import annotations

import numpy as np

from .controlled_overlap import margin_cube, score_summary
from .controlled_overlap_cache_panel import ordered_jobs
from .nonlearned_processing import FEATURES, STARTS, direction_status

VIEWS = [(s, f) for s in STARTS for f in FEATURES]


def cell_path(job):
    return f"cells/{job['phase']}/{job['map']}/{job['pair_id']}_{job['cell']}.npz"


def feature_key(start, feature):
    return f"{start}__{feature}"


def validate_features(arrays):
    if set(arrays) != {feature_key(s, f) for s, f in VIEWS}:
        raise ValueError("twelve feature/start arrays required")
    for start, feature in VIEWS:
        a = arrays[feature_key(start, feature)]
        if (
            a.dtype != np.float64
            or a.shape != ((6,) if feature == "global_moments" else (192,))
            or not np.isfinite(a).all()
        ):
            raise ValueError("registered feature dtype, shape or finiteness differs")


def reference_seal(rows, metadata):
    expected = [cell_path(j) for j in ordered_jobs(metadata)[:256]]
    if (
        len(rows) != 256
        or [r["path"] for r in rows] != expected
        or [r["ordinal"] for r in rows] != list(range(129, 385))
    ):
        raise ValueError("reference feature coverage/order differs")
    return rows


def paired_ranges(margins, spec):
    if margins.shape != (12, 4, 2, 3, 8, 2) or np.isinf(margins).any():
        raise ValueError("registered twelve-view margin cube differs")
    available = [bool(np.isfinite(np.take(v, [0, 2], axis=2)).all()) for v in margins]
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([spec["seed"]]))
    )
    draws = np.empty((spec["draws"], 12, 2))
    for i in range(spec["draws"]):
        masks, families = generator.integers(0, 4, 4), generator.integers(0, 8, 8)
        seeds = generator.integers(0, 2, (8, 2))
        total = np.zeros((12, 3))
        for mask in masks:
            destinations = generator.integers(0, 2, 2)
            block = margins[:, mask, destinations].mean(axis=1)
            total += (
                np.stack(
                    [
                        block[:, :, f, seeds[j]].mean(axis=-1)
                        for j, f in enumerate(families)
                    ],
                    axis=-1,
                ).mean(axis=-1)
                / 4
            )
        draws[i, :, 0], draws[i, :, 1] = total[:, 0], total[:, 2] - total[:, 0]
    return {
        **spec,
        "available_views": available,
        "ranges": [
            np.quantile(draws[:, i], spec["quantiles"], axis=0).T.tolist()
            if ok
            else None
            for i, ok in enumerate(available)
        ],
    }


def summarize_scores(cube, labels, families, metadata, config):
    if cube.shape != (12, 4, 2, 3, 16, 8) or np.isinf(cube).any():
        raise ValueError("registered twelve-view score cube differs")
    plan = config["planned_panel"]
    margins = margin_cube(cube, labels, families, metadata)
    ranges = paired_ranges(margins, plan["bootstrap"])
    aggregates, endpoints = [], []
    for vi, (start, feature) in enumerate(VIEWS):
        for qi, q in enumerate((0, 1, 4)):
            row = {
                "start": start,
                "feature": feature,
                "q_sixteenths": q,
                "available": bool(np.isfinite(cube[vi, :, :, qi]).all()),
            }
            if row["available"]:
                row.update(
                    score_summary(
                        cube[vi, :, :, qi].mean(axis=(0, 1)), labels, families, metadata
                    )
                )
            else:
                row["reason"] = "planned score unavailable; no filtering"
            aggregates.append(row)
        for ei, endpoint in enumerate(plan["primary_endpoints"]):
            row = {
                "start": start,
                "feature": feature,
                "endpoint": endpoint,
                "available": ranges["available_views"][vi],
                "p_value": None,
                "equivalence_tested": False,
            }
            if row["available"]:
                values = (
                    margins[vi, :, :, 0]
                    if ei == 0
                    else margins[vi, :, :, 2] - margins[vi, :, :, 0]
                )
                row.update(
                    mean=float(values.mean()),
                    family_values=dict(
                        zip(families, values.mean(axis=(0, 1, 3)).tolist())
                    ),
                    mask_values=values.mean(axis=(1, 2, 3)).tolist(),
                    exploratory_range=ranges["ranges"][vi][ei],
                )
            else:
                row["reason"] = (
                    "required zero/high score unavailable; both endpoints withheld"
                )
            endpoints.append(row)
    return aggregates, endpoints, ranges, margins


def vector_diagnostics(vector, tolerance):
    status = direction_status(vector)
    bound = float(4 * np.sqrt(vector.size) * tolerance)
    return {
        **status,
        "conservative_feature_oracle_l2_bound": bound,
        "norm_to_oracle_bound_ratio": status["norm"] / bound,
        "interpretation": "numerical_sensitivity_diagnostic_only_no_filtering",
    }
