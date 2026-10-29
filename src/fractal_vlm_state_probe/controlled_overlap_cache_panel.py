from __future__ import annotations

import copy

import numpy as np

from .controlled_overlap import margin_cube, score_summary
from .controlled_overlap_cache import (
    CELLS,
    TARGETS,
    planned_panel_blocks,
    validate_capture,
)

VIEWS = [
    (target, region) for target in TARGETS for region in ("image_tokens", "post_image")
]


def ordered_jobs(metadata):
    blocks = planned_panel_blocks(metadata)
    return [
        {"phase": phase, "map": key, "pair_id": label, "cell": cell}
        for phase, replicates in (("reference", (1, 2)), ("test", (3, 4)))
        for key, label in blocks
        if metadata[label]["replicate"] in replicates
        for cell in CELLS
    ]


def validate_panel_capture(
    run, audit, historical, historical_inputs, measured, artifact, config
):
    expected_run, expected_inputs = (
        copy.deepcopy(historical),
        copy.deepcopy(historical_inputs),
    )
    expected_run["stream_events"][0]["frame_sha256s"] = [artifact["frame_sha256"]]
    expected_inputs["pixel_values"]["sha256"] = measured["pixel_values_sha256"]
    validate_capture(run, audit, expected_run, expected_inputs, config)


def require_reference_seal(registry, metadata):
    expected = {
        f"panel/reference/{job['map']}/{job['pair_id']}_{job['cell']}.json"
        for job in ordered_jobs(metadata)
        if job["phase"] == "reference"
    }
    selected = {k: v for k, v in registry.items() if k.startswith("panel/reference/")}
    if (
        len(selected) != 256
        or set(selected) != expected
        or any(k.startswith("panel/test/") for k in registry)
    ):
        raise ValueError("reference coverage or test-before-seal violation")
    return selected


def bootstrap_ranges(margins, spec):
    if margins.shape != (6, 4, 2, 3, 8, 2) or np.isinf(margins).any():
        raise ValueError("registered cache margin layout differs")
    available = [bool(np.isfinite(np.take(v, [0, 2], axis=2)).all()) for v in margins]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([spec["seed"]])))
    draws = np.empty((spec["draws"], 6, 2))
    for i in range(spec["draws"]):
        masks, families = rng.integers(0, 4, 4), rng.integers(0, 8, 8)
        sources = rng.integers(0, 2, (8, 2))
        values = np.zeros((6, 3))
        for mask in masks:
            destinations = rng.integers(0, 2, 2)
            block = margins[:, mask, destinations].mean(axis=1)
            values += (
                np.stack(
                    [
                        block[:, :, f, sources[j]].mean(axis=-1)
                        for j, f in enumerate(families)
                    ],
                    axis=-1,
                ).mean(axis=-1)
                / 4
            )
        draws[i, :, 0] = values[:, 0]
        draws[i, :, 1] = values[:, 2] - values[:, 0]
    return {
        "draws": spec["draws"],
        "seed": spec["seed"],
        "quantiles": spec["quantiles"],
        "interpretation": spec["interpretation"],
        "available_views": available,
        "ranges": [
            np.quantile(draws[:, i], spec["quantiles"], axis=0).T.tolist()
            if ok
            else None
            for i, ok in enumerate(available)
        ],
    }


def summarize_cube(cube, labels, families, metadata, config):
    if cube.shape != (6, 4, 2, 3, 16, 8) or np.isinf(cube).any():
        raise ValueError("registered score cube differs")
    margins = margin_cube(cube, labels, families, metadata)
    bootstrap = bootstrap_ranges(margins, config["analysis"]["bootstrap"])
    aggregates, endpoints = [], []
    for vi, (target, region) in enumerate(VIEWS):
        for qi, q in enumerate((0, 1, 4)):
            valid = bool(np.isfinite(cube[vi, :, :, qi]).all())
            row = {
                "target": target,
                "region": region,
                "q_sixteenths": q,
                "available": valid,
            }
            if valid:
                row.update(
                    score_summary(
                        cube[vi, :, :, qi].mean(axis=(0, 1)), labels, families, metadata
                    )
                )
            else:
                row["reason"] = "planned map score unavailable; no filtering"
            aggregates.append(row)
        for ei, name in enumerate(config["analysis"]["primary_endpoints"]):
            row = {
                "target": target,
                "region": region,
                "endpoint": name,
                "available": bootstrap["available_views"][vi],
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
                    exploratory_range=bootstrap["ranges"][vi][ei],
                )
            else:
                row["reason"] = (
                    "required zero/high score unavailable; both endpoints withheld for view"
                )
            endpoints.append(row)
    return aggregates, endpoints, bootstrap, margins


def direction(before, after):
    a, b = before.ravel(), after.ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all() or a.shape != b.shape:
        raise ValueError("invalid direction vectors")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {
        "zero_norm": na,
        "other_norm": nb,
        "cosine": float(np.dot(a, b) / (na * nb)) if na and nb else None,
        "norm_ratio": nb / na if na else None,
        "relative_l2_difference": float(np.linalg.norm(b - a) / na) if na else None,
    }
