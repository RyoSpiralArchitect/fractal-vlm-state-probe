from __future__ import annotations

import hashlib

import numpy as np

from .controlled_overlap import margin_cube, score_summary

SITES = ("encoder", "projector")
SHAPES = {"encoder": (1, 256, 3072), "projector": (1, 256, 896)}
ENDPOINTS = ("zero_overlap_margin", "high_minus_zero_margin")
GIB = 2**30


def digest(a):
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def expand(words):
    if words.dtype != np.dtype("<u2"):
        raise ValueError("expected little-endian uint16 words")
    return (words.astype("<u4") << 16).view("<f4")


def oracle_expand(words):
    if words.dtype != np.dtype("<u2"):
        raise ValueError("invalid word dtype")
    # Write the high half of each IEEE float explicitly, independently of the shift.
    halves = np.zeros((*words.shape, 2), dtype="<u2")
    halves[..., 1] = words
    return halves.view("<f4").reshape(words.shape)


def pack(a):
    if a.dtype != np.dtype("<f4") or not np.isfinite(a).all():
        raise ValueError("finite float32 bfloat16 expansion required")
    words = (a.view("<u4") >> 16).astype("<u2")
    if expand(words).tobytes() != a.tobytes():
        raise ValueError("not an exact bfloat16 expansion")
    return words


def codec_check():
    all_words = np.arange(65536, dtype="<u2")
    finite = all_words[((all_words >> 7) & 255) != 255]
    direct = oracle_expand(finite)
    if finite.size != 65280 or not np.array_equal(pack(direct), finite):
        raise ValueError("exhaustive finite-word codec qualification failed")
    return int(finite.size)


def disk_budget(config):
    scalar_count = sum(int(np.prod(shape)) for shape in SHAPES.values())
    cache_per_context = 3 * 1 * 2 * 397 * 64 * 4
    parts = {
        "packed_panel_bytes": 2048 * scalar_count * 2,
        "panel_cache_bytes": 2048 * cache_per_context,
        "fresh_calibration_bytes": 128 * scalar_count * 4 + 384 * cache_per_context,
        "peak_scratch_bytes": 64 * int(np.prod(SHAPES["encoder"])) * 8,
        "overhead_allowance_bytes": config["overhead_allowance_bytes"],
        "reserve_bytes": config["reserve_bytes"],
    }
    return {**parts, "required_at_launch_bytes": sum(parts.values())}


def validate_config(config):
    expected = {
        "study_id": "representation_correspondence_panel_v1",
        "sites": list(SITES),
        "calibration_contexts": 384,
        "reference_cells": 256,
        "test_cells": 1792,
        "panel_cells": 2048,
        "total_source_contexts": 2432,
        "cell_arrays": 4096,
        "panel_cache_parity_pairs": 6144,
        "factorial_blocks": 512,
        "interaction_vectors": 1024,
        "score_matrices": 56,
        "direction_records": 512,
        "primary_endpoints": list(ENDPOINTS),
        "retry_or_resume": False,
        "cancelling_unit_reference_threshold": 1e-12,
        "score_absolute_tolerance": 1e-12,
        "norm_relative_tolerance": 1e-11,
        "maximum_scratch_vectors": 64,
        "reserve_bytes": 8 * GIB,
        "overhead_allowance_bytes": GIB,
        "new_direct_probe_cells": 0,
        "new_p_values": 0,
        "equivalence_tested": False,
        "confirmation_cohort": "not_generated_not_sealed",
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("registered representation panel contract differs")
    b = config["bootstrap"]
    if (
        b["draws"] != 10000
        or b["seed"] != 2026110301
        or b["quantiles"] != [0.025 / 4, 1 - 0.025 / 4]
    ):
        raise ValueError("four-endpoint bootstrap contract differs")


def run_path(job):
    return f"panel/{job['phase']}/{job['map']}/{job['pair_id']}_{job['cell']}.json"


def array_path(job):
    return f"representations/{job['phase']}/{job['map']}/{job['pair_id']}_{job['cell']}.npz"


def seal(rows, jobs):
    if len(rows) != 256 or len(jobs) != 2048:
        raise ValueError("reference denominator differs")
    for i, (row, job) in enumerate(zip(rows, jobs)):
        if (
            job["phase"] != "reference"
            or any(row[k] != v for k, v in job.items())
            or row["ordinal"] != 385 + i
            or row["path"] != run_path(job)
            or row["representation"]["path"] != array_path(job)
        ):
            raise ValueError("reference identity or order differs")
    if jobs[256]["phase"] != "test":
        raise ValueError("test must start after sealed references")
    return {
        "reference_cells": 256,
        "completed_contexts_at_seal": 640,
        "first_test_ordinal": 641,
    }


def bootstrap(margins, spec):
    if margins.shape != (2, 4, 2, 3, 8, 2) or np.isinf(margins).any():
        raise ValueError("invalid paired margin cube")
    available = np.isfinite(np.take(margins, [0, 2], axis=3)).all(axis=(1, 2, 3, 4, 5))
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([spec["seed"]]))
    )
    draws = np.empty((spec["draws"], 2, 2))
    for i in range(spec["draws"]):
        masks, families = generator.integers(0, 4, 4), generator.integers(0, 8, 8)
        seeds = generator.integers(0, 2, (8, 2))
        mf, fs = np.zeros((4, 2)), np.zeros((8, 2))
        for mask in masks:
            np.add.at(mf[mask], generator.integers(0, 2, 2), 1)
        for j, family in enumerate(families):
            np.add.at(fs[family], seeds[j], 1)
        means = np.einsum("md,fr,smdqfr->sq", mf, fs, margins) / 128
        draws[i, :, 0], draws[i, :, 1] = means[:, 0], means[:, 2] - means[:, 0]
    return {
        **spec,
        "available_views": available.tolist(),
        "ranges": [
            np.quantile(draws[:, i], spec["quantiles"], axis=0).T.tolist()
            if ok
            else None
            for i, ok in enumerate(available)
        ],
    }


def summarize(cube, labels, families, metadata, spec):
    if cube.shape != (2, 4, 2, 3, 16, 8) or np.isinf(cube).any():
        raise ValueError("invalid site score cube")
    margins = margin_cube(cube, labels, families, metadata)
    ranges = bootstrap(margins, spec)
    aggregates, endpoints = [], []
    for vi, site in enumerate(SITES):
        for qi, q in enumerate((0, 1, 4)):
            values = cube[vi, :, :, qi]
            row = {
                "site": site,
                "q_sixteenths": q,
                "available": bool(np.isfinite(values).all()),
            }
            if row["available"]:
                row.update(
                    score_summary(values.mean(axis=(0, 1)), labels, families, metadata)
                )
            else:
                row["reason"] = "unavailable map score; no filtering"
            aggregates.append(row)
        for ei, name in enumerate(ENDPOINTS):
            row = {
                "site": site,
                "endpoint": name,
                "available": ranges["available_views"][vi],
                "p_value": None,
                "equivalence_tested": False,
            }
            if row["available"]:
                zero = margins[vi, :, :, 0]
                values = zero if ei == 0 else margins[vi, :, :, 2] - zero
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
