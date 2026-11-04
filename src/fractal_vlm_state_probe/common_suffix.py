"""Complete, token-aligned cache comparisons for the fixed post-image prompt edit."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .native_suffix_replay import NAMES
from .representation_panel import oracle_expand


def panel_anchors(rows):
    if len(rows) != 96 or [r["ordinal"] for r in rows] != list(range(1, 97)):
        raise ValueError("historical row denominator/order differs")
    chosen = []
    for i in range(0, 96, 3):
        a, b, closing = rows[i : i + 3]
        if (
            [r["stage"] for r in (a, b, closing)] != ["ack_before", "yes", "ack_after"]
            or len({(r["pair_id"], r["cell"]) for r in (a, b, closing)}) != 1
            or a["prepared_ids"] != closing["prepared_ids"]
        ):
            raise ValueError("unpaired historical source")
        if (
            len(a["prepared_ids"]) != 140
            or len(b["prepared_ids"]) != 140
            or [
                (i, x, y)
                for i, (x, y) in enumerate(
                    zip(a["prepared_ids"], b["prepared_ids"], strict=True)
                )
                if x != y
            ]
            != [(132, 53763, 14080)]
        ):
            raise ValueError("not the single-token historical prompt edit")
        chosen.extend((a, b))
    if len({(r["pair_id"], r["cell"]) for r in chosen}) != 32:
        raise ValueError("duplicated historical image")
    return chosen


def expected_anchor(row, suffix):
    return {
        **row,
        "suffix_ids": list(suffix),
        "cache_state_ids": row["prepared_ids"] + list(suffix),
    }


def region_metrics(x, y):
    if (
        x.dtype != np.dtype("<u2")
        or y.dtype != x.dtype
        or x.shape != y.shape
        or not x.size
    ):
        raise ValueError("matched nonempty native-word arrays required")
    different = x != y
    a, b = (
        oracle_expand(x).astype(np.float64).ravel(),
        oracle_expand(y).astype(np.float64).ravel(),
    )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite representation")
    delta = b - a
    na, nb, nd = (
        float(np.linalg.norm(a)),
        float(np.linalg.norm(b)),
        float(np.linalg.norm(delta)),
    )
    return {
        "changed_scalars": int(different.sum()),
        "scalar_pairs": int(x.size),
        "changed_relative_positions": np.flatnonzero(
            different.any(axis=(0, 1, 3))
        ).tolist(),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "max_abs": float(np.abs(delta).max()),
        "ack_l2": na,
        "yes_l2": nb,
        "delta_l2": nd,
        "relative_l2": nd / na if na else None,
        "cosine": float(np.dot(a, b) / (na * nb)) if na and nb else None,
    }


def verify_metric(x, y, metrics, rtol, atol):
    # Separate byte scan and scalar reductions, independent of vectorized metric calculation.
    changed = [
        i
        for i in range(x.shape[-2])
        if x[..., i, :].tobytes() != y[..., i, :].tobytes()
    ]
    if changed != metrics["changed_relative_positions"]:
        raise ValueError("independent changed-position check differs")
    a = ((x.astype("<u4") * 65536).view("<f4")).astype(np.float64).ravel()
    b = ((y.astype("<u4") * 65536).view("<f4")).astype(np.float64).ravel()
    aa, bb, dd, ab = (math.fsum(v) for v in (a * a, b * b, (b - a) ** 2, a * b))
    na, nb, nd = math.sqrt(aa), math.sqrt(bb), math.sqrt(dd)
    tests = {
        "rms": nd / math.sqrt(a.size),
        "ack_l2": na,
        "yes_l2": nb,
        "delta_l2": nd,
        "relative_l2": nd / na if na else None,
        "cosine": ab / (na * nb) if na and nb else None,
    }
    for key, value in tests.items():
        if (value is None) != (metrics[key] is None) or (
            value is not None
            and not math.isclose(value, metrics[key], rel_tol=rtol, abs_tol=atol)
        ):
            raise ValueError(f"independent numeric reduction differs: {key}")


def compare_pair(ack, yes, row, cfg):
    if ack.shape != (24, 2, 1, 2, 397, 64) or yes.shape != ack.shape:
        raise ValueError("full terminal all-layer cache required")
    for i, target in enumerate(NAMES):
        a, b = ack[i // 2, i % 2], yes[i // 2, i % 2]
        positions = np.flatnonzero((a != b).any(axis=(0, 1, 3))).tolist()
        for name, (lo, hi) in cfg["regions"].items():
            x, y = a[..., lo:hi, :], b[..., lo:hi, :]
            m = region_metrics(x, y)
            verify_metric(
                x, y, m, cfg["numeric_oracle_rtol"], cfg["numeric_oracle_atol"]
            )
            yield {
                "pair_id": row["pair_id"],
                "cell": row["cell"],
                "target": target,
                "primary_target": target in cfg["primary_targets"],
                "region": name,
                "span": [lo, hi],
                "first_changed_full_cache": positions[0] if positions else None,
                "changed_positions": [
                    lo + p for p in m.pop("changed_relative_positions")
                ],
                **m,
            }


def aggregate(metrics, cfg):
    out = []
    for target in NAMES:
        for name in cfg["regions"]:
            rows = [r for r in metrics if r["target"] == target and r["region"] == name]
            if len(rows) != 32 or len({(r["pair_id"], r["cell"]) for r in rows}) != 32:
                raise ValueError(
                    "missing/duplicate paired metric; no reduced denominator"
                )
            rms = [r["rms"] for r in rows]
            out.append(
                {
                    "target": target,
                    "region": name,
                    "images": 32,
                    "primary": target in cfg["primary_targets"] and name == "suffix",
                    "word_exact": sum(r["changed_scalars"] == 0 for r in rows),
                    "rms_median": float(np.median(rms)),
                    "rms_min": min(rms),
                    "rms_max": max(rms),
                    "max_abs": max(r["max_abs"] for r in rows),
                    "first_changed_counts": dict(
                        Counter(str(r["first_changed_full_cache"]) for r in rows)
                    ),
                }
            )
    return out
