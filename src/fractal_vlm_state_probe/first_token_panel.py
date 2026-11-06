"""Prespecified whole-vocabulary prompt contrasts on the paired development images."""

from __future__ import annotations

import math
from collections import Counter
from statistics import median

import numpy as np

from .first_step_readout import logsumexp64, validate_scores
from .full_vocab_readout import _distribution_distances


def pack_scores(scores):
    validate_scores(scores, scores.size)
    words = (scores.view("<u4") >> 16).astype("<u2")
    if expand_scores(words).tobytes() != scores.tobytes():
        raise ValueError("native score words lost information")
    return words


def expand_scores(words):
    if words.dtype != np.dtype("<u2"):
        raise ValueError("little-endian native word array required")
    return (words.astype("<u4") * 65536).view("<f4")


def oracle_expand_scores(words):
    if words.dtype != np.dtype("<u2"):
        raise ValueError("native word dtype differs")
    halves = np.zeros((*words.shape, 2), dtype="<u2")
    halves[..., 1] = words
    return halves.view("<f4").reshape(words.shape)


def normalized(scores):
    validate_scores(scores, scores.size)
    values = scores.astype(np.float64)
    logs = values - logsumexp64(values)
    p = np.exp(logs)
    if abs(math.fsum(p) - 1) > 1e-12:
        raise ValueError("registered full-vocabulary renormalization failed")
    return logs, p


def paired_metrics(ack, yes, top_k):
    if ack.shape != yes.shape:
        raise ValueError("paired vocabulary sizes differ")
    a, p = normalized(ack)
    b, q = normalized(yes)
    distances = _distribution_distances(a, b)
    delta = q - p
    order = np.lexsort((np.arange(delta.size), -np.abs(delta)))[:top_k]
    maximum = float(np.abs(delta).max())
    max_ids = np.flatnonzero(np.abs(delta) == maximum)
    am = np.flatnonzero(ack == ack.max()).tolist()
    bm = np.flatnonzero(yes == yes.max()).tolist()
    return {
        "total_variation": distances["total_variation"],
        "jensen_shannon_nats": distances["jensen_shannon"],
        "hellinger": distances["hellinger"],
        "delta_l2": float(np.linalg.norm(delta)),
        "max_abs_delta": maximum,
        "max_abs_token_id": int(max_ids[0]) if maximum else None,
        "max_abs_tie_count": int(max_ids.size),
        "ack_argmax_ids": am,
        "yes_argmax_ids": bm,
        "argmax_sets_overlap": bool(set(am) & set(bm)),
        "ack_probability_sum": float(p.sum()),
        "yes_probability_sum": float(q.sum()),
        "native_score_word_equal": ack.tobytes() == yes.tobytes(),
        "top_delta_coordinates": [
            {
                "token_id": int(i),
                "ack_probability": float(p[i]),
                "yes_probability": float(q[i]),
                "yes_minus_ack": float(delta[i]),
            }
            for i in order
        ],
    }


def oracle_metrics(ack, yes):
    def probability(score):
        values = score.astype(np.float64)
        weights = np.exp(values - values.max())
        return weights / math.fsum(weights)

    p, q = probability(ack), probability(yes)
    mid = (p + q) / 2
    tv = 0.5 * math.fsum(abs(p - q))
    # No probability floor: terms outside each distribution's own support are zero.
    js = 0.5 * (
        math.fsum(p[p > 0] * np.log(p[p > 0] / mid[p > 0]))
        + math.fsum(q[q > 0] * np.log(q[q > 0] / mid[q > 0]))
    )
    return {
        "total_variation": tv,
        "jensen_shannon_nats": max(0.0, js),
        "hellinger": math.sqrt(0.5 * math.fsum((np.sqrt(p) - np.sqrt(q)) ** 2)),
        "delta_l2": math.sqrt(math.fsum((q - p) ** 2)),
        "max_abs_delta": float(abs(q - p).max()),
    }


def check_metrics(ack, yes, record, rtol, atol):
    oracle = oracle_metrics(ack, yes)
    for key, value in oracle.items():
        if not math.isclose(value, record[key], rel_tol=rtol, abs_tol=atol):
            raise ValueError(
                f"independent paired probability calculation differs: {key}"
            )
    if not (
        0 <= record["total_variation"] <= 1 + atol
        and 0 <= record["jensen_shannon_nats"] <= math.log(2) + atol
    ):
        raise ValueError("probability distance outside its declared domain")


def aggregate(pairs):
    ids = [(r["pair_id"], r["cell"]) for r in pairs]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError(
            "complete 32-image paired panel required; no reduced denominator"
        )
    families = sorted({r["pair_id"] for r in pairs})
    cells = sorted({r["cell"] for r in pairs})
    if (
        len(families) != 8
        or cells != ["jj", "jm", "mj", "mm"]
        or any(sum(r["pair_id"] == f for r in pairs) != 4 for f in families)
    ):
        raise ValueError("unbalanced source-family/cell coverage")
    keys = (
        "total_variation",
        "jensen_shannon_nats",
        "hellinger",
        "delta_l2",
        "max_abs_delta",
    )

    def summary(rows):
        return {
            key: {
                "mean": math.fsum(r[key] for r in rows) / len(rows),
                "median": median(r[key] for r in rows),
                "min": min(r[key] for r in rows),
                "max": max(r[key] for r in rows),
            }
            for key in keys
        }

    return {
        "primary_endpoint": "equal_image_mean_total_variation_over_32_pairs",
        "primary_value": math.fsum(r["total_variation"] for r in pairs) / 32,
        "paired_images": 32,
        "summary": summary(pairs),
        "argmax_sets_overlap": sum(r["argmax_sets_overlap"] for r in pairs),
        "argmax_set_size_counts": {
            "ack": dict(Counter(str(len(r["ack_argmax_ids"])) for r in pairs)),
            "yes": dict(Counter(str(len(r["yes_argmax_ids"])) for r in pairs)),
        },
        "by_source_pair": [
            {
                "pair_id": f,
                "images": 4,
                "metrics": summary([r for r in pairs if r["pair_id"] == f]),
            }
            for f in families
        ],
        "by_cell": [
            {
                "cell": c,
                "images": 8,
                "metrics": summary([r for r in pairs if r["cell"] == c]),
            }
            for c in cells
        ],
    }
