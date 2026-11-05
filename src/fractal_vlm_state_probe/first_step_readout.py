"""Unmodified native first-step scores versus a fresh, suffix-free prefill."""

from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

from .full_vocab_readout import _to_float32_numpy, write_full_vocab_logprob_sidecar
from .native_suffix_replay import sha_words


def validate_scores(scores, size):
    if (
        scores.dtype != np.float32
        or scores.shape != (size,)
        or np.isnan(scores).any()
        or np.isposinf(scores).any()
        or not np.isfinite(scores).any()
        or np.any(scores.view(np.uint32) & 65535)
    ):
        raise ValueError("invalid complete native bf16 vocabulary scores")


def validate_vectors(logits, scores, size):
    validate_scores(scores, size)
    if (
        logits.dtype != np.float32
        or logits.shape != (size,)
        or not np.isfinite(logits).all()
        or np.any(logits.view(np.uint32) & 65535)
    ):
        raise ValueError("invalid complete native bf16 vocabulary expansions")


def logsumexp64(x):
    m = float(np.max(x))
    return m + math.log(float(np.exp(x - m).sum()))


def diagnostics(logits, scores):
    s = scores.astype(np.float64)
    z = logsumexp64(s)
    p = np.exp(s - z)
    raw = logits.astype(np.float64)
    raw_p = np.exp(raw - logsumexp64(raw))
    return {
        "native_logsumexp64": z,
        "native_score_probability_sum": math.exp(z),
        "renormalized_probability_sum": float(p.sum()),
        "native_vs_raw_logits_softmax_tv": float(0.5 * np.abs(p - raw_p).sum()),
        "negative_infinity_count": int(np.isneginf(s).sum()),
        "argmax_ids": np.flatnonzero(scores == scores.max()).tolist(),
    }


def independent_diagnostics(logits, scores, expected, rtol, atol):
    native = scores.astype(np.float64)
    raw = logits.astype(np.float64)
    m, n = float(native.max()), float(raw.max())
    weights = np.exp(native - m)
    rawweights = np.exp(raw - n)
    z = m + math.log(math.fsum(weights))
    denominator, raw_denominator = math.fsum(weights), math.fsum(rawweights)
    mass = math.fsum(weights / denominator)
    tv = 0.5 * math.fsum(abs(weights / denominator - rawweights / raw_denominator))
    for key, value in (
        ("native_logsumexp64", z),
        ("native_score_probability_sum", math.exp(z)),
        ("renormalized_probability_sum", mass),
        ("native_vs_raw_logits_softmax_tv", tv),
    ):
        if not math.isclose(value, expected[key], rel_tol=rtol, abs_tol=atol):
            raise ValueError(f"independent numerical diagnostic differs: {key}")
    if abs(mass - 1) > 1e-12:
        raise ValueError("defined float64 renormalization did not sum to one")


def save_scores(root, scores, config):
    if str(scores.dtype) != config["native_dtype"] or tuple(scores.shape) != (
        config["vocab_size"],
    ):
        raise ValueError("actual native score shape or dtype differs")
    values = _to_float32_numpy(scores).reshape(-1)
    validate_scores(values, config["vocab_size"])
    meta = write_full_vocab_logprob_sidecar(
        scores, path=root / "logprobs.npz", relative_to=root
    )
    return {
        "scores_array_sha256": sha_words(values),
        "source_dtype": config["native_dtype"],
        "sidecar": meta,
    }


def reload_scores(root, receipt, config):
    from .cli.run_pairing_seed_validation import _sha

    if _sha(root / "logprobs.npz") != receipt["sidecar"]["sha256"]:
        raise ValueError("native score sidecar changed")
    with np.load(root / "logprobs.npz", allow_pickle=False) as archive:
        if archive.files != ["logprobs"]:
            raise ValueError("saved score coordinate coverage differs")
        scores = archive["logprobs"]
    validate_scores(scores, config["vocab_size"])
    if (
        sha_words(scores) != receipt["scores_array_sha256"]
        or receipt["source_dtype"] != config["native_dtype"]
    ):
        raise ValueError("native score word digest or source dtype differs")
    return scores


def save_vectors(root, logits, scores, config):
    if (
        str(logits.dtype) != config["native_dtype"]
        or str(scores.dtype) != config["native_dtype"]
        or tuple(logits.shape) != (1, config["vocab_size"])
        or tuple(scores.shape) != (config["vocab_size"],)
    ):
        raise ValueError("actual native vocabulary shape or dtype differs")
    raw = _to_float32_numpy(logits).reshape(-1)
    values = _to_float32_numpy(scores).reshape(-1)
    validate_vectors(raw, values, config["vocab_size"])
    np.savez_compressed(root / "logits.npz", logits=raw)
    meta = save_scores(root, scores, config)
    return {
        **meta,
        "logits_array_sha256": sha_words(raw),
        "diagnostics": diagnostics(raw, values),
    }


def reload_vectors(root, receipt, config):
    scores = reload_scores(root, receipt, config)
    with np.load(root / "logits.npz", allow_pickle=False) as a:
        if a.files != ["logits"]:
            raise ValueError("vocabulary coordinate coverage differs")
        logits = a["logits"]
    validate_vectors(logits, scores, config["vocab_size"])
    if (
        sha_words(logits) != receipt["logits_array_sha256"]
        or sha_words(scores) != receipt["scores_array_sha256"]
        or diagnostics(logits, scores) != receipt["diagnostics"]
    ):
        raise ValueError("raw vocabulary bytes or diagnostics changed")
    independent_diagnostics(
        logits,
        scores,
        receipt["diagnostics"],
        config["oracle_rtol"],
        config["oracle_atol"],
    )
    return logits, scores


@contextmanager
def stream_score_observation(runtime):
    first_scores = []

    def stream(*args, **kwargs):
        for i, chunk in enumerate(runtime["stream_generate"](*args, **kwargs)):
            if i == 0:
                first_scores.append(chunk.logprobs)
            yield chunk

    yield {**runtime, "stream_generate": stream}, first_scores
    if len(first_scores) != 1:
        raise ValueError("exactly one native first score object required")


@contextmanager
def native_score_observation(base, runtime):
    original_trace = base.trace_call
    outputs = []

    def factory(language, anchor, checkpoints):
        traced = original_trace(language, anchor, checkpoints)

        def observed(instance, *args, **kwargs):
            result = traced(instance, *args, **kwargs)
            if instance is language and len(checkpoints.rows) == 1:
                outputs.append(result.logits[:, -1, :])
            return result

        return observed

    old = base.trace_call
    base.trace_call = factory
    try:
        with stream_score_observation(runtime) as (observed, first_scores):
            yield observed, outputs, first_scores
        if len(outputs) != 1:
            raise ValueError(
                "exactly one native prefill and first score object required"
            )
    finally:
        base.trace_call = old


def readout_identity(row):
    return {k: row[k] for k in ("ordinal", "pair_id", "cell", "stage")}
