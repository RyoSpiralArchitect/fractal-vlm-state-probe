"""Prescribed single-token decoding and lossless all-layer cache checkpoints."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np

ARMS = ("native_plain", "native_traced", "forced_own")
OFFSETS = (395, 396, 397)
NAMES = tuple(f"{i}:{k}" for i in range(24) for k in ("keys", "values"))


def prescribed_steps(suffix, prefill, consume):
    """Consume even the last (possibly EOS) ID before yielding to the streamer."""
    if len(suffix) != 2 or any(
        type(t) is not int or not 0 <= t < 151936 for t in suffix
    ):
        raise ValueError("two registered native vocabulary IDs required")
    scores = prefill()
    for token in suffix:
        following = consume(token)
        yield token, scores
        scores = following


def forced_stepper(suffix, generator, expected_model, invocations):
    import mlx.core as mx

    def generate(
        input_ids,
        model,
        pixel_values,
        mask,
        *,
        max_tokens,
        temperature,
        prompt_cache,
        image_sizes,
        thinking_budget_criteria,
        **extra,
    ):
        if (
            invocations
            or model is not expected_model
            or max_tokens != 2
            or temperature != 0
            or thinking_budget_criteria is not None
            or extra
            or pixel_values is None
            or mask is None
            or any(
                c.offset != 0 or c.keys is not None or c.values is not None
                for c in prompt_cache
            )
        ):
            raise ValueError("unexpected teacher-forced source entry")
        invocations.append(tuple(suffix))

        def logprobs(out):
            logits = out.logits[:, -1, :]
            return (logits - mx.logsumexp(logits)).squeeze(0)

        def prefill():
            with mx.stream(generator.generation_stream):
                features = model.get_input_embeddings(
                    input_ids, pixel_values, mask=mask, image_sizes=image_sizes
                )
                emb = features.inputs_embeds
                if tuple(emb.shape) != (1, 395, 896) or any(
                    k != "inputs_embeds" and v is not None
                    for k, v in features.to_dict().items()
                ):
                    raise ValueError("unknown multimodal embedding path")
                return logprobs(
                    model.language_model(
                        input_ids,
                        inputs_embeds=emb,
                        cache=prompt_cache,
                        image_sizes=image_sizes,
                    )
                )

        def consume(token):
            with mx.stream(generator.generation_stream):
                inputs = mx.array([[token]], dtype=mx.uint32)
                return logprobs(
                    model.language_model(inputs, inputs_embeds=None, cache=prompt_cache)
                )

        yield from prescribed_steps(suffix, prefill, consume)

    return generate


@contextmanager
def replaced(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def sha_words(a):
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def compare_words(a, b, label):
    if a.shape != b.shape or a.dtype != np.dtype("<u2") or b.dtype != a.dtype:
        raise ValueError(f"{label}: cache shape or word dtype differs")
    if not np.array_equal(a, b):
        index = np.unravel_index(np.argmax(a != b), a.shape)
        raise ValueError(
            f"{label}: first unequal word at {tuple(map(int, index))}: "
            f"{int(a[index]):04x} vs {int(b[index]):04x}"
        )


def read_live_cache(cache, offset):
    import mlx.core as mx

    views = []
    if len(cache) != 24:
        raise ValueError("expected 24 distinct layer caches")
    for c in cache:
        if (
            f"{type(c).__module__}.{type(c).__name__}" != "mlx_lm.models.cache.KVCache"
            or c.offset != offset
        ):
            raise ValueError("cache class or effective offset differs")
        for x in c.state:
            if x.dtype != mx.bfloat16 or tuple(x.shape) != (1, 2, offset, 64):
                raise ValueError("effective native cache axes or dtype differ")
            views.append(x.view(mx.uint16))
    if len(views) != 48 or len({id(c) for c in cache}) != 24:
        raise ValueError("incomplete or aliased layer collection")
    mx.eval(*views)
    words = np.stack([np.asarray(x) for x in views]).reshape(24, 2, 1, 2, offset, 64)
    if words.dtype != np.dtype("<u2") or np.any((words & 0x7F80) == 0x7F80):
        raise ValueError("invalid native word array")
    return words


def tensor_digests(words):
    return {k: sha_words(words[i // 2, i % 2]) for i, k in enumerate(NAMES)}


class Checkpoints:
    def __init__(self, path):
        self.path = path
        self.last = None
        self.rows = []

    def save(self, words, detail):
        idx = len(self.rows)
        offset = words.shape[-2]
        begin = 0 if self.last is None else self.last.shape[-2]
        if begin:
            try:
                compare_words(self.last, words[..., :begin, :], "non-append update")
            except ValueError:
                np.save(self.path / f"nonappend_{idx}.npy", words, allow_pickle=False)
                raise
        name = f"kv_{idx}.npy"
        np.save(
            self.path / name,
            np.ascontiguousarray(words[..., begin:, :]),
            allow_pickle=False,
        )
        self.rows.append(
            {
                **detail,
                "offset": offset,
                "word_file": name,
                "start": begin,
                "sha256": sha_words(words),
                "tensor_sha256": tensor_digests(words),
            }
        )
        self.last = words


def trace_call(language, anchor, checkpoints):
    original = type(language).__call__
    cache_ref = []

    def traced(instance, *args, **kw):
        if instance is not language:
            return original(instance, *args, **kw)
        i = len(checkpoints.rows)
        if i >= 3 or len(args) != 1:
            raise ValueError("unexpected language-model call count/signature")
        inputs = args[0]
        actual = inputs.tolist()
        expected = anchor["prepared_ids"] if i == 0 else anchor["suffix_ids"][i - 1 : i]
        dtype = (
            anchor["input_fingerprints"]["input_ids"]["dtype"] if i == 0 else "uint32"
        )
        embeddings = kw["inputs_embeds"]
        shape = list(embeddings.shape) if embeddings is not None else None
        cache = kw["cache"]
        previous = 0 if not i else OFFSETS[i - 1]
        if (
            actual != [expected]
            or str(inputs.dtype) != "mlx.core." + dtype
            or shape != ([1, 395, 896] if i == 0 else None)
            or len(cache) != 24
            or any(c.offset != previous for c in cache)
            or (i and cache is not cache_ref[0])
            or (
                not i and any(c.keys is not None or c.values is not None for c in cache)
            )
        ):
            raise ValueError(
                f"decoder entry {i} differs: ids_match={actual == [expected]}, "
                f"dtype={inputs.dtype}, embeddings={shape}, offsets={[c.offset for c in cache]}"
            )
        if not i:
            cache_ref.append(cache)
        result = original(instance, *args, **kw)
        words = read_live_cache(cache, OFFSETS[i])
        checkpoints.save(
            words,
            {
                "step": i,
                "input_ids": actual[0],
                "input_dtype": dtype,
                "embedding_shape": shape,
                "offset_before": previous,
            },
        )
        return result

    return traced


def reconstruct(root, receipts, arm, anchor):
    """Reload the persisted additions; verify every complete boundary, not just the new row."""
    offsets = (397,) if arm == "native_plain" else OFFSETS
    if len(receipts) != len(offsets):
        raise ValueError("incomplete checkpoint coverage")
    full = None
    for i, (r, end) in enumerate(zip(receipts, offsets, strict=True)):
        start = 0 if i == 0 else offsets[i - 1]
        if r["word_file"] != f"kv_{i}.npy" or r["offset"] != end or r["start"] != start:
            raise ValueError("invalid packed boundary")
        if arm != "native_plain":
            expected = (
                anchor["prepared_ids"] if not i else anchor["suffix_ids"][i - 1 : i]
            )
            if (
                r["step"] != i
                or r["input_ids"] != expected
                or r["offset_before"] != start
                or r["embedding_shape"] != ([1, 395, 896] if not i else None)
                or r["input_dtype"]
                != (
                    anchor["input_fingerprints"]["input_ids"]["dtype"]
                    if not i
                    else "uint32"
                )
            ):
                raise ValueError("decoder-step receipt differs")
        part = np.load(root / r["word_file"], allow_pickle=False)
        if (
            part.shape != (24, 2, 1, 2, end - start, 64)
            or part.dtype != np.dtype("<u2")
            or np.any((part & 0x7F80) == 0x7F80)
        ):
            raise ValueError("stored layer words invalid")
        full = part if full is None else np.concatenate((full, part), axis=-2)
        if sha_words(full) != r["sha256"] or tensor_digests(full) != r["tensor_sha256"]:
            raise ValueError("full checkpoint digest mismatch")
        yield full
