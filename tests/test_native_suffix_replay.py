import copy
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import run_native_suffix_replay as runner
from fractal_vlm_state_probe.native_suffix_replay import (
    OFFSETS,
    Checkpoints,
    compare_words,
    forced_stepper,
    prescribed_steps,
    reconstruct,
    replaced,
)


@pytest.mark.parametrize("suffix", [[9454, 151645], [9454, 11], [785, 2168], [8, 8]])
def test_stream_consumes_each_id_even_when_it_stops_at_eos(suffix):
    entered = []

    def consume(t):
        entered.append(t)
        return 999

    stream = prescribed_steps(suffix, lambda: 777, consume)
    assert next(stream) == (suffix[0], 777) and entered == suffix[:1]
    assert next(stream) == (suffix[1], 999) and entered == suffix
    stream.close()
    assert entered == suffix


@pytest.mark.parametrize("bad", [[1], [1, 2, 2], [-200, 1], [True, 8], [151936, 1]])
def test_reject_notifications_or_out_of_vocabulary_tokens(bad):
    with pytest.raises(ValueError):
        next(prescribed_steps(bad, lambda: 0, lambda _: 0))


def test_forced_model_boundary_does_not_choose_from_logits_or_skip_eos(monkeypatch):
    # Adversarial fake model prefers ID 0 on every step, including prefill.
    mx = ModuleType("mlx.core")
    mx.stream, mx.array, mx.uint32 = lambda _: nullcontext(), np.array, np.uint32
    mx.logsumexp = lambda x: np.log(np.exp(x).sum())
    mlx = ModuleType("mlx")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    calls, embedded, used = [], [], []
    emb = SimpleNamespace(shape=(1, 395, 896))

    def embeddings(inputs, pixels, **kw):
        embedded.append((inputs, pixels, kw))
        return SimpleNamespace(
            inputs_embeds=emb, to_dict=lambda: {"inputs_embeds": emb}
        )

    def language(inputs, **kw):
        calls.append((inputs.tolist(), kw["cache"], kw["inputs_embeds"]))
        return SimpleNamespace(logits=np.array([[[5.0, 0.0, 0.0]]]))

    model = SimpleNamespace(language_model=language, get_input_embeddings=embeddings)
    cache = [SimpleNamespace(offset=0, keys=None, values=None)]
    stepper = forced_stepper(
        [9454, 151645], SimpleNamespace(generation_stream=None), model, used
    )
    args = (np.array([[3, 4]]), model, object(), object())
    kw = dict(
        prompt_cache=cache,
        max_tokens=2,
        temperature=0,
        image_sizes=[[320, 320]],
        thinking_budget_criteria=None,
    )
    actual = list(stepper(*args, **kw))
    assert [t for t, _ in actual] == [9454, 151645] and all(
        np.argmax(scores) == 0 for _, scores in actual
    )
    assert calls == [
        ([[3, 4]], cache, emb),
        ([[9454]], cache, None),
        ([[151645]], cache, None),
    ]
    assert len(embedded) == 1 and used == [(9454, 151645)]
    with pytest.raises(ValueError):
        list(stepper(*args, **kw))


def test_single_word_including_signed_zero_is_a_failure():
    a = np.zeros((2, 2), dtype="<u2")
    b = a.copy()
    b[1, 0] = 0x8000
    with pytest.raises(ValueError, match="8000"):
        compare_words(a, b, "fixture")
    with pytest.raises(ValueError, match="dtype"):
        compare_words(a, b.astype(np.uint32), "fixture")


def test_all_three_actual_boundaries_roundtrip_and_tampering_fails(tmp_path):
    a = {
        "prepared_ids": [1, 2],
        "suffix_ids": [9454, 151645],
        "input_fingerprints": {"input_ids": {"dtype": "int64"}},
    }
    cp = Checkpoints(tmp_path)
    full = np.zeros((24, 2, 1, 2, 397, 64), dtype="<u2")
    full[23, 1, 0, 1, :, 63] = np.arange(397)
    for i, end in enumerate(OFFSETS):
        cp.save(
            full[..., :end, :],
            {
                "step": i,
                "input_ids": a["prepared_ids"] if not i else a["suffix_ids"][i - 1 : i],
                "input_dtype": "uint32" if i else "int64",
                "offset_before": OFFSETS[i - 1] if i else 0,
                "embedding_shape": None if i else [1, 395, 896],
            },
        )
    for i, x in enumerate(reconstruct(tmp_path, cp.rows, "forced_own", a)):
        np.testing.assert_array_equal(x, full[..., : OFFSETS[i], :])
    for mutation in ("drop", "token", "sha", "offset"):
        bad = copy.deepcopy(cp.rows)
        if mutation == "drop":
            bad.pop()
        if mutation == "token":
            bad[2]["input_ids"] = [9454]
        if mutation == "sha":
            bad[1]["tensor_sha256"]["7:values"] = "bad"
        if mutation == "offset":
            bad[0]["offset"] = 396
        with pytest.raises(ValueError):
            list(reconstruct(tmp_path, bad, "forced_own", a))
    p = tmp_path / "kv_2.npy"
    corrupted = np.load(p, allow_pickle=False)
    corrupted[0, 1, 0, 1, 0, 0] = 1
    np.save(p, corrupted)
    with pytest.raises(ValueError, match="digest"):
        list(reconstruct(tmp_path, cp.rows, "forced_own", a))


def test_nonappend_preserves_actual_failure_and_hook_restores(tmp_path):
    cp = Checkpoints(tmp_path)
    cp.save(np.zeros((24, 2, 1, 2, 1, 64), dtype="<u2"), {})
    b = np.ones((24, 2, 1, 2, 2, 64), dtype="<u2")
    with pytest.raises(ValueError, match="non-append"):
        cp.save(b, {})
    np.testing.assert_array_equal(np.load(tmp_path / "nonappend_1.npy"), b)
    obj = SimpleNamespace(step="native")
    with pytest.raises(RuntimeError), replaced(obj, "step", "forced"):
        assert obj.step == "forced"
        raise RuntimeError
    assert obj.step == "native"


def test_fixed_capture_root_and_no_overwrite(monkeypatch, tmp_path):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError, match="canonical"):
        runner.run("capture", tmp_path / "alternate")
    with pytest.raises(FileExistsError):
        runner.run("verify", tmp_path)
