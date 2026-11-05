import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import run_first_step_readout_calibration as runner
from fractal_vlm_state_probe.first_step_readout import (
    diagnostics,
    independent_diagnostics,
    native_score_observation,
    reload_scores,
    reload_vectors,
    stream_score_observation,
    validate_scores,
    validate_vectors,
)
from fractal_vlm_state_probe.padding_cache import input_fingerprint


@pytest.fixture
def cfg():
    return runner.base._read(runner.REPO / runner.CFG)


def test_exact_native_coordinates_zero_support_and_native_mass_not_assumed_one():
    raw = np.array([0.0, -1.0, -2.0], np.float32)
    scores = np.array([0.0, -1.0, -np.inf], np.float32)
    validate_vectors(raw, scores, 3)
    d = diagnostics(raw, scores)
    independent_diagnostics(raw, scores, d, 1e-12, 1e-14)
    assert d["negative_infinity_count"] == 1 and d["argmax_ids"] == [0]
    assert d["native_score_probability_sum"] > 1
    assert d["renormalized_probability_sum"] == 1
    assert d["native_vs_raw_logits_softmax_tv"] > 0
    with pytest.raises(ValueError):
        independent_diagnostics(
            raw, scores, {**d, "native_score_probability_sum": 1}, 1e-12, 1e-14
        )
    tied = diagnostics(raw, np.array([-1.0, -1.0, -2.0], np.float32))
    assert tied["argmax_ids"] == [0, 1]


@pytest.mark.parametrize(
    "bad",
    [
        np.array([0.0, 1.1, -1.0], np.float32),
        np.array([0.0, np.nan, -1.0], np.float32),
        np.array([0.0, np.inf, -1.0], np.float32),
        np.array([-np.inf] * 3, np.float32),
        np.array([0.0, -1.0], np.float32),
        np.array([0.0, -1.0, -2.0], np.float64),
    ],
)
def test_no_vocabulary_truncation_nonfinite_replacement_or_lossy_conversion(bad):
    with pytest.raises(ValueError):
        validate_scores(bad, 3)


def test_stream_holds_first_object_only_and_keeps_final_duplicate():
    scores = [object(), object(), object()]
    chunks = [SimpleNamespace(logprobs=s) for s in scores]

    def stream():
        yield from chunks

    with stream_score_observation({"stream_generate": stream}) as (runtime, captured):
        assert list(runtime["stream_generate"]()) == chunks
        assert captured == scores[:1] and captured[0] is scores[0]
    with (
        pytest.raises(ValueError),
        stream_score_observation({"stream_generate": lambda: iter(())}),
    ):
        pass


def test_decoder_observation_holds_raw_prefill_only_and_restores():
    lm = object()
    outputs = [SimpleNamespace(logits=np.full((1, 2, 3), i)) for i in range(3)]

    def original(language, anchor, checkpoints):
        def called(instance):
            index = len(checkpoints.rows)
            checkpoints.rows.append(index)
            return outputs[index]

        return called

    api = SimpleNamespace(trace_call=original)
    chunks = [SimpleNamespace(logprobs=object()) for _ in range(3)]
    checkpoints = SimpleNamespace(rows=[])
    with native_score_observation(api, {"stream_generate": lambda: iter(chunks)}) as (
        runtime,
        raw,
        scores,
    ):
        fn = api.trace_call(lm, {}, checkpoints)
        assert [fn(lm) for _ in range(3)] == outputs
        assert len(raw) == 1 and np.shares_memory(raw[0], outputs[0].logits)
        assert (
            list(runtime["stream_generate"]()) == chunks
            and scores[0] is chunks[0].logprobs
        )
    assert api.trace_call is original
    with pytest.raises(RuntimeError), native_score_observation(api, {}):
        raise RuntimeError
    assert api.trace_call is original


@pytest.mark.parametrize(
    "key,value",
    [
        ("vocab_size", 151665),
        ("rows", 64),
        ("prefix_suffix_decoder_calls", 1),
        ("experimental_readout_panel_released", True),
        ("interprompt_effects_computed", 1),
        ("oracle_rtol", 1e-6),
        ("retry_or_resume", True),
        ("reserve_bytes", 0),
    ],
)
def test_readout_gate_cannot_be_relaxed_or_auto_release_panel(cfg, key, value):
    runner.require_config(cfg)
    with pytest.raises(ValueError):
        runner.require_config({**cfg, key: value})


def test_real_prefix_adapter_with_stub_decoder_no_weights(monkeypatch, tmp_path, cfg):
    mx = pytest.importorskip("mlx.core")
    g = pytest.importorskip("mlx_vlm.generate")
    cachemod = pytest.importorskip("mlx_lm.models.cache")
    image = tmp_path / "frame.png"
    image.write_bytes(b"stub-image-never-decoded")
    frame = {
        "path": image.name,
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }
    manifest = tmp_path / "manifest.json"
    runner.base.write_json(manifest, {"frames": [frame]})
    ids = [3] * 140
    ids[42] = -200
    payload = {
        "input_ids": mx.array([ids], dtype=mx.int64),
        "attention_mask": mx.ones((1, 140), dtype=mx.int64),
        "pixel_values": mx.zeros((1, 3, 2, 2)),
        "image_sizes": [[320, 320]],
    }
    fingerprints = input_fingerprint(payload)
    anchor = {
        "ordinal": 1,
        "pair_id": "stub",
        "cell": "mm",
        "stage": "ack_before",
        "prepared_ids": ids,
        "suffix_ids": [785, 2168],
        "input_fingerprints": fingerprints,
    }
    monkeypatch.setattr(runner, "REPO", tmp_path)
    history = runner.prior.context_dir(tmp_path / "history", anchor, anchor["stage"])
    history.mkdir(parents=True)
    source = {
        "manifest_path": str(manifest),
        "stream_events": [{"prompt": "unchanged", "frame_sha256s": [frame["sha256"]]}],
    }
    runner.base.write_json(history / "source.json", source)
    audit = {
        "schema_version": 1,
        "analysis_kind": "actual_generation_prepared_input_audit",
        "context": {"pair_id": "stub", "cell": "mm", "stage": "ack_before"},
        "native": fingerprints,
        "effective": fingerprints,
        "override_image_sizes": None,
    }
    runner.base.write_json(history / "input.json", audit)
    np.savez(
        history / "sites.npz",
        encoder=np.zeros((1, 256, 3072), dtype="<u2"),
        projector=np.zeros((1, 256, 896), dtype="<u2"),
    )
    anchor["files"] = {
        n: runner.base._sha(history / n)
        for n in ("source.json", "input.json", "sites.npz")
    }
    calls = []

    class Projector:
        def __call__(self, x):
            return mx.zeros((1, 256, 896), dtype=mx.bfloat16)

    class Language:
        def __call__(self, inputs, *, inputs_embeds, cache, **kw):
            calls.append(inputs.tolist()[0])
            assert all(c.offset == 0 for c in cache) and inputs_embeds.shape == (
                1,
                395,
                896,
            )
            for c in cache:
                c.update_and_fetch(
                    mx.zeros((1, 2, 395, 64), dtype=mx.bfloat16),
                    mx.zeros((1, 2, 395, 64), dtype=mx.bfloat16),
                )
            return SimpleNamespace(logits=mx.zeros((1, 1, 151936), dtype=mx.bfloat16))

    projector, language = Projector(), Language()

    def embeddings(*a, **kw):
        projector(mx.zeros((1, 256, 3072), dtype=mx.bfloat16))
        emb = mx.zeros((1, 395, 896), dtype=mx.bfloat16)
        return SimpleNamespace(
            inputs_embeds=emb, to_dict=lambda: {"inputs_embeds": emb}
        )

    model = SimpleNamespace(
        mm_projector=projector,
        language_model=language,
        get_input_embeddings=embeddings,
        config=SimpleNamespace(image_token_index=-200, model_type="llava_qwen2"),
    )
    runtime = {
        "model": model,
        "processor": object(),
        "model_config": model.config,
        "apply_chat_template": lambda *a, **k: "unchanged template",
    }

    def prepare(*args, **kwargs):
        return payload

    monkeypatch.setattr(g, "prepare_inputs", prepare)
    monkeypatch.setattr(g, "wired_limit", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        g.cache,
        "make_prompt_cache",
        lambda *a, **kw: [cachemod.KVCache() for _ in range(24)],
    )
    plain = Language.__call__
    receipt = runner.acquire_prefix(
        tmp_path / "prefix",
        anchor,
        runtime,
        {"config": {"capture_root": "history", "source_seed": 20260604}},
        cfg,
    )
    assert calls == [ids] and Language.__call__ is plain and g.prepare_inputs is prepare
    state = json.loads((tmp_path / "prefix/prefix.json").read_bytes())
    assert (
        state["suffix_ids"] == []
        and state["effective_length"] == 395
        and state["decoder_calls"] == 1
    )
    raw, scores = reload_vectors(tmp_path / "prefix", receipt["vocabulary"], cfg)
    assert (
        raw.shape == scores.shape == (151936,)
        and scores.tobytes()
        == reload_scores(tmp_path / "prefix", receipt["vocabulary"], cfg).tobytes()
    )
    np.savez_compressed(
        tmp_path / "prefix/logits.npz", logits=np.ones(151936, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        reload_vectors(tmp_path / "prefix", receipt["vocabulary"], cfg)


def test_single_attempt_root_offline_and_no_overwrite(monkeypatch, tmp_path):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError, match="canonical"):
        runner.run("capture", tmp_path / "other")
    with pytest.raises(FileExistsError):
        runner.run("verify", tmp_path)
