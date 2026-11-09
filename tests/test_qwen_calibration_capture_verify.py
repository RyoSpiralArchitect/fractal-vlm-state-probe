"""Software-only capture/verifier adversaries; no model or processor loaded."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe import qwen_calibration_capture as cap
from fractal_vlm_state_probe import qwen_calibration_verify as ver
from fractal_vlm_state_probe import qwen_calibration_admission as admission
from fractal_vlm_state_probe.qwen_calibration_contract import PLAN, window_order
from fractal_vlm_state_probe.qwen_calibration_native import decoder_records, first_yield
from fractal_vlm_state_probe.stat_matched_source_storage import inventory, read, save_array, write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path.resolve()
    source = root / "source"
    source.mkdir()
    plan = read(ROOT / PLAN)
    plan["jobs"].update(cases=2, source_shape_hwc=[2, 2, 3])
    plan["limits"].update(language_layers=2, kv_heads=1, head_dim=2, vocab=8,
                           vision_width=3, output_width=2, patch_size=1, temporal_patch_size=1,
                           pixel_channels=1, window_size=4, max_patch_tokens=64)
    plan["storage"].update(reserve_bytes=0, payload_allowance_bytes=2**24,
                           overhead_allowance_bytes=2**24)
    jobs = [{"case_id": str(i), "raw": save_array(source, f"{i}.npy", np.full((2, 2, 3), i, np.uint8))}
            for i in range(2)]
    return root, source, plan, jobs


class Fake:
    def __init__(self, plan, damage=None):
        self.plan, self.damage = plan, damage
        self.contexts = 0

    def processor(self, raw):
        return {"pixels": np.full((48, 1), raw[0, 0, 0], np.float32),
                "grid": np.array([[1, 6, 8]], dtype=np.int64)}

    def context(self, raw, stage, emit):
        self.contexts += 1
        for k in ("prepare_begin", "prepare_return", "decoder_begin", "decoder_return", "decoder_begin", "decoder_return"):
            emit(k)
        arrays = self.processor(raw)
        arrays["grid"] = arrays["grid"].astype(np.int32)
        arrays.update(ids=np.array([[1, 2]], np.int32), mask=np.ones((1, 2), np.int32),
                      logits_0=np.zeros(8, np.float32), logits_1=np.ones(8, np.float32),
                      scores=np.full(8, -2, np.float32))
        arrays.update({k: np.zeros((1, 1, 3, 2), np.float32) for k in cap.cache_names(self.plan)})
        if stage == "on":
            arrays["site_premerger"] = np.zeros((48, 3), np.float32)
            arrays["site_auxiliary"] = np.arange(24, dtype=np.float32).reshape(12, 2)
            arrays["site_final"] = arrays["site_auxiliary"][np.argsort(window_order(arrays["grid"], self.plan))]
        packet = {"arrays": arrays, "source_dtypes": {k: "mlx.core.float32" if a.dtype.kind == "f" else "mlx.core.int32"
                                                    for k, a in arrays.items()}, "metadata": {
            "sampled_id": 0, "decoder_inputs": [[1, 2], [0]], "cache_offsets": [3, 3], "extra_inputs": {},
            "observer_calls": {"vision": int(stage == "on"), "merger": int(stage == "on")}, "observer_restored": True}}
        if self.damage:
            self.damage(packet, raw, stage)
        return packet


def acquire(fixture, fake=None, name="capture"):
    root, source, plan, jobs = fixture
    output = root / name
    cap.capture(output, plan, jobs, source, fake or Fake(plan), {}, software_fixture=True)
    return output


def test_capture_independent_two_tree_verification_and_no_native_promotion(fixture):
    root, source, plan, jobs = fixture
    output = acquire(fixture)
    for name in ("verify1", "verify2"):
        result = ver.verify_to(output, root / name, plan, jobs, source, software_fixture=True)
        assert result["status"] == "SOFTWARE_FIXTURE_ONLY"
        assert result["native_runtime_qualified"] is result["cohort_released"] is False
        assert result["decoder_calls"] == 12 and result["processor_calls"] == 8
    assert inventory(root / "verify1") == inventory(root / "verify2")
    with pytest.raises(ValueError):
        ver.check_saved(output, plan, jobs, source)
    with pytest.raises(FileExistsError):
        acquire(fixture)
    with pytest.raises(FileExistsError):
        ver.verify_to(output, root / "verify1", plan, jobs, source, software_fixture=True)


@pytest.mark.parametrize("damage", ["signed_zero", "dtype", "cache_length", "logits", "window", "nan", "token", "restoration", "metadata", "across_case"])
def test_capture_first_failure_retained_no_remaining_contexts(fixture, damage):
    root, source, plan, jobs = fixture

    def mutate(p, raw, stage):
        if stage != "on":
            return
        a, m = p["arrays"], p["metadata"]
        if damage == "signed_zero":
            a["cache_00_keys"][0, 0, 0, 0] = -0.0
        elif damage == "dtype":
            p["source_dtypes"]["scores"] = "mlx.core.float64"
        elif damage == "cache_length":
            a["cache_00_keys"] = a["cache_00_keys"][:, :, :-1]
        elif damage == "logits":
            a["logits_1"] = a["logits_1"][:-1]
        elif damage == "window":
            a["site_final"] = a["site_auxiliary"].copy()
        elif damage == "nan":
            a["site_premerger"][0, 0] = np.nan
        elif damage == "token":
            m["decoder_inputs"][1] = [1]
        elif damage == "restoration":
            m["observer_restored"] = False
        elif damage == "metadata":
            m["extra_inputs"] = {"changed": True}
        else:
            a["ids"][0, 0] = 2

    fake = Fake(plan, mutate)
    with pytest.raises(ValueError):
        acquire(fixture, fake)
    assert fake.contexts == 2
    assert read(root / "capture/failure.json")["retry_authorized"] is False
    # The first returned failing observation is retained, not reacquired.
    assert (root / "capture/rows/0/on/cache_00_keys.npy").exists()
    from fractal_vlm_state_probe.stat_matched_source_storage import ledger
    assert ledger(root / "capture", "rows.jsonl")[-1]["stage"] == "on"
    with pytest.raises(ValueError, match="failed captures"):
        ver.check_saved(root / "capture", plan, jobs, source, software_fixture=True)


def rewrite_json(path, value):
    path.unlink()
    write(path, value)


def rewrite_rows(root, rows):
    path = root / "rows.jsonl"
    path.unlink()
    from fractal_vlm_state_probe.stat_matched_source_storage import append
    for row in rows:
        append(root, "rows.jsonl", row)


@pytest.mark.parametrize("damage", ["rehash_cache", "rehash_window", "rehash_dtype", "rehash_events", "extra_file", "raw", "duplicate", "symlink", "source_dtype", "missing_logits", "provenance", "processor_grid_float"])
def test_verifier_rejects_rehashed_corruption_not_just_hash_mismatch(fixture, damage):
    root, source, plan, jobs = fixture
    output = acquire(fixture)
    from fractal_vlm_state_probe.stat_matched_source_storage import ledger
    rows = ledger(output, "rows.jsonl")
    if damage in ("rehash_cache", "rehash_window", "rehash_dtype"):
        name = "site_final" if damage == "rehash_window" else "cache_00_keys"
        row = rows[1]
        path = row["arrays"][name]["path"]
        a = np.load(output / path)
        (output / path).unlink()
        if damage == "rehash_dtype":
            a = a.astype(np.float64)
        else:
            a.flat[0] += 1
        row["arrays"][name] = save_array(output, path, a)
    elif damage == "rehash_events":
        (output / "operations.jsonl").unlink()
        write(output / "operations.jsonl", {"sequence": 1})
    elif damage == "extra_file":
        write(output / "extra.json", {})
    elif damage == "raw":
        p = source / jobs[0]["raw"]["path"]
        p.unlink()
        save_array(source, jobs[0]["raw"]["path"], np.ones((2, 2, 3), np.uint8))
    elif damage == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif damage == "symlink":
        (output / "link").symlink_to(source)
    elif damage == "source_dtype":
        rows[1]["source_dtypes"]["scores"] = "mlx.core.float16"
    elif damage == "missing_logits":
        del rows[1]["arrays"]["logits_1"]
    elif damage == "processor_grid_float":
        processors = ledger(output, "processor.jsonl")
        p = processors[0]["arrays"]["grid"]["path"]
        grid = np.load(output / p).astype(np.float32)
        (output / p).unlink()
        processors[0]["arrays"]["grid"] = save_array(output, p, grid)
        (output / "processor.jsonl").unlink()
        from fractal_vlm_state_probe.stat_matched_source_storage import append
        for pp in processors:
            append(output, "processor.jsonl", pp)
    else:
        frozen = read(output / "frozen.json")
        frozen["software_fixture"] = False
        rewrite_json(output / "frozen.json", frozen)
    rewrite_rows(output, rows)
    result = read(output / "capture.json")
    if damage != "symlink":
        result["receipts"] = inventory(output)
        rewrite_json(output / "capture.json", result)
    with pytest.raises(ValueError):
        ver.verify_to(output, root / "failed_verification", plan, jobs, source, software_fixture=True)
    assert (root / "failed_verification/failure.json").exists()


def test_disk_guard_before_calls_and_payload_guard(fixture, monkeypatch):
    root, source, plan, jobs = fixture
    fake = Fake(plan)
    monkeypatch.setattr(cap.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="reserve"):
        acquire(fixture, fake)
    assert fake.contexts == 0
    monkeypatch.undo()
    plan["storage"]["payload_allowance_bytes"] = 1
    with pytest.raises(ValueError, match="payload"):
        acquire(fixture, fake, "payload")
    assert fake.contexts == 0


def test_decoder_recording_preserves_identity_closes_before_resume_and_restores():
    class Decoder:
        def __call__(self, value):
            return value
    target, events = Decoder(), []
    original = Decoder.__call__
    value = object()
    resumed = []

    def stream():
        try:
            assert target(value) is value
            assert target(value) is value
            yield 1
            resumed.append(True)
        finally:
            resumed.append("closed")
    with decoder_records(target, events.append) as records:
        assert first_yield(stream()) == 1
    assert resumed == ["closed"] and Decoder.__call__ is original
    assert len(records) == 2 and all(x is value and y is value for x, y in records)
    assert events == ["decoder_begin", "decoder_return"]*2
    with pytest.raises(RuntimeError):
        with decoder_records(target, events.append):
            raise RuntimeError("first failure")
    assert Decoder.__call__ is original


def test_absent_release_stops_before_environment_or_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "environment", lambda _: pytest.fail("offline environment reached"))
    with pytest.raises(FileNotFoundError):
        admission.admit(tmp_path.resolve())
    spec = importlib.util.spec_from_file_location("qwen_calibration_cli", ROOT / "scripts/run_qwen_observation_calibration.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for action in ("capture", "verify", "verify-replay"):
        with pytest.raises(FileNotFoundError):
            cli.run(tmp_path.resolve(), action)
    assert not (tmp_path / "runs").exists()


def test_verifier_has_no_capture_or_native_dependency():
    import ast
    tree = ast.parse(Path(ver.__file__).read_text())
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("capture" in x or "native" in x or "contract" in x for x in imports)


def test_pinned_native_generate_function_on_toy_arrays_and_real_adapter(fixture):
    """Execute installed control flow, not MLX/model arithmetic or a processor."""
    import ast
    import functools
    from contextlib import nullcontext
    from fractal_vlm_state_probe.qwen_calibration_native import NativeBackend
    from fractal_vlm_state_probe.stat_matched_source_storage import sha

    spec = importlib.util.find_spec("mlx_vlm")
    if spec is None:
        pytest.skip("optional installed source unavailable; toy qualification not recorded")
    path = Path(spec.origin).parent / "generate.py"
    registered = read(ROOT / PLAN)
    assert sha(path) == registered["installed_generation_source"]["sha256"]
    tree = ast.parse(path.read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "generate_step")
    # Defaults/annotations refer to unrelated runtime definitions. Body unchanged.
    for arg in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs):
        arg.annotation = None
    fn.returns = None
    fn.args.defaults = [ast.Constant(None) for _ in fn.args.defaults]
    fn.args.kw_defaults = [ast.Constant(None) for _ in fn.args.kw_defaults]
    _, _, plan, _ = fixture
    plan = copy.deepcopy(plan)
    plan["limits"]["source_dtype_allowlist"] = ["float32"]

    class Cache:
        def __init__(self):
            self.offset = 0
            self.data = np.zeros((1, 1, 256, 2), np.float32)
        @property
        def state(self):
            return self.data[:, :, :self.offset], self.data[:, :, :self.offset]

    class Decoder:
        def __call__(self, ids, *, cache, **kwargs):
            for entry in cache:
                entry.offset += ids.shape[-1]
            return SimpleNamespace(logits=np.zeros((1, ids.shape[-1], 8), np.float32),
                                   cross_attention_states=None, encoder_outputs=None)

    class Merger:
        def __call__(self, hidden):
            return np.arange(24, dtype=np.float32).reshape(12, 2)

    class Tower:
        def __init__(self):
            self.merger = Merger()
        def __call__(self, pixels, grid, *, output_hidden_states):
            aux = self.merger(np.zeros((48, 3), np.float32))
            return aux[np.argsort(window_order(grid, plan))]

    class Model:
        def __init__(self):
            self.language_model, self.vision_tower = Decoder(), Tower()
        def get_input_embeddings(self, ids, pixels, *, mask, **kwargs):
            self.vision_tower(pixels, kwargs["image_grid_thw"], output_hidden_states=False)
            return SimpleNamespace(inputs_embeds=np.zeros((1, ids.shape[1], 3), np.float32), to_dict=lambda: {})

    mx = SimpleNamespace(array=np.array, stream=lambda _: nullcontext(),
                         logsumexp=lambda x: np.log(np.exp(x).sum()), async_eval=lambda _: None,
                         eval=lambda _: None, clear_cache=lambda: None)
    namespace = {"mx": mx, "functools": functools, "generation_stream": None,
                 "make_sampler": lambda **_: lambda x: np.argmax(x, axis=-1),
                 "make_logits_processors": lambda *args: [], "maybe_quantize_kv_cache": lambda *a, **k: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), "exec"), namespace)
    backend = NativeBackend.__new__(NativeBackend)
    backend.plan, backend.mx, backend.model = plan, mx, Model()
    backend.generator = SimpleNamespace(generate_step=namespace["generate_step"],
        cache=SimpleNamespace(make_prompt_cache=lambda _: [Cache(), Cache()]))
    prepared = {"input_ids": np.array([[1, 2]], np.int32), "pixel_values": np.zeros((48, 1), np.float32),
                "attention_mask": np.ones((1, 2), np.int32), "image_grid_thw": np.array([[1, 6, 8]], np.int32)}
    backend._prepare = lambda raw: prepared.copy()
    before = (Decoder.__call__, Tower.__call__, Merger.__call__)
    packets = []
    for stage in cap.STAGES:
        events = []
        p = backend.context(None, stage, events.append)
        assert events == ["prepare_begin", "prepare_return", "decoder_begin", "decoder_return", "decoder_begin", "decoder_return"]
        cap.validate_packet(p, {"pixels": prepared["pixel_values"], "grid": prepared["image_grid_thw"]}, stage, plan)
        assert p["arrays"]["cache_00_keys"].shape == (1, 1, 3, 2)
        assert p["metadata"]["decoder_inputs"] == [[1, 2], [0]]
        assert (Decoder.__call__, Tower.__call__, Merger.__call__) == before
        packets.append(p)
    for p in packets[1:]:
        assert all(cap.words_equal(p["arrays"][k], packets[0]["arrays"][k]) for k in packets[0]["arrays"])
    backend._prepare = lambda raw: {**prepared, "input_ids": np.zeros((1, 513), np.int32)}
    events = []
    with pytest.raises(ValueError, match="limits"):
        backend.context(None, "on", events.append)
    assert events == ["prepare_begin", "prepare_return"]
    prepared["pixel_values"][0, 0] = np.nan
    backend._prepare = lambda raw: prepared.copy()
    events = []
    with pytest.raises(ValueError, match="limits"):
        backend.context(None, "on", events.append)
    assert events == ["prepare_begin", "prepare_return"]


@pytest.mark.parametrize("damage", ["grid", "nonfinite", "shape"])
def test_standalone_failure_stops_before_any_context_and_is_retained(fixture, damage):
    root, source, plan, jobs = fixture
    backend = Fake(plan)
    original = backend.processor

    def processor(raw):
        p = original(raw)
        if damage == "grid":
            p["grid"][0, 1] = 65
        elif damage == "nonfinite":
            p["pixels"][0, 0] = np.inf
        else:
            p["pixels"] = p["pixels"][:-1]
        return p

    backend.processor = processor
    with pytest.raises(ValueError):
        acquire(fixture, backend)
    assert backend.contexts == 0
    assert (root / "capture/processor/0/pixels.npy").exists()
    assert read(root / "capture/failure.json")["retained_artifact_sha256"]


@pytest.mark.parametrize("action", ["capture", "verify", "verify-replay"])
def test_cli_ordered_verification_and_consumed_failures(fixture, monkeypatch, action):
    root, source, plan, jobs = fixture
    spec = importlib.util.spec_from_file_location("qwen_cli", ROOT / "scripts/run_qwen_observation_calibration.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "admit", lambda *a, **k: (plan, jobs, {}))
    if action == "capture":
        consumed = root / plan["future_native_root"]
        consumed.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            cli.run(root, action)
    elif action == "verify-replay":
        with pytest.raises(FileNotFoundError):
            cli.run(root, action)
    else:
        replay = root / plan["future_verification_roots"][1]
        replay.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            cli.run(root, action)


def test_loading_failure_keeps_frozen_root_and_blocks_retry(fixture):
    root, source, plan, jobs = fixture
    calls = []

    def failing_loader():
        calls.append(True)
        assert (root / "loading/frozen.json").exists()
        raise RuntimeError("load failed")

    with pytest.raises(RuntimeError, match="load failed"):
        cap.capture(root / "loading", plan, jobs, source, failing_loader, {}, software_fixture=True)
    assert read(root / "loading/failure.json")["last_event_sequence"] == 0
    with pytest.raises(FileExistsError):
        cap.capture(root / "loading", plan, jobs, source, failing_loader, {}, software_fixture=True)
    assert calls == [True]


def test_capture_admission_rechecks_environment_but_verify_does_not(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    plan = read(ROOT / PLAN)
    frozen = {"status": "OFFLINE_PREFLIGHT_ONLY", "plan_sha256": admission.PLAN_SHA,
              "software": {"toy": "digest"}, "environment": {"snapshot": "toy"}}
    pf = root / admission.PREFLIGHT / "preflight.json"
    write(pf, frozen)
    from fractal_vlm_state_probe.stat_matched_source_storage import sha
    release = {"approved_for_native_execution": True, "cohort_generation_authorized": False,
               "scope": "qwen_observation_calibration_v1_only", "plan_sha256": admission.PLAN_SHA,
               "preflight_sha256": sha(pf), "software": frozen["software"]}
    write(root / admission.RELEASE, release)
    monkeypatch.setattr(admission, "registered", lambda _: (plan, []))
    monkeypatch.setattr(admission, "software", lambda _: frozen["software"])
    monkeypatch.setattr(admission, "environment", lambda _: {"snapshot": "changed"})
    with pytest.raises(ValueError, match="snapshot changed"):
        admission.admit(root)
    monkeypatch.setattr(admission, "environment", lambda _: pytest.fail("verification must not require runtime"))
    monkeypatch.setattr(admission.shutil, "disk_usage", lambda _: pytest.fail("verification must not demand launch space"))
    assert admission.admit(root, for_capture=False)[2]["environment"] == frozen["environment"]
    release["cohort_generation_authorized"] = True
    rewrite_json(root / admission.RELEASE, release)
    with pytest.raises(ValueError, match="calibration-only"):
        admission.admit(root, for_capture=False)


def test_failed_offline_preflight_is_consumed_without_runtime_load(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(admission, "registered", lambda _: (read(ROOT / PLAN), []))
    monkeypatch.setattr(admission, "software", lambda _: {})
    monkeypatch.setattr(admission, "environment", lambda _: (_ for _ in ()).throw(ValueError("offline missing")))
    with pytest.raises(ValueError, match="offline missing"):
        admission.preflight(root)
    assert read(root / admission.PREFLIGHT / "failure.json")["retry_authorized"] is False
    with pytest.raises(FileExistsError):
        admission.preflight(root)
