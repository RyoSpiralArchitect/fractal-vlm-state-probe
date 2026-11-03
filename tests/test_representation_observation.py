import copy
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import (
    run_representation_observation_calibration as runner,
)
from fractal_vlm_state_probe.cli import (
    summarize_representation_observation_calibration as publisher,
)
from fractal_vlm_state_probe.representation_observation import (
    STAGES,
    array_receipts,
    observe_projector,
    require_complete,
    validate_arrays,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def config():
    return json.loads(
        (
            Path(__file__).parents[1]
            / "configs/representation_observation_calibration_v1.json"
        ).read_text()
    )


class Projector:
    def __init__(self):
        self.calls = []
        self.result = object()

    def __call__(self, value):
        self.calls.append(value)
        return self.result


def test_observer_preserves_identity_single_call_and_other_instances():
    target, other, value = Projector(), Projector(), object()
    original = Projector.__call__
    with observe_projector(target) as records:
        assert other(value) is other.result
        assert target(value) is target.result
        assert records == [(value, target.result)]
    assert Projector.__call__ is original
    assert target.calls == [value] and other.calls == [value]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "exception", "nested"])
def test_observer_restores_on_all_failure_paths(mode):
    target, value = Projector(), object()
    original = Projector.__call__
    with pytest.raises((ValueError, RuntimeError)):
        with observe_projector(target):
            if mode == "exception":
                raise RuntimeError("retained failure")
            if mode == "nested":
                with observe_projector(target):
                    pass
            if mode == "duplicate":
                target(value)
                target(value)
    assert Projector.__call__ is original


def test_observer_does_not_materialize_lazy_values():
    class Lazy:
        def __array__(self, *args, **kwargs):
            raise AssertionError("must not materialize inside observer")

    target, value = Projector(), Lazy()
    target.result = Lazy()
    with observe_projector(target) as records:
        assert target(value) is target.result
    assert records[0][0] is value and records[0][1] is target.result


@pytest.mark.parametrize(
    "key,value",
    [
        ("retry_or_resume", True),
        ("experimental_panel_released", True),
        ("new_source_contexts", 128),
        ("on_off_tensor_comparisons", 384),
        ("new_correspondence_scores", 1),
        ("source_dtype", "mlx.core.float16"),
        ("stages", ["on", "off_before", "off_after"]),
    ],
)
def test_registration_rejects_budget_or_path_drift(config, key, value):
    validate_config(config)
    changed = copy.deepcopy(config)
    changed[key] = value
    with pytest.raises(ValueError, match="contract"):
        validate_config(changed)


@pytest.fixture
def arrays_config(config):
    config["sites"]["encoder"]["shape"] = [1, 2, 3]
    config["sites"]["projector"]["shape"] = [1, 2, 2]
    arrays = {k: np.ones(v["shape"], np.float32) for k, v in config["sites"].items()}
    return arrays, config


@pytest.mark.parametrize(
    "corruption", ["shape", "dtype", "nan", "not_bfloat16", "missing"]
)
def test_native_expansion_contract_rejects_corruption(arrays_config, corruption):
    arrays, config = arrays_config
    validate_arrays(arrays, config)
    if corruption == "shape":
        arrays["encoder"] = arrays["encoder"][:, :1]
    elif corruption == "dtype":
        arrays["encoder"] = arrays["encoder"].astype(np.float64)
    elif corruption == "nan":
        arrays["encoder"].flat[0] = np.nan
    elif corruption == "not_bfloat16":
        arrays["encoder"].flat[0] = np.nextafter(np.float32(1), np.float32(2))
    else:
        del arrays["projector"]
    with pytest.raises(ValueError):
        validate_arrays(arrays, config)


def test_saved_representations_reloaded_and_signed_zero_retained(
    tmp_path, arrays_config
):
    arrays, config = arrays_config
    arrays["encoder"].flat[0] = -0.0
    receipt = {"path": "representations.npz", "arrays": array_receipts(arrays, config)}
    np.savez_compressed(tmp_path / receipt["path"], **arrays)
    receipt["sha256"] = runner._sha(tmp_path / receipt["path"])
    assert (
        publisher.checked_representations(tmp_path, receipt, config)
        == receipt["arrays"]
    )
    arrays["encoder"].flat[0] = 0.0
    np.savez_compressed(tmp_path / receipt["path"], **arrays)
    with pytest.raises(ValueError, match="hash"):
        publisher.checked_representations(tmp_path, receipt, config)
    receipt["sha256"] = runner._sha(tmp_path / receipt["path"])
    with pytest.raises(ValueError, match="receipt"):
        publisher.checked_representations(tmp_path, receipt, config)


@pytest.fixture
def jobs_rows():
    jobs = [
        {"pair_id": f"f{f}_r{r}", "cell": c}
        for f in range(8)
        for r in range(1, 5)
        for c in ("mm", "jj", "mj", "jm")
    ]
    rows = [
        {
            **job,
            "stage": stage,
            "ordinal": i * 3 + si + 1,
            "historical_cache_exact": True,
            "observer_restored": True,
            "observer_calls": int(stage == "on"),
            "representation": {"path": "on.npz"} if stage == "on" else None,
        }
        for i, job in enumerate(jobs)
        for si, stage in enumerate(STAGES)
    ]
    return jobs, rows


@pytest.mark.parametrize(
    "corruption",
    ["missing", "duplicate", "order", "restoration", "cache", "off_capture"],
)
def test_all_128_bracketed_inputs_required(jobs_rows, corruption):
    jobs, rows = jobs_rows
    require_complete(rows, jobs)
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0]
    elif corruption == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif corruption == "restoration":
        rows[0]["observer_restored"] = False
    elif corruption == "cache":
        rows[0]["historical_cache_exact"] = False
    else:
        rows[0]["representation"] = {"path": "unexpected.npz"}
    with pytest.raises(ValueError):
        require_complete(rows, jobs)


def test_preflight_never_loads_model_and_failure_never_retries(
    tmp_path, monkeypatch, config
):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    frozen = {
        "config": {"model_id": "qualified"},
        "observation_config": config,
        "model_snapshot": {"qualified": True},
    }
    monkeypatch.setattr(runner, "qualify", lambda: frozen)
    monkeypatch.setattr(
        runner,
        "_freeze_model_snapshot",
        lambda model, path: write_json(path, frozen["model_snapshot"]),
    )

    def forbidden(*args):
        raise AssertionError("no model load permitted")

    monkeypatch.setattr(runner, "_load_mlx_runtime", forbidden)
    pre = tmp_path / "pre"
    result = runner.run("preflight", pre)
    assert result["new_model_forwards"] == 0 and not result["observer_installed"]
    result["status"] = "FAILED"
    write_json(pre / "preflight.json", result)
    with pytest.raises(ValueError, match="preflight"):
        runner.run("calibrate", tmp_path / "run", pre)
    failure = json.loads((tmp_path / "run/execution_failure.json").read_text())
    assert (
        failure["attempted_source_contexts"]
        == failure["completed_source_contexts"]
        == 0
    )
    with pytest.raises(FileExistsError):
        runner.run("preflight", pre)


def test_source_hash_guard_preserves_original_contract(tmp_path):
    path = tmp_path / "source.json"
    write_json(path, {"original": True})
    frozen = {
        "code_sha256": {},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
        "manifest_sha256": {},
    }
    runner.assert_frozen(frozen)
    write_json(path, {"original": False})
    with pytest.raises(ValueError, match="source_sha256"):
        runner.assert_frozen(frozen)


def test_bracketed_capture_runs_all_contexts_and_defers_conversion(
    tmp_path, monkeypatch, config, jobs_rows
):
    jobs, _ = jobs_rows
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")

    class Native:
        dtype = "mlx.core.bfloat16"

        def __init__(self, shape):
            self.array = np.ones(shape, np.float32)

    class RuntimeProjector(list):
        def __call__(self, value):
            return Native([1, 2, 2])

    projector = RuntimeProjector()
    original_call = RuntimeProjector.__call__
    config["projector_class"] = (
        f"{RuntimeProjector.__module__}.{RuntimeProjector.__name__}"
    )
    config["projector_components"] = []
    config["sites"]["encoder"]["shape"] = [1, 2, 3]
    config["sites"]["projector"]["shape"] = [1, 2, 2]
    old_config = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/controlled_overlap_cache_phase2_v1.json"
        ).read_text()
    )
    old_path, manifest = tmp_path / "historical.json", tmp_path / "manifest.json"
    write_json(old_path, {"historical": True})
    write_json(
        old_path.with_suffix(".input_audit.json"), {"effective": {"qualified": True}}
    )
    write_json(manifest, {"qualified": True})
    jobs = [
        {
            **j,
            "historical_run": str(old_path),
            "manifest_path": str(manifest),
            "pixel_values_sha256": "accepted",
        }
        for j in jobs
    ]
    frozen = {
        "config": old_config,
        "observation_config": config,
        "jobs": jobs,
        "model_snapshot": {"qualified": True},
        "runtime_identity": {"qualified": True},
        "registered_design_commit": "frozen_before_run",
        "code_sha256": {},
        "source_sha256": {},
        "manifest_sha256": {},
    }
    runtime = {"model": SimpleNamespace(mm_projector=projector)}
    monkeypatch.setattr(runner, "qualify", lambda: frozen)
    monkeypatch.setattr(
        runner,
        "_freeze_model_snapshot",
        lambda model, path: write_json(path, frozen["model_snapshot"]),
    )
    loads = []
    monkeypatch.setattr(
        runner, "_load_mlx_runtime", lambda model: loads.append(model) or runtime
    )
    monkeypatch.setattr(
        runner, "runtime_identity", lambda runtime: frozen["runtime_identity"]
    )
    monkeypatch.setattr(runner, "contract", lambda *args: {})
    cache = np.ones((1, 2, 397, 64), np.float32)
    monkeypatch.setattr(runner, "_validate_target_contract", lambda *args: cache)
    monkeypatch.setattr(runner, "validate_capture", lambda *args: None)
    generator = SimpleNamespace()
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: generator if name == "mlx_vlm.generate" else original_import(name),
    )
    generation_complete, calls, conversions = [False], [], []

    @contextmanager
    def audit(module, *, audit_path, context, **kwargs):
        write_json(audit_path, {"context": context})
        yield

    def fake_replay(run_config, *, mlx_runtime):
        generation_complete[0] = False
        assert mlx_runtime is runtime
        stage = next(s for s in STAGES if run_config.output_path.stem.endswith(s))
        assert (RuntimeProjector.__call__ is not original_call) == (stage == "on")
        projector(Native([1, 2, 3]))
        write_json(
            run_config.output_path, {"stream_events": [{"cache_tensor_artifacts": []}]}
        )
        calls.append(stage)
        generation_complete[0] = True

    def convert(native):
        assert generation_complete[0] and RuntimeProjector.__call__ is original_call
        conversions.append(native.array.shape)
        return native.array

    monkeypatch.setattr(runner, "prepared_input_audit", audit)
    monkeypatch.setattr(runner, "run_cumulative_replay_probe", fake_replay)
    monkeypatch.setattr(runner, "_to_float32_numpy", convert)
    pre, root = tmp_path / "pre", tmp_path / "capture"
    runner.run("preflight", pre)
    assert not loads and not calls
    result = runner.run("calibrate", root, pre)
    assert loads == [old_config["model_id"]]
    assert calls == list(STAGES) * 128
    assert len(conversions) == 256
    assert result["calibration_source_contexts"] == 384
    assert not result["experimental_panel_released"]
    rows = [
        json.loads(line) for line in (root / "cells.jsonl").read_text().splitlines()
    ]
    require_complete(rows, jobs)
    assert len(list((root / "representations").glob("*.npz"))) == 128
    assert RuntimeProjector.__call__ is original_call


def test_independent_publication_checks_complete_brackets_without_model_calls(
    tmp_path, monkeypatch, config, jobs_rows
):
    jobs, rows = jobs_rows
    root, pre = tmp_path / "capture", tmp_path / "pre"
    root.mkdir()
    pre.mkdir()
    jobs = [
        {
            **j,
            "historical_run": "/qualified/historical.json",
            "manifest_path": "/qualified/manifest.json",
            "pixel_values_sha256": "pixels",
        }
        for j in jobs
    ]
    frozen = {
        "jobs": jobs,
        "observation_config": config,
        "registered_design_commit": "registered_before_capture",
        "model_snapshot": {"qualified": True},
        "runtime_identity": {"qualified": True},
        "code_sha256": {},
        "source_sha256": {},
        "manifest_sha256": {},
    }
    for directory in (root, pre):
        write_json(directory / "frozen_specification.json", frozen)
        write_json(directory / "model_snapshot.json", frozen["model_snapshot"])
    write_json(root / "model_snapshot_after.json", frozen["model_snapshot"])
    write_json(root / "runtime_identity.json", frozen["runtime_identity"])
    write_json(
        pre / "preflight.json",
        {
            "status": "PASS",
            "new_model_forwards": 0,
            "model_weights_loaded": False,
            "observer_installed": False,
            "experimental_panel_released": False,
            "frozen_specification_sha256": runner._sha(
                pre / "frozen_specification.json"
            ),
            "model_snapshot_sha256": runner._sha(pre / "model_snapshot.json"),
        },
    )
    write_json(
        root / "preflight_receipt.json",
        {"path": str(pre), "sha256": runner._sha(pre / "preflight.json")},
    )
    arrays = {k: np.ones(v["shape"], np.float32) for k, v in config["sites"].items()}
    source_npz = root / "synthetic_representations.npz"
    np.savez_compressed(source_npz, **arrays)
    rep_receipts = array_receipts(arrays, config)
    cache = np.ones((1, 2, 397, 64), np.float32)
    array_hash = hashlib.sha256(cache.tobytes()).hexdigest()
    registry = {}
    for row in rows:
        relative = f"calibration/{row['pair_id']}_{row['cell']}_{row['stage']}.json"
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        artifacts = [
            {"layer_index": int(t.split(":")[0]), "tensor": t.split(":")[1]}
            for t in runner.TARGETS
        ]
        write_json(path, {"stream_events": [{"cache_tensor_artifacts": artifacts}]})
        receipt = {
            "run_sha256": runner._sha(path),
            "input_audit_sha256": "verified-by-fake-oracle",
            "historical_run": jobs[0]["historical_run"],
            "tensor_artifacts": artifacts,
        }
        registry[relative] = receipt
        row.update(
            path=relative,
            cache_array_sha256={t: array_hash for t in runner.TARGETS},
            **receipt,
        )
        if row["stage"] == "on":
            rep_path = f"representations/{row['pair_id']}_{row['cell']}.npz"
            (root / rep_path).parent.mkdir(exist_ok=True)
            os.link(source_npz, root / rep_path)
            row["representation"] = {
                "path": rep_path,
                "sha256": runner._sha(root / rep_path),
                "arrays": rep_receipts,
            }
    (root / "cells.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(root / "capture_registry.json", registry)
    names = (
        "frozen_specification.json",
        "model_snapshot.json",
        "model_snapshot_after.json",
        "preflight_receipt.json",
        "runtime_identity.json",
        "cells.jsonl",
        "capture_registry.json",
    )
    write_json(
        root / "summary.json",
        {
            "status": "PASS",
            "registered_design_commit": frozen["registered_design_commit"],
            "calibration_source_contexts": 384,
            "historical_source_cells": 128,
            "historical_tensor_comparisons": 1152,
            "on_off_tensor_comparisons": 768,
            "observed_source_cells": 128,
            "representation_arrays": 256,
            "new_model_forwards": 384,
            "new_experimental_panel_cells": 0,
            "new_direct_probe_cells": 0,
            "new_correspondence_scores": 0,
            "experimental_panel_released": False,
            "confirmation_cohort": "not_generated_not_sealed",
            "receipts": {n: runner._sha(root / n) for n in names},
        },
    )
    calls = []

    def verify(before, after, receipt, frozen, frozen_sha):
        calls.append(str(after))
        audit = {
            "context": {"manifest_path": jobs[0]["manifest_path"]},
            "effective": {"pixel_values": {"sha256": "pixels"}},
        }
        result = {
            t: {
                "before_array_sha256": array_hash,
                "after_array_sha256": array_hash,
                "nonzero": True,
                "scalar_count": cache.size,
            }
            for t in runner.TARGETS
        }
        return result, audit, "The image"

    def forbidden(*args):
        raise AssertionError("publisher must not load a model")

    monkeypatch.setattr(publisher, "verify_pair", verify)
    monkeypatch.setattr(publisher, "checked_array", lambda *args: cache)
    monkeypatch.setattr(runner, "_load_mlx_runtime", forbidden)
    result = publisher.publish(root, tmp_path / "public")
    assert len(calls) == 384
    assert {k: v["rows"] for k, v in result["ledgers"].items()} == {
        "cells.jsonl": 384,
        "historical_comparisons.jsonl": 1152,
        "on_off_comparisons.jsonl": 768,
        "representations.jsonl": 128,
    }
    check = result["independent_verification"]
    assert check["representation_arrays"] == 256
    assert (
        check["verification_model_forwards"]
        == check["verification_processor_calls"]
        == 0
    )
    (root / "cells.jsonl").write_text("corruption retained\n")
    with pytest.raises(ValueError, match="receipt changed"):
        publisher.publish(root, tmp_path / "rejected-publication")
