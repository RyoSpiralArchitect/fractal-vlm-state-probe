import copy
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import analyze_representation_panel as analyzer
from fractal_vlm_state_probe.cli import run_representation_panel as runner
from fractal_vlm_state_probe.cli import summarize_representation_panel as publisher
from fractal_vlm_state_probe.cli.summarize_controlled_overlap import oracle_bootstrap
from fractal_vlm_state_probe.controlled_overlap_cache_panel import ordered_jobs
from fractal_vlm_state_probe.representation_observation import array_receipts
from fractal_vlm_state_probe.representation_panel import (
    GIB,
    array_path,
    bootstrap,
    codec_check,
    disk_budget,
    expand,
    oracle_expand,
    pack,
    run_path,
    seal,
    summarize,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def config():
    return runner.read(runner.CONFIG)


@pytest.fixture
def metadata():
    return {
        f"f{f}_r{r}": {
            "broad_class": "a" if f < 4 else "b",
            "pairing_family": f"f{f}",
            "replicate": r,
        }
        for f in range(8)
        for r in range(1, 5)
    }


def test_lossless_codec_includes_every_finite_native_word():
    assert codec_check() == 65280
    words = np.array([0, 32768, 1, 32769, 0x7F7F, 0xFF7F, 0x3F80], dtype="<u2")
    np.testing.assert_array_equal(pack(oracle_expand(words)), words)
    assert expand(words).tobytes() == oracle_expand(words).tobytes()
    assert np.signbit(expand(words)[1])
    for a in (
        np.array([1.1], np.float32),
        np.array([np.nan], np.float32),
        np.ones(1, np.float64),
    ):
        with pytest.raises(ValueError):
            pack(a)
    with pytest.raises(ValueError):
        expand(words.astype(np.uint32))


def test_conservative_budget_keeps_eight_gib_without_assumed_compression(config):
    b = disk_budget(config)
    assert b["packed_panel_bytes"] == 2048 * (786432 + 229376) * 2
    assert b["peak_scratch_bytes"] == 64 * 786432 * 8
    assert b["panel_cache_bytes"] == 6144 * 50816 * 4
    assert (
        b["fresh_calibration_bytes"] == 128 * (786432 + 229376) * 4 + 1152 * 50816 * 4
    )
    assert (
        b["reserve_bytes"] == 8 * GIB and b["required_at_launch_bytes"] == 16230187008
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("panel_cells", 128),
        ("sites", ["encoder"]),
        ("retry_or_resume", True),
        ("reserve_bytes", 0),
        ("equivalence_tested", True),
        ("maximum_scratch_vectors", 128),
    ],
)
def test_registered_contract_rejects_changed_scope(config, key, value):
    validate_config(config)
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_reference_seal_order_and_membership(metadata):
    jobs = ordered_jobs(metadata)
    rows = [
        {
            **j,
            "ordinal": i + 385,
            "path": run_path(j),
            "representation": {"path": array_path(j)},
        }
        for i, j in enumerate(jobs[:256])
    ]
    assert seal(rows, jobs)["first_test_ordinal"] == 641
    for invalid in (rows[:-1], [rows[1], rows[0], *rows[2:]], [*rows[:-1], rows[0]]):
        with pytest.raises(ValueError):
            seal(invalid, jobs)


def test_retained_loader_returns_exact_runtime_and_restores_on_error(monkeypatch):
    identity = object()

    def loader(*args):
        return identity

    monkeypatch.setattr(runner.cal, "_load_mlx_runtime", loader)
    with runner.retain_loaded_runtime() as loaded:
        assert runner.cal._load_mlx_runtime("qualified") is identity
    assert loaded == [identity] and runner.cal._load_mlx_runtime is loader
    with pytest.raises(ValueError, match="more than one"):
        with runner.retain_loaded_runtime():
            runner.cal._load_mlx_runtime("qualified")
            runner.cal._load_mlx_runtime("qualified")
    assert runner.cal._load_mlx_runtime is loader


def test_bootstrap_matches_independent_nested_draws_and_retains_missingness(
    config, metadata
):
    rng = np.random.default_rng(345)
    cube = rng.normal(size=(2, 4, 2, 3, 16, 8))
    cube[0, 0, 0, 1] = np.nan
    cube[1, 0, 0, 2] = np.nan
    spec = dict(config["bootstrap"], draws=30)
    families = sorted({v["pairing_family"] for v in metadata.values()})
    labels = sorted(k for k, v in metadata.items() if v["replicate"] > 2)
    agg, ep, ranges, margins = summarize(cube, labels, families, metadata, spec)
    assert [v["available"] for v in ep] == [True, True, False, False]
    assert [v["available"] for v in agg[:3]] == [True, False, True]
    assert ranges["ranges"][1] is None and "mean" not in ep[2]
    np.testing.assert_allclose(
        ranges["ranges"][0], oracle_bootstrap(margins, spec)[0], rtol=0, atol=1e-12
    )
    margins[:] = 0
    np.testing.assert_array_equal(bootstrap(margins, spec)["ranges"], 0)


@pytest.mark.parametrize("fail", [True, False])
def test_panel_driver_gates_first_test_and_never_reloads_model(
    tmp_path, monkeypatch, metadata, config, fail
):
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(k, "1")
    jobs = ordered_jobs(metadata)
    frozen = {
        "release": config,
        "panel_jobs": jobs,
        "hierarchy_metadata": metadata,
        "disk_budget": disk_budget(config),
        "config": {"model_id": "fixture"},
        "model_snapshot": {"id": "fixture"},
        "runtime_identity": {"id": "fixture"},
        "code_sha256": {},
        "source_sha256": {},
        "manifest_sha256": {},
    }
    runtime = {
        "model": SimpleNamespace(mm_projector=lambda x: x),
        "processor": object(),
    }
    loads, called = [], []

    def cal_run(mode, root, pre=None):
        root.mkdir(parents=True)
        if mode == "preflight":
            write_json(root / "frozen_specification.json", frozen)
            write_json(root / "preflight.json", {"status": "PASS"})
        else:
            assert runner.cal._load_mlx_runtime("fixture") is runtime

    def fake_capture(root, job, r, p, original, f, sha):
        assert r is runtime and p is runtime["model"].mm_projector
        if job["phase"] == "test":
            assert (
                runner.read(root / "reference_seal.json")["first_test_ordinal"] == 641
            )
            assert len(runner.read(root / "reference_cells.json")) == 256
            if fail:
                raise ValueError("injected first-test mismatch")
        else:
            assert not (root / "reference_seal.json").exists()
        called.append(job)
        return {
            **job,
            "path": run_path(job),
            "representation": {"path": array_path(job)},
        }

    monkeypatch.setattr(
        runner.cal, "_load_mlx_runtime", lambda model: loads.append(model) or runtime
    )
    monkeypatch.setattr(runner.cal, "run", cal_run)
    monkeypatch.setattr(runner.cal, "qualify", lambda: frozen)
    monkeypatch.setattr(runner, "specification", lambda base: frozen)
    monkeypatch.setattr(runner, "check_space", lambda root, required: required + 1)
    monkeypatch.setattr(runner.cal_pub, "publish", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "recalibration_parity",
        lambda *args: {"representation_array_pairs_exact": 256},
    )
    monkeypatch.setattr(
        runner.cal, "runtime_identity", lambda r: frozen["runtime_identity"]
    )
    monkeypatch.setattr(
        runner.cal,
        "_freeze_model_snapshot",
        lambda model, path: write_json(path, frozen["model_snapshot"]),
    )
    monkeypatch.setattr(runner, "capture_cell", fake_capture)
    pre, root = tmp_path / "pre", tmp_path / "run"
    assert runner.run("preflight", pre)["new_model_forwards"] == 0
    assert not loads and not called
    if fail:
        with pytest.raises(ValueError, match="first-test"):
            runner.run("run", root, pre)
        failure = runner.read(root / "execution_failure.json")
        assert (failure["attempted_panel_cells"], failure["completed_panel_cells"]) == (
            257,
            256,
        )
        assert (
            failure["primary_results_withheld"]
            and not (root / "capture_summary.json").exists()
        )
    else:
        result = runner.run("run", root, pre)
        assert result["new_source_contexts"] == 2432 and result["same_model_instance"]
        assert called == jobs
    assert loads == ["fixture"]
    with pytest.raises(FileExistsError):
        runner.run("run", root, pre)


def test_low_disk_stops_before_any_calibration_or_model_call(tmp_path, monkeypatch):
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(k, "1")
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=8 * GIB)
    )
    monkeypatch.setattr(
        runner.cal, "run", lambda *a: pytest.fail("unexpected calibration")
    )
    with pytest.raises(ValueError, match="disk gate"):
        runner.run("preflight", tmp_path / "pre")
    failure = runner.read(tmp_path / "pre/execution_failure.json")
    assert failure["attempted_panel_cells"] == 0


@pytest.mark.parametrize("corruption", [None, "suffix", "cache", "missing_call"])
def test_single_observed_cell_matches_prior_run_before_packing(
    tmp_path, monkeypatch, corruption
):
    from fractal_vlm_state_probe.padding_cache import input_fingerprint

    cfg = runner.read(runner.REPO / "configs/controlled_overlap_cache_phase2_v1.json")
    monkeypatch.setattr(runner, "REPO", tmp_path)
    job = {
        "phase": "reference",
        "map": "m0_reference",
        "pair_id": "f0_r1",
        "cell": "mm",
    }
    path = tmp_path / "old" / run_path(job)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"registered": True})
    payload = {
        "pixel_values": np.zeros((1, 3, 2, 2), np.float32),
        "image_sizes": [[320, 320]],
        "input_ids": [[1]],
    }
    inputs = input_fingerprint(payload)
    artifacts = [
        {
            "layer_index": int(t.split(":")[0]),
            "tensor": t.split(":")[1],
            "source_dtype": cfg["expected_source_dtype"],
            "dtype": "float32",
            "shape": cfg["expected_tensor_shape"],
            "effective_sequence_length": 397,
        }
        for t in runner.cal.TARGETS
    ]
    old = {
        "model_id": cfg["model_id"],
        "manifest_path": str(manifest),
        "dry_run": False,
        "context_policy": {},
        "runtime": {},
        "reproducibility": {"seed": cfg["source_seed"]},
        "stream_events": [
            {
                "prompt": "p",
                "frame_sha256s": ["f"],
                "cache_token_layout": {},
                "assistant_text": "The image",
                "generation": {
                    "steps": [
                        {"token_id": i} for i in cfg["expected_generation_step_ids"]
                    ]
                },
                "cache_tensor_artifacts": artifacts,
            }
        ],
    }
    write_json(path, old)
    write_json(
        path.with_suffix(".input_audit.json"),
        {"context": {"manifest_path": str(manifest)}, "effective": inputs},
    )
    generator = SimpleNamespace(prepare_inputs=lambda: payload)

    class Native:
        dtype = "mlx.core.bfloat16"

        def __init__(self, channels):
            self.values = np.ones((1, 2, channels), np.float32)

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.values, dtype=dtype)

    class Projector:
        def __call__(self, x):
            return Native(2)

    projector = Projector()
    original = Projector.__call__
    rt = {"model": SimpleNamespace(mm_projector=projector)}

    def fake_replay(config, *, mlx_runtime):
        assert mlx_runtime is rt and config.source_cache_only
        generator.prepare_inputs()
        if corruption != "missing_call":
            projector(Native(3))
        current = copy.deepcopy(old)
        if corruption == "suffix":
            current["stream_events"][0]["assistant_text"] = "different"
        write_json(config.output_path, current)

    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda n: generator if n == "mlx_vlm.generate" else original_import(n),
    )
    monkeypatch.setattr(runner.cal, "run_cumulative_replay_probe", fake_replay)
    monkeypatch.setattr(runner.cal, "contract", lambda *a: {})

    def tensor(run, p, c):
        return np.array(
            [-0.0 if corruption == "cache" and p != path else 0.0], np.float32
        )

    monkeypatch.setattr(runner.cal, "_validate_target_contract", tensor)
    f = {
        "config": cfg,
        "release": {"prior_panel_root": "old"},
        "manifest_sha256": {str(manifest): runner.sha(manifest)},
        "observation_config": {
            "source_dtype": "mlx.core.bfloat16",
            "sites": {
                "encoder": {"shape": [1, 2, 3]},
                "projector": {"shape": [1, 2, 2]},
            },
        },
    }
    root = tmp_path / "new"
    if corruption:
        with pytest.raises(ValueError):
            runner.capture_cell(root, job, rt, projector, original, f, "frozen")
        assert not (root / array_path(job)).exists()
    else:
        row = runner.capture_cell(root, job, rt, projector, original, f, "frozen")
        assert row["observer_calls"] == 1 and row["observer_restored"]
        with np.load(root / row["representation"]["path"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(
                expand(saved["encoder"]), np.ones((1, 2, 3), np.float32)
            )
        assert row["pixel_values_sha256"] == inputs["pixel_values"]["sha256"]
    assert Projector.__call__ is original


def test_full_synthetic_reconstruction_independent_publisher_and_tamper_guard(
    tmp_path, monkeypatch, config, metadata
):
    root = tmp_path / "capture"
    root.mkdir()
    jobs = ordered_jobs(metadata)
    config = dict(config, bootstrap=dict(config["bootstrap"], draws=25))
    obs = {
        "sites": {"encoder": {"shape": [1, 2, 7]}, "projector": {"shape": [1, 2, 3]}},
        "source_dtype": "mlx.core.bfloat16",
        "baseline_publication": "examples/research_notes/0065_nonlearned_processing_panel/summary.json",
    }
    recipes = {
        f"m{m}_reference": {
            "mask": m,
            "q": None,
            "destination": None,
            "reference": f"m{m}_reference",
        }
        for m in range(4)
    }
    recipes.update(
        {
            f"m{m}_q{q}_d{d}": {
                "mask": m,
                "q": q,
                "destination": d,
                "reference": f"m{m}_reference",
            }
            for m in range(4)
            for d in range(2)
            for q in (0, 1, 4)
        }
    )
    frozen = {
        "release": config,
        "hierarchy_metadata": metadata,
        "observation_config": obs,
        "map_recipes": recipes,
        "disk_budget": {"reserve_bytes": 0, "peak_scratch_bytes": 0},
        "config": {
            "phase1_publication": "examples/research_notes/0056_controlled_overlap_phase1/summary.json"
        },
        "code_sha256": {},
        "source_sha256": {},
        "manifest_sha256": {},
    }
    rng = np.random.default_rng(23842)
    prior = tmp_path / "prior.json"
    write_json(prior, {"manifest_path": "accepted"})
    rows = []
    for i, j in enumerate(jobs):
        a = {
            "encoder": rng.integers(-40, 41, size=(1, 2, 7)).astype(np.float32),
            "projector": np.zeros((1, 2, 3), np.float32),
        }
        path = root / array_path(j)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{s: pack(v) for s, v in a.items()})
        rows.append(
            {
                **j,
                "ordinal": 385 + i,
                "path": run_path(j),
                "prior_run": str(prior),
                "pixel_values_sha256": "p",
                "cache_array_sha256": dict.fromkeys(runner.cal.TARGETS, "a"),
                "representation": {
                    "path": array_path(j),
                    "sha256": runner.sha(path),
                    "arrays": array_receipts(a, obs),
                },
            }
        )
    capture = {"receipts": {"frozen_specification.json": "fixture"}}
    write_json(root / "capture_summary.json", capture)
    write_json(root / "reference_seal.json", {"reference_cells": 256})
    write_json(
        root / "recalibration_parity.json", {"representation_array_pairs_exact": 256}
    )
    write_json(root / "recalibration_verified/summary.json", {"status": "PASS"})
    monkeypatch.setattr(analyzer, "load_capture", lambda _: (capture, frozen, rows))
    monkeypatch.setattr(publisher, "load_capture", analyzer.load_capture)
    monkeypatch.setattr(
        publisher,
        "recalibration_parity",
        lambda f, r: runner.read(r / "recalibration_parity.json"),
    )
    monkeypatch.setattr(
        publisher.cal_pub,
        "publish",
        lambda raw, out: write_json(out / "summary.json", {"status": "PASS"}),
    )

    def parity(*args):
        return (
            {
                t: {"after_array_sha256": "a", "scalar_count": 50816}
                for t in runner.cal.TARGETS
            },
            {
                "effective": {"pixel_values": {"sha256": "p"}},
                "context": {"manifest_path": "accepted"},
            },
            "The image",
        )

    monkeypatch.setattr(publisher, "verify_pair", parity)
    monkeypatch.setattr(
        publisher, "plot_profiles", lambda agg, p: p.write_bytes(b"fixture")
    )
    analysis = analyzer.analyze(root)
    assert [e["available"] for e in analysis["primary_endpoints"]] == [
        True,
        True,
        False,
        False,
    ]
    result = publisher.publish(root, tmp_path / "verified")
    check = result["independent_verification"]
    assert (
        check["full_contrast_byte_equalities"],
        check["score_matrices"],
        check["direction_records"],
    ) == (1024, 56, 512)
    assert check["maximum_score_abs_error"] < 1e-12 and check["model_forwards"] == 0
    assert result["reference_seal"]["reference_cells"] == 256
    first = rows[0]
    invalid = copy.deepcopy(first)
    invalid["representation"]["arrays"]["encoder"]["array_sha256"] = "bad"
    with pytest.raises(ValueError, match="source bytes"):
        analyzer.site_values(root, invalid, "encoder", [1, 2, 7], True)
    analysis["primary_endpoints"][0]["mean"] += 0.1
    write_json(root / "analysis/summary.json", analysis)
    with pytest.raises(ValueError, match="endpoint mean"):
        publisher.publish(root, tmp_path / "rejected")
    assert (tmp_path / "rejected/verification_failure.json").exists()
    assert not (tmp_path / "rejected/summary.json").exists()
