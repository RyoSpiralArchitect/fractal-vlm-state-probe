import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import run_controlled_overlap_cache_panel as runner
from fractal_vlm_state_probe.controlled_overlap_cache_panel import (
    bootstrap_ranges,
    direction,
    ordered_jobs,
    require_reference_seal,
    summarize_cube,
    validate_panel_capture,
)
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def config():
    c = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/controlled_overlap_cache_phase2_v1.json"
        ).read_text()
    )
    c["analysis"]["bootstrap"]["draws"] = 100
    return c


@pytest.fixture
def metadata():
    return {
        f"f{f}_r{r}": {
            "pairing_family": f"f{f}",
            "replicate": r,
            "broad_class": "a" if f < 4 else "b",
        }
        for f in range(8)
        for r in range(1, 5)
    }


def test_reference_first_exact_budget_and_seal(metadata):
    jobs = ordered_jobs(metadata)
    assert len(jobs) == 2048
    assert all(j["phase"] == "reference" for j in jobs[:256])
    assert all(j["phase"] == "test" for j in jobs[256:])
    paths = [
        f"panel/{j['phase']}/{j['map']}/{j['pair_id']}_{j['cell']}.json" for j in jobs
    ]
    assert len(set(paths)) == 2048
    registry = {k: {"ordinal": i + 129} for i, k in enumerate(paths[:256])}
    assert require_reference_seal(registry, metadata) == registry
    with pytest.raises(ValueError):
        require_reference_seal(dict(list(registry.items())[:-1]), metadata)
    registry[paths[256]] = {}
    with pytest.raises(ValueError):
        require_reference_seal(registry, metadata)


def oracle_bootstrap(values, spec):
    rng = np.random.default_rng(np.random.SeedSequence([spec["seed"]]))
    draws = []
    for _ in range(spec["draws"]):
        masks, families = rng.integers(0, 4, 4), rng.integers(0, 8, 8)
        sources = rng.integers(0, 2, (8, 2))
        total = np.zeros((6, 3))
        for mask in masks:
            destinations = rng.integers(0, 2, 2)
            block = np.take(values[:, mask], destinations, axis=1).mean(axis=1)
            for i, family in enumerate(families):
                total += (
                    np.take(block[:, :, family], sources[i], axis=-1).mean(axis=-1) / 32
                )
        draws.append(np.column_stack((total[:, 0], total[:, 2] - total[:, 0])))
    return np.quantile(draws, spec["quantiles"], axis=0).transpose(1, 2, 0)


def test_six_view_bootstrap_matches_independent_oracle(config):
    margins = np.random.default_rng(71).normal(size=(6, 4, 2, 3, 8, 2))
    spec = config["analysis"]["bootstrap"]
    actual = bootstrap_ranges(margins, spec)
    np.testing.assert_allclose(
        actual["ranges"], oracle_bootstrap(margins, spec), atol=1e-12
    )
    assert actual == bootstrap_ranges(margins, spec)
    margins[:, :, :, 2] = margins[:, :, :, 0]
    np.testing.assert_array_equal(
        np.asarray(bootstrap_ranges(margins, spec)["ranges"])[:, 1], 0
    )


def test_missing_primary_withholds_both_endpoints_only_for_affected_view(
    config, metadata
):
    cube = np.random.default_rng(72).normal(size=(6, 4, 2, 3, 16, 8))
    labels = sorted(k for k, v in metadata.items() if v["replicate"] in (3, 4))
    families = sorted({v["pairing_family"] for v in metadata.values()})
    _, original, _, _ = summarize_cube(cube, labels, families, metadata, config)
    cube[0, 1, 1, 0, 0, 0] = np.nan
    aggregates, endpoints, bootstrap, _ = summarize_cube(
        cube, labels, families, metadata, config
    )
    assert len(endpoints) == 12 and len(aggregates) == 18
    assert [r["available"] for r in endpoints] == [False, False] + [True] * 10
    assert bootstrap["ranges"][0] is None
    assert endpoints[2:] == original[2:]
    assert all(r["p_value"] is None and not r["equivalence_tested"] for r in endpoints)


def test_middle_level_missingness_does_not_change_primary(config, metadata):
    cube = np.random.default_rng(73).normal(size=(6, 4, 2, 3, 16, 8))
    labels = sorted(k for k, v in metadata.items() if v["replicate"] in (3, 4))
    families = sorted({v["pairing_family"] for v in metadata.values()})
    _, original, _, _ = summarize_cube(cube, labels, families, metadata, config)
    cube[1, :, :, 1] = np.nan
    aggregates, endpoints, _, _ = summarize_cube(
        cube, labels, families, metadata, config
    )
    assert endpoints == original
    assert not aggregates[4]["available"]
    assert sum(not r["available"] for r in aggregates) == 1


def test_summary_mean_respects_equal_family_weights(config, metadata):
    cube = np.zeros((6, 4, 2, 3, 16, 8))
    labels = sorted(k for k, v in metadata.items() if v["replicate"] in (3, 4))
    families = sorted({v["pairing_family"] for v in metadata.values()})
    for i, label in enumerate(labels):
        own = families.index(metadata[label]["pairing_family"])
        cube[..., i, own] = 1
    cube[:, :, :, 2] *= 3
    aggregates, endpoints, _, _ = summarize_cube(
        cube, labels, families, metadata, config
    )
    assert [e["mean"] for e in endpoints] == [1, 2] * 6
    assert [a["retrieval_correct_count"] for a in aggregates] == [16] * 18
    assert all(a["test_seed_count"] == 16 for a in aggregates)


def test_direction_zero_and_nonfinite_remain_explicit():
    a = np.arange(1, 9, dtype=float)
    assert direction(a, 2 * a)["norm_ratio"] == 2
    assert direction(a, -a)["cosine"] == pytest.approx(-1)
    assert direction(a * 0, a)["cosine"] is None
    assert direction(a * 0, a)["norm_ratio"] is None
    with pytest.raises(ValueError):
        direction(a, a * np.nan)


def test_panel_input_validation_allows_only_registered_pixels_and_frame(config):
    artifacts = [
        {
            "layer_index": int(t.split(":")[0]),
            "tensor": t.split(":")[1],
            "source_dtype": "mlx.core.bfloat16",
            "dtype": "float32",
            "shape": [1, 2, 397, 64],
            "effective_sequence_length": 397,
        }
        for t in config["targets"]
    ]
    old = {
        "model_id": config["model_id"],
        "runtime": {},
        "context_policy": {},
        "reproducibility": {"seed": 20260604},
        "stream_events": [
            {
                "prompt": "frozen",
                "frame_sha256s": ["old"],
                "cache_token_layout": {},
                "assistant_text": "The image",
                "generation": {"steps": [{"token_id": i} for i in [785, 2168, 2168]]},
                "cache_tensor_artifacts": artifacts,
            }
        ],
    }
    inputs = {
        "pixel_values": {"sha256": "old_pixels"},
        "image_sizes": {"values": [[320, 320]]},
        "input_ids": {"values": [1, 2]},
    }
    new = copy.deepcopy(old)
    new["stream_events"][0]["frame_sha256s"] = ["new"]
    desired = copy.deepcopy(inputs)
    desired["pixel_values"]["sha256"] = "new_pixels"
    audit = {
        "native": copy.deepcopy(desired),
        "effective": copy.deepcopy(desired),
        "override_image_sizes": None,
    }
    validate_panel_capture(
        new,
        audit,
        old,
        inputs,
        {"pixel_values_sha256": "new_pixels"},
        {"frame_sha256": "new"},
        config,
    )
    assert (
        old["stream_events"][0]["frame_sha256s"] == ["old"]
        and inputs["pixel_values"]["sha256"] == "old_pixels"
    )
    audit["native"]["input_ids"]["values"] = [2, 1]
    audit["effective"]["input_ids"]["values"] = [2, 1]
    with pytest.raises(ValueError):
        validate_panel_capture(
            new,
            audit,
            old,
            inputs,
            {"pixel_values_sha256": "new_pixels"},
            {"frame_sha256": "new"},
            config,
        )


def test_panel_preflight_has_no_model_loading_or_resume(tmp_path, monkeypatch, config):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    frozen = {"config": config, "model_snapshot": {"revision": "qualified"}}
    monkeypatch.setattr(runner, "specification", lambda: frozen)
    monkeypatch.setattr(
        runner,
        "_freeze_model_snapshot",
        lambda model, path: write_json(path, frozen["model_snapshot"]),
    )

    def forbidden(*args):
        raise AssertionError("no model before qualified release")

    monkeypatch.setattr(runner, "_load_mlx_runtime", forbidden)
    preflight = tmp_path / "preflight"
    result = runner.run("preflight", preflight)
    assert result["new_model_forwards"] == 0 and not result["model_weights_loaded"]
    with pytest.raises(FileExistsError):
        runner.run("preflight", preflight)
    result["status"] = "FAILED"
    write_json(preflight / "preflight.json", result)
    with pytest.raises(ValueError, match="preflight"):
        runner.run("run", tmp_path / "capture", preflight)
    failure = json.loads((tmp_path / "capture/execution_failure.json").read_text())
    assert failure["attempted"] == {"calibration": 0, "panel": 0}
    assert failure["primary_results_withheld"]
