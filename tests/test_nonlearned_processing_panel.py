import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import analyze_nonlearned_processing_panel as analyzer
from fractal_vlm_state_probe.cli import run_nonlearned_processing_panel as runner
from fractal_vlm_state_probe.cli import (
    summarize_nonlearned_processing_panel as publisher,
)
from fractal_vlm_state_probe.controlled_overlap_cache_panel import ordered_jobs
from fractal_vlm_state_probe.nonlearned_processing_panel import (
    VIEWS,
    cell_path,
    feature_key,
    paired_ranges,
    reference_seal,
    summarize_scores,
    validate_features,
    vector_diagnostics,
)
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


@pytest.fixture
def config():
    result = json.loads(
        (
            Path(__file__).parents[1] / "configs/nonlearned_processing_phase3_v1.json"
        ).read_text()
    )
    result["planned_panel"]["bootstrap"]["draws"] = 50
    return result


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


def test_reference_seal_exact_order_and_no_extra_test_cell(metadata):
    jobs = ordered_jobs(metadata)
    rows = [
        {**j, "path": cell_path(j), "ordinal": i + 129}
        for i, j in enumerate(jobs[:256])
    ]
    assert reference_seal(rows, metadata) == rows
    for broken in (
        rows[:-1],
        rows[::-1],
        [*rows, {"path": cell_path(jobs[256]), "ordinal": 385}],
    ):
        with pytest.raises(ValueError):
            reference_seal(broken, metadata)


def test_twelve_view_bootstrap_and_paired_missingness(config, metadata):
    cube = np.random.default_rng(13).normal(size=(12, 4, 2, 3, 16, 8))
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    aggregates, endpoints, ranges, margins = summarize_scores(
        cube, labels, families, metadata, config
    )
    assert len(aggregates) == 36 and len(endpoints) == 24
    np.testing.assert_allclose(
        ranges["ranges"],
        publisher.oracle_ranges(margins, config["planned_panel"]["bootstrap"]),
        rtol=0,
        atol=1e-12,
    )
    cube[0, :, :, :, :, :] = np.nan
    cube[3, :, :, 1] = np.nan
    _, modified, ranges, margins = summarize_scores(
        cube, labels, families, metadata, config
    )
    assert [r["available"] for r in modified] == [False, False] + [True] * 22
    assert modified[2:] == endpoints[2:]
    assert ranges["ranges"][0] is None
    assert (
        publisher.oracle_ranges(margins, config["planned_panel"]["bootstrap"])[0]
        is None
    )
    assert all(r["p_value"] is None and not r["equivalence_tested"] for r in modified)
    cube[1, 2, 1, 2, 0, 0] = np.nan
    _, missing, _, _ = summarize_scores(cube, labels, families, metadata, config)
    assert not missing[2]["available"] and not missing[3]["available"]


def test_zero_paired_effect_and_invalid_margin_layout(config):
    margins = np.random.default_rng(14).normal(size=(12, 4, 2, 3, 8, 2))
    margins[:, :, :, 2] = margins[:, :, :, 0]
    ranges = paired_ranges(margins, config["planned_panel"]["bootstrap"])
    np.testing.assert_array_equal(np.asarray(ranges["ranges"])[:, 1], 0)
    with pytest.raises(ValueError):
        paired_ranges(margins[:6], config["planned_panel"]["bootstrap"])
    margins[0, 0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        paired_ranges(margins, config["planned_panel"]["bootstrap"])


def test_feature_contract_and_numerical_diagnostic_not_filtering():
    features = {
        feature_key(s, f): np.zeros(6 if f == "global_moments" else 192)
        for s, f in VIEWS
    }
    validate_features(features)
    bad = copy.deepcopy(features)
    bad[next(iter(bad))] = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError):
        validate_features(bad)
    diagnostic = vector_diagnostics(np.array([1e-14]), 1e-10)
    assert diagnostic["available"] and diagnostic["norm_to_oracle_bound_ratio"] < 1
    assert not vector_diagnostics(np.zeros(6), 1e-10)["available"]


def test_preflight_no_processor_and_failure_no_resume(tmp_path, monkeypatch, config):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    frozen = {"config": config, "processor_jobs": [{"artifact": {}}]}
    monkeypatch.setattr(runner, "specification", lambda: frozen)

    def forbidden(*args):
        raise AssertionError("processor must not start in preflight")

    monkeypatch.setattr(runner, "qualified_processor", forbidden)
    pre = tmp_path / "preflight"
    result = runner.run("preflight", pre)
    assert result["new_processor_calls"] == result["new_model_forwards"] == 0
    assert not result["processor_instantiated"]
    result["status"] = "FAILED"
    write_json(pre / "preflight.json", result)
    root = tmp_path / "failed"
    with pytest.raises(ValueError, match="preflight"):
        runner.run("run", root, pre)
    failure = json.loads((root / "execution_failure.json").read_text())
    assert failure["attempted"] == {"calibration": 0, "panel": 0}
    assert failure["primary_results_withheld"]
    with pytest.raises(FileExistsError):
        runner.run("run", root, pre)


def test_processor_pixel_or_metadata_change_fails(tmp_path, monkeypatch, config):
    rgb = np.zeros((320, 320, 3), dtype=np.uint8)
    pixels = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
    monkeypatch.setattr(runner, "_checked_artifact", lambda _: (rgb, {}))
    monkeypatch.setattr(publisher, "_checked_artifact", lambda _: (rgb, {}))
    expected = {
        "pixel_values_sha256": runner.sha256_array(pixels),
        "processor_non_pixel_payload": {"image_sizes": [[320, 320]]},
    }
    payload = {"pixel_values": pixels, "image_sizes": np.array([[320, 320]])}
    runner.process_cell(lambda **k: payload, {}, expected, config)
    payload["image_sizes"] = np.array([[320, 240]])
    with pytest.raises(ValueError):
        runner.process_cell(lambda **k: payload, {}, expected, config)
    with pytest.raises(ValueError):
        publisher.verify_processor(lambda **k: payload, {}, expected, config)


def test_analysis_and_publication_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "execution_failure.json", {"status": "FAILED"})
    with pytest.raises(ValueError, match="failed capture"):
        analyzer.analyze(root)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError, match="failed capture"):
        publisher.publish(root, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_full_synthetic_analysis_and_independent_verification(
    tmp_path, monkeypatch, config, metadata
):
    root = tmp_path / "panel"
    root.mkdir()
    jobs = ordered_jobs(metadata)
    keys = sorted({j["map"] for j in jobs})
    recipes = {}
    for key in keys:
        bits = key.split("_")
        recipes[key] = {
            "recipe": {
                "mask": int(bits[0][1:]),
                "q": None if bits[1] == "reference" else int(bits[1][1:]),
                "destination": None if bits[1] == "reference" else int(bits[2][1:]),
                "reference": f"{bits[0]}_reference",
                "attempt_shared_moved": [],
            }
        }
    source = tmp_path / "phase1.json"
    write_json(source, {"map_construction": {"maps": recipes}})
    config["phase1_publication"] = str(source)
    frozen = {"config": config, "hierarchy_metadata": metadata}
    generator = np.random.default_rng(822)
    rows = []
    for i, job in enumerate(jobs):
        relative = cell_path(job)
        p = root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            feature_key(s, f): generator.normal(
                size=6 if f == "global_moments" else 192
            )
            for s, f in VIEWS
        }
        arrays[feature_key("raw_rgb", "global_moments")] = np.full(
            6, 0.0 if job["cell"] in ("mm", "jm") else 1.0
        )
        np.savez(p, **arrays)
        rows.append(
            {**job, "ordinal": i + 129, "path": relative, "sha256": sha256_file(p)}
        )
    write_json(root / "capture_summary.json", {"status": "COMPLETE"})
    monkeypatch.setattr(analyzer, "load_capture", lambda p: ({}, frozen, rows))
    result = analyzer.analyze(root)
    assert result["raw_global_moment_zero_blocks"] == 512
    assert [r["available"] for r in result["primary_endpoints"]] == [False, False] + [
        True
    ] * 22
    verified, ledgers, portable, stats = publisher.verify_analysis(root, frozen, rows)
    assert verified == result and stats["bootstrap_verified"]
    assert len(portable) == 6144 and len(ledgers["directions.jsonl"]) == 3072
    assert all(
        k.startswith(("raw_rgb__", "processor_pixel_values__")) for k in portable
    )
    vector = Path(ledgers["vectors.jsonl"][0]["path"])
    np.save(vector, np.ones(6))
    with pytest.raises(ValueError, match="interaction vector changed"):
        publisher.verify_analysis(root, frozen, rows)
