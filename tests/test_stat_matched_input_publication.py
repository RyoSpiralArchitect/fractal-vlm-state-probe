"""Public-only raw recipe and complete feature/residual replay; no processor."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe import stat_matched_input as m
from fractal_vlm_state_probe import stat_matched_input_oracle as oracle
from fractal_vlm_state_probe.cli.run_stat_matched_input_calibration import (
    require_complete,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "examples/research_notes/0087_stat_matched_input_calibration"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tables():
    s = read(PUBLIC / "summary.json")
    data = {}
    for name, meta in s["ledgers"].items():
        assert sha(PUBLIC / name) == meta["sha256"]
        data[name] = [json.loads(x) for x in (PUBLIC / name).read_bytes().splitlines()]
        assert len(data[name]) == meta["rows"]
    return s, data


def test_provenance_full_denominators_metadata_and_release_boundary():
    s, data = tables()
    cfg = read(ROOT / "configs/stat_matched_input_calibration_v1.json")
    assert (
        s["status"] == "PASS" and s["calibration_passed"] and s["first_failure"] is None
    )
    assert s["design_config_sha256"] == sha(
        ROOT / "configs/stat_matched_input_calibration_v1.json"
    )
    assert len(s["code_sha256"]) == 118
    for name, digest in s["code_sha256"].items():
        assert sha(ROOT / name) == digest
    require_complete(s["actual_counts"], cfg)
    assert (
        s["attempted_processor_calls"]
        == s["completed_processor_calls"]
        == s["planned_processor_calls"]
        == 153
    )
    assert s["model_forwards"] == s["verification_processor_calls"] == 0
    assert not s["measurement_panel_released"]
    assert s["confirmation_cohort"] == "not_generated_not_sealed"
    replay = read(PUBLIC / "verification_replay.json")
    assert (
        replay["analysis_kind"] == "stat_matched_input_calibration_verification_replay"
    )
    assert replay["status"] == "PASS" and replay["registered_outputs"] == 8
    assert replay["verification_processor_calls"] == replay["model_forwards"] == 0
    assert replay["byte_identical_files"] == {
        n: sha(PUBLIC / n) for n in ("summary.json", *s["ledgers"], *s["archives"])
    }
    historical = read(
        ROOT
        / "examples/research_notes/0063_nonlearned_processing_calibration/summary.json"
    )["frozen_specification"]["processor_jobs"]
    for i, (r, j) in enumerate(
        zip(data["historical_processor.jsonl"], historical, strict=True)
    ):
        assert r["ordinal"] == i + 1 and r["exact"]
        assert (r["pair_id"], r["cell"]) == (j["pair_id"], j["cell"])
        assert r["pixel_word_sha256"] == j["pixel_values_sha256"]
        assert r["non_pixel_payload"] == j["processor_non_pixel_payload"]
    for r, case in zip(
        data["synthetic_processor.jsonl"], data["cases.jsonl"], strict=True
    ):
        assert r["key"] == case["key"] and not case["failures"]
        assert r["shape"] == [1, 3, 1024, 1024] and r["dtype"] == "float32"
        assert (
            r["non_pixel_payload"]
            == cfg["future_calibration"]["synthetic_non_pixel_expected"]
        )
        assert r["pixel_word_sha256"] == case["pixel_word_sha256"]


def test_all_raw_images_and_maps_reconstruct_from_registered_recipe():
    _, data = tables()
    cfg = read(ROOT / "configs/stat_matched_input_calibration_v1.json")
    for row in data["coordinate_maps.jsonl"]:
        a = oracle.coordinate_map(cfg, row["shift"])
        assert m.word_sha(a) == row["word_sha256"]
        moved = int(np.count_nonzero(a.ravel() != np.arange(a.size)))
        assert moved == row["moved"] == (6912 if row["shift"] else 0)
        assert row["fixed"] + moved == 320 * 320
    for row, expected in zip(data["raw_inputs.jsonl"], m.cases(cfg), strict=True):
        key, name, shift = expected
        assert (row["key"], row["fixture"], row["shift"]) == expected
        raw = oracle.fixture(cfg, name, shift, key == "sentinel")
        assert m.word_sha(raw) == row["word_sha256"]


def test_all_300_oracle_and_216_match_vectors_and_sentinel_replay():
    s, data = tables()
    cfg = read(ROOT / "configs/stat_matched_input_calibration_v1.json")
    for name, meta in s["archives"].items():
        assert sha(PUBLIC / name) == meta["sha256"]
    with np.load(PUBLIC / "features.npz", allow_pickle=False) as z:
        features = {k: z[k] for k in z.files}
    with np.load(PUBLIC / "residuals.npz", allow_pickle=False) as z:
        residuals = {k: z[k] for k in z.files}
    assert len(features) == s["archives"]["features.npz"]["arrays"] == 600
    assert len(residuals) == s["archives"]["residuals.npz"]["arrays"] == 228
    expected_features, expected_residuals = set(), set()
    oracle_count = match_count = sentinel_count = 0
    for case in data["cases.jsonl"]:
        key = case["key"]
        for r in case["oracle_checks"]:
            st, f = r["start"], r["feature"]
            names = [f"{key}__{kind}__{st}__{f}" for kind in ("actual", "oracle")]
            expected_features.update(names)
            a, b = [features[n] for n in names]
            assert (
                a.shape
                == b.shape
                == (cfg["feature_contract"]["feature_dimensions"][f],)
            )
            assert (
                a.dtype == b.dtype == np.float64
                and np.isfinite(a).all()
                and np.isfinite(b).all()
            )
            maximum = float(np.max(np.abs(a - b)))
            assert maximum == r["max_abs"] and maximum <= 1e-10
            oracle_count += 1
        for r in case["comparisons"]:
            st, f = r["start"], r["feature"]
            a = features[f"{key}__actual__{st}__{f}"]
            b = features[f"{case['fixture']}_s0__actual__{st}__{f}"]
            delta = a - b
            n = f"{key}__{st}__{f}"
            expected_residuals.add(n)
            assert delta.tobytes() == residuals[n].tobytes()
            assert m.word_sha(delta) == r["residual_word_sha256"]
            assert float(np.max(np.abs(delta))) == r["max_abs"]
            assert math.isclose(
                float(np.linalg.norm(delta)), r["l2"], rel_tol=1e-12, abs_tol=1e-14
            )
            assert math.isclose(
                float(np.linalg.norm(a)), r["norm"], rel_tol=1e-12, abs_tol=1e-14
            )
            assert math.isclose(
                float(np.linalg.norm(b)),
                r["reference_norm"],
                rel_tol=1e-12,
                abs_tol=1e-14,
            )
            if key == "sentinel":
                if f == "grid_mean":
                    assert r["max_abs"] > 1e-3
                    sentinel_count += 1
            else:
                assert r["max_abs"] <= 1e-10
                match_count += 1
        if case["comparisons"]:
            for words in case["word_checks"].values():
                if key != "sentinel":
                    assert len(words["matched_groups"]) == 65 and all(
                        words["matched_groups"]
                    )
                    assert words["word_equal"] == (
                        case["fixture"] in ("zero", "constant")
                    )
                else:
                    assert not words["matched_groups"][0]
    assert (oracle_count, match_count, sentinel_count) == (300, 216, 2)
    assert set(features) == expected_features and set(residuals) == expected_residuals
