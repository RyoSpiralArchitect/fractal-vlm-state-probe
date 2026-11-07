"""Toy-only implementation tests; never generate the registered 320px fixtures."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe import stat_matched_input as m
from fractal_vlm_state_probe import stat_matched_input_oracle as oracle
from fractal_vlm_state_probe.cli import run_stat_matched_input_calibration as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def toy():
    s = json.loads(
        (ROOT / "configs/stat_matched_input_calibration_v1.json").read_bytes()
    )
    c = s["construction"]
    c.update(
        raw_shape_hwc=[80, 80, 3],
        map_shape=[80, 80],
        grid=[2, 2],
        content_rows=[0, 80],
        active_grid_rows=[0, 2],
        active_grid_cols=[0, 2],
        moved_sites_nonidentity=576,
        fixed_sites_nonidentity=5824,
    )
    s["fixtures"]["noise_seed"] = 17
    s["fixtures"]["sentinel"]["raw_rows"] = [0, 40]
    s["future_calibration"]["processor_shape_nchw"] = [1, 3, 80, 80]
    config = json.loads((ROOT / s["inherited_feature_config"]).read_bytes())
    config["grid"] = [2, 2]
    config["windows"]["processor_pixel_values"] = copy.deepcopy(
        config["windows"]["raw_rgb"]
    )
    return s, config


def fake_pixels(raw):
    a = (raw.astype(np.float32) / np.float32(255) - np.float32(0.5)) / np.float32(0.25)
    return np.moveaxis(a, -1, 0)[None]


def test_toy_maps_are_independently_reconstructed_bijections(toy):
    s, _ = toy
    for shift in range(4):
        a, b = m.coordinate_map(s, shift), oracle.coordinate_map(s, shift)
        row = m.map_audit(a, b, s, shift)
        assert row["moved"] == (576 if shift else 0)
        assert row["fixed"] + row["moved"] == 6400
        bad = a.copy()
        bad[0, 0] = 1
        with pytest.raises(ValueError):
            m.map_audit(bad, b, s, shift)
    with pytest.raises(ValueError):
        m.coordinate_map(s, 4)


def test_all_toy_payloads_and_sentinel_match_scalar_oracle(toy):
    s, _ = toy
    for key, name, shift in m.cases(s):
        a = m.fixture(s, name, shift, m.coordinate_map(s, shift), key == "sentinel")
        b = oracle.fixture(s, name, shift, key == "sentinel")
        assert a.shape == (80, 80, 3) and a.dtype == np.uint8
        assert a.tobytes() == b.tobytes()
        if key == "sentinel":
            assert np.all(a[:40, :40] == 255)


def test_toy_end_to_end_matching_and_expected_sentinel_rejection(toy):
    s, config = toy
    refs = {}
    comparisons = errors = groups = changes = shams = 0
    for key, name, shift in m.cases(s):
        raw = m.fixture(s, name, shift, m.coordinate_map(s, shift), key == "sentinel")
        pixels = fake_pixels(raw)
        arrays, checks = m.feature_arrays(raw, pixels, config)
        ref = refs.get(name) if shift or key == "sentinel" else None
        row, residuals = m.case_audit(
            key, name, shift, raw, pixels, arrays, checks, ref, config, s
        )
        assert not row["failures"]
        errors += len(checks)
        if ref is None:
            refs[name] = {
                start: (values, arrays, row["joint_multisets"][start])
                for start, values in zip(m.STARTS, (raw, pixels), strict=True)
            }
        elif key != "sentinel":
            comparisons += len(row["comparisons"])
            groups += sum(len(x["matched_groups"]) for x in row["word_checks"].values())
            assert all(np.max(np.abs(v)) <= 1e-10 for v in residuals.values())
            if name in ("zero", "constant"):
                shams += 2
            else:
                changes += 2
        else:
            assert all(not x["matched_groups"][0] for x in row["word_checks"].values())
            assert all(
                x["max_abs"] > 1e-3
                for x in row["comparisons"]
                if x["feature"] == "grid_mean"
            )
    assert (errors, comparisons, groups, changes, shams) == (300, 216, 180, 24, 12)


def test_joint_color_not_channel_marginals_and_native_signed_zero():
    a = np.array([[[0, 1, 2], [1, 2, 0], [2, 0, 1]]], dtype=np.uint8)
    b = np.array([[[0, 2, 1], [1, 0, 2], [2, 1, 0]]], dtype=np.uint8)
    assert np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1))
    assert m.joint_multisets(a, "raw_rgb", [1, 1]) != m.joint_multisets(
        b, "raw_rgb", [1, 1]
    )
    z = np.zeros((1, 3, 2, 2), dtype=np.float32)
    n = z.copy()
    n[0, 0, 0, 0] = -0.0
    assert m.joint_multisets(z, "processor_pixel_values", [1, 1]) != m.joint_multisets(
        n, "processor_pixel_values", [1, 1]
    )


def test_bad_native_dtype_shape_and_nonfinite_fail_closed(toy):
    s, _ = toy
    for a in (
        np.zeros((1, 3, 80, 80), dtype=np.float64),
        np.zeros((1, 3, 40, 40), dtype=np.float32),
        np.full((1, 3, 80, 80), np.nan, dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            m.checked_pixels(a, s)


def toy_runtime(toy, tmp_path, monkeypatch, failure=None):
    s, features = copy.deepcopy(toy)
    s["inherited_feature_config"] = "toy_features.json"
    cal = s["future_calibration"]
    cal.update(
        historical_processor_checks=2,
        total_processor_calls=27,
        joint_multiset_comparisons=180,
        reserve_bytes=0,
        required_launch_free_bytes=0,
    )
    cal["synthetic_non_pixel_expected"]["image_sizes"] = [[80, 80]]
    release = {"preflight_root": "preflight", "capture_root": "capture"}
    (tmp_path / "configs").mkdir()
    for name, obj in (
        (runner.DESIGN, s),
        (runner.RELEASE, release),
        ("toy_features.json", features),
    ):
        (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
    monkeypatch.setattr(runner, "REPO", tmp_path)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    images = [
        np.zeros((80, 80, 3), dtype=np.uint8),
        np.full((80, 80, 3), 93, dtype=np.uint8),
    ]
    jobs = [
        {
            "pair_id": f"toy_{i}",
            "cell": "mm",
            "artifact": i,
            "pixel_values_sha256": m.word_sha(fake_pixels(a)),
            "processor_non_pixel_payload": cal["synthetic_non_pixel_expected"],
        }
        for i, a in enumerate(images)
    ]
    spec = {
        "registered_commit": "toy",
        "config": s,
        "release": release,
        "code_sha256": {},
        "legacy": {"code_sha256": {}, "source_sha256": {}, "processor_jobs": jobs},
    }
    monkeypatch.setattr(runner, "qualify", lambda: spec)
    monkeypatch.setattr(runner.old, "_checked_artifact", lambda i: (images[i], {}))
    calls = []

    def processor(*, images, text, return_tensors):
        assert text == ["<image>"] and return_tensors == "np"
        raw = np.asarray(images)
        assert raw.shape == (80, 80, 3)
        calls.append(1)
        pixels = fake_pixels(raw)
        other = copy.deepcopy(cal["synthetic_non_pixel_expected"])
        if len(calls) == 1 and failure == "historical":
            pixels[0, 0, 0, 0] += np.float32(0.5)
        if len(calls) == 4 and failure:
            if failure == "exception":
                raise RuntimeError("toy processor interrupted")
            if failure == "metadata":
                other["image_sizes"] = [[81, 80]]
            if failure == "matching":
                pixels[0, 0, 0, 0] += np.float32(0.5)
        return {"pixel_values": pixels, **{k: np.array(v) for k, v in other.items()}}

    monkeypatch.setattr(runner, "load_processor", lambda: processor)
    return spec, calls


def test_toy_complete_lifecycle_two_identical_file_verifications(
    toy, tmp_path, monkeypatch
):
    _, calls = toy_runtime(toy, tmp_path, monkeypatch)
    assert runner.run("preflight", tmp_path / "preflight")["processor_calls"] == 0
    assert not calls
    capture = runner.run("calibrate", tmp_path / "capture")
    assert capture["status"] == "PASS" and len(calls) == 27
    bad_counts = dict(capture["counts"], sentinel_audited=False)
    with pytest.raises(ValueError):
        runner.require_complete(bad_counts, runner.read(tmp_path / runner.DESIGN))
    for name in ("verify_a", "verify_b"):
        result = runner.run("verify", tmp_path / name)
        assert result["status"] == "PASS" and result["calibration_passed"]
        assert result["actual_counts"]["feature_oracle_vector_pairs"] == 300
    assert len(calls) == 27
    for name in (*runner.LEDGERS, "features.npz", "residuals.npz", "summary.json"):
        assert (tmp_path / "verify_a" / name).read_bytes() == (
            tmp_path / "verify_b" / name
        ).read_bytes()


@pytest.mark.parametrize("failure", ["matching", "metadata", "exception"])
def test_toy_first_failure_retained_no_retry_and_never_promoted(
    toy, tmp_path, monkeypatch, failure
):
    _, calls = toy_runtime(toy, tmp_path, monkeypatch, failure)
    runner.run("preflight", tmp_path / "preflight")
    with pytest.raises((ValueError, RuntimeError)):
        runner.run("calibrate", tmp_path / "capture")
    first = (tmp_path / "capture/failure.json").read_bytes()
    assert len(calls) == 4
    with pytest.raises(FileExistsError):
        runner.run("calibrate", tmp_path / "capture")
    result = runner.run("verify", tmp_path / "verify")
    assert (
        result["status"] == "FAILED_CALIBRATION_AUDITED"
        and not result["calibration_passed"]
    )
    assert result["attempted_processor_calls"] == 4
    assert result["actual_counts"]["processor_return_receipts"] == (
        3 if failure == "exception" else 4
    )
    assert result["completed_processor_calls"] == (3 if failure == "exception" else 4)
    runner.run("verify", tmp_path / "verify_replay")
    for name in (*runner.LEDGERS, "features.npz", "residuals.npz", "summary.json"):
        assert (tmp_path / "verify" / name).read_bytes() == (
            tmp_path / "verify_replay" / name
        ).read_bytes()
    assert len(calls) == 4 and (tmp_path / "capture/failure.json").read_bytes() == first


def test_toy_changed_saved_input_fails_verification(toy, tmp_path, monkeypatch):
    toy_runtime(toy, tmp_path, monkeypatch, "matching")
    runner.run("preflight", tmp_path / "preflight")
    with pytest.raises(ValueError):
        runner.run("calibrate", tmp_path / "capture")
    p = tmp_path / "capture/cases/zero_s0/raw.npy"
    p.write_bytes(p.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="inventory"):
        runner.run("verify", tmp_path / "verify")


def test_historical_mismatch_stops_before_synthetic_calls_and_is_auditable(
    toy, tmp_path, monkeypatch
):
    _, calls = toy_runtime(toy, tmp_path, monkeypatch, "historical")
    runner.run("preflight", tmp_path / "preflight")
    with pytest.raises(ValueError, match="historical processor parity"):
        runner.run("calibrate", tmp_path / "capture")
    assert len(calls) == 1
    r = runner.run("verify", tmp_path / "verify")
    assert not r["calibration_passed"] and r["actual_counts"]["historical_exact"] == 0
    assert r["actual_counts"]["synthetic_processor_calls"] == 0
    assert r["archives"]["features.npz"]["arrays"] == 0
    assert r["acquisition_progress"]["unexpected_historical_array_saved"]
    assert len(calls) == 1


def test_offline_and_canonical_root_guards_before_acquisition(
    toy, tmp_path, monkeypatch
):
    _, calls = toy_runtime(toy, tmp_path, monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with pytest.raises(ValueError, match="offline"):
        runner.run("preflight", tmp_path / "preflight")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ValueError, match="canonical"):
        runner.run("calibrate", tmp_path / "wrong")
    assert not calls and not (tmp_path / "preflight").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("calibration_only", False),
        ("execution_authorized", False),
        ("model_forwards", 1),
        ("retry_or_resume", True),
    ],
)
def test_invalid_release_is_rejected_before_qualification(
    tmp_path, monkeypatch, key, value
):
    release = json.loads((ROOT / runner.RELEASE).read_bytes())
    release[key] = value
    (tmp_path / "configs").mkdir()
    (tmp_path / runner.RELEASE).write_text(json.dumps(release), encoding="utf-8")
    monkeypatch.setattr(runner, "REPO", tmp_path)
    with pytest.raises(ValueError, match="release contract"):
        runner.qualify()


def test_recipe_hash_cannot_be_relaxed_by_release(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / runner.RELEASE).write_bytes((ROOT / runner.RELEASE).read_bytes())
    (tmp_path / runner.DESIGN).write_bytes((ROOT / runner.DESIGN).read_bytes() + b" ")
    monkeypatch.setattr(runner, "REPO", tmp_path)
    with pytest.raises(ValueError, match="recipe changed"):
        runner.qualify()


def test_disk_thresholds_are_not_relaxed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    c = json.loads((ROOT / runner.DESIGN).read_bytes())["future_calibration"]
    for launch, threshold in ((True, 9 * 2**30), (False, 8 * 2**30)):
        monkeypatch.setattr(
            runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=threshold - 1)
        )
        with pytest.raises(ValueError, match="reserve"):
            runner.check_space(tmp_path, c, launch)
        monkeypatch.setattr(
            runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=threshold)
        )
        runner.check_space(tmp_path, c, launch)
