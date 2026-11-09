"""Analytic geometry, native decoding and fail-closed execution fixtures only."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe import reference_crossing as core
from fractal_vlm_state_probe import reference_crossing_oracle as oracle
from fractal_vlm_state_probe.cli import run_reference_crossing as cli
from fractal_vlm_state_probe.stat_matched_source_storage import (
    load_array,
    read,
    save_array,
    write,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(mode="identity"):
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    pairs = {p["pair_id"]: p for p in spec["sources"]["pairs"]}
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs.values()))

    def load(pair, shift, view):
        p = pairs[pair]
        i = families.index(p["pairing_family"])
        v = np.eye(8)[i]
        if mode == "rotation" and shift:
            v = np.eye(8)[i // 4 * 4 + (i + shift) % 4]
        if mode == "collapse" and shift:
            v = np.ones(8)
        if mode == "missing" and p["split"] == "test" and i == 0 and shift == 1:
            v *= 0
        if mode == "cancel" and p["replicate"] == 6 and i == 0:
            v *= -1
        return {"mm": np.zeros(8), "mj": np.zeros(8), "jm": np.zeros(8), "jj": v}

    return spec, load


@pytest.mark.parametrize(
    "mode", ["identity", "rotation", "collapse", "missing", "cancel"]
)
def test_full_profiles_match_independent_oracle(mode):
    spec, cells = fixture(mode)
    actual = list(core.profiles(cells, spec, "encoder"))
    expected = list(oracle.profiles(cells, spec, "encoder"))
    assert len(actual) == len(expected) == 3
    for a, b in zip(actual, expected, strict=True):
        assert core.compare(a, b) < 1e-12
        assert np.array(a["scores"], dtype=object).shape == (4, 4, 16, 8)
        assert len(a["interactions"]) == 128
        if mode == "identity":
            assert a["overall"] == {
                "fixed_change": 0,
                "matched_change": 0,
                "reference_difference": 0,
            }
        elif mode == "rotation":
            assert a["overall"]["fixed_change"] == pytest.approx(-4 / 3)
            assert a["overall"]["matched_change"] == pytest.approx(0)
            assert a["overall"]["reference_difference"] == pytest.approx(4 / 3)
        elif mode == "collapse":
            assert a["overall"]["matched_change"] == pytest.approx(-1)
        else:
            assert a["overall"]["fixed_change"] is None
            assert len(a["family_results"]) == 8
            if mode == "missing":
                assert a["available_test_pairs"]["fixed_change"] == 14
            else:
                assert a["available_test_pairs"]["fixed_change"] == 8


@pytest.mark.parametrize(
    "bad", ["shape", "nan", "infinity", "empty", "cell", "underflow"]
)
def test_invalid_quartets_fail_in_both_kernels(bad):
    c = {k: np.ones(8) for k in ("mm", "mj", "jm", "jj")}
    if bad == "shape":
        c["jj"] = np.ones(7)
    elif bad == "nan":
        c["jj"][0] = np.nan
    elif bad == "infinity":
        c["jj"][0] = np.inf
    elif bad == "empty":
        c = {k: np.array([]) for k in c}
    elif bad == "cell":
        c.pop("mm")
    else:
        c = {k: np.zeros(8) for k in c}
        c["jj"][0] = 1e-300
    for implementation in (core, oracle):
        with pytest.raises(ValueError):
            implementation.direction(c)


def test_guard_boundary_and_zero_remain_unavailable_without_tuning():
    spec, load = fixture()

    def cells(pair, shift, view):
        c = load(pair, shift, view)
        c["mm"][:] = 1e6
        c["mj"][:] = 1e6
        c["jm"][:] = 1e6
        c["jj"] += 1e6
        return c

    profiles = list(core.profiles(cells, spec, "full_raw_rgb"))
    assert profiles[0]["available_test_pairs"]["fixed_change"] == 16
    assert profiles[1]["available_test_pairs"]["fixed_change"] == 0
    for a, b in zip(
        profiles, oracle.profiles(cells, spec, "full_raw_rgb"), strict=True
    ):
        core.compare(a, b)


def test_reference_guard_requires_both_replicates_and_cancellation():
    spec, load = fixture()

    def cells(pair, shift, view):
        c = load(pair, shift, view)
        if pair.endswith("r6"):
            c["jj"] *= 0
        return c

    for p in core.profiles(cells, spec, "encoder"):
        assert all(not r["available"] for r in p["references"])
        assert all(
            r["reason"] == "unavailable_reference_replicate" for r in p["references"]
        )
        assert p["overall"]["matched_change"] is None


def test_parent_row_parity_detects_score_and_missingness_corruption():
    spec, load = fixture("rotation")
    p = next(core.profiles(load, spec, "encoder"))
    old = {k: p[k] for k in ("families", "test_pair_ids")}
    old.update(
        scores=p["scores"][0],
        margins=p["margins"][0],
        paired_margin_change=p["overall"]["fixed_change"],
        available_test_pairs=16,
    )
    core.primary_parity(p, old)
    bad = copy.deepcopy(old)
    bad["scores"][0][0][0] = None
    with pytest.raises(ValueError):
        core.primary_parity(p, bad)
    bad = copy.deepcopy(old)
    bad["paired_margin_change"] += 1e-8
    with pytest.raises(ValueError):
        core.primary_parity(p, bad)


def test_native_file_word_hash_and_bfloat16_conversion(tmp_path):
    words = np.array([0x3F80, 0xBF80, 0, 0x4000], dtype=np.uint16)
    receipt = save_array(tmp_path, "values.npy", words)
    expanded = np.array([1, -1, 0, 2], dtype=np.float32)
    receipt["expanded_word_sha256"] = hashlib.sha256(expanded.tobytes()).hexdigest()
    np.testing.assert_array_equal(oracle.load(tmp_path, receipt, "encoder"), expanded)
    np.testing.assert_array_equal(load_array(tmp_path, receipt), words)
    bad = {**receipt, "word_sha256": "0" * 64}
    with pytest.raises(ValueError):
        oracle.load(tmp_path, bad, "encoder")
    bad = {**receipt, "expanded_word_sha256": "0" * 64}
    with pytest.raises(ValueError):
        oracle.load(tmp_path, bad, "encoder")


def test_strict_comparison_checks_schema_missingness_and_nonfinite():
    for actual, expected in [
        (0, None),
        (True, 1),
        ({"a": 1}, {}),
        ([1], []),
        (float("nan"), 0.0),
        (1e-6, 0.0),
    ]:
        with pytest.raises(ValueError):
            core.compare(actual, expected)


def test_execution_retains_first_failure_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO", tmp_path)
    plan_path = (
        tmp_path
        / "examples/research_notes/0096_reference_crossing_registration/plan.json"
    )
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(
        (
            ROOT
            / "examples/research_notes/0096_reference_crossing_registration/plan.json"
        ).read_bytes()
    )

    def failure(repo):
        raise ValueError("retained synthetic preflight failure")

    monkeypatch.setattr(cli, "setup", failure)
    out = tmp_path / "runs/reference_crossing_v1"
    with pytest.raises(ValueError, match="retained synthetic"):
        cli.run("run", out)
    assert read(out / "failure.json")["retry_authorized"] is False
    with pytest.raises(ValueError, match="retained failure"):
        cli.run("run", out)
    with pytest.raises(ValueError, match="canonical"):
        cli.run("run", tmp_path / "wrong")


def software_execution(tmp_path, monkeypatch):
    spec, cells = fixture("rotation")
    plan_path = (
        tmp_path
        / "examples/research_notes/0096_reference_crossing_registration/plan.json"
    )
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(
        (
            ROOT
            / "examples/research_notes/0096_reference_crossing_registration/plan.json"
        ).read_bytes()
    )
    originals = {}
    for view in ("encoder", "projector"):
        for p in core.profiles(cells, spec, view):
            originals[view, p["relative_floor"]] = {
                "families": p["families"],
                "test_pair_ids": p["test_pair_ids"],
                "scores": p["scores"][0],
                "margins": p["margins"][0],
                "paired_margin_change": p["overall"]["fixed_change"],
                "available_test_pairs": 16,
            }
    monkeypatch.setattr(cli, "REPO", tmp_path)
    plan = read(plan_path)
    monkeypatch.setattr(
        cli,
        "setup",
        lambda repo: (plan, [], {"software_fixture": True}, spec, originals),
    )
    monkeypatch.setattr(cli, "loader", lambda *a, **k: cells)
    monkeypatch.setattr(
        cli.parent_core,
        "analyze_view",
        lambda load, spec, view, relative_floor: originals[view, relative_floor],
    )
    return cli.run("run", tmp_path / plan["future_output_root"]), plan


def test_software_run_two_fresh_verifications_byte_identical(tmp_path, monkeypatch):
    result, plan = software_execution(tmp_path, monkeypatch)
    assert result["status"] == "SCORED_UNVERIFIED"
    roots = [tmp_path / p for p in plan["future_verification_roots"]]
    for root in roots:
        checked = cli.run("verify", root)
        assert checked["status"] == "VERIFIED_NATIVE_CROSSINGS"
        assert checked["model_contexts"] == 0
    assert cli.tree(roots[0]) == cli.tree(roots[1])


def test_saved_score_tampering_detected_even_with_rehashed_output_tree(
    tmp_path, monkeypatch
):
    import json

    _, plan = software_execution(tmp_path, monkeypatch)
    source = tmp_path / plan["future_output_root"]
    path = source / "profiles.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["scores"][0][0][0][0] += 1e-4
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = read(source / "summary.json")
    hashes = cli.tree(source)
    hashes.pop("summary.json")
    summary["output_sha256"] = hashes
    (source / "summary.json").unlink()
    write(source / "summary.json", summary)
    root = tmp_path / plan["future_verification_roots"][0]
    with pytest.raises(ValueError, match="numeric"):
        cli.run("verify", root)
    assert (root / "failure.json").is_file()


def test_failed_first_verification_blocks_replay(tmp_path, monkeypatch):
    _, plan = software_execution(tmp_path, monkeypatch)
    first, second = [tmp_path / p for p in plan["future_verification_roots"]]
    first.mkdir()
    write(first / "failure.json", {"error": "retained synthetic failure"})
    with pytest.raises(ValueError, match="retained failure"):
        cli.run("verify", second)
    assert not second.exists()


def test_replay_requires_successful_first_verification(tmp_path, monkeypatch):
    _, plan = software_execution(tmp_path, monkeypatch)
    second = tmp_path / plan["future_verification_roots"][1]
    with pytest.raises(ValueError, match="successful first"):
        cli.run("verify", second)
    assert not second.exists()


def test_subtolerance_replay_difference_is_retained_and_not_qualified(
    tmp_path, monkeypatch
):
    _, plan = software_execution(tmp_path, monkeypatch)
    first, second = [tmp_path / p for p in plan["future_verification_roots"]]
    cli.run("verify", first)
    original = oracle.profiles

    def drift(*args):
        for p in original(*args):
            p["scores"][1][1][0][0] += 5e-13
            yield p

    monkeypatch.setattr(oracle, "profiles", drift)
    with pytest.raises(ValueError, match="verification trees differ"):
        cli.run("verify", second)
    assert (second / "failure.json").is_file()
    assert not (second / "summary.json").exists()


@pytest.mark.parametrize("root_kind", ["source", "first_verification"])
def test_symlinked_source_or_verification_root_is_rejected(
    tmp_path, monkeypatch, root_kind
):
    _, plan = software_execution(tmp_path, monkeypatch)
    source = tmp_path / plan["future_output_root"]
    first, second = [tmp_path / p for p in plan["future_verification_roots"]]
    if root_kind == "first_verification":
        cli.run("verify", first)
        root, output = first, second
    else:
        root, output = source, first
    target = tmp_path / "moved_original_root"
    root.rename(target)
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        cli.run("verify", output)
    with pytest.raises(ValueError, match="symlinks"):
        cli.tree(root)
    assert not output.exists()
