"""File-only registration contracts; never qualify the native runtime."""

import hashlib
import json
from pathlib import Path

import pytest

from fractal_vlm_state_probe.cli import run_stat_matched_source as cli

ROOT = Path(__file__).resolve().parents[1]
NAME = "configs/stat_matched_source_acquisition_registration_v1.json"


def read(name):
    return json.loads((ROOT / name).read_bytes())


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_registration_does_not_authorize_acquisition_or_confirmation():
    r = read(NAME)
    p = read(cli.PROFILE)
    assert r["study_id"] == p["required_separate_release_id"]
    assert r["registration_approved"] and r["zero_call_preparation_approved"]
    assert r["approved_for_execution"] is False
    assert r["status"] == "registered_plan_start_withheld"
    assert (
        r["operator_start_authorization"]
        == "withheld_pending_zero_call_preparation_review"
    )
    assert r["allowed_now"] == [
        "registration",
        "read_only_review",
        "PR_and_main_merge",
        "zero_call_preparation",
    ]
    assert set(r["forbidden_now"]) == {
        "registered_source_generation",
        "native_processor_calls",
        "model_weights_loading",
        "model_contexts",
        "capture",
        "primary_analysis",
    }
    assert r["observations_at_registration"] == {
        "registered_sources_generated": 0,
        "native_processor_calls": 0,
        "model_weights_loaded": False,
        "model_contexts": 0,
    }
    assert r["confirmation_cohort"] == "not_generated_not_sealed"
    assert not r["retry_or_resume"]


def test_complete_135_file_freeze_is_exact_and_preserves_previous_profiles():
    r, p = read(NAME), read(cli.PROFILE)
    design = read("configs/stat_matched_source_panel_v1.json")
    core = read("configs/stat_matched_source_core_v1.json")
    previous = read(design["preserve_inherited_frozen_hashes_from"])
    required = (
        set(previous["code_sha256"])
        | set(core["registration_sha256"])
        | set(p["preserved_core_sha256"])
        | set(p["implementation_files"])
    )
    assert set(r["code_sha256"]) == required and len(required) == 135
    assert r["preparation_sha256"] == digest(cli.PROFILE)
    for name, sha in r["code_sha256"].items():
        assert digest(name) == sha, name
    for group in (
        previous["code_sha256"],
        core["registration_sha256"],
        p["preserved_core_sha256"],
    ):
        assert all(r["code_sha256"][name] == sha for name, sha in group.items())
    assert not p["execution_authorized"]


def test_code_publication_receipt_is_bound_to_the_merged_implementation():
    r = read(NAME)
    receipt = read(r["public_receipt"])
    assert digest(r["public_receipt"]) == r["public_receipt_sha256"]
    assert receipt == {
        "schema_version": 1,
        "analysis_kind": "code_freeze_publication_receipt",
        "baseRefName": "main",
        "headRefOid": "c21cfcdc049e7651bb69bf6932c13d7b079cc9aa",
        "mergeCommit": {"oid": "45ad7db162194ebe7cb596526ec1953605af6700"},
        "mergedAt": "2026-11-08T09:44:34Z",
        "number": 11,
        "state": "MERGED",
        "url": "https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/11",
    }
    assert r["implementation_commit"] == receipt["mergeCommit"]["oid"]


def test_counts_roots_and_budget_are_planned_not_executed():
    r, p = read(NAME), read(cli.PROFILE)
    s = read("configs/stat_matched_source_panel_v1.json")
    assert r["planned_budget"] == {
        "source_frames": 64,
        "processor_only_calls": 665,
        "model_source_contexts": 920,
        "reference_seal_after_context": 664,
        "first_test_context": 665,
    }
    assert (
        r["required_launch_free_bytes"]
        == s["budget"]["required_launch_free_bytes"]
        == 28 * 2**30
    )
    assert r["preparation_root"] == p["preparation_root"]
    assert r["future_approved_preflight_root"] == p["preflight_root"]
    assert r["capture_root"] == p["capture_root"]
    assert r["zero_call_preparation_status_expected"] == "PREPARED_NOT_RELEASED"
    roots = [
        r["preparation_root"],
        r["future_approved_preflight_root"],
        r["capture_root"],
        *r["verification_roots"],
        r["analysis_root"],
    ]
    assert len(set(roots)) == 6
    assert "separate_explicit_operator_start_approval" in r["future_start_gates"]


def test_existing_cli_refuses_current_registration_before_any_native_qualification(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(cli, "qualification", lambda *args: calls.append(True))
    output = tmp_path / "never_created"
    with pytest.raises(ValueError, match="approved release binding"):
        cli.run("capture", output, release=ROOT / NAME)
    assert not calls and not output.exists()


def test_zero_call_preparation_remains_distinct_from_an_executable_preflight(
    tmp_path, monkeypatch
):
    import shutil
    from types import SimpleNamespace

    r = read(NAME)
    (tmp_path / "configs").mkdir()
    (tmp_path / "runs").mkdir()
    for name in (cli.PROFILE, "configs/stat_matched_source_panel_v1.json"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda *args: SimpleNamespace(free=40 * 2**30)
    )
    called = []

    def qualification(*args):
        called.append(True)
        return {"software_fixture_only": True}

    monkeypatch.setattr(cli, "qualification", qualification)
    result = cli.run("preflight", tmp_path / r["preparation_root"])
    assert called == [True] and result["status"] == "PREPARED_NOT_RELEASED"
    assert result["new_processor_calls"] == result["new_model_contexts"] == 0
    assert not result["model_weights_loaded"]
    frozen = json.loads((tmp_path / r["preparation_root"] / "frozen.json").read_bytes())
    assert (
        frozen["execution_release_sha256"] is None
        and frozen["implementation_sha256"] == {}
    )
    assert not (tmp_path / r["future_approved_preflight_root"]).exists()
    assert not (tmp_path / r["capture_root"]).exists()
