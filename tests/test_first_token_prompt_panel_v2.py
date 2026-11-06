import copy
import hashlib
import json

import pytest

from fractal_vlm_state_probe.cli import run_first_token_prompt_panel_v2 as release


@pytest.fixture
def configs():
    return (
        release.v1.base._read(release.v1.REPO / release.V1_CFG),
        release.v1.base._read(release.v1.REPO / release.CFG),
    )


def test_new_release_is_only_namespace_and_failure_provenance(configs):
    original, current = configs
    release.same_design(original, current)
    assert current["calibration_contexts"] == 288 and current["panel_contexts"] == 64
    assert current["reserve_bytes"] == 8589934592
    assert current["launch_allowance_bytes"] + current["reserve_bytes"] == 12041764864
    assert current["primary_endpoint"] == original["primary_endpoint"]
    assert current["probability_definition"] == original["probability_definition"]


def test_storage_receipt_is_bounded_and_hash_bound(configs):
    _, current = configs
    raw = (release.v1.REPO / current["storage_recovery_receipt"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == current["storage_recovery_sha256"]
    receipt = json.loads(raw)
    assert receipt["approved_scope"] == ["pip/http-v2", "pip/http"]
    assert receipt["locally_built_wheels_preserved"] == 13
    assert receipt["preserved_wheel_artifact_count"] == 25
    assert all(not v["exists"] for v in receipt["after_http_caches"].values())
    assert "preserved_wheel_files" not in receipt


@pytest.mark.parametrize(
    "key,value",
    [
        ("panel_contexts", 32),
        ("calibration_contexts", 0),
        ("launch_allowance_bytes", 1),
        ("reserve_bytes", 0),
        ("numeric_oracle_rtol", 1e-6),
        ("retry_or_resume", True),
        ("probability_definition", "raw_exp"),
        ("capture_root", "runs/first_token_prompt_panel_v1"),
    ],
)
def test_recovery_does_not_relax_inputs_metrics_budget_or_attempts(configs, key, value):
    original, current = configs
    bad = copy.deepcopy(current)
    bad[key] = value
    with pytest.raises(ValueError):
        release.same_design(original, bad)


def test_v1_failed_path_not_reused_and_all_runtime_bindings_restore(
    monkeypatch, tmp_path
):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    before = (
        release.v1.CFG,
        release.v1.CODE,
        release.v1.require_config,
        release.v1.qualify,
    )
    with pytest.raises(ValueError, match="canonical"):
        release.run("capture", release.v1.REPO / "runs/first_token_prompt_panel_v1")
    assert (
        release.v1.CFG,
        release.v1.CODE,
        release.v1.require_config,
        release.v1.qualify,
    ) == before
    with pytest.raises(FileExistsError):
        release.run("verify", tmp_path)
    assert (
        release.v1.CFG,
        release.v1.CODE,
        release.v1.require_config,
        release.v1.qualify,
    ) == before
