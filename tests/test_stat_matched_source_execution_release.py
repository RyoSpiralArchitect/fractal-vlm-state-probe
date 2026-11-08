"""Metadata and authority checks only; no real qualification or acquisition."""

import hashlib
import json
from pathlib import Path

import pytest

from fractal_vlm_state_probe.cli import run_stat_matched_source as cli

ROOT = Path(__file__).resolve().parents[1]
CURRENT = "configs/stat_matched_source_acquisition_release_v1.json"
HISTORY = "configs/stat_matched_source_acquisition_registration_v1.json"


def read(name):
    return json.loads((ROOT / name).read_bytes())


def test_original_registration_remains_byte_exact_and_start_withheld():
    old = read(HISTORY)
    assert (
        hashlib.sha256((ROOT / HISTORY).read_bytes()).hexdigest()
        == "d32f360b1bd4dfecd35cd29565cd931956f51976791b4d4e782fd51a9994692c"
    )
    assert old["approved_for_execution"] is False
    current = read(CURRENT)
    assert current["prior_registration"] == HISTORY
    assert (
        current["prior_registration_sha256"]
        == hashlib.sha256((ROOT / HISTORY).read_bytes()).hexdigest()
    )


def test_execution_authorization_changes_only_authority_not_frozen_science():
    old, current = read(HISTORY), read(CURRENT)
    changed = {k for k in set(old) | set(current) if old.get(k) != current.get(k)}
    assert changed == {
        "status",
        "approved_for_execution",
        "operator_start_authorization",
        "scope",
        "prior_registration",
        "prior_registration_sha256",
        "preparation_evidence",
        "allowed_now",
        "forbidden_now",
    }
    assert current["approved_for_execution"] is True
    assert current["status"] == "approved_for_one_bounded_acquisition"
    assert (
        current["operator_start_authorization"]
        == "explicit_operator_go_2026_11_08_after_PREPARED_NOT_RELEASED"
    )
    assert current["code_sha256"] == old["code_sha256"]
    assert len(current["code_sha256"]) == 135
    for name, digest in current["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert current["planned_budget"] == old["planned_budget"]
    assert not current["retry_or_resume"]
    assert current["confirmation_cohort"] == "not_generated_not_sealed"


def test_old_preparation_does_not_become_approved_preflight():
    r = read(CURRENT)
    assert r["preparation_root"] != r["future_approved_preflight_root"]
    assert r["preparation_evidence"] == {
        "status": "PREPARED_NOT_RELEASED",
        "preflight_sha256": "f7052314ebebc89cd4295562a689d43bbcc4b3821ac55993c18dd5833c0fcb10",
        "frozen_sha256": "48adda22d36196d7d04367fe092d3952447ac7f33d2a44ea41a2152c6498ead7",
    }
    assert "retry_or_resume" in r["forbidden_now"]
    assert "confirmation_cohort_acquisition" in r["forbidden_now"]
    assert "primary_analysis_before_two_identical_verifications" in r["forbidden_now"]


def test_committed_execution_release_still_requires_its_publication_receipt(
    monkeypatch,
):
    invoked = []
    monkeypatch.setattr(cli, "qualification", lambda *args: invoked.append(True))
    monkeypatch.setattr(
        cli.subprocess,
        "check_output",
        lambda args, **kwargs: (
            b"software_commit\n" if args[1] == "log" else (ROOT / CURRENT).read_bytes()
        ),
    )
    with pytest.raises(ValueError, match="public merged release receipt required"):
        cli.require_release(ROOT, ROOT / CURRENT)
    assert not invoked
