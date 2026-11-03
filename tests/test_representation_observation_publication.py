import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def publication():
    repo = Path(__file__).parents[1]
    root = repo / "examples/research_notes/0068_representation_observation_calibration"
    content = (root / "summary.json").read_bytes()
    assert (
        hashlib.sha256(content).hexdigest()
        == "8baf41b326186f90cba3b7666b31b48dc9ffb1fd7f5954042bd1d410acc089d5"
    )
    summary = json.loads(content)
    ledgers = {}
    assert {p.name for p in root.iterdir()} == {"summary.json", *summary["ledgers"]}
    for name, receipt in summary["ledgers"].items():
        content = (root / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == receipt["sha256"]
        ledgers[name] = [json.loads(line) for line in content.splitlines()]
        assert len(ledgers[name]) == receipt["rows"]
    return repo, summary, ledgers


def test_observation_publication_registration_scope_and_frozen_contract(publication):
    repo, summary, ledgers = publication
    assert summary["schema_version"] == 1 and summary["status"] == "PASS"
    assert (
        summary["analysis_kind"] == "representation_observation_calibration_publication"
    )
    frozen = summary["frozen_specification"]
    assert (
        summary["registered_design_commit"]
        == frozen["registered_design_commit"]
        == "7ad7d8219e4ff4ad3c760c3e508e746ea2c16ffd"
    )
    assert (
        frozen["observation_config"]["preceding_merge_commit"]
        == "dff3101fb2c0abf54048547a50d9e4d7942101e0"
    )
    assert len(frozen["code_sha256"]) == 58
    for name, digest in frozen["code_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
    assert (
        summary["new_model_forwards"] == summary["calibration_source_contexts"] == 384
    )
    assert (
        summary["model_forward_unit"]
        == "fresh_source_context_not_internal_generation_steps"
    )
    assert (
        summary["new_experimental_panel_cells"]
        == summary["new_direct_probe_cells"]
        == summary["new_correspondence_scores"]
        == 0
    )
    assert not summary["experimental_panel_released"]
    assert summary["confirmation_cohort"] == "not_generated_not_sealed"
    assert summary["source_response_counts"] == {"The image": 384}
    check = summary["independent_verification"]
    assert (
        check["historical_tensor_pairs"] == 1152 and check["on_off_tensor_pairs"] == 768
    )
    assert check["historical_scalar_pairs"] == 58540032
    assert check["on_off_scalar_pairs"] == 39026688
    assert check["representation_arrays"] == 256
    assert check["representation_scalars"] == 130023424
    assert check["maximum_cache_absolute_difference"] == 0
    assert (
        check["verification_model_forwards"]
        == check["verification_processor_calls"]
        == 0
    )
    assert not summary["representation_sidecars_published"]
    assert {k: len(v) for k, v in ledgers.items()} == {
        "cells.jsonl": 384,
        "historical_comparisons.jsonl": 1152,
        "on_off_comparisons.jsonl": 768,
        "representations.jsonl": 128,
    }


def test_portable_bracket_receipts_retain_all_sources_and_sites(publication):
    _, summary, ledgers = publication
    frozen = summary["frozen_specification"]
    jobs, rows = frozen["jobs"], ledgers["cells.jsonl"]
    stages, targets = (
        ["off_before", "on", "off_after"],
        ["1:keys", "12:keys", "23:values"],
    )
    assert len(jobs) == 128
    expected = [(j["pair_id"], j["cell"], s) for j in jobs for s in stages]
    assert [(r["pair_id"], r["cell"], r["stage"]) for r in rows] == expected
    assert [r["ordinal"] for r in rows] == list(range(1, 385))
    assert Counter(r["stage"] for r in rows) == dict.fromkeys(stages, 128)
    cells = {(r["pair_id"], r["cell"], r["stage"]): r for r in rows}
    historical = ledgers["historical_comparisons.jsonl"]
    assert {(r["pair_id"], r["cell"], r["stage"], r["target"]) for r in historical} == {
        (*key, t) for key in expected for t in targets
    }
    for row in historical:
        key = (row["pair_id"], row["cell"], row["stage"])
        assert row["independent_tensor_bytes_exact"]
        assert (
            row["before_array_sha256"]
            == row["after_array_sha256"]
            == cells[key]["cache_array_sha256"][row["target"]]
        )
    direct = ledgers["on_off_comparisons.jsonl"]
    assert {(r["pair_id"], r["cell"], r["off_stage"], r["target"]) for r in direct} == {
        (j["pair_id"], j["cell"], s, t)
        for j in jobs
        for s in ("off_before", "off_after")
        for t in targets
    }
    for row in direct:
        key = row["pair_id"], row["cell"]
        assert row["independent_tensor_bytes_exact"]
        assert (
            cells[(*key, row["off_stage"])]["cache_array_sha256"]
            == cells[(*key, "on")]["cache_array_sha256"]
        )
    reps = {(r["pair_id"], r["cell"]): r for r in ledgers["representations.jsonl"]}
    assert set(reps) == {(j["pair_id"], j["cell"]) for j in jobs}
    for row in rows:
        assert row["historical_cache_exact"] and row["observer_restored"]
        assert row["independent_input_suffix_layout_and_cache_exact"]
        assert row["observer_calls"] == int(row["stage"] == "on")
        if row["stage"] != "on":
            assert row["representation"] is None
            continue
        rep = reps[(row["pair_id"], row["cell"])]
        assert rep["path"] == row["representation"]["path"]
        assert rep["sha256"] == row["representation"]["sha256"]
        assert rep["arrays"] == rep["independent_arrays_verified"]
        assert set(rep["arrays"]) == {"encoder", "projector"}
        for site, channels in (("encoder", 3072), ("projector", 896)):
            a = rep["arrays"][site]
            assert a["shape"] == [1, 256, channels]
            assert a["scalar_count"] == 256 * channels and a["nonzero"]
            assert a["dtype"] == "float32" and a["source_dtype"] == "mlx.core.bfloat16"
            assert len(a["array_sha256"]) == 64
