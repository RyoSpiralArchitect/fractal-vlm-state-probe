import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import (
    summarize_controlled_overlap_cache_panel as publisher,
)
from fractal_vlm_state_probe.controlled_overlap_cache_panel import bootstrap_ranges
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def spec():
    config = json.loads(
        (
            Path(__file__).parents[1]
            / "configs/controlled_overlap_cache_phase2_v1.json"
        ).read_text()
    )
    result = config["analysis"]["bootstrap"]
    result["draws"] = 80
    return result


def test_publisher_bootstrap_recomputes_six_paired_views(spec):
    values = np.random.default_rng(74).normal(size=(6, 4, 2, 3, 8, 2))
    expected = publisher.oracle_ranges(values, spec)
    np.testing.assert_allclose(
        bootstrap_ranges(values, spec)["ranges"], expected, atol=1e-12
    )
    values[:, :, :, 2] = values[:, :, :, 0]
    assert all(
        np.array_equal(r[1], [0, 0]) for r in publisher.oracle_ranges(values, spec)
    )


def test_publisher_preserves_missing_primary_and_secondary_separately(spec):
    values = np.random.default_rng(75).normal(size=(6, 4, 2, 3, 8, 2))
    original = publisher.oracle_ranges(values, spec)
    values[0, :, :, 1] = np.nan
    middle = publisher.oracle_ranges(values, spec)
    np.testing.assert_array_equal(original, middle)
    # Even one draw that happens not to select the missing cell cannot qualify it.
    spec["draws"] = 1
    values[1, 2, 1, 2, 3, 0] = np.nan
    missing = publisher.oracle_ranges(values, spec)
    assert missing[1] is None
    assert all(r is not None for i, r in enumerate(missing) if i != 1)


def test_failed_capture_is_retained_without_publication(tmp_path):
    root, output = tmp_path / "run", tmp_path / "public"
    root.mkdir()
    write_json(root / "execution_failure.json", {"status": "FAILED"})
    with pytest.raises(ValueError, match="failed capture"):
        publisher.publish(root, output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        publisher.publish(root, output)


@pytest.fixture
def input_records(tmp_path, monkeypatch):
    root = tmp_path / "run"
    old_path = tmp_path / "old.json"
    receipt = {
        "kind": "panel",
        "map": "m0_q0_d0",
        "pair_id": "family_r3",
        "cell": "mm",
        "historical_run": str(old_path),
    }
    inputs = {
        "pixel_values": {
            "sha256": "old",
            "shape": [1, 3, 256, 256],
            "dtype": "float32",
        },
        "image_sizes": {"values": [[320, 320]]},
        "input_ids": {"values": [1, 2]},
    }
    targets = [
        {
            "layer_index": i,
            "tensor": kind,
            "source_dtype": "mlx.core.bfloat16",
            "dtype": "float32",
        }
        for i, kind in ((1, "keys"), (12, "keys"), (23, "values"))
    ]
    old = {
        "model_id": "fixed",
        "context_policy": {},
        "runtime": {},
        "reproducibility": {"seed": 20260604},
        "stream_events": [
            {
                "prompt": "fixed",
                "cache_token_layout": {"length": 397},
                "assistant_text": "The image",
                "frame_sha256s": ["old_frame"],
                "generation": {"steps": [{"token_id": i} for i in (785, 2168, 2168)]},
                "cache_tensor_artifacts": targets,
            }
        ],
    }
    new = copy.deepcopy(old)
    new["stream_events"][0]["frame_sha256s"] = ["new_frame"]
    effective = copy.deepcopy(inputs)
    effective["pixel_values"]["sha256"] = "new"
    audit = {
        "native": copy.deepcopy(effective),
        "effective": effective,
        "override_image_sizes": None,
        "context": {
            "manifest_path": "manifest",
            "manifest_sha256": "manifest_hash",
            "frozen_specification_sha256": "frozen_hash",
        },
    }
    frozen = {"config": {"phase1_root": "phase1"}}
    measured = {
        "map": "m0_q0_d0",
        "pair_id": "family_r3",
        "cells": {"mm": {"pixel_values_sha256": "new"}},
        "artifacts": {
            "mm": {
                "frame_sha256": "new_frame",
                "manifest_path": "manifest",
                "manifest_sha256": "manifest_hash",
            }
        },
    }
    documents = {
        publisher.REPO / "phase1/input_summary.json": {"records": [measured]},
        old_path: old,
        old_path.with_suffix(".input_audit.json"): {"effective": inputs},
        root / "new.json": new,
        root / "new.input_audit.json": audit,
    }
    monkeypatch.setattr(publisher, "_read", lambda path: documents[path])
    monkeypatch.setattr(publisher, "_sha", lambda path: "frozen_hash")
    monkeypatch.setattr(
        publisher,
        "load_cache_tensor_artifact",
        lambda *args: np.zeros((1, 2, 397, 64), dtype=np.float32),
    )
    return root, {"new.json": receipt}, frozen, new, audit


def test_publisher_accepts_pixel_only_change_but_requires_all_cells(input_records):
    root, registry, frozen, _, _ = input_records
    with pytest.raises(ValueError, match="cell coverage"):
        publisher.verify_panel_inputs(root, registry, frozen)


@pytest.mark.parametrize(
    "field", ["suffix", "token_ids", "layout", "metadata", "override", "pixels"]
)
def test_publisher_rejects_nonregistered_input_or_suffix_change(input_records, field):
    root, registry, frozen, run, audit = input_records
    event = run["stream_events"][0]
    if field == "suffix":
        event["assistant_text"] = "ACK"
    elif field == "token_ids":
        event["generation"]["steps"][0]["token_id"] = 1
    elif field == "layout":
        event["cache_token_layout"]["length"] = 398
    elif field == "override":
        audit["override_image_sizes"] = [[320, 320]]
    else:
        key = "image_sizes" if field == "metadata" else "pixel_values"
        audit["native"][key] = audit["effective"][key] = {"sha256": "different"}
    with pytest.raises(ValueError, match="source/suffix/layout|pixel-only"):
        publisher.verify_panel_inputs(root, registry, frozen)


@pytest.fixture
def publication():
    root = (
        Path(__file__).parents[1]
        / "examples/research_notes/0060_controlled_overlap_cache_panel"
    )
    raw = (root / "summary.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "1bf4f0b5fb2f6bc62b2448fe031300a18969f37a10b8a77ae6bc5d909ec801f0"
    )
    summary = json.loads(raw)
    ledgers = {}
    for name, receipt in summary["ledgers"].items():
        content = (root / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == receipt["sha256"]
        ledgers[name] = [json.loads(line) for line in content.splitlines()]
        assert len(ledgers[name]) == receipt["rows"]
    for name, digest in summary["figures"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    return summary, ledgers


def test_checked_in_panel_denominators_provenance_and_order(publication):
    summary, ledgers = publication
    capture, frozen = summary["capture_summary"], summary["frozen_specification"]
    assert summary["status"] == capture["status"] == "COMPLETE"
    assert summary["new_model_forwards"] == summary["new_p_values"] == 0
    assert capture["panel_source_cells"] == 2048
    assert capture["panel_tensor_sidecars"] == 6144
    assert capture["panel_factorial_blocks"] == 512
    assert capture["calibration_source_cells"] == 128
    assert capture["calibration_tensor_comparisons"] == 384
    assert capture["new_direct_probe_cells"] == 0
    assert capture["same_model_instance_for_calibration_and_panel"]
    assert summary["confirmation_cohort"] == "not_generated_not_sealed"
    assert summary["source_response_counts"] == {"The image": 2048}
    assert (
        frozen["registered_design_commit"]
        == "3be6645a6fb72b12935cc5c3ad30edaf05df3296"
    )
    assert len(frozen["code_sha256"]) == 38
    assert {k: len(v) for k, v in ledgers.items()} == {
        "cells.jsonl": 2048,
        "calibration.jsonl": 384,
        "factorials.jsonl": 1536,
        "vectors.jsonl": 3072,
        "scores.jsonl": 168,
        "directions.jsonl": 1536,
    }
    cells = sorted(ledgers["cells.jsonl"], key=lambda c: c["ordinal"])
    assert [c["ordinal"] for c in cells] == list(range(129, 2177))
    assert all(c["path"].startswith("panel/reference/") for c in cells[:256])
    assert all(c["path"].startswith("panel/test/") for c in cells[256:])
    assert {(c["map"], c["pair_id"], c["cell"]) for c in cells} == {
        (j["map"], j["pair_id"], j["cell"]) for j in frozen["panel_jobs"]
    }
    for c in cells:
        assert c["independent_pixel_only_check"]
        assert c["actual_source_response"] == "The image"
        assert c["generation_step_ids"] == [785, 2168, 2168]
    for c in ledgers["calibration.jsonl"]:
        assert c["bitwise_equal"] and c["max_abs_difference"] == 0
        assert c["before_array_sha256"] == c["after_array_sha256"]
    assert summary["independent_verification"]["bootstrap_verified"]
    assert (
        summary["independent_verification"]["maximum_interaction_oracle_abs_error"] == 0
    )
    assert summary["independent_verification"]["maximum_score_oracle_abs_error"] < 1e-12
    assert summary["independent_verification"]["verification_model_forwards"] == 0
    diagnostics = summary["localization_diagnostics"]
    assert diagnostics["pre_image_all_effects_zero_count"] == 1536
    assert diagnostics["interaction_argmax_image_count"] == 1536
    assert diagnostics["image_energy_fraction_above_0_9_count"] == 1536
    assert diagnostics["balanced_dominance_by_region"] == {
        "image_tokens": {"spatial_contrast": 894, "palette_contrast": 642},
        "all_effective": {"spatial_contrast": 893, "palette_contrast": 643},
    }


def test_checked_in_scores_endpoints_and_negative_cases(publication):
    summary, ledgers = publication
    metadata = summary["frozen_specification"]["hierarchy_metadata"]
    for row in ledgers["scores.jsonl"]:
        assert row["available"]
        publisher.verify_summary(
            row["summary"],
            np.asarray(row["scores"]),
            row["labels"],
            row["families"],
            metadata,
        )
    endpoints = summary["primary_endpoints"]
    assert len(endpoints) == 12 and all(e["available"] for e in endpoints)
    assert all(e["p_value"] is None and not e["equivalence_tested"] for e in endpoints)
    aggregates = {
        (a["target"], a["region"], a["q_sixteenths"]): a for a in summary["aggregates"]
    }
    for e in endpoints:
        zero = aggregates[(e["target"], e["region"], 0)]
        high = aggregates[(e["target"], e["region"], 4)]
        expected = (
            zero["mean_family_margin"]
            if e["endpoint"] == "zero_overlap_margin"
            else high["mean_family_margin"] - zero["mean_family_margin"]
        )
        assert e["mean"] == pytest.approx(expected, abs=1e-12)
        assert len(e["family_values"]) == 8 and len(e["mask_values"]) == 4
    zero = [e for e in endpoints if e["endpoint"] == "zero_overlap_margin"]
    difference = [e for e in endpoints if e["endpoint"] == "high_minus_zero_margin"]
    assert sum(e["exploratory_range"][0] > 0 for e in zero) == 5
    assert all(
        e["exploratory_range"][0] < 0 < e["exploratory_range"][1] for e in difference
    )
    assert sum(v < 0 for e in zero for v in e["family_values"].values()) == 1
    assert sum(v < 0 for e in difference for v in e["family_values"].values()) == 28
    assert sum(a["negative_family_margin_count"] for a in aggregates.values()) == 12
