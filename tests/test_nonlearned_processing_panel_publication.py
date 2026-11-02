import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.summarize_controlled_overlap import (
    oracle_summary,
    verify_summary,
)
from fractal_vlm_state_probe.cli.summarize_nonlearned_processing_panel import (
    oracle_ranges,
)
from fractal_vlm_state_probe.nonlearned_processing_panel import VIEWS


@pytest.fixture(scope="module")
def publication():
    root = (
        Path(__file__).parents[1]
        / "examples/research_notes/0065_nonlearned_processing_panel"
    )
    content = (root / "summary.json").read_bytes()
    assert (
        hashlib.sha256(content).hexdigest()
        == "01420d73cb0f67920b794a0a28c0a167c379fe6c63954e8c57a65ad3070f3b85"
    )
    summary = json.loads(content)
    assert summary["status"] == "COMPLETE"
    assert summary["analysis_kind"] == "nonlearned_processing_panel_publication"
    ledgers = {}
    for name, receipt in summary["ledgers"].items():
        content = (root / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == receipt["sha256"]
        ledgers[name] = [json.loads(line) for line in content.splitlines()]
        assert len(ledgers[name]) == receipt["rows"]
    for name, digest in summary["figures"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    archive = root / "interaction_vectors.npz"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == summary["interaction_archive_sha256"]
    )
    with np.load(archive, allow_pickle=False) as arrays:
        values = {key: arrays[key] for key in arrays.files}
    return summary, ledgers, values


def test_publication_units_and_regenerated_processor_boundary(publication):
    summary, ledgers, arrays = publication
    assert {k: len(v) for k, v in ledgers.items()} == {
        "vectors.jsonl": 6144,
        "scores.jsonl": 336,
        "directions.jsonl": 3072,
        "cells.jsonl": 2048,
        "calibration.jsonl": 128,
        "verification_calibration.jsonl": 128,
        "feature_checks.jsonl": 2048,
    }
    frozen = summary["frozen_specification"]
    assert (
        frozen["registered_design_commit"] == "99fb45ca167e93f820f50fa39e816d64f689b4af"
    )
    assert len(frozen["code_sha256"]) == 51
    assert (
        frozen["release"]["merged_commit"] == "10ae3de86f22b4f8574feb9c8e372cbd417f6a5d"
    )
    assert summary["capture_summary"]["source_cells"] == 2048
    assert summary["capture_summary"]["feature_arrays"] == 24576
    assert summary["capture_summary"][
        "same_processor_instance_for_calibration_and_panel"
    ]
    assert (
        summary["new_model_forwards"]
        == summary["new_processor_calls"]
        == summary["new_p_values"]
        == 0
    )
    assert summary["confirmation_cohort"] == "not_generated_not_sealed"
    check = summary["independent_verification"]
    assert check["historical_processor_regenerations"] == 128
    assert check["experimental_processor_regenerations"] == 2048
    assert check["verification_processor_calls"] == 2176
    assert check["real_image_feature_oracle_pairs"] == 24576
    assert check["maximum_feature_oracle_abs_error"] <= 1e-10
    assert check["maximum_score_oracle_abs_error"] <= 1e-12
    assert not check["full_processor_pixels_stored"]
    cells = ledgers["cells.jsonl"]
    assert [c["ordinal"] for c in cells] == list(range(129, 2177))
    assert all(c["phase"] == "reference" for c in cells[:256])
    assert all(c["phase"] == "test" for c in cells[256:])
    for captured, regenerated in zip(cells, ledgers["feature_checks.jsonl"]):
        assert (captured["ordinal"], captured["pixel_values_sha256"]) == (
            regenerated["ordinal"],
            regenerated["pixel_values_sha256"],
        )
        assert (
            regenerated["passed"]
            and len(regenerated["feature_oracle_abs_errors"]) == 12
        )
        assert max(regenerated["feature_oracle_abs_errors"].values()) <= 1e-10
    assert len(arrays) == 6144
    assert set(arrays) == {r["public_array_key"] for r in ledgers["vectors.jsonl"]}
    raw = [
        r
        for r in ledgers["vectors.jsonl"]
        if (r["start"], r["feature"]) == ("raw_rgb", "global_moments")
    ]
    assert len(raw) == 512
    for row in raw:
        assert not row["available"] and row["norm"] == 0
        np.testing.assert_array_equal(arrays[row["public_array_key"]], 0)


def test_portable_archive_replays_all_scores_margins_and_bootstrap(publication):
    summary, ledgers, arrays = publication
    metadata = summary["frozen_specification"]["hierarchy_metadata"]
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    vectors = {
        (r["start"], r["feature"], r["map"], r["pair_id"]): arrays[
            r["public_array_key"]
        ]
        for r in ledgers["vectors.jsonl"]
    }
    cube = np.full((12, 4, 2, 3, 16, 8), np.nan)

    def unit(a):
        n = np.sqrt(sum(float(x * x) for x in a))
        return a / n if n else None

    for row in ledgers["scores.jsonl"]:
        s, f, reference, key = (
            row[k] for k in ("start", "feature", "reference_map", "test_map")
        )
        refs = []
        for family in families:
            members = sorted(
                k
                for k, m in metadata.items()
                if m["pairing_family"] == family and m["replicate"] in (1, 2)
            )
            a, b = [unit(vectors[(s, f, reference, label)]) for label in members]
            if a is None or b is None:
                refs = None
                break
            combined = a + b
            if np.linalg.norm(combined) <= 1e-12:
                refs = None
                break
            refs.append(unit(combined))
        tests = [unit(vectors[(s, f, key, label)]) for label in labels]
        available = refs is not None and all(a is not None for a in tests)
        assert row["available"] == available
        if not available:
            assert "scores" not in row
            continue
        scored = np.array([[float(np.dot(a, b)) for b in refs] for a in tests])
        np.testing.assert_allclose(row["scores"], scored, rtol=0, atol=1e-12)
        verify_summary(row["summary"], scored, labels, families, metadata)
        if row["q"] is not None:
            cube[
                VIEWS.index((s, f)),
                row["mask"],
                row["destination"],
                (0, 1, 4).index(row["q"]),
            ] = scored
    margins = np.full((12, 4, 2, 3, 8, 2), np.nan)
    for index in np.ndindex(12, 4, 2, 3):
        if np.isfinite(cube[index]).all():
            values = oracle_summary(cube[index], labels, families, metadata)[0]
            for fi, family in enumerate(families):
                margins[index + (fi,)] = [
                    v
                    for k, v in zip(labels, values)
                    if metadata[k]["pairing_family"] == family
                ]
    spec = summary["frozen_specification"]["config"]["planned_panel"]["bootstrap"]
    ranges = oracle_ranges(margins, spec)
    endpoints = summary["primary_endpoints"]
    assert len(endpoints) == 24 and len(summary["aggregates"]) == 36
    for row in endpoints:
        vi = VIEWS.index((row["start"], row["feature"]))
        ei = ("zero_overlap_margin", "high_minus_zero_margin").index(row["endpoint"])
        assert row["available"] == (ranges[vi] is not None)
        assert row["p_value"] is None and not row["equivalence_tested"]
        if row["available"]:
            values = (
                margins[vi, :, :, 0]
                if ei == 0
                else margins[vi, :, :, 2] - margins[vi, :, :, 0]
            )
            np.testing.assert_allclose(row["mean"], values.mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                row["exploratory_range"], ranges[vi][ei], rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                row["mask_values"], values.mean(axis=(1, 2, 3)), rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                [row["family_values"][family] for family in families],
                values.mean(axis=(0, 1, 3)),
                rtol=0,
                atol=1e-12,
            )
        else:
            assert "mean" not in row
