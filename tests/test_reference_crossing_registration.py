"""Metadata and analytic-formula checks only; no real arrays or scores."""

import copy
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe import reference_crossing_registration as core
from fractal_vlm_state_probe.cli import prepare_reference_crossing as cli
from fractal_vlm_state_probe.stat_matched_source_storage import read

ROOT = Path(__file__).resolve().parents[1]


def test_registered_scope_and_all_profiles():
    p = core.validate_plan(read(ROOT / core.PLAN))
    assert p["profiles"] == len(p["views"]) * len(p["relative_floors"]) == 12
    assert p["score_shape_per_profile"] == [4, 4, 16, 8]
    assert p["absolute_check_tolerance"] == 1e-12
    assert not p["approved_for_score_execution"]
    assert p["preparation_payload_reads"] == 0
    assert all((ROOT / p).is_file() for p in core.IMPLEMENTATION)


@pytest.mark.parametrize("key,value", [
    ("approved_for_score_execution", True), ("source_generation", 1),
    ("relative_floors", [1e-6]), ("test_pairs", 15),
    ("reference_shifts", [0]), ("views", ["encoder"]),
    ("absolute_check_tolerance", 1e-6), ("p_values", []),
    ("approved_for_score_execution", 0), ("both_reference_replicates_required", 1),
])
def test_contract_changes_fail_closed(key, value):
    p = copy.deepcopy(read(ROOT / core.PLAN))
    p[key] = value
    with pytest.raises(ValueError):
        core.validate_plan(p)


def metadata_fixture():
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    capture, models, candidates = {"receipts": {}}, [], []
    for pair in spec["sources"]["pairs"]:
        for shift in range(4):
            for cell in ("mm", "mj", "jm", "jj"):
                meta = {"pair_id": pair["pair_id"], "shift": shift, "cell": cell}
                receipts = {}
                for view, (shape, dtype) in core.VIEWS.items():
                    path = f"{pair['pair_id']}/{shift}/{cell}/{view}.npy"
                    receipts[view] = {"path": path, "sha256": "a" * 64,
                                      "word_sha256": "b" * 64,
                                      "shape": shape, "dtype": dtype}
                    capture["receipts"][path] = "a" * 64
                models.append({**meta, "phase": pair["split"], "sites": {
                    k: receipts[k] for k in ("encoder", "projector")}})
                candidates.append({**meta, "split": pair["split"],
                                   "raw": receipts["full_raw_rgb"],
                                   "pixels": receipts["full_processor_pixel_values"]})
    return spec, capture, models, candidates


@pytest.mark.parametrize("damage", [None, "duplicate", "missing", "split", "shape",
                                    "digest", "unsafe", "word_hash", "view"])
def test_complete_metadata_inventory_without_payload_reads(tmp_path, monkeypatch, damage):
    spec, capture, models, candidates = metadata_fixture()
    # Only existence is consulted; a payload read must never occur here.
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("payload read"))
    if damage == "duplicate":
        models.append(models[0])
    elif damage == "missing":
        candidates.pop()
    elif damage == "split":
        models[0]["phase"] = "test"
    elif damage == "shape":
        candidates[0]["raw"]["shape"] = [3, 320, 320]
    elif damage == "digest":
        candidates[0]["pixels"]["sha256"] = "c" * 64
    elif damage == "unsafe":
        candidates[0]["raw"]["path"] = "../escape.npy"
    elif damage == "word_hash":
        candidates[0]["raw"]["word_sha256"] = "not-a-hash"
    elif damage == "view":
        models[0]["sites"].pop("encoder")
    if damage:
        with pytest.raises(ValueError):
            core.selected_inventory(tmp_path, spec, capture, models, candidates)
    else:
        rows = core.selected_inventory(tmp_path, spec, capture, models, candidates)
        assert len(rows) == 2048
        assert sum(r["split"] == "reference" for r in rows) == 1024
        assert {v: sum(r["view"] == v for r in rows) for v in core.VIEWS} == {
            v: 512 for v in core.VIEWS}


def test_no_score_execution_cli_and_canonical_fresh_root_only(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--mode", "run", "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="canonical"):
        core.prepare(ROOT, tmp_path)


def test_fixed_matched_reference_difference_identity_on_analytic_scores():
    # Independent arithmetic check of the registered endpoints, not real scores.
    m = np.array([[0.7, 0.1, 0.2, 0.3], [0.0, 0.6, 0.0, 0.0],
                  [0.0, 0.0, 0.8, 0.0], [0.0, 0.0, 0.0, 0.7]])
    fixed = sum(m[0, b] for b in (1, 2, 3)) / 3 - m[0, 0]
    matched = sum(m[b, b] for b in (1, 2, 3)) / 3 - m[0, 0]
    difference = sum(m[b, b] - m[0, b] for b in (1, 2, 3)) / 3
    assert fixed == pytest.approx(-0.5, abs=1e-12)
    assert matched == pytest.approx(0, abs=1e-12)
    assert matched - fixed == pytest.approx(difference, abs=1e-12)


def test_scale_relative_input_guard_is_not_old_feature_error_bound():
    p = read(ROOT / core.PLAN)
    assert p["direction_guard"] == "norm(z)>relative_floor*sum_four_cell_norms"
    assert "not_feature_matching_error_bound" in p["input_guard_interpretation"]
    assert "without_dividing_by_255" in p["conversion"]["full_raw_rgb"]
    assert "no_reprocessing" in p["conversion"]["full_processor_pixel_values"]
