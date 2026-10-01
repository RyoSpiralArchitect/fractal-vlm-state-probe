from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import capture_pairing_references as cli
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _validate_panel_split,
)


def _panels() -> tuple[dict, dict]:
    panels = []
    for replicates in ((1, 2), (3, 4)):
        records = []
        for family in range(8):
            for replicate in replicates:
                label = f"family_{family}_r{replicate}"
                records.append(
                    {
                        "pair_id": label,
                        "pairing_family": f"family_{family}",
                        "broad_class": "geometry" if family < 4 else "stochastic",
                        "replicate": replicate,
                        "source_a": {"first_frame_sha256": label + "_a"},
                        "source_b": {"first_frame_sha256": label + "_b"},
                    }
                )
        panels.append(
            {"analysis_kind": "generator_pairing_factorial_panel", "records": records}
        )
    return tuple(panels)


def test_panel_split_rejects_reference_test_role_exchange_and_duplicate_images() -> (
    None
):
    reference, test = _panels()
    metadata, hashes = _validate_panel_split(reference, test)
    assert len(metadata) == 32
    assert len(set(hashes)) == 64
    with pytest.raises(ValueError, match="replicate role differs"):
        _validate_panel_split(test, reference)
    test["records"][0]["source_a"] = copy.deepcopy(reference["records"][0]["source_a"])
    with pytest.raises(ValueError, match="repeated first-frame hashes"):
        _validate_panel_split(reference, test)


def test_target_contract_rejects_shape_layout_and_missing_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array = np.zeros((1, 2, 8, 4), dtype=np.float32)
    monkeypatch.setattr(cli, "load_cache_tensor_artifact", lambda *args: array)
    layout = {
        "token_count": 8,
        "image_token_runs": [{"start": 2, "end": 5, "length": 4}],
    }
    historical = {
        "layer_index": 1,
        "tensor": "keys",
        "tensor_shape": [1, 2, 8, 4],
        "cells": {"mm": {"cache_token_layout": layout}},
    }
    run = {
        "stream_events": [
            {
                "cache_tensor_artifacts": [{"layer_index": 1, "tensor": "keys"}],
                "cache_token_layout": layout,
            }
        ]
    }
    assert cli._validate_target_contract(run, Path("run.json"), historical) is array
    wrong = copy.deepcopy(historical)
    wrong["tensor_shape"][-1] = 5
    with pytest.raises(ValueError, match="tensor shape differs"):
        cli._validate_target_contract(run, Path("run.json"), wrong)
    wrong = copy.deepcopy(run)
    wrong["stream_events"][0]["cache_token_layout"]["image_token_runs"][0]["start"] = 3
    with pytest.raises(ValueError, match="token layout differs"):
        cli._validate_target_contract(wrong, Path("run.json"), historical)
    wrong["stream_events"][0]["cache_tensor_artifacts"] = []
    with pytest.raises(ValueError, match="missing or duplicate target"):
        cli._validate_target_contract(wrong, Path("run.json"), historical)


def test_failed_historical_calibration_stops_before_reference_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, test = _panels()
    reference_path, test_path = tmp_path / "reference.json", tmp_path / "test.json"
    reference_path.write_text(json.dumps(reference))
    test_path.write_text(json.dumps(test))
    cells = {}
    for cell in ("mm", "jj", "mj", "jm"):
        path = tmp_path / f"{cell}.json"
        path.write_text(json.dumps({"manifest_path": str(tmp_path / "manifest.json")}))
        cells[cell] = {
            "source_path": str(path),
            "cache_tensor_artifact": {"sha256": cell},
        }
    factorial = tmp_path / "factorial.json"
    factorial.write_text(
        json.dumps(
            {
                "model_id": "example/model",
                "layer_index": 1,
                "tensor": "keys",
                "cells": cells,
            }
        )
    )
    output = tmp_path / "capture"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "capture",
            "--model",
            "example/model",
            "--reference-panel",
            str(reference_path),
            "--test-panel",
            str(test_path),
            "--historical-factorial",
            f"1:keys={factorial}",
            "--output-root",
            str(output),
        ],
    )
    calls = []

    def capture(config, *, mlx_runtime):
        calls.append(config.output_path)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text("{}")

    monkeypatch.setattr(cli, "run_cumulative_replay_probe", capture)
    monkeypatch.setattr(cli, "_load_mlx_runtime", lambda model: {})
    monkeypatch.setattr(cli, "_freeze_model_snapshot", lambda *args: None)
    monkeypatch.setattr(cli, "_validate_run", lambda *args: None)
    monkeypatch.setattr(
        cli, "load_cache_tensor_artifact", lambda *args: np.zeros((1, 2, 8, 4))
    )
    monkeypatch.setattr(
        cli, "_validate_target_contract", lambda *args: np.ones((1, 2, 8, 4))
    )
    with pytest.raises(ValueError, match="historical calibration differs"):
        cli.main()
    assert len(calls) == 4
    assert all(p.parent.name == "historical_recheck" for p in calls)
    assert not (output / "source_runs").exists()
    assert not any(
        c["bitwise_equal"]
        for c in json.loads((output / "historical_recheck.json").read_text())["checks"]
    )
