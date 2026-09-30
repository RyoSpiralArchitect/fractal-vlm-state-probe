from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _validate_run,
)
from fractal_vlm_state_probe.cli.summarize_pairing_seed_validation import (
    main as summarize_main,
)


def test_resume_validation_rejects_a_different_runtime_or_visual_input(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"frames": [{"sha256": "expected_hash"}]}), encoding="utf-8"
    )
    reference = {
        "model_id": "example/model",
        "runtime": {"mlx_version": "0.31.1", "mlx_vlm_version": "0.4.4"},
        "context_policy": {"source_cache_only": True, "stream_temperature": 0},
        "reproducibility": {"seed": 20260604},
        "stream_events": [
            {
                "frame_sha256s": ["expected_hash"],
                "prompt": "fixed prompt",
                "assistant_text": "ACK",
                "generation": {"steps": [{"token_id": 123}, {"token_id": 2}]},
                "cache_tensor_artifacts": [],
            }
        ],
    }
    path = tmp_path / "run.json"
    _validate_run(reference, path, manifest, reference)

    wrong = copy.deepcopy(reference)
    wrong["runtime"]["mlx_version"] = "other"
    with pytest.raises(ValueError, match="runtime mlx_version differs"):
        _validate_run(wrong, path, manifest, reference)

    wrong = copy.deepcopy(reference)
    wrong["stream_events"][0]["frame_sha256s"] = ["different_hash"]
    with pytest.raises(ValueError, match="image hash/prompt differs"):
        _validate_run(wrong, path, manifest, reference)

    wrong = copy.deepcopy(reference)
    wrong["stream_events"][0]["assistant_text"] = "The image"
    with pytest.raises(ValueError, match="source suffix differs"):
        _validate_run(wrong, path, manifest, reference)

    wrong = copy.deepcopy(reference)
    wrong["stream_events"][0]["generation"]["steps"][0]["token_id"] = 124
    with pytest.raises(ValueError, match="suffix token IDs differ"):
        _validate_run(wrong, path, manifest, reference)

    wrong = copy.deepcopy(reference)
    wrong["stream_events"][0]["generation"]["steps"] = []
    with pytest.raises(ValueError, match="suffix token IDs are unavailable"):
        _validate_run(wrong, path, manifest, reference)


def test_snapshot_fingerprint_includes_processor_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshots" / "fixed_revision"
    snapshot.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "chat_template.jinja",
    ):
        (snapshot / name).write_bytes(name.encode())
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            try_to_load_from_cache=lambda *args: str(snapshot / "config.json")
        ),
    )
    output = tmp_path / "model_snapshot.json"
    _freeze_model_snapshot("example/model", output)
    record = json.loads(output.read_text())
    assert record["revision"] == "fixed_revision"
    assert (
        record["file_sha256"]["model.safetensors"]
        == hashlib.sha256(b"model.safetensors").hexdigest()
    )
    assert len(record["file_sha256"]) == 5
    _freeze_model_snapshot("example/model", output)
    (snapshot / "processor_config.json").write_bytes(b"changed processor")
    with pytest.raises(ValueError, match="snapshot differs"):
        _freeze_model_snapshot("example/model", output)


def test_snapshot_fingerprint_fails_closed_when_weights_are_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(try_to_load_from_cache=lambda *args: str(config)),
    )
    with pytest.raises(ValueError, match="no weights"):
        _freeze_model_snapshot("example/model", tmp_path / "output.json")


def test_study_summary_corrects_all_eight_tests_and_rejects_incomplete_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = []
    for layer in range(4):
        path = tmp_path / f"holdout_{layer}.json"
        path.write_text(
            json.dumps(
                {
                    "model_id": "example/model",
                    "layer_index": layer,
                    "tensor": "values",
                    "analysis_paths": {},
                    "hierarchy_metadata": {},
                    "views": [
                        {
                            "view": region,
                            "primary": True,
                            "available": True,
                            "own_family_cosine_mean": 1.0,
                            "other_family_cosine_mean": 0.0,
                            "mean_family_margin": 1.0,
                            "positive_family_margin_count": 8,
                            "retrieval_correct_count": 16,
                            "test_seed_count": 16,
                            "exact_block_test": {"p_greater": 1 / 576},
                        }
                        for region in ("image_tokens", "post_image")
                    ],
                }
            )
        )
        paths.append(str(path))
    execution = tmp_path / "execution.json"
    record = {
        "analysis_kind": "pairing_seed_validation_execution",
        "holdout_analyses": paths,
        "new_source_cells": 0,
        "new_tensor_sidecars": 0,
        "new_factorial_analyses": 0,
        "reference_recheck_cells": 4,
        "model_snapshot": {"revision": "fixed"},
    }
    execution.write_text(json.dumps(record))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize",
            "--execution",
            str(execution),
            "--output-json",
            str(output),
            "--output-md",
            str(tmp_path / "summary.md"),
        ],
    )
    summarize_main()
    result = json.loads(output.read_text())
    assert result["primary_test_count"] == 8
    assert result["primary_holm_p_below_0_05"] == 8
    assert all(
        r["holm_p_greater"] == pytest.approx(8 / 576) for r in result["primary_tests"]
    )
    assert result["reference_recheck_cells"] == 4
    record["holdout_analyses"] = paths[:3]
    execution.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="requires 8 primary tests; received 6"):
        summarize_main()
