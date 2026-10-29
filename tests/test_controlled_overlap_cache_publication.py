import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.summarize_controlled_overlap_cache_calibration import (
    checked_array,
    publish,
)
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def tensor(tmp_path):
    path = tmp_path / "tensor.npz"
    array = np.zeros((1, 2, 397, 64), dtype=np.float32)
    np.savez(path, tensor=array)
    artifact = {
        "path": "tensor.npz",
        "array_key": "tensor",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(array.shape),
        "dtype": "float32",
        "source_dtype": "mlx.core.bfloat16",
        "effective_sequence_length": 397,
    }
    return tmp_path / "run.json", artifact


def test_independent_loader_verifies_storage(tensor):
    path, artifact = tensor
    array = checked_array(path, artifact)
    assert array.shape == (1, 2, 397, 64) and array.dtype == np.float32


@pytest.mark.parametrize(
    "key,value",
    [
        ("sha256", "bad"),
        ("shape", [1, 2, 396, 64]),
        ("dtype", "float16"),
        ("source_dtype", "mlx.core.float16"),
        ("effective_sequence_length", 398),
        ("path", "../outside.npz"),
    ],
)
def test_independent_loader_rejects_corrupted_receipts(tensor, key, value):
    path, artifact = tensor
    artifact[key] = value
    with pytest.raises(ValueError):
        checked_array(path, artifact)


def test_failed_capture_is_never_published_as_pass(tmp_path):
    root, output = tmp_path / "run", tmp_path / "publication"
    root.mkdir()
    write_json(root / "execution_failure.json", {"status": "FAILED"})
    with pytest.raises(ValueError, match="failure retained"):
        publish(root, output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        publish(root, output)


def test_checked_in_calibration_receipts_and_count_boundaries():
    root = (
        Path(__file__).parents[1]
        / "examples/research_notes/0058_controlled_overlap_cache_calibration"
    )
    summary = json.loads((root / "summary.json").read_text())
    ledgers = {}
    for filename, receipt in summary["ledgers"].items():
        data = (root / filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == receipt["sha256"]
        rows = [json.loads(line) for line in data.splitlines()]
        assert len(rows) == receipt["rows"]
        ledgers[filename] = rows
    assert (
        summary["registered_design_commit"]
        == "526950da184960f9a912e2f8201539841616603d"
    )
    assert summary["new_model_forwards"] == 128
    assert (
        summary["new_experimental_panel_cells"]
        == summary["new_direct_probe_cells"]
        == 0
    )
    assert not summary["panel_started"] and not summary["panel_automatically_released"]
    assert summary["confirmation_cohort"] == "not_generated_not_sealed"
    assert summary["preflight"]["new_model_forwards"] == 0
    assert not summary["preflight"]["model_weights_loaded"]
    assert summary["source_response_counts"] == {"The image": 128}
    cells, checks = ledgers["cells.jsonl"], ledgers["comparisons.jsonl"]
    labels = summary["frozen_specification"]["hierarchy_metadata"]
    assert len(cells) == 128 and len(checks) == 384 and len(labels) == 32
    assert {(c["pair_id"], c["cell"]) for c in cells} == {
        (k, c) for k in labels for c in ("mm", "jj", "mj", "jm")
    }
    assert {(c["pair_id"], c["cell"], c["target"]) for c in checks} == {
        (k, c, t)
        for k in labels
        for c in ("mm", "jj", "mj", "jm")
        for t in ("1:keys", "12:keys", "23:values")
    }
    for row in checks:
        assert row["bitwise_equal"] and row["independent_tensor_bytes_exact"]
        assert row["max_abs_difference"] == row["rms_difference"] == 0
        assert row["before_array_sha256"] == row["after_array_sha256"]
    verification = summary["independent_verification"]
    assert verification["scalar_pairs"] == 384 * 2 * 397 * 64
    assert (
        verification["maximum_absolute_difference"]
        == verification["verification_model_forwards"]
        == 0
    )
