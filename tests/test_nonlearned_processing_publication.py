import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.summarize_nonlearned_processing_calibration import (
    publish,
)
from fractal_vlm_state_probe.nonlearned_processing_calibration import (
    fixture,
    oracle_features,
)
from fractal_vlm_state_probe.stimulus import write_json


@pytest.fixture
def publication():
    root = (
        Path(__file__).parents[1]
        / "examples/research_notes/0063_nonlearned_processing_calibration"
    )
    raw = (root / "summary.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "d78a58db2470f71239c515e684db120ca8a066c8c9a5460e219e5db7c6c5affc"
    )
    return root, json.loads(raw)


def test_failed_calibration_never_becomes_publication(tmp_path):
    root, output = tmp_path / "run", tmp_path / "public"
    root.mkdir()
    write_json(root / "execution_failure.json", {"status": "FAILED"})
    with pytest.raises(ValueError, match="failed calibration"):
        publish(root, output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        publish(root, output)


def test_checked_in_calibration_arrays_receipts_and_closed_panel(publication):
    root, summary = publication
    ledgers = {}
    for name, receipt in summary["ledgers"].items():
        raw = (root / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
        ledgers[name] = [json.loads(line) for line in raw.splitlines()]
        assert len(ledgers[name]) == receipt["rows"]
    assert len(ledgers["feature_checks.jsonl"]) == 84
    assert len(ledgers["processor_checks.jsonl"]) == 128
    assert (
        summary["registered_design_commit"]
        == "5b44429c52e4519a5328692bc2bc8b53c6bd747c"
    )
    assert summary["status"] == "PASS"
    assert len(summary["frozen_specification"]["code_sha256"]) == 43
    assert (
        summary["new_processor_calls"] == summary["historical_processor_checks"] == 128
    )
    assert (
        summary["new_model_forwards"]
        == summary["experimental_source_cells_scored"]
        == 0
    )
    assert not summary["model_weights_loaded"]
    assert (
        not summary["experimental_panel_released"]
        and not summary["encoder_or_projector_released"]
    )
    assert summary["confirmation_cohort"] == "not_generated_not_sealed"
    assert not summary["preflight"]["processor_instantiated"]
    assert summary["preflight"]["new_processor_calls"] == 0
    verification = summary["independent_verification"]
    assert verification["saved_feature_arrays"] == 84
    assert (
        verification["new_processor_calls"] == verification["new_model_forwards"] == 0
    )
    assert not verification["processor_pixels_reloaded"]
    path = root / "synthetic_features.npz"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == summary["synthetic_features_sha256"]
    )
    config = summary["frozen_specification"]["config"]
    with np.load(path, allow_pickle=False) as arrays:
        assert len(arrays.files) == 85
        for row in ledgers["feature_checks.jsonl"]:
            a = arrays[row["array_key"]]
            assert hashlib.sha256(a.tobytes()).hexdigest() == row["array_sha256"]
            expected = oracle_features(
                fixture(
                    row["fixture"],
                    row["side"],
                    row["start"],
                    config["calibration"]["synthetic_seed"],
                ),
                row["start"],
                config,
            )[row["feature"]]
            np.testing.assert_allclose(a, expected, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(arrays["raw_multiset_zero_interaction"], 0)
    jobs = summary["frozen_specification"]["processor_jobs"]
    assert len({(j["pair_id"], j["cell"]) for j in jobs}) == 128
    for row, job in zip(ledgers["processor_checks.jsonl"], jobs):
        assert row["exact"]
        for key in (
            "pair_id",
            "cell",
            "pixel_values_sha256",
            "processor_non_pixel_payload",
        ):
            assert row[key] == job[key]


def test_changed_output_receipt_rejected_before_publication(tmp_path, publication):
    _, summary = publication
    root, output = tmp_path / "changed", tmp_path / "public"
    root.mkdir()
    changed = copy.deepcopy(summary)
    changed["receipts"] = {"synthetic_features.npz": "0" * 64}
    write_json(root / "summary.json", changed)
    write_json(root / "frozen_specification.json", summary["frozen_specification"])
    write_json(root / "synthetic_calibration.json", summary["synthetic_calibration"])
    np.savez(root / "synthetic_features.npz", broken=np.ones(1))
    with pytest.raises(ValueError, match="output receipt changed"):
        publish(root, output)
    assert not output.exists()
