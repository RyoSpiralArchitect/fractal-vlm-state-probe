"""Public metadata replay is not a replacement for local native array checks."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.stat_matched_source_storage import read, sha

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0106_qwen_native_calibration_result"


def metadata():
    result = read(PUB / "summary.json")
    path = PUB / result["metadata_bundle"]
    assert sha(path) == result["metadata_bundle_sha256"]
    with tarfile.open(path) as bundle:
        names = {"capture.json", "frozen.json", "rows.jsonl", "processor.jsonl", "operations.jsonl"}
        assert set(bundle.getnames()) == names
        assert all(m.isfile() for m in bundle.getmembers())
        files = {name: bundle.extractfile(name).read() for name in names}
    return result, files


def test_bound_metadata_manifest_release_and_retained_scope():
    result, files = metadata()
    capture, frozen = json.loads(files["capture.json"]), json.loads(files["frozen.json"])
    assert hashlib.sha256(files["capture.json"]).hexdigest() == result["capture_sha256"]
    assert hashlib.sha256(files["frozen.json"]).hexdigest() == result["frozen_sha256"]
    for name in files.keys() - {"capture.json"}:
        assert hashlib.sha256(files[name]).hexdigest() == capture["receipts"][name]
    assert frozen["jobs"] == read(ROOT / "examples/research_notes/0102_qwen_calibration_registration/jobs.json")["jobs"]
    assert sha(ROOT / "examples/qwen_native_execution_release_v1.json") == result["release_sha256"]
    assert frozen["provenance"]["release_sha256"] == result["release_sha256"]
    assert frozen["provenance"]["preflight_sha256"] == result["preflight_sha256"]
    assert frozen["software_fixture"] is capture["software_fixture"] is result["software_fixture"] is False
    assert capture["status"] == result["capture_status"] == "COMPLETE_UNVERIFIED"
    assert result["verification_output"]["status"] == "NATIVE_SAVED_DATA_VERIFIED"
    assert result["verification_output"]["files_checked"] == len(capture["receipts"]) == 8266
    raw = (json.dumps(result["verification_output"], sort_keys=True, separators=(",", ":")) + "\n").encode()
    assert result["verification_tree_sha256"] == {"summary.json": hashlib.sha256(raw).hexdigest()}
    assert result["verification_trees_byte_identical"] is True
    assert result["observation_parity_verified_registered_cases_only"] is True
    assert result["native_runtime_qualified"] is result["cohort_generation_authorized"] is False
    assert result["verification_output"]["native_runtime_qualified"] is False
    assert result["verification_output"]["cohort_released"] is False
    assert result["new_source_images"] == result["new_cohort_maps"] == result["confirmation_scores"] == 0
    assert result["joint_cohort_required_launch_free_bytes"] is None
    assert result["complete_capture_tree_bytes"] == 2633447156


def test_full_operation_order_and_all_layer_layout_receipts():
    result, files = metadata()
    rows = [json.loads(x) for x in files["rows.jsonl"].splitlines()]
    processors = [json.loads(x) for x in files["processor.jsonl"].splitlines()]
    events = [json.loads(x) for x in files["operations.jsonl"].splitlines()]
    jobs = json.loads(files["frozen.json"])["jobs"]
    assert len(jobs) == len(processors) == result["cases"] == 34
    assert len(rows) == result["contexts"] == 102
    expected = []
    cache_keys = {f"cache_{i:02d}_{kind}" for i in range(36) for kind in ("keys", "values")}
    core = cache_keys | {"pixels", "ids", "mask", "grid", "logits_0", "logits_1", "scores"}
    grid = np.array([[1, 22, 22]], dtype=np.int64)
    assert result["grid_thw"] == grid.tolist()
    for i, job in enumerate(jobs):
        assert processors[i]["case_id"] == job["case_id"]
        expected.extend((job["case_id"], "processor_only", k) for k in ("begin", "return"))
        for j, stage in enumerate(("off_before", "on", "off_after")):
            row = rows[3*i+j]
            assert (row["case_id"], row["stage"]) == (job["case_id"], stage)
            assert set(row["arrays"]) == core | ({"site_premerger", "site_auxiliary", "site_final"} if j == 1 else set())
            assert row["arrays"]["ids"]["shape"] == row["arrays"]["mask"]["shape"] == [1, 261]
            assert row["metadata"]["cache_offsets"] == [262]*36
            assert row["arrays"]["grid"]["word_sha256"] == hashlib.sha256(grid.tobytes()).hexdigest()
            assert row["arrays"]["pixels"]["shape"] == [484, 1176]
            for k in cache_keys:
                assert row["arrays"][k]["shape"] == [1, 2, 262, 128]
                assert row["arrays"][k]["dtype"] == "float32"
                assert row["source_dtypes"][k] == "mlx.core.float16"
            expected.extend((job["case_id"], stage, k) for k in (
                "context_begin", "prepare_begin", "prepare_return", "decoder_begin", "decoder_return",
                "decoder_begin", "decoder_return", "context_return"))
    assert events == [{"sequence": i+1, "case_id": c, "stage": s, "kind": k}
                      for i, (c, s, k) in enumerate(expected)]
    assert len(events) == 884
