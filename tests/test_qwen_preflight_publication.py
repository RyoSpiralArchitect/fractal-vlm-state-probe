"""Replay the saved preflight/release without touching a model or runtime."""

import gzip
import hashlib
import json
from pathlib import Path

from fractal_vlm_state_probe.stat_matched_source_storage import read, sha

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0104_qwen_offline_preflight"


def test_lossless_preflight_receipt_and_committed_software():
    receipt = read(PUB / "summary.json")
    compressed = PUB / receipt["compressed_manifest"]
    assert sha(compressed) == receipt["compressed_manifest_sha256"]
    raw = gzip.decompress(compressed.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == receipt["preflight_sha256"]
    p = json.loads(raw)
    assert p["status"] == "OFFLINE_PREFLIGHT_ONLY"
    assert p["native_execution_released"] is False
    assert p["model_contexts"] == p["processor_calls"] == 0
    assert p["free_bytes"] == receipt["free_bytes_at_preflight"] >= receipt["required_launch_free_bytes"]
    assert len(p["software"]) == receipt["software_files"] == 185
    assert len(p["environment"]["snapshot_files"]) == receipt["snapshot_files"] == 11
    assert {k: v["version"] for k, v in p["environment"]["packages"].items()} == receipt["package_versions"]
    assert {k: len(v["files"]) for k, v in p["environment"]["packages"].items()} == receipt["installed_files_by_package"]
    for path, digest in p["software"].items():
        assert sha(ROOT / path) == digest


def test_separate_release_is_bound_and_cannot_release_cohort():
    receipt = read(PUB / "summary.json")
    preflight = json.loads(gzip.decompress((PUB / receipt["compressed_manifest"]).read_bytes()))
    release = read(ROOT / "examples/qwen_native_execution_release_v1.json")
    assert release["preflight_sha256"] == receipt["preflight_sha256"]
    assert release["plan_sha256"] == receipt["plan_sha256"] == preflight["plan_sha256"]
    assert release["software"] == preflight["software"]
    assert release["approved_for_native_execution"] is True
    assert release["cohort_generation_authorized"] is False
    assert release["scope"] == "qwen_observation_calibration_v1_only"
    assert release["model_contexts_upper"] == 102
    assert release["decoder_calls_upper"] == 204
    assert release["processor_calls_upper"] == 136
