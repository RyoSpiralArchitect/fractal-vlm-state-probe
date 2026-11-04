"""Public-only joins, denominators and aggregate replay for the fixed suffix panel."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/research_notes"
PUBLIC = BASE / "0078_common_suffix_prompt_control"


def read(p):
    return json.loads(p.read_bytes())


def rows(p, sha=None):
    data = p.read_bytes()
    if sha is not None:
        assert hashlib.sha256(data).hexdigest() == sha
    return [json.loads(line) for line in data.splitlines()]


def test_all_images_actual_ids_calibration_and_prior_failures_are_retained():
    s = read(PUBLIC / "summary.json")
    cfg = read(ROOT / "configs/common_suffix_prompt_control_v1.json")
    assert s["status"] == "VERIFIED" and s["source_contexts"] == 352
    assert (
        s["calibration_contexts"] == 288
        and s["panel_contexts"] == 64
        and s["panel_images"] == 32
    )
    assert s["common_suffix_ids"] == [785, 2168] and s["same_model_instance"]
    assert (
        s["preceding_qualification_summary_sha256"]
        == cfg["qualification_summary_sha256"]
    )
    assert (
        s["direct_readout_contexts"] == s["equivalence_tests"] == s["new_p_values"] == 0
    )
    assert (
        s["verification_model_calls"] == 0
        and s["confirmation_cohort"] == "not_generated_not_sealed"
    )
    for name, sha in s["code_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
    for name, meta in s["ledgers"].items():
        assert len(rows(PUBLIC / name, meta["sha256"])) == meta["rows"]
    old = read(BASE / "0072_prompt_only_source_calibration/summary.json")
    anchors = rows(
        BASE / "0072_prompt_only_source_calibration/contexts.jsonl",
        old["ledgers"]["contexts.jsonl"]["sha256"],
    )
    chosen = [r for r in anchors if r["stage"] != "ack_after"]
    cells = rows(PUBLIC / "panel_cells.jsonl")
    assert len(cells) == len(chosen) == 64
    for r, a in zip(cells, chosen, strict=True):
        assert all(r[k] == a[k] for k in ("pair_id", "cell", "stage", "prepared_ids"))
        assert r["anchor_ordinal"] == a["ordinal"]
        assert r["natural_suffix_ids"] == a["suffix_ids"]
        assert r["supplied_suffix_ids"] == [785, 2168]
        assert r["actual_cache_ids"] == a["prepared_ids"] + [785, 2168]
        assert r["site_array_sha256"] == a["site_array_sha256"]
        assert r["spans"] == a["spans"]
        assert (
            r["pixel_values_sha256"]
            == a["input_fingerprints"]["pixel_values"]["sha256"]
        )
        assert r["forced_text_not_readout"] == "The image"
    assert len({r["pixel_values_sha256"] for r in cells}) == 32
    assert sum(r["natural_suffix_ids"] != r["supplied_suffix_ids"] for r in cells) == 30
    previous = BASE / "0076_native_suffix_replay_parity"
    assert rows(PUBLIC / "calibration_boundaries.jsonl") == rows(
        previous / "boundaries.jsonl"
    )
    cal = rows(PUBLIC / "calibration_arms.jsonl")
    prior = rows(previous / "arms.jsonl")
    assert len(cal) == len(prior) == 288
    for a, b in zip(cal, prior, strict=True):
        assert {k: v for k, v in a.items() if k != "source_files"} == {
            k: v for k, v in b.items() if k != "source_files"
        }
    failed = read(BASE / "0074_native_suffix_replay_entry_failure/summary.json")
    assert failed["status"] == "NOT_QUALIFIED" and failed["parity_result"] is None


def test_all_48_targets_regions_changed_positions_and_32_image_summary():
    s = read(PUBLIC / "summary.json")
    cfg = read(ROOT / "configs/common_suffix_prompt_control_v1.json")
    cells = rows(PUBLIC / "panel_cells.jsonl")
    pairs = {(r["pair_id"], r["cell"]) for r in cells}
    region = rows(PUBLIC / "regions.jsonl")
    names = {f"{i}:{k}" for i in range(24) for k in ("keys", "values")}
    assert (
        len(region)
        == len({(r["pair_id"], r["cell"], r["target"], r["region"]) for r in region})
        == 12288
    )
    assert set(r["target"] for r in region) == names
    assert len(s["summaries"]) == 48 * 8
    assert sum(r["primary"] for r in s["summaries"]) == 3
    for r in region:
        lo, hi = cfg["regions"][r["region"]]
        assert r["span"] == [lo, hi] and (r["pair_id"], r["cell"]) in pairs
        assert r["primary_target"] == (r["target"] in cfg["primary_targets"])
        assert r["scalar_pairs"] == (hi - lo) * 128
        assert r["changed_positions"] == sorted(set(r["changed_positions"]))
        assert all(lo <= p < hi for p in r["changed_positions"])
        assert (
            len(r["changed_positions"])
            <= r["changed_scalars"]
            <= 128 * len(r["changed_positions"])
        )
        assert 0 <= r["rms"] <= r["max_abs"]
        assert abs(r["delta_l2"] - r["rms"] * r["scalar_pairs"] ** 0.5) <= max(
            1e-14, r["delta_l2"] * 1e-12
        )
        assert (r["relative_l2"] is None) == (r["ack_l2"] == 0)
        assert (r["cosine"] is None) == (r["ack_l2"] == 0 or r["yes_l2"] == 0)
        if r["relative_l2"] is not None:
            assert abs(r["relative_l2"] - r["delta_l2"] / r["ack_l2"]) < 1e-12
    for stats in s["summaries"]:
        group = [
            r
            for r in region
            if r["target"] == stats["target"] and r["region"] == stats["region"]
        ]
        assert len(group) == 32 and {(r["pair_id"], r["cell"]) for r in group} == pairs
        values = [r["rms"] for r in group]
        assert stats["rms_median"] == median(values)
        assert stats["rms_min"] == min(values) and stats["rms_max"] == max(values)
        assert stats["max_abs"] == max(r["max_abs"] for r in group)
        assert stats["word_exact"] == sum(r["changed_scalars"] == 0 for r in group)
        assert stats["first_changed_counts"] == dict(
            Counter(str(r["first_changed_full_cache"]) for r in group)
        )
    previous = rows(BASE / "0072_prompt_only_source_calibration/regions.jsonl")
    matched = [r for r in previous if r["role"] == "suffix" and r["available"]]
    assert len(matched) == 6
    for old in matched:
        current = next(
            r
            for r in region
            if r["region"] == "suffix"
            and all(r[k] == old[k] for k in ("pair_id", "cell", "target"))
        )
        assert all(
            current[k] == old[k]
            for k in ("rms", "max_abs", "changed_positions", "changed_scalars")
        )


def test_actual_decoder_boundaries_and_word_audit_match_the_published_cells():
    independent = read(PUBLIC / "independent_word_check.json")
    assert (
        independent["status"] == "PASS"
        and independent["model_calls"] == independent["processor_calls"] == 0
    )
    assert (
        independent["verified_summary_sha256"]
        == hashlib.sha256((PUBLIC / "summary.json").read_bytes()).hexdigest()
    )
    checkpoints = rows(
        PUBLIC / "decoder_boundaries.jsonl", independent["decoder_boundaries_sha256"]
    )
    cells = rows(PUBLIC / "panel_cells.jsonl")
    assert len(checkpoints) == independent["actual_decoder_boundaries"] == 192
    expected_targets = {f"{i}:{k}" for i in range(24) for k in ("keys", "values")}
    for i, cell in enumerate(cells):
        for step, r in enumerate(checkpoints[3 * i : 3 * i + 3]):
            assert all(
                r[k] == cell[k] for k in ("anchor_ordinal", "pair_id", "cell", "stage")
            )
            assert r["step"] == step and r["offset"] == 395 + step
            assert r["input_ids"] == (
                cell["prepared_ids"] if not step else [785, 2168][step - 1 : step]
            )
            assert r["embedding_shape"] == ([1, 395, 896] if not step else None)
            assert set(r["tensor_sha256"]) == expected_targets
            if step == 2:
                assert r["sha256"] == cell["terminal_sha256"]
    assert independent["two_singleton_calls_per_context"] == 128
    assert independent["matching_own_suffix_exact_tensor_comparisons"] == 34 * 3 * 48
    assert (
        independent["sham_tensor_comparisons_including_prefill"]
        == 64 * 48 + 34 * 2 * 48
    )
    assert independent["independent_paired_region_byte_and_metric_checks"] == 12288
