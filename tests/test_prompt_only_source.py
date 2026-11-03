import copy
import json

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import run_prompt_only_source as runner
from fractal_vlm_state_probe.padding_cache import input_fingerprint
from fractal_vlm_state_probe.prompt_only import (
    ROLES,
    STAGES,
    aggregate,
    check_inputs,
    region_rows,
    source_prompt,
    token_roles,
)


@pytest.fixture
def cfg():
    return json.loads((runner.REPO / runner.CFG).read_text())


def inputs(cfg, edited=False):
    ids = np.zeros((1, 140), dtype=np.int32)
    ids[0, 42] = -200
    ids[0, 132] = cfg["token_edit"]["new_id" if edited else "old_id"]
    return input_fingerprint(
        {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "pixel_values": np.ones(1, dtype=np.float32),
            "image_sizes": [[320, 320]],
        }
    )


def make_event(prepared, suffix, repeated):
    n = len(prepared) + len(suffix)
    return {
        "assistant_text": "a",
        "generation": {
            "steps": [
                {"token_id": i} for i in (suffix + suffix[-1:] if repeated else suffix)
            ],
            "prompt_tokens": len(prepared),
            "generation_tokens": len(suffix),
            "total_tokens": n,
        },
        "cache_token_layout": {
            "token_count": n,
            "marker_token_ids": {"image_token_index": -200},
            "marker_positions": {"image_token_index": [42]},
            "cache_position_layout": {
                "available": True,
                "image_partition_available": True,
                "strategy": "llava_qwen2_single_image_run_replacement",
                "image_position_runs": [{"start": 42, "end": 297, "length": 256}],
                "processor_token_count": n,
                "cache_sequence_lengths": [n + 255],
                "cache_sequence_length": n + 255,
            },
        },
    }


def roles(cfg, suffix, edited=False):
    p = inputs(cfg, edited)["input_ids"]["values"][0]
    return token_roles(p, p + suffix, make_event(p, suffix, True), cfg["token_edit"])


def test_literal_edit_only_and_restore_on_exception(cfg):
    sentence = "Read. Return ACK only."
    assert [source_prompt(sentence, cfg, stage) for stage in STAGES] == [
        sentence,
        "Read. Return YES only.",
        sentence,
    ]
    for bad in (sentence + " ", sentence + sentence):
        with pytest.raises(ValueError):
            source_prompt(bad, cfg, "yes")
    original = runner.replay.cumulative_replay_prompt
    with pytest.raises(RuntimeError), runner.prompt_condition(cfg, "yes"):
        assert runner.replay.cumulative_replay_prompt is not original
        raise RuntimeError
    assert runner.replay.cumulative_replay_prompt is original


@pytest.mark.parametrize("stage", STAGES)
def test_only_one_actual_input_token_may_change(cfg, stage):
    old, actual = inputs(cfg), inputs(cfg, stage == "yes")
    audit = {"native": actual, "effective": actual, "override_image_sizes": None}
    ids = check_inputs(audit, old, cfg["token_edit"], stage)
    assert ids[132] == (14080 if stage == "yes" else 53763)
    for key in actual:
        damaged = copy.deepcopy(audit)
        damaged["effective"][key]["sha256"] = "bogus"
        damaged["native"] = damaged["effective"]
        with pytest.raises(ValueError):
            check_inputs(damaged, old, cfg["token_edit"], stage)


@pytest.mark.parametrize(
    "suffix,repeated",
    [([785, 2168], True), ([7, 7], True), ([1], False), ([7, 8], False)],
)
def test_token_roles_use_cache_state_not_deduplicated_notifications(
    cfg, suffix, repeated
):
    p = inputs(cfg)["input_ids"]["values"][0]
    ev = make_event(p, suffix, repeated)
    r = token_roles(p, p + suffix, ev, cfg["token_edit"])
    assert r["suffix_ids"] == suffix
    assert list(r["spans"].values()) == [
        [0, 42],
        [42, 298],
        [298, 387],
        [387, 388],
        [388, 395],
        [395, 395 + len(suffix)],
    ]
    for bad in (p[1:] + suffix, p + suffix + [3], p + [42] * len(suffix)):
        with pytest.raises(ValueError):
            token_roles(p, bad, ev, cfg["token_edit"])
    ev["cache_token_layout"]["cache_position_layout"]["cache_sequence_lengths"] = [123]
    with pytest.raises(ValueError):
        token_roles(p, p + suffix, ev, cfg["token_edit"])


@pytest.mark.parametrize("suffix", [[785, 2168], [785], [785, 7]])
def test_matching_spans_signed_zero_and_unavailable_suffix(cfg, suffix):
    left, right = roles(cfg, [785, 2168]), roles(cfg, suffix, True)
    a = np.zeros((1, 2, 397, 64), np.float32)
    b = np.zeros((1, 2, 395 + len(suffix), 64), np.float32)
    b[0, 1, 387, 2] = -0.0
    b[0, 0, 394, 0] = 2
    rr = list(region_rows(a, b, left, right))
    assert [r["role"] for r in rr] == list(ROLES)
    assert rr[3]["changed_positions"] == [387] and rr[3]["rms"] == 0
    assert rr[4]["changed_positions"] == [394] and rr[4]["max_abs"] == 2
    assert rr[-1]["available"] == (suffix == [785, 2168])
    if not rr[-1]["available"]:
        assert rr[-1]["rms"] is None and rr[-1]["changed_positions"] is None
    for r in rr:
        runner.verify_region(r, a, b, left, right)
    with pytest.raises(ValueError):
        runner.verify_region({**rr[3], "changed_positions": []}, a, b, left, right)


def test_incomplete_or_duplicate_jobs_rejected(cfg):
    rows = [
        {"phase": "reference", "map": "m0_reference", "pair_id": p, "cell": c}
        for p in cfg["pair_ids"]
        for c in cfg["cells"]
    ]
    assert runner.select_jobs(rows[::-1], cfg) == rows
    for wrong in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError):
            runner.select_jobs(wrong, cfg)


def test_registered_roots_fail_closed_without_forwards(cfg, monkeypatch, tmp_path):
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(k, "1")
    with pytest.raises(ValueError, match="fixed single-attempt"):
        runner.run("capture", tmp_path / "other")
    assert not (tmp_path / "other").exists()
    with pytest.raises(FileExistsError):
        runner.run("verify", tmp_path)


def test_empty_suffix_group_has_no_imputed_numeric_endpoint():
    row = {
        "role": "suffix",
        "target": "1:keys",
        "available": False,
        "rms": None,
        "max_abs": None,
    }
    stats = aggregate(
        [row], [{"first_changed_comparable_position": {"1:keys": None}}], ["1:keys"]
    )
    suffix = stats["1:keys"]["regions"]["suffix"]
    assert suffix == {
        "planned": 1,
        "available": 0,
        "unavailable": 1,
        "bitwise_exact": 0,
        "median_cell_rms": None,
        "maximum_abs": None,
    }
