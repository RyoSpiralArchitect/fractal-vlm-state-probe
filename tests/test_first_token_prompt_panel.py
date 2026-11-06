import copy
import math

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import run_first_token_prompt_panel as runner
from fractal_vlm_state_probe.first_token_panel import (
    aggregate,
    check_metrics,
    expand_scores,
    normalized,
    oracle_expand_scores,
    oracle_metrics,
    pack_scores,
    paired_metrics,
)


@pytest.fixture
def cfg():
    return runner.base._read(runner.REPO / runner.CFG)


@pytest.mark.parametrize(
    "ack,yes,tv,js",
    [
        ([0.0, -np.inf], [-np.inf, 0.0], 1.0, math.log(2)),
        ([0.0, 0.0], [0.0, 0.0], 0.0, 0.0),
        ([0.0, -1000.0], [-1000.0, 0.0], 1.0, math.log(2)),
    ],
)
def test_full_probability_distances_include_zero_support_and_tail_underflow(
    ack, yes, tv, js
):
    a, b = np.array(ack, np.float32), np.array(yes, np.float32)
    r = paired_metrics(a, b, 20)
    assert r["total_variation"] == tv
    assert r["jensen_shannon_nats"] == js
    check_metrics(a, b, r, 1e-12, 1e-14)
    assert len(r["top_delta_coordinates"]) == 2
    assert [x["token_id"] for x in r["top_delta_coordinates"]] == [0, 1]
    assert oracle_metrics(a, b)["total_variation"] == tv
    with pytest.raises(ValueError):
        check_metrics(a, b, {**r, "total_variation": tv + 0.1}, 1e-12, 1e-14)


def test_native_score_offsets_are_not_probability_mass_or_an_effect():
    a, b = np.array([0.0, 1.0, -2.0], np.float32), np.array([2.0, 3.0, 0.0], np.float32)
    r = paired_metrics(a, b, 3)
    check_metrics(a, b, r, 1e-12, 1e-14)
    assert r["total_variation"] < 1e-14 and not r["native_score_word_equal"]
    assert abs(r["ack_probability_sum"] - 1) < 1e-14
    assert abs(r["yes_probability_sum"] - 1) < 1e-14
    assert math.isclose(
        float(np.exp(b.astype(np.float64)).sum() / np.exp(a.astype(np.float64)).sum()),
        math.exp(2),
    )
    assert normalized(a)[1].shape == (3,)


def test_same_argmax_and_a_changed_tail_still_have_nonzero_full_vocabulary_effect():
    a, b = (
        np.array([0.0, -1.0, -2.0, -20.0], np.float32),
        np.array([0.0, -1.0, -2.0, -10.0], np.float32),
    )
    r = paired_metrics(a, b, 2)
    check_metrics(a, b, r, 1e-12, 1e-14)
    assert r["argmax_sets_overlap"] and r["ack_argmax_ids"] == r["yes_argmax_ids"] == [
        0
    ]
    assert r["total_variation"] > 0 and r["max_abs_token_id"] == 3
    assert [x["token_id"] for x in r["top_delta_coordinates"]][0] == 3


def test_complete_native_word_codec_retains_infinity_signed_zero_and_low_coordinates():
    scores = np.array([0.0, -0.0, -1.0, -np.inf, -1000.0], np.float32)
    packed = pack_scores(scores)
    assert packed.dtype == np.dtype("<u2") and packed[1] == 32768
    assert (
        expand_scores(packed).tobytes()
        == scores.tobytes()
        == oracle_expand_scores(packed).tobytes()
    )
    for bad in (
        np.array([1.1], np.float32),
        np.array([np.nan], np.float32),
        np.array([np.inf], np.float32),
    ):
        with pytest.raises(ValueError):
            pack_scores(bad)
    with pytest.raises(ValueError):
        expand_scores(packed.astype(np.uint32))


def fixture_pairs():
    return [
        {
            "pair_id": f"f{i // 4}",
            "cell": ("mm", "jj", "mj", "jm")[i % 4],
            "total_variation": i / 32,
            "jensen_shannon_nats": 0.1,
            "hellinger": 0.2,
            "delta_l2": 0.3,
            "max_abs_delta": 0.1,
            "argmax_sets_overlap": i % 2 == 0,
            "ack_argmax_ids": [1],
            "yes_argmax_ids": [1, 2] if i % 2 == 0 else [2],
        }
        for i in range(32)
    ]


def test_balanced_primary_mean_and_all_family_cell_rows_without_confidence_claim():
    p = fixture_pairs()
    a = aggregate(p)
    assert a["primary_value"] == sum(range(32)) / 32**2
    assert len(a["by_source_pair"]) == 8 and len(a["by_cell"]) == 4
    assert a["argmax_sets_overlap"] == 16
    assert a["argmax_set_size_counts"]["yes"] == {"2": 16, "1": 16}
    for group in a["by_source_pair"]:
        group_rows = [r for r in p if r["pair_id"] == group["pair_id"]]
        assert group["images"] == 4
        assert (
            group["metrics"]["total_variation"]["mean"]
            == math.fsum(r["total_variation"] for r in group_rows) / 4
        )
    for wrong in (p[:-1], p + [p[0]], p[:-1] + [p[0]]):
        with pytest.raises(ValueError):
            aggregate(wrong)
    wrong = copy.deepcopy(p)
    wrong[0]["cell"] = "unregistered"
    with pytest.raises(ValueError):
        aggregate(wrong)


@pytest.mark.parametrize(
    "key,value",
    [
        ("vocab_size", 151665),
        ("panel_contexts", 62),
        ("calibration_contexts", 0),
        ("p_values", 1),
        ("confidence_intervals", 1),
        ("retry_or_resume", True),
        ("numeric_oracle_rtol", 1e-6),
        ("reserve_bytes", 0),
        ("launch_allowance_bytes", 1),
        ("probability_definition", "raw_exp_without_normalization"),
    ],
)
def test_frozen_full_panel_stopping_probability_definition_and_budget(cfg, key, value):
    runner.require_config(cfg)
    with pytest.raises(ValueError):
        runner.require_config({**cfg, key: value})


def test_conservative_budget_no_compression_or_deletion_assumed(cfg):
    b = runner.storage_budget()
    assert (
        b["calibration_payload_bytes"] == 2396504064
        and b["panel_payload_bytes"] == 518455296
    )
    assert sum(b.values()) == 12041764864
    assert cfg["launch_allowance_bytes"] + cfg["reserve_bytes"] == sum(b.values())


def test_canonical_one_attempt_and_output_not_overwritten(monkeypatch, tmp_path):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError, match="canonical"):
        runner.run("capture", tmp_path / "alternative")
    assert not (tmp_path / "alternative").exists()
    with pytest.raises(FileExistsError):
        runner.run("verify", tmp_path)
