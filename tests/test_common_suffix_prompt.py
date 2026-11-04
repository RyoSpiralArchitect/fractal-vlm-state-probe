import copy
import json

import numpy as np
import pytest

from fractal_vlm_state_probe.cli import run_common_suffix_prompt as runner
from fractal_vlm_state_probe.common_suffix import (
    aggregate,
    compare_pair,
    expected_anchor,
    panel_anchors,
    region_metrics,
    verify_metric,
)
from fractal_vlm_state_probe.native_suffix_replay import NAMES


@pytest.fixture
def cfg():
    return json.loads((runner.REPO / runner.CFG).read_text())


@pytest.fixture
def anchors():
    p = (
        runner.REPO
        / "examples/research_notes/0072_prompt_only_source_calibration/contexts.jsonl"
    )
    return [json.loads(line) for line in p.read_text().splitlines()]


def test_keep_all_32_images_even_when_native_suffix_differs(anchors):
    cells = panel_anchors(anchors)
    assert len(cells) == 64 and len({r["frame_sha256"] for r in cells}) == 32
    assert sum(r["suffix_ids"] == [785, 2168] for r in cells) == 34
    assert sum(r["stage"] == "yes" for r in cells) == 32
    for original in cells:
        saved = copy.deepcopy(original)
        expected = expected_anchor(original, [785, 2168])
        assert expected["suffix_ids"] == [785, 2168]
        assert expected["cache_state_ids"] == original["prepared_ids"] + [785, 2168]
        assert expected["prepared_ids"] == original["prepared_ids"]
        assert original == saved


@pytest.mark.parametrize("kind", ["drop", "duplicate", "stage", "extra_token"])
def test_incomplete_duplicate_or_misaligned_images_not_filtered(anchors, kind):
    broken = copy.deepcopy(anchors)
    if kind == "drop":
        broken.pop()
    if kind == "duplicate":
        broken[3:6] = copy.deepcopy(broken[:3])
    if kind == "stage":
        broken[0]["stage"] = "ack_after"
    if kind == "extra_token":
        broken[1]["prepared_ids"][30] += 1
    with pytest.raises(ValueError):
        panel_anchors(broken)


def test_word_difference_and_signed_zero_separate_from_numeric_change(cfg):
    a = np.zeros((1, 2, 2, 64), dtype="<u2")
    b = a.copy()
    b[0, 1, 1, 3] = 0x8000
    m = region_metrics(a, b)
    assert m["changed_scalars"] == 1 and m["changed_relative_positions"] == [1]
    assert m["rms"] == m["max_abs"] == 0
    assert m["cosine"] is m["relative_l2"] is None
    verify_metric(a, b, m, cfg["numeric_oracle_rtol"], cfg["numeric_oracle_atol"])
    with pytest.raises(ValueError):
        verify_metric(a, b, {**m, "changed_relative_positions": []}, 1e-12, 1e-14)


@pytest.mark.parametrize("zeros", ["neither", "ack", "yes", "both"])
def test_independent_scalar_metrics_and_explicit_zero_norms(cfg, zeros):
    rng = np.random.default_rng(2)
    a = (rng.integers(1, 40, (1, 2, 7, 64)) + 15000).astype("<u2")
    b = (rng.integers(1, 40, a.shape) + 16000).astype("<u2")
    if zeros in ("ack", "both"):
        a.fill(0)
    if zeros in ("yes", "both"):
        b.fill(0)
    m = region_metrics(a, b)
    verify_metric(a, b, m, 1e-12, 1e-14)
    assert (m["relative_l2"] is None) == (zeros in ("ack", "both"))
    assert (m["cosine"] is None) == (zeros != "neither")
    if m["rms"]:
        with pytest.raises(ValueError):
            verify_metric(a, b, {**m, "rms": m["rms"] * 2}, 1e-12, 1e-14)


def test_all_layer_region_denominators_positions_and_no_change_endpoint(cfg, anchors):
    a = np.full((24, 2, 1, 2, 397, 64), 0x3F80, dtype="<u2")
    b = a.copy()
    b[23, 1, 0, 1, 396, 2] = 0x4000
    cfg = {**cfg, "numeric_oracle_rtol": 1e-12}
    pair = list(compare_pair(a, b, anchors[0], cfg))
    assert len(pair) == 48 * 8
    suffix = next(
        r for r in pair if r["target"] == "23:values" and r["region"] == "suffix"
    )
    assert (
        suffix["changed_positions"] == [396]
        and suffix["first_changed_full_cache"] == 396
    )
    assert suffix["rms"] == 1 / 16 and suffix["max_abs"] == 1
    assert suffix["changed_scalars"] == 1 and suffix["scalar_pairs"] == 256
    metrics = [
        {**r, "pair_id": f"family{i // 4}", "cell": f"cell{i % 4}"}
        for i in range(32)
        for r in pair
    ]
    summary = aggregate(metrics, cfg)
    assert len(summary) == 48 * 8 and sum(r["primary"] for r in summary) == 3
    row = next(
        r for r in summary if r["target"] == "23:values" and r["region"] == "suffix"
    )
    assert (
        row["images"] == 32 and row["word_exact"] == 0 and row["rms_median"] == 1 / 16
    )
    assert set(r["target"] for r in summary) == set(NAMES)
    with pytest.raises(ValueError, match="denominator"):
        aggregate(metrics[:-1], cfg)


@pytest.mark.parametrize(
    "key,value",
    [
        ("common_suffix_ids", [9454, 11]),
        ("panel_contexts", 62),
        ("calibration_contexts", 0),
        ("numeric_oracle_rtol", 1e-6),
        ("retry_or_resume", True),
        ("reserve_bytes", 0),
        ("p_values", 1),
        ("regions", {"suffix": [395, 396]}),
    ],
)
def test_frozen_scope_rejects_changed_condition_and_tolerance(cfg, key, value):
    runner.require_config(cfg)
    changed = {**cfg, key: value}
    with pytest.raises(ValueError):
        runner.require_config(changed)


def test_original_profiles_not_modified_and_alternate_attempt_refused(
    monkeypatch, tmp_path
):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    original = (
        runner.base.CFG,
        runner.base.CODE,
        runner.base.forced_stepper,
        runner.base.qualify,
    )
    with pytest.raises(ValueError, match="canonical"):
        runner.run("capture", tmp_path / "alternate")
    with pytest.raises(FileExistsError):
        runner.run("verify", tmp_path)
    assert not (tmp_path / "alternate").exists()
    assert (
        runner.base.CFG,
        runner.base.CODE,
        runner.base.forced_stepper,
        runner.base.qualify,
    ) == original


@pytest.mark.parametrize("natural", [[785, 2168], [9454, 151645]])
def test_common_shams_compare_actual_prefill_words_and_all_matching_own_boundaries(
    monkeypatch, tmp_path, cfg, natural
):
    anchor = {"ordinal": 1, "suffix_ids": natural}
    reference = np.zeros((1, 1, 397, 1), dtype="<u2")
    calls = []

    def reload(path, receipts, arm, a):
        calls.append((path, arm, a))
        for end in (395, 396, 397):
            yield reference[..., :end, :]

    monkeypatch.setattr(runner, "reconstruct", reload)
    terminal = reference.copy()
    root = tmp_path / "panel" / "001"
    runner.compare_own_sham(root, anchor, cfg, [], terminal)
    assert (
        calls[-1][0] == tmp_path / "recalibration" / "contexts" / "001" / "forced_own"
    )
    terminal[..., 396, 0] = 1
    if natural == [785, 2168]:
        with pytest.raises(ValueError):
            runner.compare_own_sham(root, anchor, cfg, [], terminal)
    else:
        runner.compare_own_sham(root, anchor, cfg, [], terminal)
    terminal[..., 394, 0] = 1
    with pytest.raises(ValueError):
        runner.compare_own_sham(root, anchor, cfg, [], terminal)
