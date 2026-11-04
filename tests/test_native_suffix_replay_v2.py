import copy
from pathlib import Path

import pytest

from fractal_vlm_state_probe.cli import run_native_suffix_replay_v2 as v2


def test_inactive_entry_normalizes_only_omitted_criterion():
    received = []

    def step(a, *, thinking_budget_criteria, sentinel):
        received.append((a, thinking_budget_criteria, sentinel))
        return a

    wrapper = v2.inactive_entry(step)
    for options in ({}, {"thinking_budget_criteria": None}):
        assert wrapper(97, sentinel="keep", **options) == 97
    for active in (False, 0, "", object()):
        with pytest.raises(ValueError, match="active thinking"):
            wrapper(97, sentinel="keep", thinking_budget_criteria=active)
    with pytest.raises(TypeError):
        wrapper(97, sentinel="keep", unregistered=True)
    assert received == [(97, None, "keep")] * 2


def test_installed_outer_stream_inactive_dispatch_and_consumed_eos():
    pytest.importorskip("mlx_vlm.generate")
    records = v2.stream_fixture()
    assert len(records) == 9
    for row in records:
        first, second = row["suffix"]
        assert ("thinking_budget_criteria" in row["dispatch_keys"]) == (
            row["mode"] != "omitted"
        )
        if row["mode"] == "active":
            assert not row["actual_stub_decoder_inputs"] and row["cached_ids"] is None
        else:
            assert row["actual_stub_decoder_inputs"][-2:] == [[first], [second]]
            assert row["cached_ids"][-2:] == [first, second]
            assert len(row["notifications"]) == (2 if second == 151645 else 3)


@pytest.mark.parametrize(
    "key",
    [
        "arms",
        "layers",
        "source_contexts",
        "offsets",
        "retry_or_resume",
        "cross_prompt_suffixes",
        "capture_root",
    ],
)
def test_v2_does_not_change_old_design(key):
    v1 = v2.base._read(v2.base.REPO / v2.V1_CFG)
    current = v2.base._read(v2.base.REPO / v2.CFG)
    v2.same_design(v1, current)
    broken = copy.deepcopy(current)
    broken[key] = "wrong"
    with pytest.raises(ValueError):
        v2.same_design(v1, broken)


def test_profile_restores_v1_and_refuses_its_output_root(monkeypatch):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    before = (v2.base.CFG, v2.base.CODE, v2.base.qualify, v2.base.forced_stepper)
    with pytest.raises(ValueError, match="canonical"):
        v2.run(
            "capture", v2.base.REPO / Path("runs/native_suffix_replay_calibration_v1")
        )
    assert (
        v2.base.CFG,
        v2.base.CODE,
        v2.base.qualify,
        v2.base.forced_stepper,
    ) == before
