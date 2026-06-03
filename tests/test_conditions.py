from __future__ import annotations

import pytest

from fractal_vlm_state_probe.conditions import (
    StimulusCondition,
    condition_from_config,
)


def test_condition_from_config_uses_explicit_metadata() -> None:
    condition = condition_from_config(
        {
            "stimulus_condition": {
                "condition_id": "natural_cat_ordered",
                "condition_family": "natural",
                "temporal_policy": "ordered",
                "semantic_load": "high",
                "deterministic": False,
                "source_kind": "external_frames",
            }
        }
    )
    assert condition.condition_family == "natural"
    assert condition.semantic_load == "high"


def test_condition_rejects_invalid_family() -> None:
    with pytest.raises(ValueError, match="unsupported condition_family"):
        StimulusCondition.from_dict(
            {
                "condition_id": "bad",
                "condition_family": "dream",
                "temporal_policy": "ordered",
                "semantic_load": "low",
                "deterministic": True,
                "source_kind": "generated",
            }
        )

