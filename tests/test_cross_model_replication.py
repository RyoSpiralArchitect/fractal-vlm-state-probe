from __future__ import annotations

from pathlib import Path

import pytest

from fractal_vlm_state_probe.cross_model_replication import (
    analyze_cross_model_replication,
)


def test_cross_model_replication_separates_exact_and_component_stability() -> None:
    family = _trajectory(
        [
            _point("model_a_b_c_1f", "model/a", "b", "c", 33, "values", -10.0),
            _point("model_a_c_d_1f", "model/a", "c", "d", 33, "values", -20.0),
            _point("model_b_b_c_1f", "model/b", "b", "c", 1, "keys", 5.0),
            _point("model_b_c_d_1f", "model/b", "c", "d", 2, "keys", -6.0),
        ]
    )
    frequency = _trajectory(
        [
            _point("model_a_b_c_1f", "model/a", "b", "c", 33, "values", -10.0),
            _point("model_a_c_d_1f", "model/a", "c", "d", 33, "values", -20.0),
            _point("model_b_b_c_1f", "model/b", "b", "c", 1, "keys", 5.0),
            _point("model_b_c_d_1f", "model/b", "c", "d", 2, "keys", -6.0),
        ]
    )

    result = analyze_cross_model_replication(
        family,
        frequency,
        family_path=Path("family.json"),
        frequency_path=Path("frequency.json"),
    )

    summaries = {record["model_id"]: record for record in result["model_summaries"]}
    stable = summaries["model/a"]
    assert stable["exact_cache_location_consistent"] is True
    assert stable["cache_component_consistent"] is True
    assert stable["cache_interaction_sign_consistent"] is True
    assert stable["dominant_cache_location_share"] == pytest.approx(1)

    component_only = summaries["model/b"]
    assert component_only["exact_cache_location_consistent"] is False
    assert component_only["cache_component_consistent"] is True
    assert component_only["cache_interaction_sign_consistent"] is False
    assert component_only["dominant_cache_location_share"] == pytest.approx(0.5)
    source_pairs = {
        record["source_pair_id"]: record for record in result["source_pair_summaries"]
    }
    assert source_pairs["b_c"]["family_generated_pattern_consistent"] is True


def test_cross_model_replication_rejects_cache_location_mismatch() -> None:
    family = _trajectory(
        [_point("model_a_b_c_1f", "model/a", "b", "c", 3, "values", 1.0)]
    )
    frequency = _trajectory(
        [_point("model_a_b_c_1f", "model/a", "b", "c", 4, "values", 1.0)]
    )

    with pytest.raises(ValueError, match="cache argmax mismatch"):
        analyze_cross_model_replication(
            family,
            frequency,
            family_path=Path("family.json"),
            frequency_path=Path("frequency.json"),
        )


def _trajectory(points: list[dict]) -> dict:
    return {"points": points}


def _point(
    label: str,
    model: str,
    mm: str,
    jj: str,
    layer: int,
    tensor: str,
    interaction: float,
) -> dict:
    return {
        "label": label,
        "series_id": (f"{model}__mandelbrot_zoom_{mm}_50f__julia_zoom_{jj}_50f"),
        "model_id": model,
        "frame_count": 1,
        "scalar_argmax": {
            "layer_index": layer,
            "tensor": tensor,
            "interaction_effect": interaction,
            "relative_abs_interaction": 0.1,
            "normalized_depth": layer / 40,
        },
        "family_labels": {"mm": " A", "jj": "A", "mj": "A", "jm": "A"},
        "readout_cell_invariant": True,
        "full_vocab_readout": {
            "max_pair_jensen_shannon": 0.01,
            "interaction_l1_norm": abs(interaction) / 100,
            "interaction_max_abs": abs(interaction) / 200,
            "interaction_argmax_token": "A",
        },
    }
