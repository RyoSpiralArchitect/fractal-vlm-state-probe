from __future__ import annotations

from fractal_vlm_state_probe.factorial_contrast import (
    analyze_factorial_cache_contrast,
    format_factorial_contrast_markdown,
)


def test_factorial_cache_contrast_reports_main_and_interaction_effects() -> None:
    analysis = analyze_factorial_cache_contrast(
        mm=_run("mm", keys_l2=10.0, values_l2=5.0),
        jj=_run("jj", keys_l2=16.0, values_l2=11.0),
        mj=_run("mj", keys_l2=13.0, values_l2=6.0),
        jm=_run("jm", keys_l2=11.0, values_l2=10.0),
    )

    keys_l2 = [
        record
        for record in analysis["scalar_records"]
        if record["phase"] == "mid"
        and record["probe_id"] == "choice"
        and record["tensor"] == "keys"
        and record["field"] == "l2_norm"
    ][0]

    assert keys_l2["spatial_main_effect"] == 2.0
    assert keys_l2["palette_main_effect"] == 4.0
    assert keys_l2["interaction_effect"] == 2.0
    assert analysis["interaction_argmax_summary"][0]["same_location"] is True
    assert "2x2 Factorial Cache Contrast" in format_factorial_contrast_markdown(analysis)


def _run(condition_id: str, *, keys_l2: float, values_l2: float) -> dict:
    return {
        "run_kind": "fractal_vlm_stream_probe",
        "model_id": "model-a",
        "reproducibility": {"seed": 7, "probe_seed": 0},
        "stimulus": {
            "condition": {"condition_id": condition_id},
        },
        "probes": {
            "mid": [_probe("choice", keys_l2=keys_l2, values_l2=values_l2)],
            "after": [_probe("choice", keys_l2=keys_l2 + 1.0, values_l2=values_l2 + 1.0)],
        },
    }


def _probe(probe_id: str, *, keys_l2: float, values_l2: float) -> dict:
    return {
        "probe_id": probe_id,
        "source_cache_summary_before_probe": {
            "available": True,
            "token_count": 10,
            "layers": [
                {
                    "layer_index": 0,
                    "keys": _tensor(keys_l2),
                    "values": _tensor(values_l2),
                }
            ],
        },
    }


def _tensor(l2_norm: float) -> dict:
    return {
        "mean": 0.0,
        "variance": 1.0,
        "std": 1.0,
        "abs_mean": 0.5,
        "l2_norm": l2_norm,
        "sequence_position_stats": [
            {
                "position": 0,
                "mean": 0.0,
                "variance": 1.0,
                "abs_mean": 0.5,
                "l2_norm": l2_norm / 2.0,
            }
        ],
    }
