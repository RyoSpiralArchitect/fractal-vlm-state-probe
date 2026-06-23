from __future__ import annotations

from fractal_vlm_state_probe.probe_readout import analyze_first_token_readout_contrast


def test_first_token_readout_contrast_uses_common_top_k_tokens() -> None:
    analysis = analyze_first_token_readout_contrast(
        mm=_run("mandelbrot", -0.1, generated="A"),
        jj=_run("julia", -0.4, generated="B"),
        mj=_run("m_spatial_j_palette", -0.2, generated="A"),
        jm=_run("j_spatial_m_palette", -0.6, generated="B"),
    )

    record = analysis["records"][0]
    assert record["available"] is True
    assert record["mean_top_k_jaccard"] == 1.0
    assert record["common_top_k_token_count"] == 2
    assert record["generated_tokens"]["mm"]["token"] == "A"

    top = record["top_common_token_effects"][0]
    assert top["token_id"] == 1
    assert round(top["interaction_effect"], 3) == 0.3
    assert analysis["top_common_token_effects"][0]["phase"] == "after"


def test_first_token_readout_contrast_marks_old_runs_unavailable() -> None:
    analysis = analyze_first_token_readout_contrast(
        mm=_old_run("mandelbrot"),
        jj=_old_run("julia"),
        mj=_old_run("m_spatial_j_palette"),
        jm=_old_run("j_spatial_m_palette"),
    )

    record = analysis["records"][0]
    assert record["available"] is False
    assert record["missing_cells"] == ["mm", "jj", "mj", "jm"]
    assert record["top_common_token_effects"] == []


def _run(condition_id: str, token_1_logprob: float, *, generated: str) -> dict:
    generated_id = 1 if generated == "A" else 2
    return {
        "stimulus": {"condition": {"condition_id": condition_id}},
        "probes": {
            "after": [
                {
                    "probe_id": "forced_family_choice",
                    "generation": {
                        "steps": [
                            {
                                "step_index": 0,
                                "token_id": generated_id,
                                "token": generated,
                                "token_logprob": token_1_logprob if generated_id == 1 else -1.0,
                                "top_logprobs": [
                                    {
                                        "token_id": 1,
                                        "token": "A",
                                        "logprob": token_1_logprob,
                                    },
                                    {
                                        "token_id": 2,
                                        "token": "B",
                                        "logprob": -1.0,
                                    },
                                ],
                            }
                        ]
                    },
                }
            ]
        },
    }


def _old_run(condition_id: str) -> dict:
    return {
        "stimulus": {"condition": {"condition_id": condition_id}},
        "probes": {
            "after": [
                {
                    "probe_id": "forced_family_choice",
                    "generation": {"prompt_tokens": 10, "generation_tokens": 1},
                }
            ]
        },
    }
