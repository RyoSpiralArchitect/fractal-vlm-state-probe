from __future__ import annotations

from fractal_vlm_state_probe.compare_runs import compare_runs, format_comparison_markdown


def test_compare_runs_marks_matched_context() -> None:
    left_cache = _cache_summary(keys_l2=10.0, values_l2=3.0)
    right_cache = _cache_summary(keys_l2=14.5, values_l2=4.0)
    base = {
        "run_kind": "fractal_vlm_stream_probe",
        "model_id": "model-a",
        "adapter_capabilities": {"adapter_id": "mlx_vlm"},
        "reproducibility": {"seed": 7},
        "context_policy": {"probe_cache_policy": "isolated"},
        "stimulus_delivery": {"mode": "visual_stream"},
        "stimulus": {
            "frame_count_selected": 2,
            "source_frame_count_selected": 2,
            "condition": {"condition_id": "left", "condition_family": "fractal"},
        },
        "probes": {
            "before": [
                {
                    "probe_id": "p",
                    "assistant_text": "left text",
                    "generation": {"generation_tokens": 2},
                    "source_cache_summary_before_probe": left_cache,
                }
            ]
        },
        "stream_events": [
            {"frame_index": 0, "cache_summary": left_cache},
        ],
    }
    right = {
        **base,
        "stimulus": {
            "frame_count_selected": 2,
            "source_frame_count_selected": 2,
            "condition": {"condition_id": "right", "condition_family": "fractal"},
        },
        "stimulus_delivery": {"mode": "text_only_stream"},
        "probes": {
            "before": [
                {
                    "probe_id": "p",
                    "assistant_text": "right text",
                    "generation": {"generation_tokens": 2},
                    "source_cache_summary_before_probe": right_cache,
                }
            ]
        },
        "stream_events": [
            {"frame_index": 0, "cache_summary": right_cache},
        ],
    }
    comparison = compare_runs(base, right)
    assert comparison["matched_context"]["same_seed"]
    assert not comparison["matched_context"]["same_delivery_mode"]
    assert not comparison["probe_comparison"][0]["same_text"]
    assert comparison["stream_cache_comparison"][0]["cache_stat_delta"]["max_abs_l2_delta"] == 4.5
    assert (
        comparison["stream_cache_comparison"][0]["cache_stat_delta"][
            "max_abs_sequence_position_l2_delta"
        ]
        == 2.25
    )
    assert comparison["probe_source_cache_comparison"][0]["source_cache_delta"]["max_abs_l2_delta"] == 4.5
    markdown = format_comparison_markdown(comparison)
    assert "left text" in markdown
    assert "right text" in markdown
    assert "max_abs_l2=4.5" in markdown
    assert "top_sequence_l2_delta" in markdown
    assert "visual_stream" in markdown
    assert "text_only_stream" in markdown


def test_compare_runs_reports_first_step_top_k_readout_delta() -> None:
    base = {
        "run_kind": "hf_vlm_stream_probe",
        "model_ref": "model-a",
        "adapter_capabilities": {"adapter_id": "hf_transformers"},
        "reproducibility": {"seed": 7},
        "context_policy": {"probe_cache_policy": "isolated"},
        "stimulus_delivery": {"mode": "visual_stream"},
        "stimulus": {
            "frame_count_selected": 2,
            "source_frame_count_selected": 2,
            "condition": {"condition_id": "left", "condition_family": "fractal"},
        },
        "probes": {
            "after": [
                {
                    "probe_id": "choice",
                    "assistant_text": "A",
                    "generation": {
                        "generation_tokens": 1,
                        "steps": [
                            {
                                "step_index": 0,
                                "top_logprobs": [
                                    {"token_id": 10, "token": "A", "logprob": -0.1},
                                    {"token_id": 11, "token": "B", "logprob": -1.5},
                                ],
                            }
                        ],
                    },
                }
            ]
        },
        "stream_events": [],
    }
    right = {
        **base,
        "stimulus": {
            "frame_count_selected": 2,
            "source_frame_count_selected": 2,
            "condition": {"condition_id": "right", "condition_family": "fractal"},
        },
        "probes": {
            "after": [
                {
                    "probe_id": "choice",
                    "assistant_text": "A",
                    "generation": {
                        "generation_tokens": 1,
                        "steps": [
                            {
                                "step_index": 0,
                                "top_logprobs": [
                                    {"token_id": 10, "token": "A", "logprob": -0.4},
                                    {"token_id": 12, "token": "C", "logprob": -1.2},
                                ],
                            }
                        ],
                    },
                }
            ]
        },
    }

    comparison = compare_runs(base, right)
    readout = comparison["probe_comparison"][0]["generation_readout_delta"]

    assert readout["available"] is True
    assert readout["top1_same_token_id"] is True
    assert readout["top_k_jaccard"] == 1 / 3
    assert readout["max_abs_shared_logprob_delta"] == 0.30000000000000004
    assert "First-step top-k readout" in format_comparison_markdown(comparison)


def _cache_summary(*, keys_l2: float, values_l2: float) -> dict:
    return {
        "available": True,
        "token_count": 10,
        "layers": [
            {
                "layer_index": 0,
                "keys": {
                    "mean": 0.0,
                    "variance": 1.0,
                    "std": 1.0,
                    "abs_mean": 0.5,
                    "l2_norm": keys_l2,
                    "sequence_position_stats": [
                        {
                            "position": 0,
                            "mean": 0.0,
                            "variance": 1.0,
                            "abs_mean": 0.5,
                            "l2_norm": keys_l2 / 2,
                        }
                    ],
                },
                "values": {
                    "mean": 0.0,
                    "variance": 1.0,
                    "std": 1.0,
                    "abs_mean": 0.5,
                    "l2_norm": values_l2,
                    "sequence_position_stats": [
                        {
                            "position": 0,
                            "mean": 0.0,
                            "variance": 1.0,
                            "abs_mean": 0.5,
                            "l2_norm": values_l2 / 2,
                        }
                    ],
                },
            }
        ],
    }
