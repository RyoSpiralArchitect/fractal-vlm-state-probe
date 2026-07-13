from __future__ import annotations

from fractal_vlm_state_probe.cache_prefix_audit import analyze_cache_prefix_audits


def test_cache_prefix_audit_counts_unsafe_multimodal_reuse() -> None:
    analysis = analyze_cache_prefix_audits(
        {
            "qwen": {
                "model_id": "example/qwen",
                "stream_events": [],
                "probes": {
                    "after": [
                        {
                            "probe_id": "forced",
                            "cache_prefix_audit": {
                                "available": True,
                                "source_token_count": 241,
                                "formatted_prompt_token_count": 198,
                                "common_prefix_token_count": 42,
                                "common_prefix_fraction": 42 / 241,
                                "cache_sequence_lengths": [256],
                                "token_cache_length_aligned": False,
                                "reuse_safe_under_token_prefix_contract": False,
                            },
                        }
                    ]
                },
            }
        }
    )

    assert analysis["available_record_count"] == 1
    assert analysis["safe_record_count"] == 0
    assert analysis["unsafe_record_count"] == 1
    assert analysis["records"][0]["location"] == "probe:after:forced"
