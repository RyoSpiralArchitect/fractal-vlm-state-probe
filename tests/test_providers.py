from __future__ import annotations

import pytest

from fractal_vlm_state_probe.providers import get_capabilities


def test_provider_capabilities_include_tiers() -> None:
    assert get_capabilities("mlx_vlm").tier == "T1_LOCAL_INTERNAL_TRACE"
    assert get_capabilities("openai_logprob").tier == "T2_REMOTE_LOGPROB_PROBE"
    assert (
        get_capabilities("instrumented_openai_compatible").tier
        == "T3_INSTRUMENTED_SERVING"
    )


def test_unknown_provider_capability_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown adapter_id"):
        get_capabilities("missing")

