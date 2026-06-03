from __future__ import annotations

from fractal_vlm_state_probe.mlx_stream import (
    _mid_probe_after_position,
    _should_summarize_cache,
)


def test_mid_probe_after_position_consumes_half_stream() -> None:
    assert _mid_probe_after_position(0) is None
    assert _mid_probe_after_position(1) == 0
    assert _mid_probe_after_position(11) == 5
    assert _mid_probe_after_position(12) == 5


def test_should_summarize_cache_keeps_key_positions() -> None:
    assert _should_summarize_cache(0, frame_count=12, every=10)
    assert _should_summarize_cache(5, frame_count=12, every=10)
    assert _should_summarize_cache(10, frame_count=12, every=10)
    assert _should_summarize_cache(11, frame_count=12, every=10)
    assert not _should_summarize_cache(4, frame_count=12, every=10)
    assert not _should_summarize_cache(0, frame_count=12, every=0)
