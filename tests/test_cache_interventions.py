from __future__ import annotations

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_interventions import (
    CacheTensorSwapSpec,
    swap_prompt_cache_tensor,
)


class CacheEntry:
    def __init__(self, keys: np.ndarray, values: np.ndarray) -> None:
        self.keys = keys
        self.values = values


class PromptCacheState:
    def __init__(self) -> None:
        self.token_ids = None
        self.cache = None


def test_swap_prompt_cache_tensor_replaces_only_target_values() -> None:
    source_value = np.ones((1, 2, 3, 4))
    donor_value = np.full((1, 2, 3, 4), 7.0)
    source = _state(
        [
            CacheEntry(np.zeros((1, 2, 3, 4)), np.zeros((1, 2, 3, 4))),
            CacheEntry(np.full((1, 2, 3, 4), 3.0), source_value),
        ]
    )
    donor = _state(
        [
            CacheEntry(np.zeros((1, 2, 3, 4)), np.zeros((1, 2, 3, 4))),
            CacheEntry(np.full((1, 2, 3, 4), 9.0), donor_value),
        ]
    )

    swapped, record = swap_prompt_cache_tensor(
        source,
        donor,
        CacheTensorSwapSpec(layer_index=1, tensor="values"),
    )

    assert swapped is not source
    assert swapped.cache[1] is not source.cache[1]
    assert swapped.cache[1].values is donor_value
    assert swapped.cache[1].keys is source.cache[1].keys
    assert source.cache[1].values is source_value
    assert record["status"] == "applied"
    assert record["layer_index"] == 1
    assert record["tensor"] == "values"
    assert record["token_ids_equal"] is True
    assert record["source_shape"] == [1, 2, 3, 4]


def test_swap_prompt_cache_tensor_rejects_token_mismatch() -> None:
    source = _state([CacheEntry(np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1, 1)))], token_ids=[1])
    donor = _state([CacheEntry(np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1, 1)))], token_ids=[2])

    with pytest.raises(ValueError, match="token_ids differ"):
        swap_prompt_cache_tensor(source, donor, CacheTensorSwapSpec(layer_index=0))


def test_swap_prompt_cache_tensor_rejects_shape_mismatch() -> None:
    source = _state([CacheEntry(np.zeros((1, 1, 2, 1)), np.zeros((1, 1, 2, 1)))])
    donor = _state([CacheEntry(np.zeros((1, 1, 3, 1)), np.zeros((1, 1, 3, 1)))])

    with pytest.raises(ValueError, match="shapes differ"):
        swap_prompt_cache_tensor(source, donor, CacheTensorSwapSpec(layer_index=0))


def _state(entries: list[CacheEntry], *, token_ids: list[int] | None = None) -> PromptCacheState:
    state = PromptCacheState()
    state.token_ids = token_ids if token_ids is not None else [1, 2, 3]
    state.cache = entries
    return state
