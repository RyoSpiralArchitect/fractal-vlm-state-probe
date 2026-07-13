from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .mlx_stream import clone_prompt_cache_state

CacheTensorName = Literal["keys", "values"]


@dataclass(frozen=True)
class CacheTensorSwapSpec:
    layer_index: int
    tensor: CacheTensorName = "values"
    require_token_match: bool = True
    require_shape_match: bool = True


def swap_prompt_cache_tensor(
    source_cache_state: Any,
    donor_cache_state: Any,
    spec: CacheTensorSwapSpec,
) -> tuple[Any, dict[str, Any]]:
    """Clone source cache state and replace one layer tensor with donor tensor."""
    source_entries = _cache_entries(source_cache_state, label="source")
    donor_entries = _cache_entries(donor_cache_state, label="donor")
    _validate_layer_index(spec.layer_index, source_entries, donor_entries)

    token_ids_equal = _token_ids(source_cache_state) == _token_ids(donor_cache_state)
    if spec.require_token_match and not token_ids_equal:
        raise ValueError("source and donor PromptCacheState token_ids differ")

    source_entry = source_entries[spec.layer_index]
    donor_entry = donor_entries[spec.layer_index]
    source_tensor = _cache_tensor(source_entry, spec.tensor, label="source")
    donor_tensor = _cache_tensor(donor_entry, spec.tensor, label="donor")
    source_shape = _tensor_shape(source_tensor)
    donor_shape = _tensor_shape(donor_tensor)
    if spec.require_shape_match and source_shape != donor_shape:
        raise ValueError(
            f"source and donor layer {spec.layer_index} {spec.tensor} shapes differ: "
            f"{source_shape} != {donor_shape}"
        )

    swapped = clone_prompt_cache_state(source_cache_state)
    if swapped is None:
        raise ValueError("could not clone source PromptCacheState for intervention")
    swapped_entries = _cache_entries(swapped, label="swapped")
    setattr(swapped_entries[spec.layer_index], spec.tensor, donor_tensor)

    return swapped, {
        "kind": "prompt_cache_tensor_swap",
        "layer_index": spec.layer_index,
        "tensor": spec.tensor,
        "require_token_match": spec.require_token_match,
        "require_shape_match": spec.require_shape_match,
        "token_ids_equal": token_ids_equal,
        "source_token_count": len(_token_ids(source_cache_state) or []),
        "donor_token_count": len(_token_ids(donor_cache_state) or []),
        "source_total_layers": len(source_entries),
        "donor_total_layers": len(donor_entries),
        "source_shape": source_shape,
        "donor_shape": donor_shape,
        "status": "applied",
    }


def _cache_entries(prompt_cache_state: Any, *, label: str) -> list[Any]:
    if prompt_cache_state is None or getattr(prompt_cache_state, "cache", None) is None:
        raise ValueError(f"{label} PromptCacheState has no cache")
    entries = list(prompt_cache_state.cache)
    if not entries:
        raise ValueError(f"{label} PromptCacheState cache is empty")
    return entries


def _validate_layer_index(layer_index: int, source_entries: list[Any], donor_entries: list[Any]) -> None:
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    max_layers = min(len(source_entries), len(donor_entries))
    if layer_index >= max_layers:
        raise ValueError(
            f"layer_index {layer_index} is out of range for source/donor caches "
            f"({len(source_entries)} / {len(donor_entries)} layers)"
        )


def _cache_tensor(entry: Any, tensor: CacheTensorName, *, label: str) -> Any:
    value = getattr(entry, tensor, None)
    if value is None:
        raise ValueError(f"{label} cache entry has no {tensor} tensor")
    return value


def _token_ids(prompt_cache_state: Any) -> list[int] | None:
    token_ids = getattr(prompt_cache_state, "token_ids", None)
    return list(token_ids) if token_ids is not None else None


def _tensor_shape(tensor: Any) -> list[int] | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    return [int(value) for value in shape]
