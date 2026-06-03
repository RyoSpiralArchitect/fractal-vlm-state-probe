from __future__ import annotations

from typing import Any


def select_layer_indices(total_layers: int, max_layers: int | None) -> list[int]:
    if total_layers <= 0:
        return []
    if max_layers is None or max_layers >= total_layers:
        return list(range(total_layers))
    if max_layers < 0:
        raise ValueError("max_layers must be non-negative or None")
    if max_layers == 0:
        return []
    if max_layers == 1:
        return [total_layers - 1]

    # Spread the sample across the stack, preserving first and final layers.
    denominator = max_layers - 1
    return sorted({round(index * (total_layers - 1) / denominator) for index in range(max_layers)})


def select_sequence_positions(sequence_length: int) -> list[int]:
    if sequence_length <= 0:
        return []
    if sequence_length == 1:
        return [0]
    positions = {0, sequence_length - 1, sequence_length // 2}
    if sequence_length > 3:
        positions.add(max(0, sequence_length - 2))
    return sorted(positions)


def json_safe_number(value: Any) -> float | int | str:
    try:
        return float(value)
    except Exception:
        return str(value)
