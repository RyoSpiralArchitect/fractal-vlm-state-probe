from __future__ import annotations

import hashlib

import numpy as np

from .frequency_control import CELLS, permute_cells, rgb_multiset_sha256
from .image_stats import _frame_image_stats
from .padding_policy import (
    marginal_audit,
    measure_processor,
    pad_to_square,
    padding_gate,
)

NUMERATORS = (0, 1, 2, 4, 6, 7, 8)
DENOMINATOR = 8


def validate_permutation(permutation: np.ndarray) -> None:
    if (
        permutation.ndim != 1
        or permutation.dtype.kind not in "iu"
        or not len(permutation)
        or not np.array_equal(np.sort(permutation), np.arange(len(permutation)))
    ):
        raise ValueError("expected a nonempty integer bijection")


def restricted_permutation(
    full: np.ndarray, selection_order: np.ndarray, selected_count: int
) -> np.ndarray:
    validate_permutation(full)
    validate_permutation(selection_order)
    if len(full) != len(selection_order) or not 0 <= selected_count <= len(full):
        raise ValueError("selection size differs from the full permutation")
    selected = np.zeros(len(full), dtype=bool)
    selected[selection_order[:selected_count]] = True
    result = np.arange(len(full))
    seen = np.zeros(len(full), dtype=bool)
    # Contract each original cycle through its unselected sites, fixing those sites.
    for start in range(len(full)):
        if seen[start]:
            continue
        cycle, position = [], start
        while not seen[position]:
            seen[position] = True
            if selected[position]:
                cycle.append(position)
            position = full[position]
        if cycle:
            result[cycle] = np.roll(cycle, -1)
    return result


def mapping_metrics(permutation: np.ndarray, shape: tuple[int, int]) -> dict:
    validate_permutation(permutation)
    height, width = shape
    if height < 2 or width < 2 or height * width != len(permutation):
        raise ValueError("mapping requires a matching two-dimensional grid")
    identity = np.arange(len(permutation))
    rows, columns = np.divmod(permutation.reshape(shape), width)
    dr = rows.ravel() - identity // width
    dc = columns.ravel() - identity % width
    distance = np.sqrt(dr * dr + dc * dc)
    horizontal = np.abs(np.diff(rows, axis=1)) + np.abs(np.diff(columns, axis=1))
    vertical = np.abs(np.diff(rows, axis=0)) + np.abs(np.diff(columns, axis=0))
    return {
        "moved_index_fraction": float(np.mean(permutation != identity)),
        "mean_displacement_pixels": float(distance.mean()),
        "mean_displacement_over_diagonal": float(
            distance.mean() / np.hypot(height - 1, width - 1)
        ),
        "retained_undirected_grid_edge_fraction": float(
            ((horizontal == 1).sum() + (vertical == 1).sum())
            / (horizontal.size + vertical.size)
        ),
    }


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def audit_level(
    originals: dict[str, np.ndarray],
    permutation: np.ndarray,
    fills: dict[str, np.ndarray],
    expected_multisets: dict[str, str],
    expected_payloads: dict[str, dict],
    *,
    processor,
    normalization: dict,
) -> tuple[dict, dict[str, np.ndarray]]:
    if set(originals) != set(CELLS):
        raise ValueError("graded permutation requires all four cells")
    content = permute_cells(originals, permutation)
    inverse = np.argsort(permutation)
    canvases, cells = {}, {}
    for cell in CELLS:
        rgb = content[cell]
        color = fills["mm" if cell in ("mm", "jm") else "jj"]
        canvas, bounds = pad_to_square(rgb, color)
        marginal = marginal_audit(rgb, canvas, color, bounds)
        expanded_hash = rgb_multiset_sha256(canvas)
        inverse_equal = np.array_equal(
            rgb.reshape(-1, 3)[inverse].reshape(rgb.shape), originals[cell]
        )
        if (
            expanded_hash != expected_multisets[cell]
            or not inverse_equal
            or not marginal["content_pixels_unchanged"]
            or not marginal["padding_pixels_equal_registered_fill"]
        ):
            raise ValueError("graded transform changed a fixed marginal or padding")
        measured, _ = measure_processor(canvas, processor, normalization)
        if measured["processor_non_pixel_payload"] != expected_payloads[cell]:
            raise ValueError("graded transform changed non-pixel processor fields")
        measured.update(
            {
                "raw": _frame_image_stats(canvas.astype(np.float64) / 255),
                "canvas_rgb_multiset_sha256": expanded_hash,
                "registered_expanded_palette_verified": True,
                "non_pixel_payload_unchanged": True,
                "inverse_roundtrip_exact": inverse_equal,
                "canvas_rgb_bytes_sha256": array_sha256(canvas),
                "content_rgb_bytes_sha256": array_sha256(rgb),
                "changed_rgb_site_fraction": float(
                    np.any(rgb != originals[cell], axis=2).mean()
                ),
                "marginal_audit": marginal,
            }
        )
        cells[cell], canvases[cell] = measured, canvas
    return {"cells": cells, "gate": padding_gate(cells)}, canvases
