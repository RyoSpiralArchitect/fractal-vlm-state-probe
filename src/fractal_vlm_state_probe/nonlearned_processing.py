from __future__ import annotations

import numpy as np

FEATURES = (
    "global_moments",
    "grid_mean",
    "small_mean",
    "small_energy",
    "large_mean",
    "large_energy",
)
STARTS = ("raw_rgb", "processor_pixel_values")


def checked_image(values):
    a = np.asarray(values)
    if a.ndim != 3 or a.shape[-1] != 3 or not a.size or not np.isfinite(a).all():
        raise ValueError("expected finite HWC three-channel image")
    return a.astype(np.float64, copy=False)


def box_mean(values, width):
    a = checked_image(values)
    if not isinstance(width, int) or width < 1 or width % 2 != 1:
        raise ValueError("box width must be a positive odd integer")
    radius = width // 2
    for axis in (0, 1):
        padding = [(0, 0)] * 3
        padding[axis] = (radius, radius)
        padded = np.pad(a, padding, mode="edge")
        sums = np.cumsum(padded, axis=axis, dtype=np.float64)
        zero = list(sums.shape)
        zero[axis] = 1
        sums = np.concatenate((np.zeros(zero), sums), axis=axis)
        a = (
            np.take(sums, np.arange(width, sums.shape[axis]), axis=axis)
            - np.take(sums, np.arange(sums.shape[axis] - width), axis=axis)
        ) / width
    return a


def grid_pool(values, grid):
    a = checked_image(values)
    rows, cols = grid
    h, w, _ = a.shape
    if rows <= 0 or cols <= 0 or h % rows or w % cols:
        raise ValueError("equal-area grid must divide both image dimensions")
    return a.reshape(rows, h // rows, cols, w // cols, 3).mean(axis=(1, 3)).ravel()


def local_features(values, widths, grid):
    a = checked_image(values)
    result = {"grid_mean": grid_pool(a, grid)}
    for name in ("small", "large"):
        mixed = box_mean(a, widths[name])
        result[f"{name}_mean"] = grid_pool(mixed, grid)
        result[f"{name}_energy"] = grid_pool(mixed * mixed, grid)
    return result


def extract_features(values, start, config):
    if start not in STARTS:
        raise ValueError("unregistered feature start")
    source = np.asarray(values)
    if start == "raw_rgb":
        if source.dtype != np.uint8:
            raise ValueError("raw features require uint8 RGB")
        checked_image(source)
        integers = source.astype(np.uint64)
        count = source.shape[0] * source.shape[1]
        moments = np.concatenate(
            (
                integers.sum(axis=(0, 1), dtype=np.uint64).astype(np.float64)
                / (count * 255),
                (integers * integers)
                .sum(axis=(0, 1), dtype=np.uint64)
                .astype(np.float64)
                / (count * 255**2),
            )
        )
        a = source.astype(np.float64) / 255
    else:
        if source.ndim != 4 or source.shape[:2] != (1, 3) or source.dtype != np.float32:
            raise ValueError("processor features require native NCHW float32")
        a = checked_image(np.moveaxis(source[0], 0, -1))
        moments = np.concatenate((a.mean(axis=(0, 1)), (a * a).mean(axis=(0, 1))))
    out = {
        "global_moments": moments,
        **local_features(a, config["windows"][start], config["grid"]),
    }
    if set(out) != set(FEATURES) or any(not np.isfinite(v).all() for v in out.values()):
        raise ValueError("feature coverage or finiteness differs")
    return {k: out[k] for k in FEATURES}


def factorial_interaction(cells):
    if set(cells) != {"mm", "jj", "mj", "jm"}:
        raise ValueError("all four feature cells required")
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in cells.items()}
    if len({v.shape for v in arrays.values()}) != 1 or any(
        not np.isfinite(v).all() for v in arrays.values()
    ):
        raise ValueError("factorial shapes or finiteness differ")
    return (arrays["jj"] - arrays["mj"]) - (arrays["jm"] - arrays["mm"])


def direction_status(vector):
    a = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite direction")
    norm = float(np.linalg.norm(a))
    if not np.isfinite(norm) or (norm == 0 and np.any(a)):
        raise ValueError("direction norm overflow or underflow")
    return {
        "available": norm != 0,
        "norm": norm,
        "reason": None if norm else "exact_zero_direction_unavailable",
    }
