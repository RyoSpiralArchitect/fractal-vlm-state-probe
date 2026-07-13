from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .cache_tensor_artifact import load_cache_tensor_artifact
from .probe_readout import CELL_KEYS
from .stimulus import write_json


REGION_ORDER = (
    "all_effective",
    "image_tokens",
    "non_image_tokens",
    "pre_image",
    "post_image",
)


def analyze_cache_tensor_factorial(
    *,
    runs: dict[str, dict[str, Any]],
    run_paths: dict[str, Path],
    layer_index: int,
    tensor: str = "values",
) -> dict[str, Any]:
    if set(runs) != set(CELL_KEYS) or set(run_paths) != set(CELL_KEYS):
        raise ValueError(
            f"runs and run_paths must contain exactly: {', '.join(CELL_KEYS)}"
        )
    if tensor not in {"keys", "values"}:
        raise ValueError("tensor must be keys or values")

    model_ids = {str(run.get("model_id")) for run in runs.values()}
    if len(model_ids) != 1:
        raise ValueError("all factorial cells must use the same model")

    artifacts = {
        cell: _find_artifact(runs[cell], layer_index=layer_index, tensor=tensor)
        for cell in CELL_KEYS
    }
    arrays = {
        cell: load_cache_tensor_artifact(run_paths[cell], artifacts[cell])
        for cell in CELL_KEYS
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"cache tensor shapes do not align: {sorted(shapes)}")
    shape = next(iter(shapes))
    if len(shape) < 3:
        raise ValueError(f"cache tensor has no sequence axis at -2: {shape}")

    layouts = {
        cell: _source_event(runs[cell]).get("cache_token_layout") or {}
        for cell in CELL_KEYS
    }
    regions_by_cell = {
        cell: cache_tensor_regions(layouts[cell], sequence_length=shape[-2])
        for cell in CELL_KEYS
    }
    reference_regions = regions_by_cell[CELL_KEYS[0]]
    for cell in CELL_KEYS[1:]:
        if regions_by_cell[cell] != reference_regions:
            raise ValueError(f"cache token regions do not align for cell {cell}")

    region_records = []
    for region_name in REGION_ORDER:
        positions = reference_regions.get(region_name) or []
        if not positions:
            continue
        cell_arrays = {
            cell: np.take(arrays[cell], positions, axis=-2).astype(
                np.float64, copy=False
            )
            for cell in CELL_KEYS
        }
        effects = factorial_effect_vectors(cell_arrays)
        balanced_contrasts = balanced_factorial_contrast_vectors(cell_arrays)
        pairwise = _pairwise_distances(cell_arrays)
        mean_cell_rms = float(
            np.mean([_rms(values) for values in cell_arrays.values()])
        )
        mean_pairwise_rms = float(np.mean([record["rms"] for record in pairwise]))
        effect_metrics = {
            name: _vector_metrics(
                values,
                positions=positions,
                mean_cell_rms=mean_cell_rms,
                mean_pairwise_rms=mean_pairwise_rms,
            )
            for name, values in effects.items()
        }
        balanced_contrast_metrics = {
            name: _vector_metrics(
                values,
                positions=positions,
                mean_cell_rms=mean_cell_rms,
                mean_pairwise_rms=mean_pairwise_rms,
            )
            for name, values in balanced_contrasts.items()
        }
        region_records.append(
            {
                "region": region_name,
                "sequence_position_count": len(positions),
                "element_count": int(next(iter(cell_arrays.values())).size),
                "first_sequence_position": positions[0],
                "last_sequence_position": positions[-1],
                "mean_cell_rms": mean_cell_rms,
                "mean_pairwise_rms": mean_pairwise_rms,
                "pairwise_distances": pairwise,
                "effects": effect_metrics,
                "balanced_contrasts": balanced_contrast_metrics,
                "balanced_contrast_energy": _balanced_contrast_energy(
                    balanced_contrast_metrics
                ),
                "effect_direction_cosines": {
                    "spatial_vs_palette": _cosine(
                        effects["spatial_main_effect"],
                        effects["palette_main_effect"],
                    ),
                    "spatial_vs_interaction": _cosine(
                        effects["spatial_main_effect"], effects["interaction"]
                    ),
                    "palette_vs_interaction": _cosine(
                        effects["palette_main_effect"], effects["interaction"]
                    ),
                },
            }
        )

    by_region = {record["region"]: record for record in region_records}
    partition = _interaction_partition(by_region)
    return {
        "schema_version": 1,
        "analysis_kind": "source_cache_tensor_factorial_contrast",
        "model_id": model_ids.pop(),
        "source_pair_id": _source_pair_id(runs),
        "layer_index": layer_index,
        "tensor": tensor,
        "tensor_shape": list(shape),
        "sequence_axis": -2,
        "cell_semantics": {
            "mm": "Mandelbrot spatial x Mandelbrot palette",
            "jj": "Julia spatial x Julia palette",
            "mj": "Mandelbrot spatial x Julia palette",
            "jm": "Julia spatial x Mandelbrot palette",
        },
        "cells": {
            cell: {
                "source_path": str(run_paths[cell]),
                "condition_id": (
                    (runs[cell].get("stimulus") or {}).get("condition") or {}
                ).get("condition_id"),
                "cache_tensor_artifact": artifacts[cell],
                "cache_token_layout": layouts[cell],
            }
            for cell in CELL_KEYS
        },
        "contrast_formulas": {
            "spatial_main_effect": "((JM - MM) + (JJ - MJ)) / 2",
            "palette_main_effect": "((MJ - MM) + (JJ - JM)) / 2",
            "interaction": "JJ - JM - MJ + MM",
        },
        "balanced_contrast_formulas": {
            "spatial_contrast": "JJ + JM - MM - MJ",
            "palette_contrast": "JJ + MJ - MM - JM",
            "interaction_contrast": "JJ + MM - JM - MJ",
        },
        "regions": region_records,
        "interaction_partition": partition,
        "interpretation_notes": [
            "Tensor sidecars are trimmed to each cache entry's effective offset before analysis.",
            "When available, image and non-image regions use processor-expanded image-token positions from the saved source token layout.",
            "If no image-token positions are identified, only the complete effective region is reported and raw positions remain role-unassigned.",
            "RMS permits region-size comparison; L2 retains total interaction energy and therefore scales with element count.",
            "Balanced spatial, palette, and interaction contrasts all use +/-1 cell coefficients; their energy shares sum to one when any cell contrast is nonzero.",
            "An interaction energy share of 1/3 is the exchangeable isotropic reference across the three balanced factorial axes, not a fitted null distribution.",
            "These are full source-cache tensor contrasts from fresh ACK forwards, not cache-reuse or intervention effects.",
        ],
    }


def cache_tensor_regions(
    token_layout: dict[str, Any],
    *,
    sequence_length: int,
) -> dict[str, list[int]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    image_positions = []
    for run in token_layout.get("image_token_runs") or []:
        start = int(run["start"])
        end = int(run["end"])
        if start < 0 or end < start or end >= sequence_length:
            raise ValueError(
                f"invalid image-token run {start}:{end} for length {sequence_length}"
            )
        image_positions.extend(range(start, end + 1))
    image_positions = sorted(set(image_positions))
    image_set = set(image_positions)
    all_positions = list(range(sequence_length))
    if image_positions:
        non_image = [
            position for position in all_positions if position not in image_set
        ]
        first_image = image_positions[0]
        last_image = image_positions[-1]
        pre_image = list(range(first_image))
        post_image = list(range(last_image + 1, sequence_length))
    else:
        non_image = []
        pre_image = []
        post_image = []
    return {
        "all_effective": all_positions,
        "image_tokens": image_positions,
        "non_image_tokens": non_image,
        "pre_image": pre_image,
        "post_image": post_image,
    }


def factorial_effect_vectors(
    cells: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if set(cells) != set(CELL_KEYS):
        raise ValueError(f"cells must contain exactly: {', '.join(CELL_KEYS)}")
    mm = cells["mm"]
    jj = cells["jj"]
    mj = cells["mj"]
    jm = cells["jm"]
    return {
        "spatial_main_effect": ((jm - mm) + (jj - mj)) / 2.0,
        "palette_main_effect": ((mj - mm) + (jj - jm)) / 2.0,
        "interaction": jj - jm - mj + mm,
    }


def balanced_factorial_contrast_vectors(
    cells: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    effects = factorial_effect_vectors(cells)
    return {
        "spatial_contrast": 2.0 * effects["spatial_main_effect"],
        "palette_contrast": 2.0 * effects["palette_main_effect"],
        "interaction_contrast": effects["interaction"],
    }


def write_cache_tensor_factorial_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_cache_tensor_factorial_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_cache_tensor_factorial_markdown(analysis), encoding="utf-8")


def format_cache_tensor_factorial_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Source-Cache Tensor Factorial Contrast",
        "",
        f"- Model: `{analysis['model_id']}`",
        f"- Target: layer `{analysis['layer_index']}` `{analysis['tensor']}`",
        f"- Shape: `{analysis['tensor_shape']}`",
        "",
        "## Region Effects",
        "",
        "| Region | Positions | Elements | Spatial RMS | Palette RMS | Interaction RMS | Interaction / pairwise | Interaction max abs | Argmax position |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in analysis["regions"]:
        effects = record["effects"]
        interaction = effects["interaction"]
        lines.append(
            f"| `{record['region']}` | {record['sequence_position_count']} | "
            f"{record['element_count']} | "
            f"{_fmt(effects['spatial_main_effect']['rms'])} | "
            f"{_fmt(effects['palette_main_effect']['rms'])} | "
            f"{_fmt(interaction['rms'])} | "
            f"{_fmt(interaction['relative_rms_to_mean_pairwise_rms'])} | "
            f"{_fmt(interaction['max_abs'])} | "
            f"{interaction['argmax_sequence_position']} |"
        )

    partition = analysis.get("interaction_partition") or {}
    lines.extend(
        [
            "",
            "## Interaction Partition",
            "",
            f"- Image interaction energy fraction: `{_fmt(partition.get('image_energy_fraction'))}`",
            f"- Non-image interaction energy fraction: `{_fmt(partition.get('non_image_energy_fraction'))}`",
            f"- Partition closes: `{partition.get('partition_closes')}`",
            "",
            "## Balanced Contrast Calibration",
            "",
            "All three contrasts below use the same `+1/+1/-1/-1` coefficient scale.",
            "The exchangeable isotropic reference share is `0.33333333`.",
            "",
            "| Region | Spatial energy share | Palette energy share | Interaction energy share | Interaction minus 1/3 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in analysis["regions"]:
        calibration = record["balanced_contrast_energy"]
        shares = calibration["energy_shares"]
        lines.append(
            f"| `{record['region']}` | {_fmt(shares['spatial_contrast'])} | "
            f"{_fmt(shares['palette_contrast'])} | "
            f"{_fmt(shares['interaction_contrast'])} | "
            f"{_fmt(calibration['interaction_share_minus_exchangeable_reference'])} |"
        )

    lines.extend(
        [
            "",
            "## Direction Cosines",
            "",
            "| Region | Spatial vs palette | Spatial vs interaction | Palette vs interaction |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for record in analysis["regions"]:
        cosines = record["effect_direction_cosines"]
        lines.append(
            f"| `{record['region']}` | {_fmt(cosines['spatial_vs_palette'])} | "
            f"{_fmt(cosines['spatial_vs_interaction'])} | "
            f"{_fmt(cosines['palette_vs_interaction'])} |"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _source_event(run: dict[str, Any]) -> dict[str, Any]:
    events = run.get("stream_events") or []
    if len(events) != 1:
        raise ValueError(
            "cache tensor analysis requires exactly one source replay event"
        )
    return events[0]


def _source_pair_id(runs: dict[str, dict[str, Any]]) -> str:
    condition_ids = {}
    for cell in ("mm", "jj"):
        condition = ((runs[cell].get("stimulus") or {}).get("condition") or {}).get(
            "condition_id"
        )
        condition_ids[cell] = str(condition)
    mm = condition_ids["mm"].removeprefix("mandelbrot_zoom_").removesuffix("_50f")
    jj = condition_ids["jj"].removeprefix("julia_zoom_").removesuffix("_50f")
    return f"{mm}_{jj}"


def _find_artifact(
    run: dict[str, Any],
    *,
    layer_index: int,
    tensor: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in _source_event(run).get("cache_tensor_artifacts") or []
        if int(record.get("layer_index", -1)) == layer_index
        and record.get("tensor") == tensor
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one cache tensor artifact for layer {layer_index} {tensor}, found {len(matches)}"
        )
    return matches[0]


def _pairwise_distances(cells: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    records = []
    for left, right in combinations(CELL_KEYS, 2):
        delta = cells[left] - cells[right]
        records.append(
            {
                "left": left,
                "right": right,
                "l2_norm": float(np.linalg.norm(delta.reshape(-1))),
                "rms": _rms(delta),
                "cosine": _cosine(cells[left], cells[right]),
            }
        )
    return records


def _vector_metrics(
    values: np.ndarray,
    *,
    positions: list[int],
    mean_cell_rms: float,
    mean_pairwise_rms: float,
) -> dict[str, Any]:
    flat = values.reshape(-1)
    argmax_flat = int(np.argmax(np.abs(flat)))
    argmax_index = [int(value) for value in np.unravel_index(argmax_flat, values.shape)]
    sequence_axis = values.ndim - 2
    local_position = argmax_index[sequence_axis]
    rms = _rms(values)
    return {
        "l1_norm": float(np.sum(np.abs(flat))),
        "l2_norm": float(np.linalg.norm(flat)),
        "rms": rms,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "max_abs": float(abs(flat[argmax_flat])),
        "argmax_signed_effect": float(flat[argmax_flat]),
        "argmax_index": argmax_index,
        "argmax_sequence_position": positions[local_position],
        "relative_rms_to_mean_cell_rms": _safe_ratio(rms, mean_cell_rms),
        "relative_rms_to_mean_pairwise_rms": _safe_ratio(rms, mean_pairwise_rms),
    }


def _interaction_partition(
    regions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = {"all_effective", "image_tokens", "non_image_tokens"}
    if not required.issubset(regions):
        return {"available": False, "reason": "image/non-image partition unavailable"}
    all_l2 = float(regions["all_effective"]["effects"]["interaction"]["l2_norm"])
    image_l2 = float(regions["image_tokens"]["effects"]["interaction"]["l2_norm"])
    non_image_l2 = float(
        regions["non_image_tokens"]["effects"]["interaction"]["l2_norm"]
    )
    total_energy = all_l2**2
    partition_energy = image_l2**2 + non_image_l2**2
    return {
        "available": True,
        "image_energy_fraction": _safe_ratio(image_l2**2, total_energy),
        "non_image_energy_fraction": _safe_ratio(non_image_l2**2, total_energy),
        "partition_energy_relative_error": _safe_ratio(
            abs(partition_energy - total_energy), total_energy
        ),
        "partition_closes": math.isclose(
            partition_energy, total_energy, rel_tol=1e-9, abs_tol=1e-9
        ),
    }


def _balanced_contrast_energy(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    energies = {name: float(record["l2_norm"]) ** 2 for name, record in metrics.items()}
    total_energy = sum(energies.values())
    shares = {
        name: _safe_ratio(energy, total_energy) for name, energy in energies.items()
    }
    interaction_share = shares["interaction_contrast"]
    reference = 1.0 / 3.0
    return {
        "coefficient_scale": "+1/+1/-1/-1 for every contrast",
        "l2_squared": energies,
        "total_l2_squared": total_energy,
        "energy_shares": shares,
        "exchangeable_isotropic_reference_share": reference,
        "interaction_share_minus_exchangeable_reference": (
            None if interaction_share is None else interaction_share - reference
        ),
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator == 0.0:
        return None
    return float(np.dot(left_flat, right_flat) / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.8g}"
