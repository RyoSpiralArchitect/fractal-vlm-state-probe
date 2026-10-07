from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cli.analyze_pairing_input_holdout import (
    _ModelProcessorPixelView,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read
from fractal_vlm_state_probe.frequency_control import CELLS, _load_cell, frequency_gate
from fractal_vlm_state_probe.image_stats import _frame_image_stats, _luminance
from fractal_vlm_state_probe.processor_image_stats import extract_processor_pixel_tensor
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def support_bounds(probe: np.ndarray) -> tuple[int, int, int, int]:
    support = np.any(np.abs(probe) > 1e-12, axis=0)
    ys, xs = np.where(support)
    if not len(ys):
        raise ValueError("uniform-white probe has no active processor pixels")
    bounds = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    top, bottom, left, right = bounds
    if not support[top:bottom, left:right].all():
        raise ValueError("this padding audit requires rectangular processor support")
    return bounds


def padding_variance_decomposition(rgb: np.ndarray, bounds: tuple) -> dict:
    top, bottom, left, right = bounds
    luminance = _luminance(rgb)
    mask = np.zeros(luminance.shape, dtype=bool)
    mask[top:bottom, left:right] = True
    outside = luminance[~mask]
    if outside.size and not np.all(outside == 0):
        raise ValueError("processor exterior is not exactly black under probe support")
    content = luminance[mask]
    fraction = float(mask.mean())
    between = fraction * (1 - fraction) * float(content.mean()) ** 2
    within = fraction * float(content.var())
    total = float(luminance.var())
    return {
        "content_fraction": fraction,
        "black_padding_fraction": 1 - fraction,
        "content_luminance_mean": float(content.mean()),
        "content_luminance_std": float(content.std()),
        "whole_luminance_variance": total,
        "between_content_and_black_variance": between,
        "within_content_variance": within,
        "variance_decomposition_abs_error": abs(total - between - within),
        "between_region_variance_fraction": between / total if total > 1e-18 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc, input-only black-padding support audit."
    )
    parser.add_argument("--input-acceptance", required=True, type=Path)
    parser.add_argument("--fastvlm-model-processor-snapshot", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    receipt = _read(args.input_acceptance)
    snapshot = args.fastvlm_model_processor_snapshot
    expected = receipt["processor_provenance"]
    if (
        sha256_file(snapshot / "processing_fastvlm.py")
        != expected["implementation_sha256"]
    ):
        raise ValueError("processor source differs from qualified input receipt")
    import torch
    from mlx_vlm.utils import load_processor

    torch.set_num_threads(1)
    processor = load_processor(snapshot, add_detokenizer=False, trust_remote_code=True)
    cls = type(processor.image_processor)
    actual = {
        "snapshot_revision": snapshot.name,
        "preprocessor_config_sha256": sha256_file(
            snapshot / "preprocessor_config.json"
        ),
        "implementation": f"{cls.__module__}.{cls.__name__}",
        "implementation_sha256": sha256_file(Path(inspect.getfile(cls))),
        "processor_mode": "model_loading_path",
    }
    if actual != expected:
        raise ValueError("processor provenance differs from input receipt")
    normalization = _read(snapshot / "preprocessor_config.json")
    pixel_view = _ModelProcessorPixelView(processor)
    mean = np.asarray(normalization["image_mean"])[:, None, None]
    std = np.asarray(normalization["image_std"])[:, None, None]
    first = receipt["records"][0]["arms"]["shared_pixel_permutation"]["artifacts"]["mm"]
    rgb, _ = _load_cell(Path(first["manifest_path"]))
    white = (
        extract_processor_pixel_tensor(
            pixel_view, Image.fromarray(np.full_like(rgb, 255))
        )
        * std
        + mean
    )
    black = (
        extract_processor_pixel_tensor(pixel_view, Image.fromarray(np.zeros_like(rgb)))
        * std
        + mean
    )
    if np.any(black != 0):
        raise ValueError("uniform-black probe does not map to black processor pixels")
    bounds = support_bounds(white)
    top, bottom, left, right = bounds
    records = []
    for record in receipt["records"]:
        print(f"padding audit {record['pair_id']}", flush=True)
        selected = record["arms"]["shared_pixel_permutation"]
        cropped_stats, cells = {}, {}
        for cell in CELLS:
            artifact = selected["artifacts"][cell]
            path = Path(artifact["manifest_path"])
            if sha256_file(path) != artifact["manifest_sha256"]:
                raise ValueError("frozen permutation manifest changed")
            rgb, manifest = _load_cell(path)
            if manifest["frames"][0]["sha256"] != artifact["frame_sha256"]:
                raise ValueError("frozen permutation image changed")
            tensor = extract_processor_pixel_tensor(pixel_view, Image.fromarray(rgb))
            processed = np.moveaxis(tensor * std + mean, 0, -1)
            whole = _frame_image_stats(processed)
            if whole != selected["stats"][cell]["processor"]:
                raise ValueError(
                    "whole-processor input statistics no longer reproduce exactly"
                )
            content = processed[top:bottom, left:right]
            cropped_stats[cell] = {
                **selected["stats"][cell],
                "processor": _frame_image_stats(content),
                "pixel_values_shape": [3, *content.shape[:2]],
            }
            cells[cell] = {
                "whole_stats_exactly_reproduced": True,
                "support_only_stats": cropped_stats[cell]["processor"],
                "variance_decomposition": padding_variance_decomposition(
                    processed, bounds
                ),
            }
        records.append(
            {
                "pair_id": record["pair_id"],
                "pairing_family": record["pairing_family"],
                "replicate": record["replicate"],
                "registered_whole_gate": selected["gate"],
                "support_only_diagnostic_gate": frequency_gate(cropped_stats),
                "cells": cells,
            }
        )
    write_json(
        args.output_json,
        {
            "schema_version": 1,
            "analysis_kind": "posthoc_processor_black_padding_audit",
            "input_acceptance_sha256": sha256_file(args.input_acceptance),
            "processor_provenance": actual,
            "code_sha256": sha256_file(Path(__file__)),
            "cache_forwards": 0,
            "model_weights_loaded": False,
            "support_bounds_y0_y1_x0_x1": list(bounds),
            "whole_processor_shape": list(white.shape),
            "uniform_black_probe_exactly_zero": True,
            "whole_stats_exactly_reproduced_cells": len(records) * 4,
            "registered_accepted_blocks": sum(
                r["registered_whole_gate"]["accepted"] for r in records
            ),
            "support_only_diagnostic_passes": sum(
                r["support_only_diagnostic_gate"]["accepted"] for r in records
            ),
            "eligible_for_registered_frequency_control": False,
            "records": records,
            "claim_boundaries": [
                "Post-hoc support masking is a different diagnostic, never a replacement acceptance gate.",
                "No padding-removal intervention or model-cache measurement is performed.",
                "Variance decomposition is not a decomposition of the FFT centroid or causal model attribution.",
            ],
        },
    )
    print(f"wrote post-hoc padding audit to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
