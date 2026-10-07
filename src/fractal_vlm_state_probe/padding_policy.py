from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .frequency_control import (
    CELLS,
    _load_cell,
    _materialize_cells,
    frequency_gate,
    rgb_multiset_sha256,
)
from .image_stats import _frame_image_stats
from .stimulus import sha256_file, write_json

POLICIES = ("black", "fixed_gray", "palette_mean")
CONTENT_STATES = ("original", "frozen_low_pass", "shared_pixel_permutation")


def fill_color(policy: str, palette: np.ndarray) -> np.ndarray:
    if policy == "black":
        return np.zeros(3, dtype=np.uint8)
    if policy == "fixed_gray":
        return np.full(3, 128, dtype=np.uint8)
    if policy == "palette_mean":
        return np.rint(palette.mean(axis=(0, 1))).astype(np.uint8)
    raise ValueError(f"unknown padding policy: {policy}")


def pad_to_square(rgb: np.ndarray, color: np.ndarray) -> tuple[np.ndarray, tuple]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("padding requires a uint8 RGB image")
    if color.dtype != np.uint8 or color.shape != (3,):
        raise ValueError("padding color must be three uint8 channels")
    height, width = rgb.shape[:2]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    canvas = np.broadcast_to(color, (side, side, 3)).copy()
    canvas[top : top + height, left : left + width] = rgb
    return canvas, (top, top + height, left, left + width)


def marginal_audit(
    content: np.ndarray, canvas: np.ndarray, color: np.ndarray, bounds: tuple
) -> dict:
    top, bottom, left, right = bounds
    source = content.astype(np.float64) / 255
    expanded = canvas.astype(np.float64) / 255
    fraction = content.shape[0] * content.shape[1] / (canvas.shape[0] * canvas.shape[1])
    mu = source.mean(axis=(0, 1))
    var = source.var(axis=(0, 1))
    fill = color.astype(float) / 255
    expected_mean = fraction * mu + (1 - fraction) * fill
    expected_var = fraction * var + fraction * (1 - fraction) * (mu - fill) ** 2
    source_fill_mass = float(np.all(content == color, axis=2).mean())
    exterior = np.ones(canvas.shape[:2], dtype=bool)
    exterior[top:bottom, left:right] = False
    return {
        "content_pixels_unchanged": bool(
            np.array_equal(canvas[top:bottom, left:right], content)
        ),
        "padding_pixels_equal_registered_fill": bool(np.all(canvas[exterior] == color)),
        "content_fraction": fraction,
        "added_padding_pixels": int(exterior.sum()),
        "fill_rgb_u8": color.tolist(),
        "original_rgb_mean": mu.tolist(),
        "expected_canvas_rgb_mean": expected_mean.tolist(),
        "expected_canvas_rgb_variance": expected_var.tolist(),
        "mean_max_abs_error": float(
            np.max(np.abs(expanded.mean(axis=(0, 1)) - expected_mean))
        ),
        "variance_max_abs_error": float(
            np.max(np.abs(expanded.var(axis=(0, 1)) - expected_var))
        ),
        "joint_rgb_total_variation_from_content": (1 - fraction)
        * (1 - source_fill_mass),
        "marginal_formula": "s * original_palette + (1-s) * point_mass(fill_rgb_u8)",
        "original_pixel_count": int(content.shape[0] * content.shape[1]),
        "canvas_pixel_count": int(canvas.shape[0] * canvas.shape[1]),
    }


def measure_processor(
    rgb: np.ndarray, processor: Any, normalization: dict
) -> tuple[dict, np.ndarray]:
    payload = processor(
        images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
    )
    pixels = np.asarray(payload["pixel_values"])
    if pixels.ndim != 4 or pixels.shape[:2] != (1, 3) or not np.isfinite(pixels).all():
        raise ValueError(
            "padding audit requires one finite three-channel processor image"
        )
    mean = np.asarray(normalization["image_mean"])[:, None, None]
    std = np.asarray(normalization["image_std"])[:, None, None]
    processed = np.moveaxis(pixels[0].astype(np.float64) * std + mean, 0, -1)
    record = {
        "processor": _frame_image_stats(processed),
        "pixel_values_shape": list(pixels.shape[1:]),
        "pixel_values_dtype": str(pixels.dtype),
        "pixel_values_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "processor_non_pixel_payload": {
            k: np.asarray(v).tolist() for k, v in payload.items() if k != "pixel_values"
        },
    }
    return record, pixels


def padding_gate(cells: dict[str, dict]) -> dict:
    gate = frequency_gate(
        {
            c: {
                **r,
                "rgb_multiset_preserved": r["registered_expanded_palette_verified"],
            }
            for c, r in cells.items()
        }
    )
    gate["registered_expanded_rgb_multisets_match"] = gate.pop(
        "rgb_multisets_preserved"
    )
    gate["content_and_fill_verified"] = all(
        r["marginal_audit"]["content_pixels_unchanged"]
        and r["marginal_audit"]["padding_pixels_equal_registered_fill"]
        and r["marginal_audit"]["mean_max_abs_error"] <= 1e-12
        and r["marginal_audit"]["variance_max_abs_error"] <= 1e-12
        for r in cells.values()
    )
    gate["accepted"] = gate["accepted"] and gate["content_and_fill_verified"]
    gate["marginal_condition"] = "registered_expanded_palette"
    return gate


def _checked_artifact(artifact: dict) -> tuple[np.ndarray, dict]:
    path = Path(artifact["manifest_path"])
    if sha256_file(path) != artifact["manifest_sha256"]:
        raise ValueError("prior input manifest changed")
    rgb, manifest = _load_cell(path)
    if manifest["frames"][0]["sha256"] != artifact["frame_sha256"]:
        raise ValueError("prior input frame changed")
    return rgb, manifest


def run_padding_policy_study(
    *,
    source_receipt_path: Path,
    published_source_path: Path,
    config: dict,
    processor: Any,
    provenance: dict,
    normalization: dict,
    runtime: dict,
    output_root: Path,
) -> dict:
    if (
        config.get("schema_version") != 1
        or config.get("policies") != list(POLICIES)
        or config.get("content_states") != list(CONTENT_STATES)
        or config.get("source_shape_hwc") != [240, 320, 3]
        or config.get("centroid_relative_tolerance") != 0.05
        or config.get("hf_absolute_tolerance") != 0.02
    ):
        raise ValueError("padding study differs from the registered specification")
    published = json.loads(published_source_path.read_text())
    if sha256_file(source_receipt_path) != published["input_receipt_sha256"]:
        raise ValueError("input receipt differs from the published Note 0044 snapshot")
    source = json.loads(source_receipt_path.read_text())
    if provenance != source["processor_provenance"]:
        raise ValueError("processor differs from qualified source receipt")
    if len(source["records"]) != 32:
        raise ValueError("registered padding study requires all 32 source-pair blocks")
    family_replicates: dict[str, list] = {}
    for r in source["records"]:
        family_replicates.setdefault(r["pairing_family"], []).append(r["replicate"])
    if len(family_replicates) != 8 or any(
        sorted(v) != [1, 2, 3, 4] for v in family_replicates.values()
    ):
        raise ValueError("registered padding hierarchy differs")
    if output_root.exists():
        raise FileExistsError("use a new padding study output root")
    output_root.mkdir(parents=True)
    code_files = [
        Path(__file__),
        Path(__file__).parent / "frequency_control.py",
        Path(__file__).parent / "image_stats.py",
        Path(__file__).parent / "cli" / "run_padding_policy_study.py",
    ]
    frozen = {
        "config": config,
        "source_receipt_sha256": sha256_file(source_receipt_path),
        "published_source_sha256": sha256_file(published_source_path),
        "processor_provenance": provenance,
        "runtime": runtime,
        "code_sha256": {
            str(p.relative_to(Path(__file__).parents[2])): sha256_file(p)
            for p in code_files
        },
    }
    write_json(output_root / "frozen_specification.json", frozen)
    records = []
    for original in source["records"]:
        pair_id = original["pair_id"]
        print(f"padding comparison {pair_id}", flush=True)
        record = {
            k: original[k]
            for k in ("pair_id", "pairing_family", "broad_class", "replicate")
        }
        record["conditions"] = {}
        for state in CONTENT_STATES:
            if state == "original":
                artifacts = original["sham_artifacts"]
                expected_stats = original["original_stats"]
                if not all(original["sham_pixels_equal"].values()) or any(
                    a["frame_sha256"] != a["original_frame_sha256"]
                    for a in artifacts.values()
                ):
                    raise ValueError(
                        "original-content sham must identify exact historical PNGs"
                    )
            else:
                prior_arm = "rank_low_pass" if state == "frozen_low_pass" else state
                artifacts = original["arms"][prior_arm]["artifacts"]
                expected_stats = original["arms"][prior_arm]["stats"]
            loaded = {c: _checked_artifact(a) for c, a in artifacts.items()}
            content = {c: v[0] for c, v in loaded.items()}
            sources = {c: v[1] for c, v in loaded.items()}
            if any(
                list(a.shape) != config["source_shape_hwc"] for a in content.values()
            ):
                raise ValueError("source content shape differs")
            state_records = {p: {"cells": {}, "gate": None} for p in POLICIES}
            canvases = {p: {} for p in POLICIES}
            expected_multisets = {}
            for policy in POLICIES:
                expected_multisets[policy] = {
                    c: rgb_multiset_sha256(
                        pad_to_square(content[c], fill_color(policy, content[c]))[0]
                    )
                    for c in ("mm", "jj")
                }
            for cell in CELLS:
                baseline, baseline_pixels = measure_processor(
                    content[cell], processor, normalization
                )
                if baseline["processor"] != expected_stats[cell]["processor"]:
                    raise ValueError(
                        f"historical input statistics differ: {pair_id}/{state}/{cell}"
                    )
                donor = "mm" if cell in ("mm", "jm") else "jj"
                for policy in POLICIES:
                    color = fill_color(policy, content[donor])
                    canvas, bounds = pad_to_square(content[cell], color)
                    measured, pixels = measure_processor(
                        canvas, processor, normalization
                    )
                    audit = marginal_audit(content[cell], canvas, color, bounds)
                    marginal_hash = rgb_multiset_sha256(canvas)
                    measured.update(
                        {
                            "raw": _frame_image_stats(canvas.astype(np.float64) / 255),
                            "content_frame_sha256": artifacts[cell]["frame_sha256"],
                            "content_manifest_sha256": artifacts[cell][
                                "manifest_sha256"
                            ],
                            "canvas_rgb_multiset_sha256": marginal_hash,
                            "registered_expanded_palette_verified": marginal_hash
                            == expected_multisets[policy][donor],
                            "marginal_audit": audit,
                        }
                    )
                    if policy == "black":
                        equal = (
                            pixels.dtype == baseline_pixels.dtype
                            and pixels.shape == baseline_pixels.shape
                            and pixels.tobytes() == baseline_pixels.tobytes()
                        )
                        metadata_diff = sorted(
                            k
                            for k in set(baseline["processor_non_pixel_payload"])
                            | set(measured["processor_non_pixel_payload"])
                            if baseline["processor_non_pixel_payload"].get(k)
                            != measured["processor_non_pixel_payload"].get(k)
                        )
                        measured["black_pad_sham"] = {
                            "pixel_values_bitwise_equal": equal,
                            "baseline_pixel_values_sha256": baseline[
                                "pixel_values_sha256"
                            ],
                            "changed_non_pixel_payload_keys": metadata_diff,
                            "baseline_non_pixel_payload": baseline[
                                "processor_non_pixel_payload"
                            ],
                            "whole_processor_payload_equal": equal
                            and not metadata_diff,
                        }
                        if not equal:
                            raise ValueError(
                                "external black-pad pixel sham does not reproduce the actual processor"
                            )
                    state_records[policy]["cells"][cell] = measured
                    canvases[policy][cell] = canvas
            for policy in POLICIES:
                condition = state_records[policy]
                condition["gate"] = padding_gate(condition["cells"])
                condition["artifacts"] = _materialize_cells(
                    canvases[policy],
                    sources,
                    output_root / "inputs" / state / policy / pair_id,
                    {
                        "arm": f"{state}_{policy}",
                        "padding_policy": policy,
                        "content_state": state,
                        "marginal_condition": "registered_expanded_palette",
                        "fill_rgb_u8_by_cell": {
                            c: condition["cells"][c]["marginal_audit"]["fill_rgb_u8"]
                            for c in CELLS
                        },
                    },
                )
                for cell, artifact in condition["artifacts"].items():
                    materialized, _ = _checked_artifact(artifact)
                    if not np.array_equal(materialized, canvases[policy][cell]):
                        raise ValueError("serialized padding canvas changed")
                record["conditions"][f"{state}/{policy}"] = condition
        records.append(record)
        write_json(output_root / "blocks" / f"{pair_id}.json", record)
    conditions = {}
    for state in CONTENT_STATES:
        for policy in POLICIES:
            key = f"{state}/{policy}"
            accepted = [r for r in records if r["conditions"][key]["gate"]["accepted"]]
            complete = [
                f
                for f in family_replicates
                if {r["replicate"] for r in accepted if r["pairing_family"] == f}
                == {1, 2, 3, 4}
            ]
            conditions[key] = {
                "accepted_blocks": len(accepted),
                "total_blocks": len(records),
                "accepted_pair_ids": [r["pair_id"] for r in accepted],
                "complete_four_seed_families": sorted(complete),
                "all_input_blocks_accepted": len(accepted) == len(records),
                "new_cache_forwards": 0,
            }
    result = {
        "schema_version": 1,
        "analysis_kind": "padding_policy_input_comparison",
        "date": config["registered_date"],
        "frozen_specification": frozen,
        "frozen_specification_sha256": sha256_file(
            output_root / "frozen_specification.json"
        ),
        "block_count": len(records),
        "conditions": conditions,
        "records": records,
        "new_cache_forwards": 0,
        "model_weights_loaded": False,
        "comparison_scope": "paired padding-policy interventions on fixed input content",
        "claim_boundaries": [
            "The full image marginal is explicitly changed by adding 25 percent constant-color pixels.",
            "Input frequency acceptance does not establish matched cache correspondence.",
            "The black-pad sham checks pixels; changed processor metadata requires a separate model-cache calibration.",
            "Frozen low-pass choices were selected under the previous black-padding sweep and are not retuned here.",
            "The common permutation destroys geometry and retains source-palette differences.",
            "These are paired views of existing sources, not independent new families or images.",
        ],
    }
    write_json(output_root / "padding_policy_summary.json", result)
    return result
