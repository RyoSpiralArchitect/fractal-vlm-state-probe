from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .control_stimulus import _frequency_mask, _luminance_key
from .image_stats import _frame_image_stats
from .processor_image_stats import extract_processor_pixel_tensor
from .stimulus import sha256_file, sha256_json, validate_manifest, write_json

CELLS = ("mm", "jj", "mj", "jm")
ARMS = ("rank_low_pass", "shared_pixel_permutation")
CENTROID_TOLERANCE = 0.05
HF_TOLERANCE = 0.02


def rgb_multiset_sha256(rgb: np.ndarray) -> str:
    import hashlib

    pixels = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    order = np.lexsort((pixels[:, 2], pixels[:, 1], pixels[:, 0]))
    return hashlib.sha256(pixels[order].tobytes()).hexdigest()


def low_pass_rank_cells(
    originals: dict[str, np.ndarray], cutoff_a: float, cutoff_b: float
) -> dict[str, np.ndarray]:
    def order_for(rgb: np.ndarray, cutoff: float) -> np.ndarray:
        if not 0 < cutoff <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        field = _luminance_key(rgb)
        if cutoff < 1:
            field = np.fft.irfft2(
                np.fft.rfft2(field)
                * _frequency_mask(field.shape, mode="low_pass", cutoff=cutoff),
                s=field.shape,
            )
        return np.argsort(field.ravel(), kind="mergesort")

    orders = {
        "a": order_for(originals["mm"], cutoff_a),
        "b": order_for(originals["jj"], cutoff_b),
    }
    palettes = {
        role: rgb.reshape(-1, 3)[
            np.argsort(_luminance_key(rgb).ravel(), kind="mergesort")
        ]
        for role, rgb in (("a", originals["mm"]), ("b", originals["jj"]))
    }
    result = {}
    for cell, spatial, palette in (
        ("mm", "a", "a"),
        ("jj", "b", "b"),
        ("mj", "a", "b"),
        ("jm", "b", "a"),
    ):
        output = np.empty_like(palettes[palette])
        output[orders[spatial]] = palettes[palette]
        result[cell] = output.reshape(originals["mm"].shape)
    return result


def permute_cells(
    originals: dict[str, np.ndarray], permutation: np.ndarray
) -> dict[str, np.ndarray]:
    size = originals["mm"].shape[0] * originals["mm"].shape[1]
    if not np.array_equal(np.sort(permutation), np.arange(size)):
        raise ValueError("pixel permutation is not a bijection")
    if len({rgb.shape for rgb in originals.values()}) != 1:
        raise ValueError("all four cell shapes must match")
    return {
        cell: rgb.reshape(-1, 3)[permutation].reshape(rgb.shape)
        for cell, rgb in originals.items()
    }


def frequency_gate(stats: dict[str, dict]) -> dict[str, Any]:
    if set(stats) != set(CELLS):
        raise ValueError("frequency gate requires all four cells")
    centroids = [s["processor"]["spectral_centroid"] for s in stats.values()]
    high_frequency = [
        s["processor"]["high_frequency_energy_ratio"] for s in stats.values()
    ]
    valid = all(
        np.isfinite(c)
        and c > 1e-12
        and np.isfinite(h)
        and 0 <= h <= 1
        and s["processor"]["luminance_std"] > 1e-12
        for c, h, s in zip(centroids, high_frequency, stats.values())
    )
    target = float(np.mean(centroids)) if valid else None
    errors = {
        c: abs(s["processor"]["spectral_centroid"] / target - 1) if valid else None
        for c, s in stats.items()
    }
    max_error = max(errors.values()) if valid else None
    spread = float(max(high_frequency) - min(high_frequency)) if valid else None
    marginal = all(s["rgb_multiset_preserved"] for s in stats.values())
    shapes_equal = len({tuple(s["pixel_values_shape"]) for s in stats.values()}) == 1
    return {
        "common_centroid_target": target,
        "common_hf_target": float(np.mean(high_frequency)) if valid else None,
        "centroid_relative_errors": errors,
        "max_centroid_relative_error": max_error,
        "hf_max_pairwise_absolute_difference": spread,
        "centroid_relative_tolerance": CENTROID_TOLERANCE,
        "hf_absolute_tolerance": HF_TOLERANCE,
        "nondegenerate_finite_spectra": bool(valid),
        "rgb_multisets_preserved": marginal,
        "processor_shapes_equal": shapes_equal,
        "accepted": bool(
            valid
            and marginal
            and shapes_equal
            and max_error <= CENTROID_TOLERANCE
            and spread <= HF_TOLERANCE
        ),
    }


def cell_statistics(
    rgb: np.ndarray,
    *,
    processor: Any,
    normalization: dict,
    donor_multiset_sha256: str,
) -> dict[str, Any]:
    tensor = extract_processor_pixel_tensor(processor, Image.fromarray(rgb))
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("frequency controls require one three-channel processor image")
    mean = np.asarray(normalization["image_mean"])[:, None, None]
    std = np.asarray(normalization["image_std"])[:, None, None]
    processed_rgb = np.moveaxis(tensor * std + mean, 0, -1)
    if not np.all(np.isfinite(processed_rgb)):
        raise ValueError("nonfinite processor pixels")
    return {
        "raw": _frame_image_stats(rgb.astype(np.float64) / 255),
        "processor": _frame_image_stats(processed_rgb),
        "tensor_mean": float(tensor.mean()),
        "tensor_std": float(tensor.std()),
        "pixel_values_shape": list(tensor.shape),
        "rgb_multiset_sha256": rgb_multiset_sha256(rgb),
        "rgb_multiset_preserved": rgb_multiset_sha256(rgb) == donor_multiset_sha256,
    }


def prepare_frequency_controls(
    *,
    reference_panel: Path,
    test_panel: Path,
    config: dict,
    processor: Any,
    provenance: dict,
    normalization: dict,
    output_root: Path,
) -> dict[str, Any]:
    if config.get("schema_version") != 1 or config.get("arms") != list(ARMS):
        raise ValueError("unsupported frequency-control specification")
    cutoffs = config["low_pass_cutoffs"]
    if (
        not cutoffs
        or len(cutoffs) != len(set(cutoffs))
        or 1.0 not in cutoffs
        or any(not 0 < c <= 1 for c in cutoffs)
    ):
        raise ValueError("invalid frozen cutoff grid")
    if (
        config["centroid_relative_tolerance"] != CENTROID_TOLERANCE
        or config["hf_absolute_tolerance"] != HF_TOLERANCE
    ):
        raise ValueError("registered frequency tolerances cannot be relaxed")
    if output_root.exists():
        raise FileExistsError("use a new frequency-control output root")
    output_root.mkdir(parents=True)
    panels = [json.loads(p.read_text()) for p in (reference_panel, test_panel)]
    write_json(
        output_root / "frozen_input_specification.json",
        {
            "config": config,
            "processor_provenance": provenance,
            "reference_panel_sha256": sha256_file(reference_panel),
            "test_panel_sha256": sha256_file(test_panel),
            "code_sha256": sha256_file(Path(__file__)),
        },
    )
    records, output_records = [], {arm: [[], []] for arm in ARMS}
    global_shape = None
    permutation = None
    for panel_index, panel in enumerate(panels):
        for record in panel["records"]:
            pair_id = record["pair_id"]
            print(f"input audit {pair_id}", flush=True)
            originals, source_manifests = {}, {}
            for cell in CELLS:
                path = Path(record["factorial"]["manifests"][cell]["path"])
                originals[cell], source_manifests[cell] = _load_cell(path)
            shape = originals["mm"].shape
            if len({a.shape for a in originals.values()}) != 1:
                raise ValueError("source factorial shapes differ")
            if global_shape is None:
                global_shape = shape
                permutation = np.random.default_rng(
                    config["permutation_seed"]
                ).permutation(shape[0] * shape[1])
                np.save(output_root / "shared_pixel_permutation.npy", permutation)
            elif shape != global_shape:
                raise ValueError(
                    "global permutation requires one panel-wide image shape"
                )
            donors = {
                "a": rgb_multiset_sha256(originals["mm"]),
                "b": rgb_multiset_sha256(originals["jj"]),
            }

            def statistics(cells: dict[str, np.ndarray]) -> dict:
                return {
                    c: cell_statistics(
                        rgb,
                        processor=processor,
                        normalization=normalization,
                        donor_multiset_sha256=donors["a" if c in ("mm", "jm") else "b"],
                    )
                    for c, rgb in cells.items()
                }

            baseline = statistics(originals)
            # Each spatial field is shared across both palettes; only input metrics choose cutoffs.
            a_stats, b_stats = {}, {}
            for cutoff in cutoffs:
                cells = low_pass_rank_cells(originals, cutoff, cutoff)
                a_stats[cutoff] = statistics({c: cells[c] for c in ("mm", "mj")})
                b_stats[cutoff] = statistics({c: cells[c] for c in ("jj", "jm")})
            candidates = []
            for ca, cb in product(cutoffs, repeat=2):
                stats = {**a_stats[ca], **b_stats[cb]}
                gate = frequency_gate(stats)
                candidates.append({"cutoff_a": ca, "cutoff_b": cb, "gate": gate})

            def candidate_order(candidate: dict) -> tuple:
                gate = candidate["gate"]
                error = gate["max_centroid_relative_error"]
                spread = gate["hf_max_pairwise_absolute_difference"]
                violation = (
                    max(error / CENTROID_TOLERANCE, spread / HF_TOLERANCE)
                    if error is not None
                    else float("inf")
                )
                return (
                    not gate["accepted"],
                    0 if gate["accepted"] else violation,
                    -candidate["cutoff_a"] - candidate["cutoff_b"],
                    -candidate["cutoff_a"],
                    -candidate["cutoff_b"],
                )

            selected = min(candidates, key=candidate_order)
            ca, cb = selected["cutoff_a"], selected["cutoff_b"]
            low_cells = low_pass_rank_cells(originals, ca, cb)
            shuffled = permute_cells(originals, permutation)
            inverse = np.argsort(permutation)
            sham = permute_cells(shuffled, inverse)
            sham_equal = {c: sham[c].tobytes() == originals[c].tobytes() for c in CELLS}
            if not all(sham_equal.values()):
                raise ValueError("transform-only roundtrip sham failed")
            sham_artifacts = _materialize_cells(
                sham,
                source_manifests,
                output_root / "sham" / pair_id,
                {"arm": "permutation_inverse_roundtrip"},
            )
            audit = {
                k: record[k]
                for k in ("pair_id", "pairing_family", "broad_class", "replicate")
            }
            audit.update(
                {
                    "original_stats": baseline,
                    "original_gate": frequency_gate(baseline),
                    "sham_pixels_equal": sham_equal,
                    "sham_artifacts": sham_artifacts,
                    "low_pass_candidate_gates": candidates,
                    "low_pass_spatial_stats": {"a": a_stats, "b": b_stats},
                    "arms": {},
                }
            )
            for arm, cells, parameters, stats in (
                (
                    "rank_low_pass",
                    low_cells,
                    {"cutoff_a": ca, "cutoff_b": cb},
                    {**a_stats[ca], **b_stats[cb]},
                ),
                (
                    "shared_pixel_permutation",
                    shuffled,
                    {"permutation_seed": config["permutation_seed"]},
                    statistics(shuffled),
                ),
            ):
                gate = frequency_gate(stats)
                artifacts = _materialize_cells(
                    cells,
                    source_manifests,
                    output_root / arm / pair_id,
                    {"arm": arm, **parameters},
                )
                audit["arms"][arm] = {
                    "parameters": parameters,
                    "stats": stats,
                    "gate": gate,
                    "artifacts": artifacts,
                }
                transformed = {
                    k: record[k]
                    for k in ("pair_id", "pairing_family", "broad_class", "replicate")
                }
                transformed["factorial"] = {
                    "pair_id": pair_id,
                    "manifests": {
                        c: {
                            "path": a["manifest_path"],
                            "condition_id": f"{arm}_{pair_id}_{c}",
                        }
                        for c, a in artifacts.items()
                    },
                }
                for role, cell in (("source_a", "mm"), ("source_b", "jj")):
                    transformed[role] = {
                        **copy.deepcopy(record[role]),
                        "original_source": record[role],
                        "manifest_path": artifacts[cell]["manifest_path"],
                        "first_frame_sha256": artifacts[cell]["frame_sha256"],
                        "condition_id": f"{arm}_{pair_id}_{cell}",
                        "generated_by_panel": False,
                    }
                output_records[arm][panel_index].append(transformed)
            records.append(audit)
            write_json(output_root / "input_audits" / f"{pair_id}.json", audit)
    summary = {
        "schema_version": 1,
        "analysis_kind": "frequency_control_input_acceptance",
        "cache_forwards": 0,
        "config": config,
        "processor_provenance": provenance,
        "input_specification_sha256": sha256_file(
            output_root / "frozen_input_specification.json"
        ),
        "permutation_sha256": sha256_file(output_root / "shared_pixel_permutation.npy"),
        "block_count": len(records),
        "records": records,
        "arms": {
            arm: {
                "accepted_blocks": sum(
                    r["arms"][arm]["gate"]["accepted"] for r in records
                ),
                "all_blocks_accepted": all(
                    r["arms"][arm]["gate"]["accepted"] for r in records
                ),
            }
            for arm in ARMS
        },
        "claim_boundaries": [
            "Two luminance spectral summaries are matched operationally, not the full spectrum.",
            "The shared permutation destroys geometry and is not a structure-preserving frequency intervention.",
            "Input acceptance is not a cache result; no cache forward is performed by this tool.",
            "A partial accepted panel does not qualify the registered eight-family held-out design.",
        ],
    }
    receipt = output_root / "input_acceptance.json"
    write_json(receipt, summary)
    for arm in ARMS:
        for index, role in enumerate(("reference", "test")):
            panel = {
                **panels[index],
                "panel_id": f"{config['study_id']}_{arm}_{role}",
                "records": output_records[arm][index],
                "output_root": str(output_root / arm),
                "frequency_control": {
                    "arm": arm,
                    "receipt_path": str(receipt.resolve()),
                    "receipt_sha256": sha256_file(receipt),
                    "all_blocks_accepted": summary["arms"][arm]["all_blocks_accepted"],
                    "processor_provenance": provenance,
                },
            }
            # Original raw marginal summaries and command suggestions describe other images.
            for key in (
                "raw_marginal_audit",
                "factorial_batch_summary_json",
                "interpretation_notes",
            ):
                panel.pop(key, None)
            write_json(output_root / arm / f"{role}_panel.json", panel)
    return summary


def validate_frequency_panel(panel: dict) -> None:
    control = panel.get("frequency_control")
    if control is None:
        return
    receipt = Path(control["receipt_path"])
    if sha256_file(receipt) != control["receipt_sha256"]:
        raise ValueError("frequency acceptance receipt hash differs")
    summary = json.loads(receipt.read_text())
    arm = control["arm"]
    if (
        arm not in ARMS
        or not summary["arms"][arm]["all_blocks_accepted"]
        or not control["all_blocks_accepted"]
    ):
        raise ValueError("unmatched frequency panel: stop before cache forwards")
    if control["processor_provenance"] != summary["processor_provenance"]:
        raise ValueError("frequency processor provenance differs")
    audits = {r["pair_id"]: r for r in summary["records"]}
    for record in panel["records"]:
        audit = audits[record["pair_id"]]
        if any(
            record[k] != audit[k]
            for k in ("pairing_family", "broad_class", "replicate")
        ):
            raise ValueError("frequency hierarchy differs from acceptance audit")
        selected = audit["arms"][arm]
        if (
            frequency_gate(selected["stats"]) != selected["gate"]
            or not selected["gate"]["accepted"]
        ):
            raise ValueError("frequency gate no longer qualifies")
        for cell in CELLS:
            artifact = selected["artifacts"][cell]
            path = Path(record["factorial"]["manifests"][cell]["path"])
            if (
                str(path) != artifact["manifest_path"]
                or sha256_file(path) != artifact["manifest_sha256"]
            ):
                raise ValueError("frequency cell manifest differs")
            rgb, manifest = _load_cell(path)
            if (
                manifest["frames"][0]["sha256"] != artifact["frame_sha256"]
                or rgb_multiset_sha256(rgb)
                != selected["stats"][cell]["rgb_multiset_sha256"]
            ):
                raise ValueError("frequency cell image differs")
        for role, cell in (("source_a", "mm"), ("source_b", "jj")):
            if (
                record[role]["first_frame_sha256"]
                != selected["artifacts"][cell]["frame_sha256"]
            ):
                raise ValueError("frequency source hash differs")


def _load_cell(path: Path) -> tuple[np.ndarray, dict]:
    issues = validate_manifest(path)
    if issues:
        raise ValueError(f"invalid source manifest: {path}: {issues}")
    manifest = json.loads(path.read_text())
    if len(manifest.get("frames", [])) != 1:
        raise ValueError("frequency controls require exactly one source frame")
    with Image.open(path.parent / manifest["frames"][0]["path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    return rgb, manifest


def _materialize_cells(cells: dict, sources: dict, root: Path, transform: dict) -> dict:
    artifacts = {}
    for cell, rgb in cells.items():
        directory = (root / cell).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        image_path = directory / "frame.png"
        Image.fromarray(rgb).save(image_path)
        source = sources[cell]
        config = {
            "transform": transform,
            "original_manifest": source,
            "original_frame_sha256": source["frames"][0]["sha256"],
        }
        condition = {
            **source["stimulus_condition"],
            "condition_id": f"{transform['arm']}_{root.name}_{cell}",
            "source_kind": "external_frames",
            "comparison_role": "frequency_control",
            "description": f"{transform['arm']} of the recorded source factorial cell.",
        }
        manifest = {
            "schema_version": 1,
            "generator": "fractal_vlm_state_probe.frequency_control",
            "stimulus_condition": condition,
            "stimulus_config": config,
            "stimulus_config_sha256": sha256_json(config),
            "frames": [
                {
                    "index": 0,
                    "t_seconds": 0.0,
                    "path": "frame.png",
                    "sha256": sha256_file(image_path),
                    "width": rgb.shape[1],
                    "height": rgb.shape[0],
                }
            ],
        }
        path = directory / "manifest.json"
        write_json(path, manifest)
        artifacts[cell] = {
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
            "frame_sha256": sha256_file(image_path),
            "original_frame_sha256": config["original_frame_sha256"],
        }
    return artifacts
