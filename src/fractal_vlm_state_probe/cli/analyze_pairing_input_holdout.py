from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cache_direction_holdout import evaluate_pairing_holdout
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _read,
    _sha,
    _validate_panel_split,
)
from fractal_vlm_state_probe.processor_image_stats import extract_processor_pixel_tensor
from fractal_vlm_state_probe.stimulus import validate_manifest, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score input-space interaction directions with the frozen pairing split."
    )
    parser.add_argument("--reference-panel", required=True, type=Path)
    parser.add_argument("--test-panel", required=True, type=Path)
    parser.add_argument("--fastvlm-processor-snapshot", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    reference, test = _read(args.reference_panel), _read(args.test_panel)
    metadata, hashes = _validate_panel_split(reference, test)
    processor = None
    provenance = None
    if args.fastvlm_processor_snapshot is not None:
        from mlx_vlm.models.fastvlm import FastVLMImageProcessor

        snapshot = args.fastvlm_processor_snapshot
        if _read(snapshot / "config.json")["model_type"] != "llava_qwen2":
            raise ValueError(
                "processor baseline requires the registered FastVLM model type"
            )
        processor = FastVLMImageProcessor.from_pretrained(snapshot)
        source = Path(inspect.getfile(FastVLMImageProcessor))
        provenance = {
            "snapshot_revision": snapshot.name,
            "preprocessor_config_sha256": _sha(snapshot / "preprocessor_config.json"),
            "implementation": f"{FastVLMImageProcessor.__module__}.{FastVLMImageProcessor.__name__}",
            "implementation_sha256": _sha(source),
        }
    results = []
    for view in ("raw_rgb", "processor_pixel_values"):
        if view == "processor_pixel_values" and processor is None:
            continue
        vectors, manifests = {}, {}
        for panel in (reference, test):
            for record in panel["records"]:
                label = record["pair_id"]
                images = {}
                manifests[label] = {}
                for cell in ("mm", "jj", "mj", "jm"):
                    path = Path(record["factorial"]["manifests"][cell]["path"])
                    issues = validate_manifest(path)
                    if issues:
                        raise ValueError(f"invalid input manifest: {path}: {issues}")
                    manifest = _read(path)
                    frame = manifest["frames"][0]
                    with Image.open(path.parent / frame["path"]) as image:
                        images[cell] = (
                            np.asarray(image.convert("RGB"), dtype=np.float64) / 255
                            if view == "raw_rgb"
                            else extract_processor_pixel_tensor(
                                processor, image.convert("RGB")
                            )
                        )
                    manifests[label][cell] = {
                        "manifest_sha256": _sha(path),
                        "frame_sha256": frame["sha256"],
                    }
                if len({a.shape for a in images.values()}) != 1:
                    raise ValueError(f"input factorial shapes differ: {label}")
                vectors[label] = input_interaction(images)
                print(f"{view}: {label}", flush=True)
        results.append(
            {
                "view": view,
                "primary": False,
                "interaction_shape": list(next(iter(vectors.values())).shape),
                "manifests": manifests,
                **evaluate_pairing_holdout(vectors, metadata),
            }
        )
        del vectors
    write_json(
        args.output_json,
        {
            "schema_version": 1,
            "analysis_kind": "pairing_input_direction_holdout",
            "reference_panel_sha256": _sha(args.reference_panel),
            "test_panel_sha256": _sha(args.test_panel),
            "source_hash_count": len(hashes),
            "unique_source_hash_count": len(set(hashes)),
            "hierarchy_metadata": metadata,
            "processor_provenance": provenance,
            "views": results,
            "claim_boundaries": [
                "Input-space diagnostics are exploratory and do not change the registered cache-test family.",
                "No accepted frequency matching or cache mediation is measured.",
                "Input and cache vectors have different coordinates; margins are descriptive, not a shared-vector comparison.",
                "The processor baseline is eligible only if its native implementation matches the actual model processor.",
            ],
        },
    )
    print(f"wrote input holdout baseline to {args.output_json}", flush=True)


def input_interaction(cells: dict[str, np.ndarray]) -> np.ndarray:
    if set(cells) != {"mm", "jj", "mj", "jm"}:
        raise ValueError("input factorial requires all four cells")
    if len({a.shape for a in cells.values()}) != 1:
        raise ValueError("input factorial shapes differ")
    return cells["jj"] - cells["jm"] - cells["mj"] + cells["mm"]


if __name__ == "__main__":
    main()
