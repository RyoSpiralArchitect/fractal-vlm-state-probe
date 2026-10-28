from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cache_direction_holdout import (
    validate_pairing_holdout_metadata,
)
from fractal_vlm_state_probe.cli.analyze_pairing_input_holdout import input_interaction
from fractal_vlm_state_probe.controlled_overlap import (
    construct_maps,
    coordinate_counts,
    verify_coordinate_recipe,
)
from fractal_vlm_state_probe.frequency_control import CELLS, _materialize_cells
from fractal_vlm_state_probe.graded_permutation import (
    array_sha256,
    audit_level,
    mapping_metrics,
)
from fractal_vlm_state_probe.padding_policy import _checked_artifact, fill_color
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "configs/controlled_overlap_phase1_v1.json"
BASELINE = REPO / "examples/research_notes/0050_map_seed_input_baseline/summary.json"
CODE = [
    "src/fractal_vlm_state_probe/source_coordinate_overlap.py",
    "src/fractal_vlm_state_probe/controlled_overlap.py",
    "src/fractal_vlm_state_probe/cli/run_controlled_overlap.py",
    "src/fractal_vlm_state_probe/cli/analyze_controlled_overlap.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def committed_config():
    config = read(CONFIG)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    code = dict(read(BASELINE)["frozen_specification"]["code_sha256"])
    for name in CODE:
        code[name] = sha256_file(REPO / name)
    for name, digest in {
        **code,
        str(CONFIG.relative_to(REPO)): sha256_file(CONFIG),
    }.items():
        stored = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)
        if (
            hashlib.sha256(stored).hexdigest() != digest
            or sha256_file(REPO / name) != digest
        ):
            raise ValueError(
                f"commit current qualified code/config before execution: {name}"
            )
    return config, commit, code


def prepare_maps(root):
    config, commit, code = committed_config()
    if root.exists():
        raise FileExistsError("use a fresh coordinate-only root")
    maps, recipes = construct_maps(config)
    verified = verify_coordinate_recipe(config, maps, recipes)
    root.mkdir(parents=True)
    receipts = {}
    for key, mapping in maps.items():
        path = root / f"{key}.npy"
        np.save(path, mapping)
        reference = maps[recipes[key]["reference"]]
        receipts[key] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "recipe": recipes[key],
            "counts": coordinate_counts(reference, mapping),
            "metrics": mapping_metrics(mapping, (240, 320)),
        }
    report = {
        "schema_version": 1,
        "analysis_kind": "controlled_overlap_coordinate_construction",
        "config": config,
        "registered_design_commit": commit,
        "code_sha256": code,
        "maps": receipts,
        "independent_verification": verified,
        "new_images": 0,
        "new_processor_calls": 0,
        "new_model_forwards": 0,
    }
    write_json(root / "coordinate_report.json", report)
    print(json.dumps(verified), flush=True)


def load_maps(path):
    report = read(path)
    config, _, code = committed_config()
    if report["config"] != config or report["code_sha256"] != code:
        raise ValueError("coordinate registration changed")
    maps = {}
    for key, item in report["maps"].items():
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("map hash differs")
        maps[key] = np.load(item["path"], allow_pickle=False)
    recipes = {k: r["recipe"] for k, r in report["maps"].items()}
    verify_coordinate_recipe(config, maps, recipes)
    for k, mapping in maps.items():
        if report["maps"][k]["counts"] != coordinate_counts(
            maps[recipes[k]["reference"]], mapping
        ):
            raise ValueError("coordinate counts differ")
    return report, maps


def qualified_sources():
    baseline = read(BASELINE)
    f = baseline["frozen_specification"]
    input_path = Path(f["input_summary"])
    if sha256_file(input_path) != f["source_sha256"][str(input_path)]:
        raise ValueError("historical input receipt changed")
    prior = read(input_path)
    graded_path = Path(prior["frozen_specification"]["graded_receipt_path"])
    if (
        sha256_file(graded_path)
        != prior["frozen_specification"]["graded_receipt_sha256"]
    ):
        raise ValueError("historical graded receipt changed")
    graded = read(graded_path)
    padding_path = Path(
        graded["frozen_specification"]["source_paths"]["padding_receipt"]
    )
    if (
        sha256_file(padding_path)
        != graded["frozen_specification"]["source_sha256"]["padding_receipt"]
    ):
        raise ValueError("historical padding receipt changed")
    padding = read(padding_path)
    sources = {r["pair_id"]: r for r in padding["records"]}
    metadata = {
        r["pair_id"]: {k: r[k] for k in ("pairing_family", "broad_class", "replicate")}
        for r in prior["records"]
    }
    validate_pairing_holdout_metadata(metadata)
    if set(metadata) != set(sources) or metadata != f["hierarchy_metadata"]:
        raise ValueError("source hierarchy changed")
    hashes = {
        str(p): sha256_file(p)
        for p in (BASELINE, input_path, graded_path, padding_path)
    }
    return baseline, prior, sources, metadata, hashes


def qualified_processor(baseline):
    f = baseline["frozen_specification"]
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    if runtime != f["runtime"]:
        raise ValueError("qualified processor runtime changed")
    path, expected = Path(f["processor_snapshot"]), f["processor_provenance"]
    if (
        path.name != expected["snapshot_revision"]
        or sha256_file(path / "processing_fastvlm.py")
        != expected["implementation_sha256"]
        or sha256_file(path / "preprocessor_config.json")
        != expected["preprocessor_config_sha256"]
    ):
        raise ValueError("qualified processor snapshot changed")
    import torch
    from mlx_vlm.utils import load_processor

    torch.set_num_threads(1)
    processor = load_processor(path, add_detokenizer=False, trust_remote_code=True)
    cls = type(processor.image_processor)
    if (
        f"{cls.__module__}.{cls.__name__}" != expected["implementation"]
        or sha256_file(Path(inspect.getfile(cls))) != expected["implementation_sha256"]
    ):
        raise ValueError("loaded processor implementation changed")
    return processor, read(path / "preprocessor_config.json")


class PixelRecorder:
    """Retain the actual qualified processor return, without a second conversion."""

    def __init__(self, processor):
        self.processor, self.pixels = processor, []

    def __call__(self, **kwargs):
        result = self.processor(**kwargs)
        pixels = np.asarray(result["pixel_values"])
        if (
            pixels.shape != (1, 3, 1024, 1024)
            or str(pixels.dtype) != "float32"
            or not np.isfinite(pixels).all()
        ):
            raise ValueError("qualified pixel contract differs")
        self.pixels.append(pixels.copy())
        return result


def measure(map_report, root):
    if root.exists():
        raise FileExistsError("use a fresh measurement root")
    config, commit, code = committed_config()
    if shutil.disk_usage(REPO).free < config["minimum_free_disk_gib"] * 1024**3:
        raise ValueError("insufficient registered disk reserve before capture")
    report, maps = load_maps(map_report)
    baseline, prior, sources, metadata, hashes = qualified_sources()
    processor, normalization = qualified_processor(baseline)
    root.mkdir(parents=True)
    frozen = {
        "config": config,
        "registered_design_commit": commit,
        "code_sha256": code,
        "source_sha256": hashes,
        "map_report": str(map_report.resolve()),
        "map_report_sha256": sha256_file(map_report),
        "hierarchy_metadata": metadata,
        "runtime": baseline["frozen_specification"]["runtime"],
        "processor_provenance": baseline["frozen_specification"][
            "processor_provenance"
        ],
    }
    write_json(root / "frozen_specification.json", frozen)
    try:
        calibration = []
        for record in prior["records"]:
            anchor = record["maps"]["historical_3q"]
            for cell in CELLS:
                rgb, _ = _checked_artifact(anchor["artifacts"][cell])
                payload = processor(
                    images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
                )
                pixels = np.asarray(payload["pixel_values"])
                other = {
                    k: np.asarray(v).tolist()
                    for k, v in payload.items()
                    if k != "pixel_values"
                }
                expected = anchor["cells"][cell]
                if (
                    array_sha256(pixels) != expected["pixel_values_sha256"]
                    or other != expected["processor_non_pixel_payload"]
                ):
                    raise ValueError("historical processor replay differs")
                calibration.append(
                    {
                        "pair_id": record["pair_id"],
                        "cell": cell,
                        "pixel_values_sha256": array_sha256(pixels),
                        "exact": True,
                    }
                )
        write_json(root / "calibration.json", {"checks": calibration})
        records, vectors = [], []
        for label, meta in metadata.items():
            source = sources[label]["conditions"]["original/palette_mean"]
            loaded = {c: _checked_artifact(a) for c, a in source["artifacts"].items()}
            originals = {c: a[0][40:280].copy() for c, a in loaded.items()}
            fills = {c: fill_color("palette_mean", originals[c]) for c in ("mm", "jj")}
            multisets = {
                c: s["canvas_rgb_multiset_sha256"] for c, s in source["cells"].items()
            }
            payloads = {
                c: s["processor_non_pixel_payload"] for c, s in source["cells"].items()
            }
            for key, mapping in maps.items():
                print(f"all-cell audit {label}/{key}", flush=True)
                recorder = PixelRecorder(processor)
                measured, canvases = audit_level(
                    originals,
                    mapping,
                    fills,
                    multisets,
                    payloads,
                    processor=recorder,
                    normalization=normalization,
                )
                if len(recorder.pixels) != 4:
                    raise ValueError("processor call denominator differs")
                measured["artifacts"] = _materialize_cells(
                    canvases,
                    {c: a[1] for c, a in loaded.items()},
                    root / "inputs" / key / label,
                    {
                        "arm": key,
                        "mapping_sha256": report["maps"][key]["sha256"],
                        "padding_policy": "palette_mean",
                    },
                )
                row = {"pair_id": label, **meta, "map": key, **measured}
                records.append(row)
                write_json(root / "blocks" / key / f"{label}.json", row)
                if key.endswith("reference") or meta["replicate"] in (3, 4):
                    arrays = {
                        "raw_rgb": {
                            c: canvases[c].astype(np.float64) / 255 for c in CELLS
                        },
                        "processor_pixel_values": {
                            c: p.astype(np.float64)
                            for c, p in zip(CELLS, recorder.pixels)
                        },
                    }
                    for view, cells in arrays.items():
                        value = input_interaction(cells)
                        path = root / "vectors" / view / key / f"{label}.npy"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(path, value)
                        vectors.append(
                            {
                                "view": view,
                                "map": key,
                                "pair_id": label,
                                "path": str(path.resolve()),
                                "sha256": sha256_file(path),
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                                "norm": float(np.linalg.norm(value)),
                            }
                        )
        if (
            len(records) * 4 != config["input_cells"]
            or len(vectors) != 2 * config["vectors_per_view"]
            or len(calibration) != 128
        ):
            raise ValueError("measurement denominator differs")
        write_json(root / "vectors.json", {"vectors": vectors})
        counts = {
            key: {
                "accepted_blocks": sum(
                    r["gate"]["accepted"] for r in records if r["map"] == key
                ),
                "total_blocks": 32,
            }
            for key in maps
        }
        summary = {
            "schema_version": 1,
            "analysis_kind": "controlled_overlap_input_audit",
            "frozen_specification": frozen,
            "frozen_specification_sha256": sha256_file(
                root / "frozen_specification.json"
            ),
            "vectors_sha256": sha256_file(root / "vectors.json"),
            "calibration_sha256": sha256_file(root / "calibration.json"),
            "records": records,
            "maps": counts,
            "input_cells": len(records) * 4,
            "historical_processor_checks": len(calibration),
            "processor_calls": len(records) * 4 + len(calibration),
            "cache_panel_eligible": all(
                c["accepted_blocks"] == 32 for c in counts.values()
            ),
            "new_model_forwards": 0,
            "model_weights_loaded": False,
            "confirmation_cohort": "not_generated_not_sealed",
        }
        write_json(root / "input_summary.json", summary)
        print(
            json.dumps(
                {
                    "input_cells": summary["input_cells"],
                    "cache_panel_eligible": summary["cache_panel_eligible"],
                }
            ),
            flush=True,
        )
    except Exception as exc:
        write_json(
            root / "execution_failure.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Registered development-only controlled overlap input study."
    )
    parser.add_argument("mode", choices=("maps", "measure"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--map-report", type=Path)
    args = parser.parse_args()
    if args.mode == "maps":
        prepare_maps(args.output_root.resolve())
    else:
        if args.map_report is None:
            parser.error("measure requires --map-report")
        measure(args.map_report.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
