from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import platform
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe.cache_direction_holdout import (
    validate_pairing_holdout_metadata,
)
from fractal_vlm_state_probe.cli.analyze_pairing_input_holdout import input_interaction
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.cli.summarize_map_seed_input_study import (
    verify_panel_counts,
)
from fractal_vlm_state_probe.frequency_control import CELLS
from fractal_vlm_state_probe.map_seed_input_baseline import (
    VIEWS,
    analyze_vectors,
    validate_config,
)
from fractal_vlm_state_probe.padding_policy import _checked_artifact
from fractal_vlm_state_probe.permutation_maps import MAPS
from fractal_vlm_state_probe.stimulus import write_json


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def checked_pixels(rgb, processor, expected, audit):
    payload = processor(
        images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
    )
    pixels = np.asarray(payload["pixel_values"])
    non_pixels = {
        k: np.asarray(v).tolist() for k, v in payload.items() if k != "pixel_values"
    }
    digest = array_hash(pixels)
    if (
        pixels.shape != (1, 3, 1024, 1024)
        or str(pixels.dtype) != "float32"
        or not np.isfinite(pixels).all()
        or list(pixels.shape[1:]) != expected["pixel_values_shape"]
        or str(pixels.dtype) != expected["pixel_values_dtype"]
        or digest != expected["pixel_values_sha256"]
        or non_pixels != expected["processor_non_pixel_payload"]
        or digest != audit["effective"]["pixel_values"]["sha256"]
        or audit["native"] != audit["effective"]
        or audit["override_image_sizes"] is not None
    ):
        raise ValueError("processor pixels or metadata differ from qualified inputs")
    return pixels, digest


def prepare(args):
    config = _read(args.config)
    validate_config(config)
    study, published, cache = (
        _read(p)
        for p in (args.input_summary, args.published_input, args.cache_publication)
    )
    if (
        _sha(args.input_summary) != published["source_report_sha256"]
        or study["frozen_specification"] != published["frozen_specification"]
        or _sha(args.cache_root / "study_summary.json")
        != cache["source_summary_sha256"]
        or cache["frozen_specification"]["input_summary_sha256"]
        != _sha(args.input_summary)
        or cache["calibration_status"] != "PASS"
        or not verify_panel_counts(study["records"], study["maps"])
    ):
        raise ValueError("qualified input/cache receipt differs")
    metadata = {
        r["pair_id"]: {k: r[k] for k in ("pairing_family", "broad_class", "replicate")}
        for r in study["records"]
    }
    validate_pairing_holdout_metadata(metadata)
    if metadata != cache["frozen_specification"]["hierarchy_metadata"]:
        raise ValueError("input/cache hierarchy differs")
    registry_path = args.cache_root / "capture_registry.json"
    if _sha(registry_path) != cache["capture_registry_sha256"]:
        raise ValueError("cache capture registry changed")
    registry = _read(registry_path)
    repo = Path(__file__).resolve().parents[3]
    code = dict(cache["frozen_specification"]["code_sha256"])
    code.update(study["frozen_specification"]["code_sha256"])
    for name, digest in code.items():
        if _sha(repo / name) != digest:
            raise ValueError(f"qualified code changed: {name}")
    for name in (
        "src/fractal_vlm_state_probe/map_seed_input_baseline.py",
        "src/fractal_vlm_state_probe/cli/run_map_seed_input_baseline.py",
        "src/fractal_vlm_state_probe/cli/analyze_pairing_input_holdout.py",
        "src/fractal_vlm_state_probe/cli/summarize_map_seed_input_study.py",
    ):
        code[name] = _sha(repo / name)
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    if runtime != study["frozen_specification"]["runtime"]:
        raise ValueError("qualified input runtime changed")
    snapshot = args.processor_snapshot.resolve()
    expected = study["frozen_specification"]["processor_provenance"]
    if (
        snapshot.name != expected["snapshot_revision"]
        or _sha(snapshot / "processing_fastvlm.py") != expected["implementation_sha256"]
        or _sha(snapshot / "preprocessor_config.json")
        != expected["preprocessor_config_sha256"]
    ):
        raise ValueError("qualified processor snapshot changed")
    maps = {}
    for key, receipt in study["frozen_specification"]["maps"].items():
        if _sha(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("registered map changed")
        maps[key] = np.load(receipt["path"], allow_pickle=False)
    if set(maps) != set(MAPS):
        raise ValueError("map coverage differs")
    sources = {
        str(p.resolve()): _sha(p)
        for p in (
            args.config,
            args.input_summary,
            args.published_input,
            args.cache_publication,
            args.cache_root / "study_summary.json",
            registry_path,
        )
    }
    frozen = {
        "config": config,
        "runtime": runtime,
        "processor_provenance": expected,
        "processor_snapshot": str(snapshot),
        "hierarchy_metadata": metadata,
        "maps": study["frozen_specification"]["maps"],
        "code_sha256": code,
        "source_sha256": sources,
        "cache_root": str(args.cache_root.resolve()),
        "input_summary": str(args.input_summary.resolve()),
        "cache_publication": str(args.cache_publication.resolve()),
    }
    return study, registry, maps, frozen


def execute(args):
    study, registry, maps, frozen = prepare(args)
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    for name, digest in {
        **frozen["code_sha256"],
        str(args.config.resolve().relative_to(repo)): _sha(args.config),
    }.items():
        if (
            hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
            ).hexdigest()
            != digest
        ):
            raise ValueError(f"register current code/config before measurement: {name}")
    if args.output_root.exists():
        raise FileExistsError("use a fresh input baseline root")
    import torch
    from mlx_vlm.utils import load_processor

    torch.set_num_threads(1)
    processor = load_processor(
        args.processor_snapshot, add_detokenizer=False, trust_remote_code=True
    )
    cls, expected = type(processor.image_processor), frozen["processor_provenance"]
    if (
        f"{cls.__module__}.{cls.__name__}" != expected["implementation"]
        or _sha(Path(inspect.getfile(cls))) != expected["implementation_sha256"]
    ):
        raise ValueError("loaded processor identity differs")
    root = args.output_root.resolve()
    root.mkdir(parents=True)
    args.output_owned = True
    frozen["registered_design_commit"] = commit
    write_json(root / "frozen_specification.json", frozen)
    cells, vectors = [], []
    paths = {v: {m: {} for m in MAPS} for v in VIEWS}
    for phase, replicates in (("reference", (1, 2)), ("test", (3, 4))):
        for record in study["records"]:
            if record["replicate"] not in replicates:
                continue
            label = record["pair_id"]
            for key in MAPS:
                print(f"input vectors {phase}/{key}/{label}", flush=True)
                measured = record["maps"][key]
                arrays = {v: {} for v in VIEWS}
                for cell in CELLS:
                    artifact, expected = (
                        measured["artifacts"][cell],
                        measured["cells"][cell],
                    )
                    rgb, _ = _checked_artifact(artifact)
                    if (
                        rgb.shape != (320, 320, 3)
                        or array_hash(rgb) != expected["canvas_rgb_bytes_sha256"]
                    ):
                        raise ValueError("raw canvas differs from the input audit")
                    receipt = registry[f"panel/{phase}/{key}/{label}_{cell}.json"]
                    audit_path = Path(receipt["input_audit_path"])
                    if _sha(audit_path) != receipt["input_audit_sha256"]:
                        raise ValueError("actual cache-input audit changed")
                    pixels, digest = checked_pixels(
                        rgb, processor, expected, _read(audit_path)
                    )
                    arrays["raw_rgb"][cell] = rgb.astype(np.float64) / 255
                    arrays["processor_pixel_values"][cell] = pixels.astype(np.float64)
                    cells.append(
                        {
                            "pair_id": label,
                            "map": key,
                            "cell": cell,
                            "phase": phase,
                            "artifact": artifact,
                            "canvas_rgb_bytes_sha256": array_hash(rgb),
                            "processor_pixels_sha256": digest,
                            "pixel_values_shape": list(pixels.shape),
                            "pixel_values_dtype": str(pixels.dtype),
                            "cache_input_audit_path": str(audit_path),
                            "cache_input_audit_sha256": _sha(audit_path),
                            "raw_exact": True,
                            "processor_exact": True,
                        }
                    )
                for view in VIEWS:
                    vector = input_interaction(arrays[view])
                    if not np.isfinite(vector).all():
                        raise ValueError("nonfinite input interaction")
                    path = root / "vectors" / view / key / f"{label}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, vector)
                    paths[view][key][label] = str(path)
                    vectors.append(
                        {
                            "view": view,
                            "map": key,
                            "pair_id": label,
                            "phase": phase,
                            "path": str(path),
                            "sha256": _sha(path),
                            "shape": list(vector.shape),
                            "dtype": str(vector.dtype),
                            "l2_norm": float(np.linalg.norm(vector)),
                        }
                    )
                del arrays
        if phase == "reference":
            if len(vectors) != 128 or len(cells) != 256:
                raise ValueError("reference denominator differs")
            write_json(root / "frozen_references.json", vectors)
    if len(vectors) != 256 or len(cells) != 512:
        raise ValueError("input capture denominator differs")
    write_json(root / "cells.json", cells)
    write_json(root / "vectors.json", vectors)
    print("computing registered input map matrix", flush=True)
    result = analyze_vectors(paths, frozen["hierarchy_metadata"], maps)
    result.update(
        schema_version=1,
        analysis_kind="fixed_fraction_map_seed_input_baseline",
        date=frozen["config"]["registered_date"],
        frozen_specification_sha256=_sha(root / "frozen_specification.json"),
        frozen_references_sha256=_sha(root / "frozen_references.json"),
        cells_sha256=_sha(root / "cells.json"),
        vectors_sha256=_sha(root / "vectors.json"),
        input_cells=512,
        interaction_vectors=256,
        new_cache_forwards=0,
        model_weights_loaded=False,
    )
    write_json(root / "study_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Measure fixed-map correspondence in qualified input coordinates."
    )
    for name in (
        "config",
        "input-summary",
        "published-input",
        "cache-root",
        "cache-publication",
        "processor-snapshot",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args)
    except Exception as exc:
        if getattr(args, "output_owned", False):
            write_json(
                args.output_root / "execution_failure.json",
                {"error": str(exc), "type": type(exc).__name__},
            )
        raise
    print(
        f"completed {result['input_cells']} cells and {result['interaction_vectors']} input vectors",
        flush=True,
    )


if __name__ == "__main__":
    main()
