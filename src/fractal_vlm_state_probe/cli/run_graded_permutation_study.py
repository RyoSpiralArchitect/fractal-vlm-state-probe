from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.frequency_control import CELLS, _materialize_cells
from fractal_vlm_state_probe.graded_permutation import (
    DENOMINATOR,
    NUMERATORS,
    audit_level,
    mapping_metrics,
    restricted_permutation,
    validate_permutation,
)
from fractal_vlm_state_probe.padding_policy import _checked_artifact, fill_color
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def read(path):
    return json.loads(Path(path).read_text())


def validate_config(config):
    required = {
        "schema_version": 1,
        "study_id": "graded_permutation_fastvlm_v1",
        "level_numerators": list(NUMERATORS),
        "level_denominator": DENOMINATOR,
        "selection_seed": 20261008,
        "permutation_rule": "restrict_historical_cycles_to_nested_selected_sites",
        "padding_policy": "palette_mean",
        "source_shape_hwc": [240, 320, 3],
        "centroid_relative_tolerance": 0.05,
        "hf_absolute_tolerance": 0.02,
        "expected_blocks": 32,
        "expected_input_cells": 896,
        "new_cache_forwards": 0,
    }
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("graded study differs from the registered specification")


def run(args):
    config = read(args.config)
    validate_config(config)
    prior, published, frequency = (
        read(args.padding_receipt),
        read(args.published_padding),
        read(args.frequency_receipt),
    )
    if (
        sha256_file(args.padding_receipt) != published["source_report_sha256"]
        or sha256_file(args.frequency_receipt)
        != prior["frozen_specification"]["source_receipt_sha256"]
        or sha256_file(args.full_permutation) != frequency["permutation_sha256"]
        or prior["frozen_specification"] != published["frozen_specification"]
    ):
        raise ValueError("historical input receipts differ from publication")
    root = Path(__file__).resolve().parents[3]
    for path, digest in prior["frozen_specification"]["code_sha256"].items():
        if sha256_file(root / path) != digest:
            raise ValueError(f"qualified input measurement code changed: {path}")
    records = prior["records"]
    hierarchy = {}
    for r in records:
        hierarchy.setdefault(r["pairing_family"], []).append(r["replicate"])
    if (
        len(records) != 32
        or len({r["pair_id"] for r in records}) != 32
        or len(hierarchy) != 8
        or any(sorted(reps) != [1, 2, 3, 4] for reps in hierarchy.values())
    ):
        raise ValueError("all eight four-seed families are required")
    full = np.load(args.full_permutation, allow_pickle=False)
    validate_permutation(full)
    if full.shape != (240 * 320,):
        raise ValueError("historical permutation shape differs")
    selection = np.random.default_rng(config["selection_seed"]).permutation(len(full))
    maps = {
        f"selected_{n:02d}_of_08": restricted_permutation(
            full, selection, len(full) * n // DENOMINATOR
        )
        for n in NUMERATORS
    }
    if not np.array_equal(maps["selected_08_of_08"], full):
        raise ValueError("full endpoint permutation does not reproduce")
    if args.output_root.exists():
        raise FileExistsError("use a fresh graded-study output root")
    snapshot = args.fastvlm_model_processor_snapshot
    expected = prior["frozen_specification"]["processor_provenance"]
    if (
        sha256_file(snapshot / "processing_fastvlm.py")
        != expected["implementation_sha256"]
    ):
        raise ValueError("qualified processor implementation changed")
    import torch
    from mlx_vlm.utils import load_processor

    torch.set_num_threads(1)
    processor = load_processor(snapshot, add_detokenizer=False, trust_remote_code=True)
    cls = type(processor.image_processor)
    provenance = {
        "snapshot_revision": snapshot.name,
        "preprocessor_config_sha256": sha256_file(
            snapshot / "preprocessor_config.json"
        ),
        "implementation": f"{cls.__module__}.{cls.__name__}",
        "implementation_sha256": sha256_file(Path(inspect.getfile(cls))),
        "processor_mode": "model_loading_path",
    }
    if provenance != expected:
        raise ValueError("qualified processor provenance changed")
    normalization = read(snapshot / "preprocessor_config.json")
    if (
        not normalization["do_normalize"]
        or not normalization["do_rescale"]
        or normalization["rescale_factor"] != 1 / 255
    ):
        raise ValueError("unsupported normalization")
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    if runtime != prior["frozen_specification"]["runtime"]:
        raise ValueError("qualified processor runtime changed")
    # Verify every fixed endpoint before creating a new measurement root.
    for record in records:
        for state in ("original", "shared_pixel_permutation"):
            endpoint = record["conditions"][f"{state}/palette_mean"]
            for artifact in endpoint["artifacts"].values():
                rgb, _ = _checked_artifact(artifact)
                if rgb.shape != (320, 320, 3):
                    raise ValueError("historical square endpoint shape differs")
    args.output_root.mkdir(parents=True)
    np.save(args.output_root / "selection_order.npy", selection)
    map_records = {}
    for n, (key, mapping) in zip(NUMERATORS, maps.items()):
        path = args.output_root / f"{key}.npy"
        np.save(path, mapping)
        map_records[key] = {
            "selected_numerator": n,
            "selected_denominator": DENOMINATOR,
            "selected_count": len(full) * n // DENOMINATOR,
            "selected_fraction": n / DENOMINATOR,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            **mapping_metrics(mapping, (240, 320)),
        }
    code_paths = [
        "src/fractal_vlm_state_probe/graded_permutation.py",
        "src/fractal_vlm_state_probe/cli/run_graded_permutation_study.py",
        "src/fractal_vlm_state_probe/stimulus.py",
        *prior["frozen_specification"]["code_sha256"],
    ]
    frozen = {
        "config": config,
        "registered_design_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_paths": {
            k: str(getattr(args, k).resolve())
            for k in (
                "padding_receipt",
                "published_padding",
                "frequency_receipt",
                "full_permutation",
            )
        },
        "source_sha256": {
            k: sha256_file(getattr(args, k))
            for k in (
                "padding_receipt",
                "published_padding",
                "frequency_receipt",
                "full_permutation",
            )
        },
        "code_sha256": {p: sha256_file(root / p) for p in code_paths},
        "processor_provenance": provenance,
        "runtime": runtime,
        "selection_order_sha256": sha256_file(args.output_root / "selection_order.npy"),
        "maps": map_records,
    }
    write_json(args.output_root / "frozen_specification.json", frozen)
    measured_records = []
    for record in records:
        pair = record["pair_id"]
        baseline = record["conditions"]["original/palette_mean"]
        endpoint = record["conditions"]["shared_pixel_permutation/palette_mean"]
        loaded = {c: _checked_artifact(a) for c, a in baseline["artifacts"].items()}
        originals = {c: value[0][40:280].copy() for c, value in loaded.items()}
        sources = {c: value[1] for c, value in loaded.items()}
        fills = {c: fill_color("palette_mean", originals[c]) for c in ("mm", "jj")}
        multiset = {
            c: r["canvas_rgb_multiset_sha256"] for c, r in baseline["cells"].items()
        }
        payloads = {
            c: r["processor_non_pixel_payload"] for c, r in baseline["cells"].items()
        }
        result = {
            k: record[k]
            for k in ("pair_id", "pairing_family", "broad_class", "replicate")
        }
        result["levels"] = {}
        for key, mapping in maps.items():
            print(f"input audit {pair}/{key}", flush=True)
            measured, canvases = audit_level(
                originals,
                mapping,
                fills,
                multiset,
                payloads,
                processor=processor,
                normalization=normalization,
            )
            expected_endpoint = (
                baseline
                if key == "selected_00_of_08"
                else endpoint
                if key == "selected_08_of_08"
                else None
            )
            for cell in CELLS:
                measured["cells"][cell]["endpoint_pixel_and_stats_reproduced"] = None
                if expected_endpoint is not None:
                    expected_row = expected_endpoint["cells"][cell]
                    old_canvas, _ = _checked_artifact(
                        expected_endpoint["artifacts"][cell]
                    )
                    if (
                        not np.array_equal(canvases[cell], old_canvas)
                        or any(
                            measured["cells"][cell][k] != expected_row[k]
                            for k in (
                                "pixel_values_sha256",
                                "pixel_values_shape",
                                "pixel_values_dtype",
                                "processor",
                                "raw",
                                "processor_non_pixel_payload",
                            )
                        )
                        or measured["gate"] != expected_endpoint["gate"]
                    ):
                        raise ValueError(
                            f"historical endpoint differs: {pair}/{key}/{cell}"
                        )
                    measured["cells"][cell]["endpoint_pixel_and_stats_reproduced"] = (
                        True
                    )
            measured["artifacts"] = _materialize_cells(
                canvases,
                sources,
                args.output_root / "inputs" / key / pair,
                {
                    "arm": key,
                    "padding_policy": "palette_mean",
                    "mapping_sha256": map_records[key]["sha256"],
                    "marginal_condition": "fixed_registered_expanded_palette",
                },
            )
            for cell, artifact in measured["artifacts"].items():
                serialized, _ = _checked_artifact(artifact)
                if not np.array_equal(serialized, canvases[cell]):
                    raise ValueError("serialized graded canvas differs")
            result["levels"][key] = measured
        measured_records.append(result)
        write_json(args.output_root / "blocks" / f"{pair}.json", result)
    levels = {}
    for key in maps:
        accepted = [r for r in measured_records if r["levels"][key]["gate"]["accepted"]]
        complete = sorted(
            f
            for f in hierarchy
            if {r["replicate"] for r in accepted if r["pairing_family"] == f}
            == {1, 2, 3, 4}
        )
        levels[key] = {
            "accepted_blocks": len(accepted),
            "total_blocks": len(records),
            "accepted_pair_ids": [r["pair_id"] for r in accepted],
            "complete_four_seed_families": complete,
            "all_input_blocks_accepted": len(accepted) == 32,
        }
    summary = {
        "schema_version": 1,
        "analysis_kind": "graded_permutation_input_audit",
        "date": config["registered_date"],
        "frozen_specification": frozen,
        "frozen_specification_sha256": sha256_file(
            args.output_root / "frozen_specification.json"
        ),
        "block_count": len(records),
        "input_cells": len(records) * 4 * len(maps),
        "levels": levels,
        "records": measured_records,
        "new_cache_forwards": 0,
        "model_weights_loaded": False,
        "claim_boundaries": [
            "Selection fraction is not changed-RGB fraction or a monotone spectral/semantic dose.",
            "Every level is one panel-wide common bijection with a nested selected-site set.",
            "Expanded RGB marginals are fixed across levels, not equal between different palette donors or to unpadded inputs.",
            "Only two within-level frequency summaries are gated; cross-level spectral equality is not established.",
            "The same 32 previously observed blocks are reused, not independent new image samples.",
            "Input eligibility does not predict cache correspondence; no cache result is added here.",
        ],
    }
    write_json(args.output_root / "graded_permutation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Audit graded common permutations under fixed palette-mean padding."
    )
    for name in (
        "config",
        "padding-receipt",
        "published-padding",
        "frequency-receipt",
        "full-permutation",
        "fastvlm-model-processor-snapshot",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    owned_root = not args.output_root.exists()
    try:
        summary = run(args)
    except Exception as error:
        if owned_root and args.output_root.exists():
            write_json(
                args.output_root / "execution_failure.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    for key, level in summary["levels"].items():
        print(
            f"{key}: {level['accepted_blocks']}/32 blocks, {len(level['complete_four_seed_families'])}/8 complete families",
            flush=True,
        )
    print(f"wrote {args.output_root / 'graded_permutation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
