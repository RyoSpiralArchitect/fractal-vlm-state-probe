from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import platform
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_graded_permutation_study import read
from fractal_vlm_state_probe.frequency_control import CELLS, _materialize_cells
from fractal_vlm_state_probe.graded_permutation import audit_level, mapping_metrics
from fractal_vlm_state_probe.padding_policy import _checked_artifact, fill_color
from fractal_vlm_state_probe.permutation_maps import (
    ANCHOR,
    MAPS,
    NEW_SEEDS,
    map_overlap,
    seeded_map,
    summarize_input_maps,
)
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def validate_config(config):
    required = {
        "schema_version": 1,
        "study_id": "map_seed_input_fastvlm_v1",
        "selected_numerator": 6,
        "selected_denominator": 8,
        "anchor_level": "selected_06_of_08",
        "new_map_seeds": list(NEW_SEEDS),
        "rng_rule": "SeedSequence_spawn_2_selection_then_full_PCG64",
        "permutation_rule": "restrict_cycles_to_selected_sites",
        "padding_policy": "palette_mean",
        "source_shape_hwc": [240, 320, 3],
        "centroid_relative_tolerance": 0.05,
        "hf_absolute_tolerance": 0.02,
        "expected_blocks": 32,
        "expected_input_cells": 512,
        "cache_gate": "all_three_new_maps_and_anchor_accept_all_32_blocks",
        "new_cache_forwards": 0,
    }
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError("map input study differs from registration")


def run(args):
    config = read(args.config)
    validate_config(config)
    prior, publication = read(args.graded_receipt), read(args.published_graded)
    if sha256_file(args.graded_receipt) != publication["source_report_sha256"]:
        raise ValueError("graded receipt differs from publication")
    old = prior["frozen_specification"]
    if old != publication["frozen_specification"]:
        raise ValueError("graded specification differs")
    repo = Path(__file__).resolve().parents[3]
    for path, digest in old["code_sha256"].items():
        if sha256_file(repo / path) != digest:
            raise ValueError(f"qualified code changed: {path}")
    for key, digest in old["source_sha256"].items():
        if sha256_file(Path(old["source_paths"][key])) != digest:
            raise ValueError(f"historical input changed: {key}")
    baseline = read(old["source_paths"]["padding_receipt"])
    sources = {r["pair_id"]: r for r in baseline["records"]}
    if len(prior["records"]) != 32 or len(sources) != 32:
        raise ValueError("expected 32 input blocks")
    anchor_receipt = old["maps"][config["anchor_level"]]
    if sha256_file(Path(anchor_receipt["path"])) != anchor_receipt["sha256"]:
        raise ValueError("historical anchor mapping changed")
    anchor = np.load(anchor_receipt["path"], allow_pickle=False)
    if anchor.shape != (76800,):
        raise ValueError("anchor map shape differs")
    maps, ingredients = {ANCHOR: anchor}, {}
    for seed in NEW_SEEDS:
        key = f"seed_{seed}"
        full, selection, maps[key] = seeded_map(seed, len(anchor))
        ingredients[key] = {"full": full, "selection": selection}
    if len({m.tobytes() for m in maps.values()}) != len(MAPS):
        raise ValueError("registered maps are not distinct")
    expected = old["processor_provenance"]
    snapshot = args.fastvlm_model_processor_snapshot
    if (
        snapshot.name != expected["snapshot_revision"]
        or sha256_file(snapshot / "processing_fastvlm.py")
        != expected["implementation_sha256"]
        or sha256_file(snapshot / "preprocessor_config.json")
        != expected["preprocessor_config_sha256"]
    ):
        raise ValueError("qualified processor snapshot changed")
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    if runtime != old["runtime"]:
        raise ValueError("qualified processor runtime changed")
    import torch
    from mlx_vlm.utils import load_processor

    torch.set_num_threads(1)
    processor = load_processor(snapshot, add_detokenizer=False, trust_remote_code=True)
    cls = type(processor.image_processor)
    if (
        f"{cls.__module__}.{cls.__name__}" != expected["implementation"]
        or sha256_file(Path(inspect.getfile(cls))) != expected["implementation_sha256"]
    ):
        raise ValueError("loaded processor identity differs")
    normalization = read(snapshot / "preprocessor_config.json")
    root = args.output_root
    if root.exists():
        raise FileExistsError("use a fresh map-input study root")
    root.mkdir(parents=True)
    args.output_owned = True
    map_receipts = {}
    for key, mapping in maps.items():
        path = root / f"{key}.npy"
        np.save(path, mapping)
        components = {}
        for name, value in ingredients.get(key, {}).items():
            component_path = root / f"{key}_{name}.npy"
            np.save(component_path, value)
            components[name] = {
                "path": str(component_path.resolve()),
                "sha256": sha256_file(component_path),
            }
        map_receipts[key] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "selected_count": 57600,
            "selected_fraction": 0.75,
            "ingredients": components,
            **mapping_metrics(mapping, (240, 320)),
        }
    code_paths = [
        "src/fractal_vlm_state_probe/permutation_maps.py",
        "src/fractal_vlm_state_probe/cli/run_map_seed_input_study.py",
        *old["code_sha256"],
    ]
    frozen = {
        "config": config,
        "registered_design_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "graded_receipt_path": str(args.graded_receipt.resolve()),
        "graded_receipt_sha256": sha256_file(args.graded_receipt),
        "published_graded_path": str(args.published_graded.resolve()),
        "published_graded_sha256": sha256_file(args.published_graded),
        "processor_provenance": expected,
        "runtime": runtime,
        "maps": map_receipts,
        "code_sha256": {p: sha256_file(repo / p) for p in code_paths},
    }
    write_json(root / "frozen_specification.json", frozen)
    measured_records, anchor_checks = [], 0
    for record in prior["records"]:
        pair = record["pair_id"]
        baseline = sources[pair]["conditions"]["original/palette_mean"]
        loaded = {c: _checked_artifact(a) for c, a in baseline["artifacts"].items()}
        originals = {c: v[0][40:280].copy() for c, v in loaded.items()}
        fills = {c: fill_color("palette_mean", originals[c]) for c in ("mm", "jj")}
        hashes = {
            c: r["canvas_rgb_multiset_sha256"] for c, r in baseline["cells"].items()
        }
        payloads = {
            c: r["processor_non_pixel_payload"] for c, r in baseline["cells"].items()
        }
        result = {
            k: record[k]
            for k in ("pair_id", "pairing_family", "broad_class", "replicate")
        }
        if any(result[k] != sources[pair][k] for k in result):
            raise ValueError("source hierarchy differs")
        result["maps"] = {}
        for key, mapping in maps.items():
            print(f"input audit {pair}/{key}", flush=True)
            measured, canvases = audit_level(
                originals,
                mapping,
                fills,
                hashes,
                payloads,
                processor=processor,
                normalization=normalization,
            )
            if key == ANCHOR:
                previous = record["levels"][config["anchor_level"]]
                if measured["gate"] != previous["gate"]:
                    raise ValueError("historical anchor gate changed")
                for cell in CELLS:
                    expected_row = dict(previous["cells"][cell])
                    expected_row.pop("endpoint_pixel_and_stats_reproduced", None)
                    rgb, _ = _checked_artifact(previous["artifacts"][cell])
                    if measured["cells"][cell] != expected_row or not np.array_equal(
                        rgb, canvases[cell]
                    ):
                        raise ValueError(
                            "historical anchor pixels or statistics differ"
                        )
                    anchor_checks += 1
            measured["artifacts"] = _materialize_cells(
                canvases,
                {c: v[1] for c, v in loaded.items()},
                root / "inputs" / key / pair,
                {
                    "arm": key,
                    "padding_policy": "palette_mean",
                    "mapping_sha256": map_receipts[key]["sha256"],
                    "selected_fraction": 0.75,
                },
            )
            for cell, artifact in measured["artifacts"].items():
                if not np.array_equal(_checked_artifact(artifact)[0], canvases[cell]):
                    raise ValueError("serialized map pixels differ")
            result["maps"][key] = measured
        measured_records.append(result)
        write_json(root / "blocks" / f"{pair}.json", result)
    counts = summarize_input_maps(measured_records)
    summary = {
        "schema_version": 1,
        "analysis_kind": "fixed_fraction_map_seed_input_audit",
        "date": config["registered_date"],
        "frozen_specification": frozen,
        "frozen_specification_sha256": sha256_file(root / "frozen_specification.json"),
        "records": measured_records,
        "maps": counts,
        "input_cells": 512,
        "anchor_exact_cell_checks": anchor_checks,
        "cache_panel_eligible": all(
            v["accepted_blocks"] == 32 for v in counts.values()
        ),
        "map_overlap": [
            {"left": a, "right": b, **map_overlap(maps[a], maps[b])}
            for a, b in combinations(MAPS, 2)
        ],
        "new_cache_forwards": 0,
        "model_weights_loaded": False,
        "claim_boundaries": [
            "Both selected positions and permutation destinations change with map seed; their separate effects are not identified.",
            "Fixed expanded RGB marginals and padding do not fix processor spectra between maps.",
            "All 32 reused source blocks and all four maps remain in the denominator; no replacement seeds or tolerance changes.",
            "Cache comparison requires all four maps to accept all 32 blocks; no cache inference is made by input eligibility.",
        ],
    }
    write_json(root / "map_seed_input_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Audit three new common maps at a fixed 3/4 selected-site fraction."
    )
    for name in (
        "config",
        "graded-receipt",
        "published-graded",
        "fastvlm-model-processor-snapshot",
        "output-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as error:
        if getattr(args, "output_owned", False):
            write_json(
                args.output_root / "execution_failure.json",
                {"type": type(error).__name__, "message": str(error)},
            )
        raise
    for key, result in summary["maps"].items():
        print(
            f"{key}: {result['accepted_blocks']}/32 accepted; {len(result['complete_four_seed_families'])}/8 complete families"
        )
    print(f"cache_panel_eligible={summary['cache_panel_eligible']}")


if __name__ == "__main__":
    main()
