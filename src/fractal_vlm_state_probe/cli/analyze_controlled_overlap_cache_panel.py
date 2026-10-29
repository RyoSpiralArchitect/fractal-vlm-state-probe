from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_tensor_artifact import load_cache_tensor_artifact
from fractal_vlm_state_probe.cache_tensor_factorial import (
    analyze_cache_tensor_factorial,
)
from fractal_vlm_state_probe.cli.analyze_controlled_overlap import pair_scores
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.controlled_overlap_cache import (
    CELLS,
    TARGETS,
    planned_panel_blocks,
    require_complete_calibration,
)
from fractal_vlm_state_probe.controlled_overlap_cache_panel import (
    VIEWS,
    direction,
    ordered_jobs,
    summarize_cube,
)
from fractal_vlm_state_probe.stimulus import write_json


def load_capture(root):
    if (root / "execution_failure.json").exists():
        raise ValueError("failed capture cannot acquire primary results")
    summary, frozen = (
        _read(root / "capture_summary.json"),
        _read(root / "frozen_specification.json"),
    )
    if (
        summary["status"] != "COMPLETE"
        or summary["panel_source_cells"] != 2048
        or summary["panel_tensor_sidecars"] != 6144
        or not summary["same_model_instance_for_calibration_and_panel"]
        or summary["calibration_source_cells"] != 128
        or summary["calibration_tensor_comparisons"] != 384
    ):
        raise ValueError("capture completion/counts differ")
    for filename, digest in summary["receipts"].items():
        if _sha(root / filename) != digest:
            raise ValueError("capture receipt changed")
    for group in ("code_sha256", "source_sha256", "manifest_sha256"):
        for name, digest in frozen[group].items():
            if _sha(REPO / name if group == "code_sha256" else Path(name)) != digest:
                raise ValueError("frozen code/source changed")
    registry = _read(root / "capture_registry.json")
    journal = [
        json.loads(line) for line in (root / "captures.jsonl").read_text().splitlines()
    ]
    if (
        len(journal) != 2176
        or len(registry) != 2176
        or {r["path"]: r for r in journal} != registry
    ):
        raise ValueError("capture journal coverage differs")
    jobs = ordered_jobs(frozen["hierarchy_metadata"])
    expected_paths = [
        f"calibration/{j['pair_id']}_{j['cell']}.json" for j in frozen["jobs"]
    ]
    expected_paths += [
        f"panel/{j['phase']}/{j['map']}/{j['pair_id']}_{j['cell']}.json" for j in jobs
    ]
    if [r["path"] for r in journal] != expected_paths or [
        r["ordinal"] for r in journal
    ] != list(range(1, 2177)):
        raise ValueError("calibration/reference/test capture order differs")
    refs = _read(root / "frozen_panel_references.json")
    seal = _read(root / "reference_seal.json")
    if (
        refs != {r["path"]: r for r in journal[128:384]}
        or len(refs) != 256
        or seal
        != {
            "sha256": _sha(root / "frozen_panel_references.json"),
            "reference_cells": 256,
            "completed_cells_at_seal": 384,
            "first_test_ordinal": 385,
        }
    ):
        raise ValueError("reference seal differs")
    calibration = _read(root / "calibration.json")
    if calibration["status"] != "PASS":
        raise ValueError("historical calibration failed")
    require_complete_calibration(calibration["checks"], frozen["hierarchy_metadata"])
    for filename, receipt in registry.items():
        p = root / filename
        if (
            _sha(p) != receipt["run_sha256"]
            or _sha(p.with_suffix(".input_audit.json")) != receipt["input_audit_sha256"]
        ):
            raise ValueError("capture output changed")
        if (
            _read(p)["stream_events"][0]["cache_tensor_artifacts"]
            != receipt["tensor_artifacts"]
        ):
            raise ValueError("tensor registry differs")
        for artifact in receipt["tensor_artifacts"]:
            load_cache_tensor_artifact(p, artifact)
    return summary, frozen, registry


def analyze(root):
    output = root / "analysis"
    if output.exists():
        raise FileExistsError("analysis output already exists; retain it")
    capture, frozen, registry = load_capture(root)
    output.mkdir()
    try:
        metadata, config = frozen["hierarchy_metadata"], frozen["config"]
        labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
        families = sorted({m["pairing_family"] for m in metadata.values()})
        phase1 = _read(REPO / config["phase1_publication"])
        maps = phase1["map_construction"]["maps"]
        vector_paths = {f"{t}/{r}": {key: {} for key in maps} for t, r in VIEWS}
        vectors, factorials = [], []
        for index, (key, label) in enumerate(planned_panel_blocks(metadata)):
            phase = "reference" if metadata[label]["replicate"] in (1, 2) else "test"
            paths = {c: root / f"panel/{phase}/{key}/{label}_{c}.json" for c in CELLS}
            runs = {c: _read(p) for c, p in paths.items()}
            for target in TARGETS:
                layer, kind = target.split(":")
                analysis = analyze_cache_tensor_factorial(
                    runs=runs, run_paths=paths, layer_index=int(layer), tensor=kind
                )
                a_path = output / "factorials" / key / label / f"{layer}_{kind}.json"
                write_json(a_path, analysis)
                cells = {
                    c: load_cache_tensor_artifact(
                        paths[c], analysis["cells"][c]["cache_tensor_artifact"]
                    ).astype(np.float64)
                    for c in CELLS
                }
                interaction = cells["jj"] - cells["jm"] - cells["mj"] + cells["mm"]
                pre = next(r for r in analysis["regions"] if r["region"] == "pre_image")
                factorials.append(
                    {
                        "map": key,
                        "pair_id": label,
                        "target": target,
                        "path": str(a_path),
                        "sha256": _sha(a_path),
                        "pre_image_all_effects_zero": all(
                            v["l2_norm"] == 0 for v in pre["effects"].values()
                        ),
                        "interaction_image_energy_fraction": analysis[
                            "interaction_partition"
                        ]["image_energy_fraction"],
                    }
                )
                for region in config["regions"]:
                    start, stop = config["region_positions"][region]
                    v = interaction[:, :, start:stop].copy()
                    path = (
                        output
                        / "vectors"
                        / f"{layer}_{kind}"
                        / region
                        / key
                        / f"{label}.npy"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, v)
                    vector_paths[f"{target}/{region}"][key][label] = path
                    vectors.append(
                        {
                            "target": target,
                            "region": region,
                            "map": key,
                            "pair_id": label,
                            "path": str(path),
                            "sha256": _sha(path),
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "norm": float(np.linalg.norm(v)),
                        }
                    )
            if index % 32 == 31:
                print(f"factorial blocks {index + 1}/512", flush=True)
        cube = np.full((6, 4, 2, 3, 16, 8), np.nan)
        scores = []
        for vi, (target, region) in enumerate(VIEWS):
            for key, item in maps.items():
                recipe, reference = item["recipe"], item["recipe"]["reference"]
                paths = vector_paths[f"{target}/{region}"]
                result = pair_scores(paths[reference], paths[key], metadata)
                scores.append(
                    {
                        "target": target,
                        "region": region,
                        "test_map": key,
                        "reference_map": reference,
                        **recipe,
                        **result,
                    }
                )
                if result["available"] and recipe["q"] is not None:
                    cube[
                        vi,
                        recipe["mask"],
                        recipe["destination"],
                        (0, 1, 4).index(recipe["q"]),
                    ] = result["scores"]
        aggregates, endpoints, bootstrap, margins = summarize_cube(
            cube, labels, families, metadata, config
        )
        np.save(output / "paired_family_seed_margins.npy", margins)
        directions = []
        for target, region in VIEWS:
            paths = vector_paths[f"{target}/{region}"]
            for mask in range(4):
                for destination in range(2):
                    zero = f"m{mask}_q0_d{destination}"
                    for label in labels:
                        a = np.load(paths[zero][label], allow_pickle=False)
                        for q in (1, 4):
                            key = f"m{mask}_q{q}_d{destination}"
                            directions.append(
                                {
                                    "target": target,
                                    "region": region,
                                    "mask": mask,
                                    "destination": destination,
                                    "pair_id": label,
                                    "q_sixteenths": q,
                                    **direction(
                                        a,
                                        np.load(paths[key][label], allow_pickle=False),
                                    ),
                                }
                            )
        ledgers = {}
        for name, rows in (
            ("vectors.jsonl", vectors),
            ("factorials.jsonl", factorials),
            ("scores.jsonl", scores),
            ("directions.jsonl", directions),
        ):
            path = output / name
            path.write_text(
                "".join(
                    json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n"
                    for r in rows
                )
            )
            ledgers[name] = {"rows": len(rows), "sha256": _sha(path)}
        result = {
            "schema_version": 1,
            "analysis_kind": "controlled_overlap_cache_correspondence",
            "status": "COMPLETE",
            "capture_summary_sha256": _sha(root / "capture_summary.json"),
            "ledgers": ledgers,
            "paired_margins_sha256": _sha(output / "paired_family_seed_margins.npy"),
            "aggregates": aggregates,
            "primary_endpoints": endpoints,
            "bootstrap": bootstrap,
            "pre_image_all_effects_zero_count": sum(
                r["pre_image_all_effects_zero"] for r in factorials
            ),
            "factorial_target_analyses": len(factorials),
            "new_model_forwards": 0,
            "new_p_values": 0,
            "confirmation_cohort": "not_generated_not_sealed",
        }
        write_json(output / "summary.json", result)
        print(json.dumps(endpoints, indent=2), flush=True)
        return result
    except Exception as error:
        write_json(
            output / "analysis_failure.json",
            {
                "schema_version": 1,
                "type": type(error).__name__,
                "message": str(error),
                "primary_results_withheld": True,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run_root.resolve())


if __name__ == "__main__":
    main()
