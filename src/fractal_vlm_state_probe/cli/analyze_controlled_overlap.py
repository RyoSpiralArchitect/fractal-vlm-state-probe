from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_controlled_overlap import (
    committed_config,
    load_maps,
    read,
)
from fractal_vlm_state_probe.controlled_overlap import (
    margin_cube,
    paired_bootstrap,
    score_summary,
)
from fractal_vlm_state_probe.source_coordinate_overlap import COMPONENTS, partition_maps
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def pair_scores(reference, test, metadata, *, categories=None):
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})

    def unit(path):
        array = np.load(path, allow_pickle=False).astype(np.float64, copy=False).ravel()
        if not np.isfinite(array).all():
            raise ValueError("nonfinite interaction")
        norm = float(np.linalg.norm(array))
        return array / norm if norm else None

    references = []
    for f in families:
        inputs = [
            unit(reference[k])
            for k, meta in sorted(metadata.items())
            if meta["pairing_family"] == f and meta["replicate"] in (1, 2)
        ]
        if any(x is None for x in inputs):
            return {"available": False, "reason": "zero reference"}
        direction = inputs[0] + inputs[1]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return {"available": False, "reason": "cancelling reference"}
        references.append(direction / norm)
    tests = [unit(test[k]) for k in labels]
    if any(x is None for x in tests):
        return {"available": False, "reason": "zero test"}
    refs, tests = np.stack(references), np.stack(tests)
    scores = tests @ refs.T
    result = {
        "available": True,
        "labels": labels,
        "families": families,
        "scores": scores.tolist(),
        "summary": score_summary(scores, labels, families, metadata),
    }
    if categories is not None:
        codes = np.repeat(categories.ravel(), 3)
        parts = {
            c: (tests[:, codes == i] @ refs[:, codes == i].T)
            for i, c in enumerate(COMPONENTS)
        }
        error = float(np.max(np.abs(sum(parts.values()) - scores)))
        if error > 1e-12:
            raise ValueError("raw components do not reconstruct full scores")
        result["component_scores"] = {c: v.tolist() for c, v in parts.items()}
        result["reconstruction_max_abs_error"] = error
    return result


def analyze(root):
    destination = root / "analysis_summary.json"
    if destination.exists():
        raise FileExistsError("analysis already exists")
    config, _, code = committed_config()
    study = read(root / "input_summary.json")
    frozen = study["frozen_specification"]
    if (
        (root / "execution_failure.json").exists()
        or frozen["config"] != config
        or frozen["code_sha256"] != code
    ):
        raise ValueError("failed measurement or changed registration")
    for file, field in (
        ("frozen_specification.json", "frozen_specification_sha256"),
        ("vectors.json", "vectors_sha256"),
        ("calibration.json", "calibration_sha256"),
    ):
        if sha256_file(root / file) != study[field]:
            raise ValueError("measurement receipt changed")
    if sha256_file(Path(frozen["map_report"])) != frozen["map_report_sha256"]:
        raise ValueError("map report changed")
    report, maps = load_maps(Path(frozen["map_report"]))
    metadata = frozen["hierarchy_metadata"]
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    entries = read(root / "vectors.json")["vectors"]
    paths = {v: {k: {} for k in maps} for v in config["views"]}
    for item in entries:
        path = Path(item["path"])
        if (
            sha256_file(path) != item["sha256"]
            or item["pair_id"] in paths[item["view"]][item["map"]]
        ):
            raise ValueError("vector hash or duplicate differs")
        paths[item["view"]][item["map"]][item["pair_id"]] = path
    for view in paths:
        for key, values in paths[view].items():
            expected = set(metadata) if key.endswith("reference") else set(labels)
            if set(values) != expected:
                raise ValueError("saved vector coverage differs")
    rows, aggregates, endpoints = [], [], []
    cube = np.full((2, 4, 2, 3, 16, 8), np.nan)
    for vi, view in enumerate(config["views"]):
        for key, receipt in report["maps"].items():
            recipe, reference = receipt["recipe"], receipt["recipe"]["reference"]
            print(f"full-vector scoring {view}/{reference}/{key}", flush=True)
            result = pair_scores(
                paths[view][reference],
                paths[view][key],
                metadata,
                categories=partition_maps(maps[reference], maps[key])
                if view == "raw_rgb"
                else None,
            )
            row = {
                "view": view,
                "reference_map": reference,
                "test_map": key,
                **recipe,
                **result,
            }
            rows.append(row)
            if result["available"] and recipe["q"] is not None:
                cube[
                    vi,
                    recipe["mask"],
                    recipe["destination"],
                    config["overlap_sixteenths"].index(recipe["q"]),
                ] = result["scores"]
        for qi, q in enumerate(config["overlap_sixteenths"]):
            available = bool(np.isfinite(cube[vi, :, :, qi]).all())
            out = {"view": view, "q_sixteenths": q, "available": available}
            if available:
                averaged = cube[vi, :, :, qi].mean(axis=(0, 1))
                out.update(score_summary(averaged, labels, families, metadata))
                if view == "raw_rgb":
                    selected = [r for r in rows if r["view"] == view and r["q"] == q]
                    out["components"] = {
                        c: score_summary(
                            np.mean(
                                [r["component_scores"][c] for r in selected], axis=0
                            ),
                            labels,
                            families,
                            metadata,
                        )
                        for c in COMPONENTS
                    }
            else:
                out["reason"] = (
                    "at least one planned map view unavailable; no filtering"
                )
            aggregates.append(out)
    bootstrap = None
    if np.isfinite(cube).all():
        margins = margin_cube(cube, labels, families, metadata)
        bootstrap = paired_bootstrap(margins, config)
        np.save(root / "paired_family_seed_margins.npy", margins)
        for vi, view in enumerate(config["views"]):
            for ei, name in enumerate(config["primary_endpoints"]):
                values = (
                    margins[vi, :, :, 0]
                    if ei == 0
                    else margins[vi, :, :, 2] - margins[vi, :, :, 0]
                )
                per_family = values.mean(axis=(0, 1, 3))
                endpoints.append(
                    {
                        "view": view,
                        "endpoint": name,
                        "available": True,
                        "mean": float(values.mean()),
                        "family_values": dict(zip(families, per_family.tolist())),
                        "mask_values": values.mean(axis=(1, 2, 3)).tolist(),
                        "exploratory_range": bootstrap["ranges"][vi][ei],
                        "p_value": None,
                        "equivalence_tested": False,
                    }
                )
    else:
        endpoints = [
            {
                "view": view,
                "endpoint": name,
                "available": False,
                "reason": "incomplete planned paired panel",
            }
            for view in config["views"]
            for name in config["primary_endpoints"]
        ]
    ledger = root / "scores.jsonl"
    ledger.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    )
    result = {
        "schema_version": 1,
        "analysis_kind": "controlled_overlap_input_correspondence",
        "input_summary_sha256": sha256_file(root / "input_summary.json"),
        "scores_sha256": sha256_file(ledger),
        "score_views": len(rows),
        "aggregates": aggregates,
        "primary_endpoints": endpoints,
        "bootstrap": bootstrap,
        "paired_margins_sha256": sha256_file(root / "paired_family_seed_margins.npy")
        if bootstrap
        else None,
        "cache_panel_eligible": study["cache_panel_eligible"],
        "new_model_forwards": 0,
        "new_p_values": 0,
        "confirmation_cohort": "not_generated_not_sealed",
    }
    write_json(destination, result)
    print(json.dumps(endpoints, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.run_root.resolve())


if __name__ == "__main__":
    main()
