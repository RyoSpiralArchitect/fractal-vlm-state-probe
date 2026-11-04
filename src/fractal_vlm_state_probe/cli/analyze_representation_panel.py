from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from fractal_vlm_state_probe.cli.analyze_controlled_overlap import pair_scores
from fractal_vlm_state_probe.cli.analyze_nonlearned_processing_panel import (
    read_jsonl,
    write_ledger,
)
from fractal_vlm_state_probe.cli.run_representation_panel import (
    REPO,
    cal,
    check_space,
    read,
    run_path,
    sha,
)
from fractal_vlm_state_probe.cli.summarize_controlled_overlap import oracle_score
from fractal_vlm_state_probe.controlled_overlap_cache import CELLS, planned_panel_blocks
from fractal_vlm_state_probe.controlled_overlap_cache_panel import (
    direction,
    ordered_jobs,
)
from fractal_vlm_state_probe.nonlearned_processing import (
    direction_status,
    factorial_interaction,
)
from fractal_vlm_state_probe.representation_panel import (
    SITES,
    array_path,
    digest,
    expand,
    oracle_expand,
    seal,
    summarize,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json


def load_capture(root):
    if (root / "execution_failure.json").exists() or (
        root / "recalibration/execution_failure.json"
    ).exists():
        raise ValueError("failed capture; no primary results")
    capture, frozen = (
        read(root / "capture_summary.json"),
        read(root / "frozen_specification.json"),
    )
    expected = {
        "status": "COMPLETE",
        "calibration_contexts": 384,
        "panel_cells": 2048,
        "new_source_contexts": 2432,
        "site_arrays": 4096,
        "panel_cache_parity_pairs": 6144,
        "same_model_instance": True,
        "new_direct_probe_cells": 0,
    }
    names = {
        "frozen_specification.json",
        "preflight_receipt.json",
        "recalibration_parity.json",
        "runtime_receipt.json",
        "reference_cells.json",
        "reference_seal.json",
        "cells.jsonl",
        "model_snapshot_after.json",
    }
    if (
        any(capture.get(k) != v for k, v in expected.items())
        or set(capture["receipts"]) != names
    ):
        raise ValueError("capture completion or denominator differs")
    for name, h in capture["receipts"].items():
        if sha(root / name) != h:
            raise ValueError("capture receipt changed")
    validate_config(frozen["release"])
    cal.assert_frozen(frozen)
    pre_rec = read(root / "preflight_receipt.json")
    pre_dir = Path(pre_rec["path"])
    pre = read(pre_dir / "preflight.json")
    if (
        sha(pre_dir / "preflight.json") != pre_rec["sha256"]
        or (pre_dir / "execution_failure.json").exists()
        or pre["status"] != "PASS"
        or pre["new_model_forwards"]
        or pre["model_weights_loaded"]
        or pre["observer_installed"]
        or pre["frozen_specification_sha256"]
        != sha(pre_dir / "frozen_specification.json")
        or read(pre_dir / "frozen_specification.json") != frozen
        or pre["available_bytes"] < frozen["disk_budget"]["required_at_launch_bytes"]
        or pre["calibration_preflight_sha256"]
        != sha(pre_dir / "calibration_preflight/preflight.json")
        or read(root / "model_snapshot_after.json") != frozen["model_snapshot"]
    ):
        raise ValueError("preflight or final model fingerprint differs")
    calrec = read(root / "recalibration_parity.json")
    runtime = read(root / "runtime_receipt.json")
    if (
        calrec["source_cells"] != 128
        or calrec["representation_array_pairs_exact"] != 256
        or calrec["verified_summary_sha256"]
        != sha(root / "recalibration_verified/summary.json")
        or calrec["calibration_summary_sha256"]
        != sha(root / "recalibration/summary.json")
        or not runtime["calibration_and_panel_use_exact_loader_return"]
        or runtime["model_loads"] != 1
    ):
        raise ValueError("same-runtime recalibration receipt differs")
    jobs = ordered_jobs(frozen["hierarchy_metadata"])
    rows = read_jsonl(root / "cells.jsonl")
    if jobs != frozen["panel_jobs"] or len(rows) != 2048:
        raise ValueError("panel identity coverage differs")
    receipt = seal(rows[:256], jobs)
    if read(root / "reference_cells.json") != rows[:256] or read(
        root / "reference_seal.json"
    ) != {**receipt, "sha256": sha(root / "reference_cells.json")}:
        raise ValueError("reference seal differs")
    for i, (j, row) in enumerate(zip(jobs, rows)):
        rel = run_path(j)
        old = REPO / frozen["release"]["prior_panel_root"] / rel
        p = root / rel
        rep = row["representation"]
        if (
            row["ordinal"] != i + 385
            or any(row[k] != v for k, v in j.items())
            or row["path"] != rel
            or row["prior_run"] != str(old)
            or row["observer_calls"] != 1
            or not row["observer_restored"]
            or rep["path"] != array_path(j)
            or rep["packed_dtype"] != "<u2"
            or sha(root / rep["path"]) != rep["sha256"]
            or sha(p) != row["run_sha256"]
            or sha(p.with_suffix(".input_audit.json")) != row["input_audit_sha256"]
            or set(rep["arrays"]) != set(SITES)
        ):
            raise ValueError("panel cell mapping/hash/observer differs")
    return capture, frozen, rows


def site_values(root, row, site, shape, oracle):
    rep = row["representation"]
    with np.load(root / rep["path"], allow_pickle=False) as archive:
        if len(archive.files) != 2 or set(archive.files) != set(SITES):
            raise ValueError("two packed sites required")
        a = (oracle_expand if oracle else expand)(archive[site])
    rec = rep["arrays"][site]
    if (
        list(a.shape) != shape
        or a.dtype != np.float32
        or not np.isfinite(a).all()
        or rec
        != {
            "shape": shape,
            "dtype": "float32",
            "source_dtype": "mlx.core.bfloat16",
            "array_sha256": digest(a),
            "scalar_count": a.size,
            "nonzero": bool(np.any(a)),
        }
    ):
        raise ValueError("lossless representation source bytes/layout differ")
    return a.ravel()


def oracle_direction(a, b):
    na, nb = (float(np.sqrt(np.sum(z * z))) for z in (a, b))
    return {
        "zero_norm": na,
        "other_norm": nb,
        "cosine": float(np.sum(a * b) / (na * nb)) if na and nb else None,
        "norm_ratio": nb / na if na else None,
        "relative_l2_difference": float(np.sqrt(np.sum((b - a) ** 2)) / na)
        if na
        else None,
    }


def scan(root, frozen, rows, scratch, *, oracle=False):
    metadata, recipes = frozen["hierarchy_metadata"], frozen["map_recipes"]
    labels = sorted(k for k, v in metadata.items() if v["replicate"] in (3, 4))
    families = sorted({v["pairing_family"] for v in metadata.values()})
    blocks = {}
    for row in rows:
        cells = blocks.setdefault((row["map"], row["pair_id"]), {})
        if row["cell"] in cells:
            raise ValueError("duplicate factorial cell")
        cells[row["cell"]] = row
    if set(blocks) != set(planned_panel_blocks(metadata)) or any(
        set(r) != set(CELLS) for r in blocks.values()
    ):
        raise ValueError("incomplete factorial block grid")
    vectors, scores, directions = [], [], []
    cube = np.full((2, 4, 2, 3, 16, 8), np.nan)

    def materialize(site, key, directory):
        paths = {}
        shape = frozen["observation_config"]["sites"][site]["shape"]
        for label in sorted(metadata) if key.endswith("_reference") else labels:
            cells = {
                c: site_values(root, blocks[(key, label)][c], site, shape, oracle)
                for c in CELLS
            }
            if oracle:
                left = np.subtract(cells["jj"], cells["mj"], dtype=np.float64)
                right = np.subtract(cells["jm"], cells["mm"], dtype=np.float64)
                vector = left - right
                norm = float(np.sqrt(np.einsum("i,i->", vector, vector)))
                if not np.isfinite(norm) or (norm == 0 and np.any(vector)):
                    raise ValueError("invalid oracle interaction norm")
            else:
                vector = factorial_interaction(cells)
                norm = direction_status(vector)["norm"]
            vectors.append(
                {
                    "site": site,
                    "map": key,
                    "pair_id": label,
                    "dtype": "float64",
                    "dimensions": vector.size,
                    "array_sha256": digest(vector),
                    "norm": norm,
                    "available": norm != 0,
                    "reason": None if norm else "exact_zero_direction_unavailable",
                }
            )
            paths[label] = Path(directory) / f"{label}.npy"
            np.save(paths[label], vector, allow_pickle=False)
        return paths

    def score(site, key, ref, test):
        recipe = recipes[key]
        if oracle:
            result = oracle_score(ref, test, labels, families, metadata)
            scored = (
                {"available": False}
                if result is None
                else {
                    "available": True,
                    "labels": labels,
                    "families": families,
                    "scores": result[0].tolist(),
                }
            )
        else:
            scored = pair_scores(ref, test, metadata)
        scores.append(
            {
                "site": site,
                "map": key,
                "reference_map": recipe["reference"],
                **recipe,
                **scored,
            }
        )
        if scored["available"] and recipe["q"] is not None:
            cube[
                SITES.index(site),
                recipe["mask"],
                recipe["destination"],
                (0, 1, 4).index(recipe["q"]),
            ] = scored["scores"]

    for site in SITES:
        for mask in range(4):
            check_space(
                scratch,
                frozen["disk_budget"]["reserve_bytes"]
                + frozen["disk_budget"]["peak_scratch_bytes"],
            )
            with TemporaryDirectory(dir=scratch) as ref_dir:
                key = f"m{mask}_reference"
                ref = materialize(site, key, ref_dir)
                score(site, key, ref, ref)
                for d in range(2):
                    with TemporaryDirectory(dir=scratch) as zero_dir:
                        key = f"m{mask}_q0_d{d}"
                        zero = materialize(site, key, zero_dir)
                        score(site, key, ref, zero)
                        for q in (1, 4):
                            with TemporaryDirectory(dir=scratch) as other_dir:
                                key = f"m{mask}_q{q}_d{d}"
                                other = materialize(site, key, other_dir)
                                score(site, key, ref, other)
                                for label in labels:
                                    a, b = (
                                        np.load(p[label], allow_pickle=False)
                                        for p in (zero, other)
                                    )
                                    val = (
                                        oracle_direction(a, b)
                                        if oracle
                                        else direction(a, b)
                                    )
                                    directions.append(
                                        {
                                            "site": site,
                                            "mask": mask,
                                            "destination": d,
                                            "pair_id": label,
                                            "q_sixteenths": q,
                                            **val,
                                        }
                                    )
            print(
                f"{'verification' if oracle else 'analysis'}: {site}, mask {mask + 1}/4",
                flush=True,
            )
    if (len(vectors), len(scores), len(directions)) != (1024, 56, 512):
        raise ValueError("analysis coverage differs")
    return vectors, scores, directions, cube, labels, families


def analyze(root):
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("fresh analysis required")
    out.mkdir()
    try:
        _, frozen, rows = load_capture(root)
        vectors, scores, directions, cube, labels, families = scan(
            root, frozen, rows, out
        )
        agg, ep, boot, margins = summarize(
            cube,
            labels,
            families,
            frozen["hierarchy_metadata"],
            frozen["release"]["bootstrap"],
        )
        ledgers = {
            n: write_ledger(out / n, v)
            for n, v in (
                ("vectors.jsonl", vectors),
                ("scores.jsonl", scores),
                ("directions.jsonl", directions),
            )
        }
        np.save(out / "paired_margins.npy", margins, allow_pickle=False)
        result = {
            "schema_version": 1,
            "analysis_kind": "representation_panel_analysis",
            "status": "COMPLETE",
            "capture_summary_sha256": sha(root / "capture_summary.json"),
            "ledgers": ledgers,
            "paired_margins_sha256": sha(out / "paired_margins.npy"),
            "aggregates": agg,
            "primary_endpoints": ep,
            "bootstrap": boot,
            "new_model_forwards": 0,
            "new_p_values": 0,
            "confirmation_cohort": "not_generated_not_sealed",
        }
        write_json(out / "summary.json", result)
        return result
    except Exception as e:
        write_json(
            out / "analysis_failure.json",
            {"status": "FAILED", "message": str(e), "primary_results_withheld": True},
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    a = parser.parse_args()
    result = analyze(a.run_root.resolve())
    print(
        f"wrote {len(result['primary_endpoints'])} endpoints; independent verification pending"
    )


if __name__ == "__main__":
    main()
