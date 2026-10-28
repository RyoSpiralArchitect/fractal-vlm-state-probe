from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.permutation_maps import MAPS
from fractal_vlm_state_probe.source_coordinate_overlap import analyze, validate_config
from fractal_vlm_state_probe.stimulus import write_json


def load_sources(root, publication):
    baseline = _read(publication / "summary.json")
    frozen = _read(root / "frozen_specification.json")
    if (
        _sha(root / "study_summary.json") != baseline["source_summary_sha256"]
        or frozen != baseline["frozen_specification"]
        or _sha(root / "frozen_specification.json")
        != baseline["frozen_specification_sha256"]
        or _sha(root / "vectors.json") != baseline["vectors_sha256"]
    ):
        raise ValueError("input baseline receipt changed")
    ledger = publication / "vectors.jsonl"
    published_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if _sha(ledger) != baseline["ledgers"]["vectors.jsonl"][
        "sha256"
    ] or published_rows != _read(root / "vectors.json"):
        raise ValueError("published vector ledger differs")
    repo = Path(__file__).resolve().parents[3]
    for name, digest in frozen["code_sha256"].items():
        if _sha(repo / name) != digest:
            raise ValueError(f"qualified baseline code changed: {name}")
    metadata = frozen["hierarchy_metadata"]
    vectors = {k: {} for k in MAPS}
    receipts = [r for r in published_rows if r["view"] == "raw_rgb"]
    if len(receipts) != 128 or {(r["map"], r["pair_id"]) for r in receipts} != {
        (m, label) for m in MAPS for label in metadata
    }:
        raise ValueError("raw vector denominator differs")
    for row in receipts:
        path = Path(row["path"])
        if _sha(path) != row["sha256"]:
            raise ValueError("raw vector bytes changed")
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        if (
            value.shape != (320, 320, 3)
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise ValueError("raw vector shape/dtype/finiteness differs")
        vectors[row["map"]][row["pair_id"]] = value
    maps = {}
    for key, row in frozen["maps"].items():
        if _sha(Path(row["path"])) != row["sha256"]:
            raise ValueError("input map changed")
        maps[key] = np.load(row["path"], allow_pickle=False)
    return baseline, vectors, maps, metadata, receipts


def execute(args):
    config = _read(args.config)
    validate_config(config)
    baseline, vectors, maps, metadata, receipts = load_sources(
        args.baseline_root, args.baseline_publication
    )
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    code = {
        **baseline["frozen_specification"]["code_sha256"],
        **{
            name: _sha(repo / name)
            for name in (
                "src/fractal_vlm_state_probe/source_coordinate_overlap.py",
                "src/fractal_vlm_state_probe/cli/run_source_coordinate_overlap.py",
            )
        },
    }
    for name, digest in {
        **code,
        str(args.config.resolve().relative_to(repo)): _sha(args.config),
    }.items():
        if (
            hashlib.sha256(
                subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
            ).hexdigest()
            != digest
        ):
            raise ValueError(
                f"register current implementation/config before analysis: {name}"
            )
    if args.output_root.exists():
        raise FileExistsError("use a fresh overlap study root")
    args.output_root.mkdir(parents=True)
    args.output_owned = True
    frozen = {
        "config": config,
        "registered_design_commit": commit,
        "code_sha256": code,
        "hierarchy_metadata": metadata,
        "maps": baseline["frozen_specification"]["maps"],
        "baseline_root": str(args.baseline_root.resolve()),
        "baseline_publication": str(args.baseline_publication.resolve()),
        "source_sha256": {
            str(p.resolve()): _sha(p)
            for p in (
                args.config,
                args.baseline_root / "study_summary.json",
                args.baseline_root / "frozen_specification.json",
                args.baseline_root / "vectors.json",
                args.baseline_publication / "summary.json",
                args.baseline_publication / "vectors.jsonl",
            )
        },
    }
    write_json(args.output_root / "frozen_specification.json", frozen)
    write_json(args.output_root / "source_vectors.json", receipts)
    result = analyze(vectors, maps, metadata, baseline)
    dots = args.output_root / "dot_products.jsonl"
    with dots.open("w") as handle:
        for row in result.pop("dot_products"):
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    result.update(
        schema_version=1,
        analysis_kind="raw_source_coordinate_overlap_decomposition",
        date=config["registered_date"],
        frozen_specification_sha256=_sha(
            args.output_root / "frozen_specification.json"
        ),
        source_vectors_sha256=_sha(args.output_root / "source_vectors.json"),
        dot_products_sha256=_sha(dots),
        new_input_vectors=0,
        new_processor_calls=0,
        new_model_forwards=0,
    )
    write_json(args.output_root / "study_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Decompose raw correspondence by source-coordinate coincidence."
    )
    for name in ("config", "baseline-root", "baseline-publication", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args)
    except Exception as exc:
        if getattr(args, "output_owned", False):
            write_json(
                args.output_root / "execution_failure.json",
                {"type": type(exc).__name__, "error": str(exc)},
            )
        raise
    print(
        f"decomposed {result['available_dot_products']}/{result['planned_dot_products']} dot products",
        flush=True,
    )


if __name__ == "__main__":
    main()
