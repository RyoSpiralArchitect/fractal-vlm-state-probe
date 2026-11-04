"""Package two matching Note 72 verifier outputs without changing the frozen verifier."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def package(first, second, destination):
    if first.resolve() == second.resolve() or any(
        (p / "failure.json").exists() for p in (first, second)
    ):
        raise ValueError("two separate successful verification roots required")
    raw = (first / "summary.json").read_bytes()
    summary = json.loads(raw)
    names = {"contexts.jsonl", "pairs.jsonl", "regions.jsonl"}
    if (
        summary["status"] != "VERIFIED"
        or summary["study_id"] != "prompt_only_source_calibration_v1"
        or set(summary["ledgers"]) != names
        or raw != (second / "summary.json").read_bytes()
    ):
        raise ValueError("verifier summaries differ or are incomplete")
    ledgers = {name: (first / name).read_bytes() for name in names}
    for name, data in ledgers.items():
        if (
            data != (second / name).read_bytes()
            or hashlib.sha256(data).hexdigest() != summary["ledgers"][name]["sha256"]
        ):
            raise ValueError(f"verified ledger differs: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in ledgers.items():
        (destination / name).write_bytes(data)
    with (destination / "verified_summary.json.gz").open("wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as archive:
            archive.write(raw)
    catalog = {
        **summary,
        "analysis_kind": "prompt_only_source_calibration",
        "verified_summary_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (destination / "summary.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "second", "destination"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    package(args.first, args.second, args.destination)
