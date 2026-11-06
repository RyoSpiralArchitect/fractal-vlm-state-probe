"""Receipt-only observer for the operator-approved pip HTTP cache cleanup."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

TARGET = Path("/Users/ryohiga/Library/Caches/pip")


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(4 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def wheels():
    root = TARGET / "wheels"
    return {
        str(p.relative_to(root)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


def http():
    result = {}
    for name in ("http-v2", "http"):
        root = TARGET / name
        if root.is_symlink():
            raise ValueError("do not follow a cache-directory symlink")
        files = (
            [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
            if root.exists()
            else []
        )
        result[name] = {
            "exists": root.exists(),
            "files": len(files),
            "logical_bytes": sum(p.stat().st_size for p in files),
            "allocated_bytes": sum(p.stat().st_blocks * 512 for p in files),
        }
    return result


def write(p, value):
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("before", "after"))
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "before":
        args.output_root.mkdir(parents=True, exist_ok=False)
        write(
            args.output_root / "before.json",
            {
                "schema_version": 1,
                "analysis_kind": "approved_pip_http_cleanup_before",
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "free_bytes": shutil.disk_usage(TARGET).free,
                "approved_targets": [str(TARGET / "http-v2"), str(TARGET / "http")],
                "http_caches": http(),
                "preserved_wheel_files": wheels(),
            },
        )
    else:
        before = read(args.output_root / "before.json")
        if wheels() != before["preserved_wheel_files"]:
            raise ValueError("preserved locally-built wheels changed")
        after = http()
        if any(v["exists"] for v in after.values()):
            raise ValueError("an approved HTTP cache target still exists")
        write(
            args.output_root / "summary.json",
            {
                "schema_version": 1,
                "analysis_kind": "approved_pip_http_cleanup_receipt",
                "status": "COMPLETE",
                "before_sha256": sha(args.output_root / "before.json"),
                "observed_utc": datetime.now(timezone.utc).isoformat(),
                "approved_targets": before["approved_targets"],
                "before_http_caches": before["http_caches"],
                "after_http_caches": after,
                "free_before_bytes": before["free_bytes"],
                "free_after_bytes": shutil.disk_usage(TARGET).free,
                "preserved_wheel_files": before["preserved_wheel_files"],
                "installed_packages_modified": False,
                "model_cache_modified": False,
                "research_runs_modified": False,
                "cleanup_observer_deletes_files": False,
                "note": "Free-byte changes may also reflect other processes; no precise attribution is inferred.",
            },
        )
    print(args.mode, args.output_root)
