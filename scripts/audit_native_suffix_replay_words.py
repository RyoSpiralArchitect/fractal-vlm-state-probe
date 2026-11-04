"""Independent file-only check of an accepted Note 75 calibration; never loads a model."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def ledger(path, sha):
    data = path.read_bytes()
    if digest(data) != sha:
        raise ValueError(f"changed ledger: {path}")
    return [json.loads(line) for line in data.splitlines()]


def audit(cap, verified, note72):
    s, seal, old = (
        read(verified / "summary.json"),
        read(cap / "capture.json"),
        read(note72 / "summary.json"),
    )
    if (
        (cap / "failure.json").exists()
        or (verified / "failure.json").exists()
        or s["status"] != "VERIFIED"
        or seal["status"] != "COMPLETE"
        or digest((cap / "capture.json").read_bytes()) != s["capture_sha256"]
        or s["source_summary_sha256"] != old["verified_summary_sha256"]
    ):
        raise ValueError("complete matching calibration and original anchors required")
    anchors = ledger(
        note72 / "contexts.jsonl", old["ledgers"]["contexts.jsonl"]["sha256"]
    )
    boundaries = ledger(
        verified / "boundaries.jsonl", s["ledgers"]["boundaries.jsonl"]["sha256"]
    )
    arms = ledger(verified / "arms.jsonl", seal["arms_sha256"])
    if (len(anchors), len(arms), len(boundaries)) != (96, 288, 288):
        raise ValueError("incomplete rows")
    scalar_counts, tensors, original_checks = Counter(), Counter(), 0
    for i, a in enumerate(anchors):
        d = cap / "contexts" / f"{i + 1:03d}"
        terminal = np.load(
            d / "native_plain" / "kv_0.npy", mmap_mode="r", allow_pickle=False
        )
        if terminal.shape != (24, 2, 1, 2, 397, 64) or terminal.dtype != np.dtype(
            "<u2"
        ):
            raise ValueError("invalid native terminal")
        # Independent unpack of three historical float32 fingerprints (no acquisition helpers).
        for label in ("1:keys", "12:keys", "23:values"):
            layer, kind = label.split(":")
            words = terminal[int(layer), int(kind == "values")]
            expanded = (words.astype("<u4") * 65536).view("<f4")
            if digest(expanded.tobytes()) != a["cache_array_sha256"][label]:
                raise ValueError("historical selected-word mismatch")
            original_checks += 1
        partial = []
        for arm, row in zip(
            ("native_plain", "native_traced", "forced_own"),
            arms[3 * i : 3 * i + 3],
            strict=True,
        ):
            if (
                row["arm"] != arm
                or row["ordinal"] != i + 1
                or row["actual_suffix_ids"] != a["suffix_ids"]
                or any(row[key] != a[key] for key in ("pair_id", "cell", "stage"))
                or row["terminal_sha256"] != digest(terminal.tobytes())
            ):
                raise ValueError("arm or terminal seal mismatch")
            journal = d / arm / "checkpoints.json"
            if digest(journal.read_bytes()) != row["checkpoints_sha256"]:
                raise ValueError("live checkpoint receipt changed")
            if arm != "native_plain":
                partial.append((arm, read(journal)))
        for j, (lo, hi) in enumerate(((0, 395), (395, 396), (396, 397))):
            b = boundaries[3 * i + j]
            expected = a["prepared_ids"] if not j else a["suffix_ids"][j - 1 : j]
            prefix = terminal[..., :hi, :]
            h = digest(prefix.tobytes())
            if (
                b["ordinal"] != i + 1
                or b["step"] != j
                or b["offset"] != hi
                or b["actual_input_ids"] != expected
                or b["native_sha256"] != h
                or b["forced_sha256"] != h
            ):
                raise ValueError("independent native prefix/ID mismatch")
            all_hashes = {}
            for layer in range(24):
                for k, kind in enumerate(("keys", "values")):
                    all_hashes[f"{layer}:{kind}"] = digest(prefix[layer, k].tobytes())
            if b["tensor_sha256"] != all_hashes or b["scalar_pairs"] != prefix.size:
                raise ValueError("independent per-tensor or scalar ledger mismatch")
            for arm, receipts in partial:
                r = receipts[j]
                chunk = np.load(
                    d / arm / f"kv_{j}.npy", mmap_mode="r", allow_pickle=False
                )
                source = terminal[..., lo:hi, :]
                if (
                    chunk.shape != source.shape
                    or chunk.dtype != terminal.dtype
                    or chunk.tobytes() != source.tobytes()
                    or r["tensor_sha256"] != all_hashes
                    or r["sha256"] != h
                    or r["input_ids"] != expected
                    or r["offset"] != hi
                ):
                    raise ValueError(
                        f"word/decoder mismatch at ordinal={i + 1}, arm={arm}, offset={hi}"
                    )
            tensors[hi] += len(all_hashes)
            scalar_counts[hi] += prefix.size
    return {
        "schema_version": 1,
        "analysis_kind": "independent_native_suffix_word_file_check",
        "status": "PASS",
        "verified_summary_sha256": digest((verified / "summary.json").read_bytes()),
        "anchor_rows": len(anchors),
        "direct_historical_terminal_checks": original_checks,
        "native_forced_tensor_pairs_by_offset": dict(tensors),
        "native_forced_scalar_pairs_by_offset": dict(scalar_counts),
        "actual_model_calls": 0,
        "processor_calls": 0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("capture", "verified", "note72", "output"):
        ap.add_argument(key, type=Path)
    a = ap.parse_args()
    result = audit(a.capture, a.verified, a.note72)
    with a.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(result["status"], a.output)
