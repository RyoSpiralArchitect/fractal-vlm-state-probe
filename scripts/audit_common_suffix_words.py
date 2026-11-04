"""Independent NPY/JSON-only check of every common-suffix decoder boundary."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def word_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def rows(p, expected):
    if sha(p) != expected:
        raise ValueError(f"ledger changed: {p}")
    return [json.loads(line) for line in p.read_bytes().splitlines()]


def all_boundaries(path, anchor):
    records = read(path / "checkpoints.json")
    if len(records) != 3:
        raise ValueError("three actual decoder calls required")
    full = None
    for i, end in enumerate((395, 396, 397)):
        r = records[i]
        expected_ids = anchor["prepared_ids"] if not i else [785, 2168][i - 1 : i]
        part = np.load(path / f"kv_{i}.npy", allow_pickle=False)
        if (
            part.dtype != np.dtype("<u2")
            or part.shape != (24, 2, 1, 2, 395 if i == 0 else 1, 64)
            or np.any((part & 0x7F80) == 0x7F80)
            or r["step"] != i
            or r["offset"] != end
            or r["start"] != (0 if not i else end - 1)
            or r["offset_before"] != (0 if not i else end - 1)
            or r["input_ids"] != expected_ids
            or r["input_dtype"] != ("int64" if not i else "uint32")
            or r["embedding_shape"] != ([1, 395, 896] if not i else None)
        ):
            raise ValueError(
                "NPY or observed decoder input/offset does not match assigned trajectory"
            )
        full = part if not i else np.concatenate((full, part), axis=-2)
        if word_sha(full) != r["sha256"] or r["tensor_sha256"] != {
            f"{layer}:{kv}": word_sha(full[layer, k])
            for layer in range(24)
            for k, kv in enumerate(("keys", "values"))
        }:
            raise ValueError("complete all-layer boundary did not reproduce its hashes")
        yield r, full


def audit(capture, verified, note72, destination):
    summary = read(verified / "summary.json")
    seal = read(capture / "capture.json")
    cfg = read(
        Path(__file__).resolve().parents[1]
        / "configs/common_suffix_prompt_control_v1.json"
    )
    if (
        (capture / "failure.json").exists()
        or summary["status"] != "VERIFIED"
        or seal["status"] != "COMPLETE"
        or summary["capture_sha256"] != sha(capture / "capture.json")
        or summary["common_suffix_ids"] != [785, 2168]
    ):
        raise ValueError("complete unchanged verified capture required")
    catalog = read(note72 / "summary.json")
    all_anchors = rows(
        note72 / "contexts.jsonl", catalog["ledgers"]["contexts.jsonl"]["sha256"]
    )
    anchors = [a for a in all_anchors if a["stage"] != "ack_after"]
    cells = rows(
        verified / "panel_cells.jsonl",
        summary["ledgers"]["panel_cells.jsonl"]["sha256"],
    )
    metrics = rows(
        verified / "regions.jsonl", summary["ledgers"]["regions.jsonl"]["sha256"]
    )
    if len(cells) != 64 or len(anchors) != 64 or len(metrics) != 12288:
        raise ValueError("wrong panel or metric denominator")
    by = {(r["pair_id"], r["cell"], r["target"], r["region"]): r for r in metrics}
    observations = []
    word_comparisons = own_shams = target_regions = 0
    for i, (cell, anchor) in enumerate(zip(cells, anchors, strict=True)):
        if any(
            cell[k] != anchor[k] for k in ("pair_id", "cell", "stage", "prepared_ids")
        ) or (
            cell["natural_suffix_ids"] != anchor["suffix_ids"]
            or cell["supplied_suffix_ids"] != [785, 2168]
            or cell["anchor_ordinal"] != anchor["ordinal"]
        ):
            raise ValueError("panel anchor identity differs")
        path = capture / "panel" / f"{anchor['ordinal']:03d}"
        state = read(path / "state.json")
        if (
            state["cache_state_ids"] != anchor["prepared_ids"] + [785, 2168]
            or state["cache_state_ids"] != cell["actual_cache_ids"]
        ):
            raise ValueError("actual terminal IDs differ")
        if sha(path / "checkpoints.json") != cell["checkpoints_sha256"]:
            raise ValueError("checkpoint receipt changed")
        own = (
            capture
            / "recalibration"
            / "contexts"
            / f"{anchor['ordinal']:03d}"
            / "forced_own"
        )
        baseline = None
        native_records = read(own / "checkpoints.json")
        for step, (r, full) in enumerate(all_boundaries(path, anchor)):
            part = np.load(own / f"kv_{step}.npy", allow_pickle=False)
            baseline = part if not step else np.concatenate((baseline, part), axis=-2)
            if step == 0 or anchor["suffix_ids"] == [785, 2168]:
                if full.tobytes() != baseline.tobytes() or r != native_records[step]:
                    raise ValueError("whole-word same-process sham mismatch")
                word_comparisons += 48
                if anchor["suffix_ids"] == [785, 2168]:
                    own_shams += 48
            observations.append(
                {
                    "anchor_ordinal": anchor["ordinal"],
                    "pair_id": anchor["pair_id"],
                    "cell": anchor["cell"],
                    "stage": anchor["stage"],
                    **r,
                }
            )
        if word_sha(full) != cell["terminal_sha256"]:
            raise ValueError("terminal words differ")
        if not i % 2:
            ack = full
            continue
        yes = full
        for layer in range(24):
            for k, kv in enumerate(("keys", "values")):
                a, b = ack[layer, k], yes[layer, k]
                changes = np.where((a != b).any(axis=(0, 1, 3)))[0]
                for role, (lo, hi) in cfg["regions"].items():
                    r = by[(cell["pair_id"], cell["cell"], f"{layer}:{kv}", role)]
                    x, y = a[..., lo:hi, :], b[..., lo:hi, :]
                    af = (
                        (x.astype("<u4") * 65536).view("<f4").astype(np.float64).ravel()
                    )
                    bf = (
                        (y.astype("<u4") * 65536).view("<f4").astype(np.float64).ravel()
                    )
                    delta = bf - af
                    mask = x != y
                    if (
                        r["changed_positions"]
                        != (lo + np.flatnonzero(mask.any(axis=(0, 1, 3)))).tolist()
                        or r["changed_scalars"] != int(mask.sum())
                        or r["scalar_pairs"] != x.size
                        or r["first_changed_full_cache"]
                        != (int(changes[0]) if changes.size else None)
                        or not np.isclose(
                            r["rms"],
                            np.linalg.norm(delta) / np.sqrt(x.size),
                            rtol=1e-12,
                            atol=1e-14,
                        )
                        or r["max_abs"] != float(np.abs(delta).max())
                    ):
                        raise ValueError(
                            "independent region byte mask or metric differs"
                        )
                    target_regions += 1
    with (destination / "decoder_boundaries.jsonl").open("x") as f:
        for row in observations:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    result = {
        "schema_version": 1,
        "analysis_kind": "common_suffix_independent_word_audit",
        "status": "PASS",
        "verified_summary_sha256": sha(verified / "summary.json"),
        "contexts": len(cells),
        "actual_decoder_boundaries": len(observations),
        "two_singleton_calls_per_context": 128,
        "sham_tensor_comparisons_including_prefill": word_comparisons,
        "matching_own_suffix_exact_tensor_comparisons": own_shams,
        "independent_paired_region_byte_and_metric_checks": target_regions,
        "decoder_boundaries_sha256": sha(destination / "decoder_boundaries.jsonl"),
        "model_calls": 0,
        "processor_calls": 0,
    }
    with (destination / "independent_word_check.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "verified", "note72", "destination"):
        ap.add_argument(name, type=Path)
    args = ap.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    result = audit(args.capture, args.verified, args.note72, args.destination)
    print(result["status"], args.destination)
