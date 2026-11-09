"""Saved-data verifier independent of acquisition and native runtime modules."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .stat_matched_source_storage import inside, inventory, ledger, read, sha, write

STAGES = ("off_before", "on", "off_after")
SITES = ("site_premerger", "site_auxiliary", "site_final")


def _equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def _array(root, receipt, expected_path):
    if receipt["path"] != expected_path:
        raise ValueError("noncanonical array path")
    path = inside(root, expected_path)
    if sha(path) != receipt["sha256"]:
        raise ValueError("independent file digest mismatch")
    value = np.load(path, allow_pickle=False)
    if (list(value.shape) != receipt["shape"] or str(value.dtype) != receipt["dtype"]
            or hashlib.sha256(value.tobytes(order="C")).hexdigest() != receipt["word_sha256"]):
        raise ValueError("independent array receipt mismatch")
    return value


def _order(grid, c):
    if grid.shape != (1, 3) or grid.dtype.kind not in "iu":
        raise ValueError("integer grid required")
    t, h, w = map(int, grid[0])
    m = c["spatial_merge"]
    if t != 1 or min(h, w) <= 0 or h % m or w % m or h*w > c["max_patch_tokens"]:
        raise ValueError("grid exceeds frozen bound")
    # Padded block transpose is deliberately separate from capture's window loops.
    h, w = h//m, w//m
    block = c["window_size"] // m // c["patch_size"]
    raster = np.arange(h*w).reshape(h, w)
    padded = np.pad(raster, ((0, (-h) % block), (0, (-w) % block)), constant_values=-1)
    windows = padded.reshape(padded.shape[0]//block, block,
                             padded.shape[1]//block, block).transpose(0, 2, 1, 3).ravel()
    return windows[windows >= 0]


def check_saved(root, plan, jobs, source_root, *, software_fixture=False, provenance=None):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("symlink verification root forbidden")
    if (root / "failure.json").exists():
        raise ValueError("failed captures cannot qualify")
    frozen, result = read(root / "frozen.json"), read(root / "capture.json")
    if (frozen["plan"] != plan or frozen["jobs"] != jobs
            or frozen["software_fixture"] is not software_fixture
            or result["software_fixture"] is not software_fixture
            or (provenance is not None and frozen["provenance"] != provenance)):
        raise ValueError("frozen plan/jobs/provenance differ")
    if not software_fixture and provenance is None:
        raise ValueError("native provenance must be independently admitted")
    if (result["status"] != ("SOFTWARE_CAPTURE_UNVERIFIED" if software_fixture else "COMPLETE_UNVERIFIED")
            or result["cases"] != len(jobs) or result["contexts"] != len(jobs)*3
            or len(jobs) != plan["jobs"]["cases"] or len({j["case_id"] for j in jobs}) != len(jobs)
            or inventory(root) != result["receipts"]):
        raise ValueError("complete capture inventory/count/status differs")
    c = plan["limits"]
    caches = {f"cache_{i:02d}_{k}" for i in range(c["language_layers"]) for k in ("keys", "values")}
    core = {"pixels", "grid", "ids", "mask", "logits_0", "logits_1", "scores"} | caches
    processors, rows = ledger(root, "processor.jsonl"), ledger(root, "rows.jsonl")
    if len(processors) != len(jobs) or len(rows) != 3*len(jobs):
        raise ValueError("incomplete processor/context ledger")
    files = {"frozen.json", "operations.jsonl", "processor.jsonl", "rows.jsonl"}
    events, common, site_dtype = [], None, None
    for i, job in enumerate(jobs):
        pp = processors[i]
        if pp["case_id"] != job["case_id"] or set(pp["arrays"]) != {"pixels", "grid"}:
            raise ValueError("processor ledger identity/coverage differs")
        raw = _array(root, pp["raw"], f"raw/{i}.npy")
        original = _array(source_root, job["raw"], job["raw"]["path"])
        if (not _equal(raw, original) or raw.dtype != np.uint8
                or list(raw.shape) != plan["jobs"]["source_shape_hwc"]):
            raise ValueError("original raw source mismatch")
        files.add(f"raw/{i}.npy")
        processor = {k: _array(root, r, f"processor/{i}/{k}.npy") for k, r in pp["arrays"].items()}
        _order(processor["grid"], c)
        files.update(f"processor/{i}/{k}.npy" for k in processor)
        events.extend((job["case_id"], "processor_only", k) for k in ("begin", "return"))
        baseline = None
        for s, stage in enumerate(STAGES):
            row = rows[i*3+s]
            names = core | (set(SITES) if stage == "on" else set())
            if (row["case_id"] != job["case_id"] or row["stage"] != stage
                    or set(row["arrays"]) != names or set(row["source_dtypes"]) != names):
                raise ValueError("context ledger coverage/order differs")
            a = {k: _array(root, r, f"rows/{i}/{stage}/{k}.npy") for k, r in row["arrays"].items()}
            files.update(f"rows/{i}/{stage}/{k}.npy" for k in names)
            ids, mask, grid = a["ids"], a["mask"], a["grid"]
            order = _order(grid, c)
            n = len(order) * c["spatial_merge"]**2
            length = ids.shape[1] if ids.ndim == 2 else 0
            if (ids.shape != (1, length) or not 0 < length <= c["max_prefix_sequence"]
                    or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= c["vocab"])
                    or mask.shape != ids.shape or mask.dtype.kind not in "biuf"
                    or mask.dtype.itemsize > 8 or not np.isin(mask, [0, 1]).all()
                    or not _equal(a["pixels"], processor["pixels"])
                    or not np.array_equal(grid, processor["grid"])):
                raise ValueError("independent prepared input check failed")
            shapes = {"pixels": (n, c["pixel_channels"]*c["temporal_patch_size"]*c["patch_size"]**2),
                      **{k: (c["vocab"],) for k in ("logits_0", "logits_1", "scores")},
                      **{k: (1, c["kv_heads"], length+1, c["head_dim"]) for k in caches}}
            if stage == "on":
                shapes.update(site_premerger=(n, c["vision_width"]),
                              site_auxiliary=(len(order), c["output_width"]),
                              site_final=(len(order), c["output_width"]))
                if not _equal(a["site_final"], a["site_auxiliary"][np.argsort(order)]):
                    raise ValueError("independent inverse-window words differ")
                dtype = {k: row["source_dtypes"][k] for k in SITES}
                if site_dtype is not None and dtype != site_dtype:
                    raise ValueError("across-case site dtype mismatch")
                site_dtype = dtype
            for k, shape in shapes.items():
                if (a[k].shape != shape or a[k].dtype != np.float32 or not np.isfinite(a[k]).all()
                        or row["source_dtypes"][k] not in c["source_dtype_allowlist"]):
                    raise ValueError("independent shape/source-dtype/finite check failed")
            meta = row["metadata"]
            if len(json.dumps(meta, allow_nan=False).encode()) > plan["storage"]["metadata_and_headers_per_context_allowance_bytes"]:
                raise ValueError("independent metadata cap exceeded")
            sampled = meta["sampled_id"]
            if (type(sampled) is not int or not 0 <= sampled < c["vocab"]
                    or meta["decoder_inputs"] != [ids.ravel().tolist(), [sampled]]
                    or meta["cache_offsets"] != [length+1]*c["language_layers"]
                    or meta["observer_calls"] != {"vision": int(stage == "on"), "merger": int(stage == "on")}
                    or meta["observer_restored"] is not True):
                raise ValueError("independent native call/restoration check failed")
            comparable = {k: v for k, v in meta.items() if k != "observer_calls"}
            current = ({k: a[k] for k in core}, {k: row["source_dtypes"][k] for k in core}, comparable)
            if baseline is None:
                baseline = current
            elif (any(not _equal(current[0][k], baseline[0][k]) for k in core)
                  or current[1:] != baseline[1:]):
                raise ValueError("independent OFF/ON/OFF word/dtype/metadata mismatch")
            inputs = {k: a[k] for k in ("ids", "mask", "grid")}
            if common is None:
                common = inputs
            elif any(not _equal(inputs[k], common[k]) for k in inputs):
                raise ValueError("across-case fixed layout mismatch")
            events.extend((job["case_id"], stage, k) for k in (
                "context_begin", "prepare_begin", "prepare_return", "decoder_begin", "decoder_return",
                "decoder_begin", "decoder_return", "context_return"))
    actual = ledger(root, "operations.jsonl")
    expected = [{"sequence": i+1, "case_id": case, "stage": stage, "kind": kind}
                for i, (case, stage, kind) in enumerate(events)]
    if actual != expected or set(inventory(root)) != files:
        raise ValueError("event trace or exhaustive file coverage differs")
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > (
            plan["storage"]["payload_allowance_bytes"] + plan["storage"]["overhead_allowance_bytes"]):
        raise ValueError("saved tree exceeds registered cap")
    return {"schema_version": 1, "status": "SOFTWARE_FIXTURE_ONLY" if software_fixture else "NATIVE_SAVED_DATA_VERIFIED",
            "native_runtime_qualified": False, "cohort_released": False,
            "cases": len(jobs), "contexts": len(rows), "decoder_calls": 6*len(jobs),
            "processor_calls": 4*len(jobs), "capture_sha256": sha(root / "capture.json"),
            "frozen_sha256": sha(root / "frozen.json"), "files_checked": len(files)}


def verify_to(root, output, plan, jobs, source_root, **kwargs):
    output = Path(output)
    if output.is_symlink() or any(p.is_symlink() for p in output.parents):
        raise ValueError("symlink verification output forbidden")
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = check_saved(root, plan, jobs, source_root, **kwargs)
        write(output / "summary.json", result)
        return result
    except Exception as error:
        write(output / "failure.json", {"error": str(error), "retry_authorized": False})
        raise
