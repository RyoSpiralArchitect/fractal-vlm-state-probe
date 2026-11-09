"""Bounded calibration capture with injected backend; no implicit native release."""

import json
import shutil
from pathlib import Path

import numpy as np

from .qwen_calibration_contract import validate_saved_layout, window_order
from .stat_matched_source_storage import append, inventory, load_array, save_array, write

STAGES = ("off_before", "on", "off_after")
SITES = ("site_premerger", "site_auxiliary", "site_final")


def cache_names(plan):
    return [f"cache_{i:02d}_{k}" for i in range(plan["limits"]["language_layers"])
            for k in ("keys", "values")]


def words_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def validate_processor(processor, plan):
    c = plan["limits"]
    if set(processor) != {"pixels", "grid"}:
        raise ValueError("standalone processor fields differ")
    order = window_order(processor["grid"], plan)
    shape = (len(order)*c["spatial_merge"]**2,
             c["pixel_channels"]*c["temporal_patch_size"]*c["patch_size"]**2)
    pixels = processor["pixels"]
    if pixels.shape != shape or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError("standalone processor layout/dtype/finiteness differs")


def validate_packet(packet, processor, stage, plan):
    c = plan["limits"]
    arrays, meta = packet["arrays"], packet["metadata"]
    base = {"pixels", "grid", "ids", "mask", "logits_0", "logits_1", "scores"} | set(cache_names(plan))
    required = base | (set(SITES) if stage == "on" else set())
    if set(arrays) != required or set(packet["source_dtypes"]) != required:
        raise ValueError("complete packet array/dtype coverage required")
    ids, mask = arrays["ids"], arrays["mask"]
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype.kind not in "iu"
            or not 0 < ids.shape[1] <= c["max_prefix_sequence"]
            or np.any(ids >= c["vocab"]) or np.any(ids < 0)
            or mask.shape != ids.shape or mask.dtype.itemsize > 8
            or mask.dtype.kind not in "biuf" or not np.isin(mask, [0, 1]).all()):
        raise ValueError("prepared token/mask contract differs")
    order = window_order(arrays["grid"], plan)
    n = len(order) * c["spatial_merge"]**2
    dim = c["pixel_channels"] * c["temporal_patch_size"] * c["patch_size"]**2
    if (arrays["pixels"].shape != (n, dim) or arrays["pixels"].dtype != np.float32
            or not words_equal(arrays["pixels"], processor["pixels"])
            or not np.array_equal(arrays["grid"], processor["grid"])):
        raise ValueError("processor-only/native pixel words or grid values differ")
    for name, value in arrays.items():
        if name not in ("ids", "mask", "grid"):
            if (value.dtype != np.float32 or not np.isfinite(value).all()
                    or packet["source_dtypes"][name] not in c["source_dtype_allowlist"]):
                raise ValueError("native floating dtype/finiteness differs")
    for name in ("logits_0", "logits_1", "scores"):
        if arrays[name].shape != (c["vocab"],):
            raise ValueError("complete last-position vocabulary required")
    shape = (1, c["kv_heads"], ids.shape[1] + 1, c["head_dim"])
    if any(arrays[k].shape != shape for k in cache_names(plan)):
        raise ValueError("full active prefix-plus-one cache layout differs")
    sampled = meta["sampled_id"]
    if (type(sampled) is not int or not 0 <= sampled < c["vocab"]
            or meta["decoder_inputs"] != [ids.ravel().tolist(), [sampled]]
            or meta["cache_offsets"] != [ids.shape[1] + 1] * c["language_layers"]
            or meta["observer_calls"] != {"vision": int(stage == "on"), "merger": int(stage == "on")}
            or meta["observer_restored"] is not True):
        raise ValueError("first-yield counters, sampled token or restoration differ")
    if len(json.dumps(meta, allow_nan=False).encode()) > plan["storage"]["metadata_and_headers_per_context_allowance_bytes"]:
        raise ValueError("context metadata cap exceeded")
    if stage == "on":
        validate_saved_layout(arrays["pixels"], arrays["grid"],
                              *(arrays[k] for k in SITES), plan)


class Writer:
    def __init__(self, root, plan):
        self.root, self.plan = root, plan
        self.bytes = 0

    def array(self, relative, a):
        storage = self.plan["storage"]
        if self.bytes + a.nbytes + 4096 > storage["payload_allowance_bytes"]:
            raise ValueError("array payload allowance breached")
        receipt = save_array(self.root, relative, a, reserve=storage["reserve_bytes"])
        self.bytes += (self.root / relative).stat().st_size
        return receipt

    def guard(self):
        storage = self.plan["storage"]
        if shutil.disk_usage(self.root).free < storage["reserve_bytes"] + 2**20:
            raise ValueError("disk reserve breached")
        if sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) > (
                storage["payload_allowance_bytes"] + storage["overhead_allowance_bytes"]):
            raise ValueError("complete capture allowance breached")


def capture(root, plan, jobs, source_root, backend, provenance, *, software_fixture=False):
    """Internal engine; native CLI must perform admission before backend creation."""
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        # Resolve system temporary roots at the fixture caller, not here.
        raise ValueError("symlink capture root forbidden")
    root.mkdir(parents=True, exist_ok=False)
    writer = Writer(root, plan)
    sequence = 0

    def event(case_id, stage, kind):
        nonlocal sequence
        writer.guard()
        sequence += 1
        append(root, "operations.jsonl", {"sequence": sequence, "case_id": case_id,
                                         "stage": stage, "kind": kind})

    try:
        write(root / "frozen.json", {"plan": plan, "jobs": jobs, "provenance": provenance,
              "software_fixture": software_fixture})
        if len(jobs) != plan["jobs"]["cases"] or len({j["case_id"] for j in jobs}) != len(jobs):
            raise ValueError("complete registered job count required")
        writer.guard()
        backend = backend() if callable(backend) else backend
        common_inputs, site_dtypes = None, None
        for i, job in enumerate(jobs):
            raw = load_array(source_root, job["raw"])
            if list(raw.shape) != plan["jobs"]["source_shape_hwc"] or raw.dtype != np.uint8:
                raise ValueError("existing raw source shape/dtype differs")
            raw_receipt = writer.array(f"raw/{i}.npy", raw)
            event(job["case_id"], "processor_only", "begin")
            processor = backend.processor(raw)
            event(job["case_id"], "processor_only", "return")
            pp_receipts = {k: writer.array(f"processor/{i}/{k}.npy", a)
                           for k, a in processor.items()}
            append(root, "processor.jsonl", {"case_id": job["case_id"], "raw": raw_receipt,
                                            "arrays": pp_receipts})
            validate_processor(processor, plan)
            baseline = None
            for stage in STAGES:
                event(job["case_id"], stage, "context_begin")
                seen = []

                def emit(kind):
                    seen.append(kind)
                    event(job["case_id"], stage, kind)

                packet = backend.context(raw, stage, emit)
                event(job["case_id"], stage, "context_return")
                # Keep bounded returned evidence before judging parity. A failed
                # root can never pass the saved-data verifier or be retried.
                if len(json.dumps(packet["metadata"], allow_nan=False).encode()) > plan["storage"]["metadata_and_headers_per_context_allowance_bytes"]:
                    raise ValueError("returned metadata exceeds persistence cap")
                receipts = {k: writer.array(f"rows/{i}/{stage}/{k}.npy", a)
                            for k, a in packet["arrays"].items()}
                append(root, "rows.jsonl", {"case_id": job["case_id"], "stage": stage,
                      "arrays": receipts, "metadata": packet["metadata"],
                      "source_dtypes": packet["source_dtypes"]})
                writer.guard()
                if seen != ["prepare_begin", "prepare_return", "decoder_begin", "decoder_return",
                            "decoder_begin", "decoder_return"]:
                    raise ValueError("actual native preparation/decoder event counts differ")
                validate_packet(packet, processor, stage, plan)
                arrays = packet["arrays"]
                core_names = set(arrays) - set(SITES)
                if baseline is None:
                    baseline = packet
                else:
                    if (any(not words_equal(arrays[k], baseline["arrays"][k]) for k in core_names)
                            or any(packet["source_dtypes"][k] != baseline["source_dtypes"][k] for k in core_names)
                            or packet["metadata"] != (baseline["metadata"] | {
                                "observer_calls": packet["metadata"]["observer_calls"]})):
                        raise ValueError("OFF/ON/OFF exact parity failed")
                inputs = {k: arrays[k] for k in ("ids", "mask", "grid")}
                if common_inputs is None:
                    common_inputs = {k: a.copy() for k, a in inputs.items()}
                elif any(not words_equal(inputs[k], common_inputs[k]) for k in inputs):
                    raise ValueError("across-case prepared layout/IDs differ")
                if stage == "on":
                    current = {k: packet["source_dtypes"][k] for k in SITES}
                    if site_dtypes is not None and current != site_dtypes:
                        raise ValueError("across-case site dtypes differ")
                    site_dtypes = current
                writer.guard()
        result = {"schema_version": 1, "analysis_kind": "qwen_observation_capture",
                  "status": "SOFTWARE_CAPTURE_UNVERIFIED" if software_fixture else "COMPLETE_UNVERIFIED",
                  "cases": len(jobs), "contexts": len(jobs) * 3,
                  "software_fixture": software_fixture, "receipts": inventory(root)}
        write(root / "capture.json", result)
        return result
    except Exception as error:
        write(root / "failure.json", {"error": str(error), "retry_authorized": False,
                                      "last_event_sequence": sequence,
                                      "retained_artifact_sha256": inventory(root)})
        raise
