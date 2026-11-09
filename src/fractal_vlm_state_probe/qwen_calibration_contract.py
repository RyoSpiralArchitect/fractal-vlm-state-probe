"""Calibration job metadata, byte bounds and window-order oracle; no native calls."""

import shutil
import importlib.util
from pathlib import Path

import numpy as np

from .stat_matched_source_storage import inside, ledger, read, sha, write

PLAN = "examples/research_notes/0102_qwen_calibration_registration/plan.json"


def jobs(spec, candidates, synthetic):
    selected = [p for p in spec["sources"]["pairs"] if p["replicate"] == 5]
    if len(selected) != 8 or len({p["pairing_family"] for p in selected}) != 8:
        raise ValueError("eight development reference families required")
    index = {}
    for row in candidates:
        key = row["pair_id"], row["shift"], row["cell"]
        if key in index:
            raise ValueError("duplicate development cell")
        index[key] = row
    result = []
    for p in selected:
        for cell in ("mm", "mj", "jm", "jj"):
            key = p["pair_id"], 0, cell
            if key not in index:
                raise ValueError("required development cell missing")
            row = index[key]
            if row["split"] != "reference":
                raise ValueError("development reference split required")
            result.append(
                {
                    "case_id": row["identity"],
                    "kind": "development",
                    "family": p["pairing_family"],
                    "raw": row["raw"],
                }
            )
    for name in ("zero", "constant"):
        matches = [r for r in synthetic if r["fixture"] == name and r["shift"] == 0]
        if len(matches) != 1:
            raise ValueError("exactly one existing sham required")
        row = matches[0]
        result.append(
            {
                "case_id": row["identity"],
                "kind": "sham",
                "fixture": name,
                "raw": row["raw"],
            }
        )
    if len({r["case_id"] for r in result}) != 34:
        raise ValueError("34 unique development cases required")
    for r in result:
        if r["raw"]["shape"] != [320, 320, 3] or r["raw"]["dtype"] != "uint8":
            raise ValueError("fixed development raw shape/dtype differs")
    return result


def storage_bound(plan):
    c, j, s = plan["limits"], plan["jobs"], plan["storage"]
    n, seq = c["max_patch_tokens"], c["max_language_sequence"]
    patch_dim = c["pixel_channels"] * c["temporal_patch_size"] * c["patch_size"] ** 2
    pixels = n * patch_dim * 4
    cache = (
        c["language_layers"] * 2 * c["batch"] * c["kv_heads"] * seq * c["head_dim"] * 4
    )
    ids_mask_grid = (2 * seq + 3) * 8
    vocab = 3 * c["vocab"] * 4
    positions = 3 * seq * 8
    sites = (
        n * c["vision_width"] * 4
        + 2 * (n // c["spatial_merge"] ** 2) * c["output_width"] * 4
    )
    raw = 320 * 320 * 3
    bound = j["model_contexts"] * (
        pixels
        + cache
        + ids_mask_grid
        + vocab
        + positions
        + s["metadata_and_headers_per_context_allowance_bytes"]
    ) + j["cases"] * (sites + pixels + raw)
    if bound > s["payload_allowance_bytes"]:
        raise ValueError("registered byte caps exceed calibration payload budget")
    if s["required_launch_free_bytes"] != sum(
        s[k]
        for k in (
            "payload_allowance_bytes",
            "reserve_bytes",
            "overhead_allowance_bytes",
        )
    ):
        raise ValueError("calibration disk accounting differs")
    return {
        "upper_payload_bytes_from_caps": bound,
        "cache_bytes_per_context_cap": cache,
        "pixel_bytes_per_preparation_cap": pixels,
        "on_stage_site_bytes_per_case_cap": sites,
        "payload_allowance_bytes": s["payload_allowance_bytes"],
        "required_launch_free_bytes": s["required_launch_free_bytes"],
        "qualified_runtime_or_joint_budget": False,
    }


def window_order(grid, plan):
    c = plan["limits"]
    a = np.asarray(grid)
    if a.shape != (1, 3) or a.dtype.kind not in "iu":
        raise ValueError("single integer image grid required")
    t, h, w = map(int, a[0])
    merge = c["spatial_merge"]
    if (
        t != 1
        or h <= 0
        or w <= 0
        or h % merge
        or w % merge
        or h * w > c["max_patch_tokens"]
    ):
        raise ValueError("registered single-image grid limits exceeded")
    rows, cols = h // merge, w // merge
    window = c["window_size"] // merge // c["patch_size"]
    order = []
    for y0 in range(0, rows, window):
        for x0 in range(0, cols, window):
            for y in range(y0, min(y0 + window, rows)):
                for x in range(x0, min(x0 + window, cols)):
                    order.append(y * cols + x)
    return np.array(order, dtype=np.int64)


def validate_saved_layout(pixels, grid, premerger, merger_return, vision_return, plan):
    c = plan["limits"]
    order = window_order(grid, plan)
    patches = int(np.prod(grid))
    patch_dim = c["pixel_channels"] * c["temporal_patch_size"] * c["patch_size"] ** 2
    expected = [
        (pixels, (patches, patch_dim)),
        (premerger, (patches, c["vision_width"])),
        (merger_return, (len(order), c["output_width"])),
        (vision_return, (len(order), c["output_width"])),
    ]
    if any(
        a.dtype != np.float32 or a.shape != shape or not np.isfinite(a).all()
        for a, shape in expected
    ):
        raise ValueError("complete saved layout/dtype/finiteness differs")
    expected_words = np.ascontiguousarray(merger_return[np.argsort(order)])
    if expected_words.tobytes() != vision_return.tobytes(order="C"):
        raise ValueError("independent window-inverse word parity failed")
    return {
        "patch_tokens": patches,
        "merged_tokens": len(order),
        "window_words_exact": True,
    }


def prepare(repo, output):
    repo, output = Path(repo).resolve(), Path(output).absolute()
    plan = read(repo / PLAN)
    if (
        plan["approved_for_native_execution"] is not False
        or plan["cohort_generation_authorized"] is not False
        or plan["status"] != "REGISTERED_PREPARATION_ONLY_NATIVE_EXECUTION_NOT_RELEASED"
    ):
        raise ValueError("preparation-only calibration contract required")
    if output != repo / plan["preparation_root"]:
        raise ValueError("canonical fresh metadata preparation root required")
    inside(repo, plan["preparation_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        for path, digest in plan["sources"].items():
            if sha(inside(repo, path)) != digest:
                raise ValueError(f"pinned calibration source differs: {path}")
        observer = read(
            repo / "examples/research_notes/0101_qwen_observer_preparation/summary.json"
        )
        for path, digest in observer["software_sha256"].items():
            if sha(inside(repo, path)) != digest:
                raise ValueError("Qwen observer software changed")
        package = importlib.util.find_spec("mlx_vlm")
        if package is None or package.origin is None:
            raise ValueError("installed generation source unavailable for static audit")
        if (
            sha(Path(package.origin).parent / "generate.py")
            != plan["installed_generation_source"]["sha256"]
        ):
            raise ValueError("installed first-yield/prefetch implementation changed")
        root = inside(repo, plan["capture_root"])
        selected = jobs(
            read(repo / "configs/stat_matched_source_panel_v1.json"),
            ledger(root, "candidates.jsonl"),
            ledger(root, "synthetic.jsonl"),
        )
        manifest = read(root / "capture.json")
        for row in selected:
            raw = row["raw"]
            if (
                manifest["receipts"].get(raw["path"]) != raw["sha256"]
                or sha(inside(root, raw["path"])) != raw["sha256"]
            ):
                raise ValueError("existing raw file differs from source receipt")
        budget = storage_bound(plan)
        write(
            output / "jobs.json",
            {
                "schema_version": 1,
                "analysis_kind": "qwen_calibration_job_metadata",
                "jobs": selected,
            },
        )
        write(
            output / "budget.json",
            {
                "schema_version": 1,
                "analysis_kind": "qwen_calibration_byte_bound",
                **budget,
            },
        )
        free = shutil.disk_usage(repo).free
        result = {
            "schema_version": 1,
            "analysis_kind": "qwen_calibration_preparation",
            "status": "METADATA_PREPARED_NATIVE_NOT_RELEASED",
            "plan_sha256": sha(repo / PLAN),
            "cases": len(selected),
            "future_model_contexts": 102,
            "free_bytes_observed": free,
            "calibration_disk_gate_met_at_preparation": free
            >= budget["required_launch_free_bytes"],
            "joint_two_model_threshold": None,
            "native_runtime_qualified": False,
            "installed_generation_source_hash_exact": True,
            "model_snapshot_or_weights_integrity_qualified": False,
            "model_weights_loaded": False,
            "processor_instantiated": False,
            "existing_raw_files_hash_checked": len(selected),
            "raw_arrays_decoded": 0,
            "new_source_images": 0,
            "new_cohort_maps": 0,
            "processor_calls": 0,
            "model_contexts": 0,
            "confirmation_scores": 0,
            "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        }
        write(output / "summary.json", result)
        return result
    except Exception as error:
        write(output / "failure.json", {"error": str(error), "retry_authorized": False})
        raise
