"""Payload-first construction and fixed-reference analysis, without acquisition."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .control_stimulus import luminance_quantile_match_rgb
from .nonlearned_processing import FEATURES, STARTS, factorial_interaction
from .stat_matched_input import joint_multisets, word_sha

CELLS = ("mm", "mj", "jm", "jj")
FLOORS = (1e-8, 1e-6, 1e-4)
BASELINE_SHAPES = {
    f"{start}/{feature}": (6,) if feature == "global_moments" else (192,)
    for start in STARTS
    for feature in FEATURES
}
RNG_RULES = {
    "voronoi": ([0, 3], []),
    "quasicrystal": ([5], []),
    "checkerboard": ([0], []),
    "square_tiling": ([0, 17], []),
    "triangle_tiling": ([1, 17], []),
    "hex_tiling": ([2, 17], []),
    "white_noise": ([], [101]),
    "blue_noise": ([], [211]),
    "random_dots": ([0], [307]),
}


def qualify_core(root):
    """Check file contracts only, never claim a runtime/acquisition preflight."""
    root = Path(root)
    profile = json.loads(
        (root / "configs/stat_matched_source_core_v1.json").read_bytes()
    )
    if (
        profile["study_id"] != "stat_matched_source_core_v1"
        or profile["acquisition_released"]
        or profile["execution_runner"] is not None
        or profile["design_config"] != "configs/stat_matched_source_panel_v1.json"
        or set(profile["registration_sha256"])
        != {
            "configs/stat_matched_source_panel_v1.json",
            "docs/research_notes/0088_stat_matched_source_registration.md",
            "tests/test_stat_matched_source_registration.py",
        }
    ):
        raise ValueError("core-only profile boundary differs")

    def check(path, expected):
        p = root / path
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen file differs: {path}")

    for path, digest in profile["registration_sha256"].items():
        check(path, digest)
    spec = json.loads((root / profile["design_config"]).read_bytes())
    for path, digest in spec["pinned_artifacts"].items():
        check(path, digest)
    previous = json.loads(
        (root / spec["preserve_inherited_frozen_hashes_from"]).read_bytes()
    )
    if previous["status"] != "PASS" or len(previous["code_sha256"]) != 118:
        raise ValueError("inherited calibration denominator/status differs")
    for path, digest in previous["code_sha256"].items():
        check(path, digest)
    return {
        "status": "FILE_CONTRACTS_PASS_NOT_ACQUISITION_RELEASE",
        "inherited_files": 118,
        "registration_files": 3,
        "acquisition_released": False,
        "runtime_qualified": False,
        "new_processor_calls": 0,
        "new_model_contexts": 0,
    }


def decoded_hash_audit(records, historical):
    """Compare supplied decoded-RGB receipts, not PNG/container hashes."""
    seen, old, collisions = {}, {}, []
    for row in [*historical, *records]:
        digest = row["decoded_rgb_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(x not in "0123456789abcdef" for x in digest)
        ):
            raise ValueError("decoded RGB SHA-256 required")
        if not isinstance(row["identity"], str) or not row["identity"]:
            raise ValueError("source identity required")
    if len({r["identity"] for r in records}) != len(records):
        raise ValueError("new source identities repeated")
    for row in historical:
        old.setdefault(row["decoded_rgb_sha256"], []).append(row["identity"])
    for row in records:
        digest = row["decoded_rgb_sha256"]
        if digest in old or digest in seen:
            collisions.append(
                {
                    "identity": row["identity"],
                    "new_conflict": seen.get(digest),
                    "historical_conflicts": old.get(digest, []),
                }
            )
        seen.setdefault(digest, row["identity"])
    return {
        "supplied_new_sources": len(records),
        "supplied_historical_sources": len(historical),
        "collisions": collisions,
        "historical_inventory_complete": False,
        "acquisition_released": False,
    }


def rng_seeds(source, defaults):
    """Enumerate frame-zero RNG initializers; never draw from an RNG."""
    values = {**defaults, **source}
    if values["total_frames"] != 1 or source["kind"] not in RNG_RULES:
        raise ValueError(
            "only the registered single-frame generator inventory is supported"
        )
    seed = values["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("nonnegative integer generator seed required")
    offsets, salts = RNG_RULES[source["kind"]]
    return [seed + x for x in offsets] + [
        (seed * 1000003 + salt) % 4294967295 for salt in salts
    ]


def source_spec_audit(spec, historical):
    """Audit supplied metadata only; this is not a complete historical inventory."""
    ordered_jobs(spec)
    defaults = spec["sources"]["defaults"]
    seen, old, collisions = {}, {}, []
    for row in historical:
        for seed in rng_seeds(row["source"], row["defaults"]):
            old.setdefault(seed, []).append(row["identity"])
    for pair in spec["sources"]["pairs"]:
        for role in ("source_a", "source_b"):
            identity = f"{pair['pair_id']}/{role}"
            for seed in rng_seeds(pair[role], defaults):
                if seed in seen or seed in old:
                    collisions.append(
                        {
                            "identity": identity,
                            "rng_seed": seed,
                            "new_conflict": seen.get(seed),
                            "historical_conflicts": old.get(seed, []),
                        }
                    )
                seen.setdefault(seed, identity)
    return {
        "derived_initializers": sum(
            len(rng_seeds(p[r], defaults))
            for p in spec["sources"]["pairs"]
            for r in ("source_a", "source_b")
        ),
        "unique_initializers": len(seen),
        "supplied_historical_sources": len(historical),
        "collisions": collisions,
        "historical_inventory_complete": False,
        "decoded_RGB_collision_audit_completed": False,
        "acquisition_released": False,
    }


def pack_payload(source, spec, geometry):
    c = geometry["construction"]
    if (
        source.dtype != np.uint8
        or list(source.shape) != spec["construction"]["payload_source_shape_hwc"]
    ):
        raise ValueError("source must have the specified uint8 RGB shape")
    tiles = []
    ch, cw = c["cell_shape_hw"]
    th, tw = c["tile_shape_hw"]
    for r in range(c["active_grid_rows"][1] - c["active_grid_rows"][0]):
        for col in range(c["active_grid_cols"][1] - c["active_grid_cols"][0]):
            for dy, dx in c["tile_offsets_yx"]:
                tile = source[
                    r * ch + dy : r * ch + dy + th, col * cw + dx : col * cw + dx + tw
                ]
                if tile.shape != (th, tw, 3):
                    raise ValueError("payload tile is outside the source")
                tiles.append(tile)
    packed = np.stack(tiles).reshape(spec["construction"]["packed_shape_hwc"])
    if packed.size != spec["construction"]["payload_pixels"] * 3:
        raise ValueError("payload denominator differs")
    return packed


def embed_payload(packed, spec, geometry):
    c = geometry["construction"]
    if (
        packed.dtype != np.uint8
        or list(packed.shape) != spec["construction"]["packed_shape_hwc"]
    ):
        raise ValueError("packed payload shape/dtype differs")
    shape = spec["construction"]["raw_shape_hwc"]
    if shape != c["raw_shape_hwc"]:
        raise ValueError("canvas and qualified geometry differ")
    out = np.empty(shape, dtype=np.uint8)
    out[:] = spec["construction"]["background_rgb"]
    ch, cw = c["cell_shape_hw"]
    th, tw = c["tile_shape_hw"]
    tiles = packed.reshape(-1, th, tw, 3)
    i = 0
    for r in range(*c["active_grid_rows"]):
        for col in range(*c["active_grid_cols"]):
            for dy, dx in c["tile_offsets_yx"]:
                out[
                    r * ch + dy : r * ch + dy + th, col * cw + dx : col * cw + dx + tw
                ] = tiles[i]
                i += 1
    if i != len(tiles):
        raise ValueError("tile count differs")
    return out


def identity_quartet(a, b, spec, geometry):
    pa, pb = (pack_payload(x, spec, geometry) for x in (a, b))
    payloads = {
        "mm": pa,
        "mj": luminance_quantile_match_rgb(pa, reference=pb),
        "jm": luminance_quantile_match_rgb(pb, reference=pa),
        "jj": pb,
    }
    return {k: embed_payload(v, spec, geometry) for k, v in payloads.items()}


def rearrange(identity, mapping):
    """Consume an externally qualified map, rather than inventing a new map."""
    if identity.dtype != np.uint8 or identity.ndim != 3 or identity.shape[-1] != 3:
        raise ValueError("uint8 HWC RGB identity required")
    if mapping.dtype != np.dtype("<u4") or mapping.shape != identity.shape[:2]:
        raise ValueError("uint32 canvas map required")
    if not np.array_equal(np.sort(mapping.ravel()), np.arange(mapping.size)):
        raise ValueError("map is not a bijection")
    return identity.reshape(-1, 3)[mapping.ravel()].reshape(identity.shape)


def palette_audit(quartet):
    if set(quartet) != set(CELLS):
        raise ValueError("complete quartet required")
    hashes = {k: joint_multisets(v, "raw_rgb", [1, 1])[0] for k, v in quartet.items()}
    return {
        "hashes": hashes,
        "matches": [hashes[a] == hashes[b] for a, b in (("mm", "jm"), ("mj", "jj"))],
    }


def ordered_jobs(spec):
    pairs = spec["sources"]["pairs"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    if (
        len(pairs) != 32
        or len(families) != 8
        or len({p["pair_id"] for p in pairs}) != 32
    ):
        raise ValueError("registered pair/family denominator differs")
    for family in families:
        group = [p for p in pairs if p["pairing_family"] == family]
        if (
            Counter(p["replicate"] for p in group) != {5: 1, 6: 1, 7: 1, 8: 1}
            or len({p["broad_class"] for p in group}) != 1
        ):
            raise ValueError("family split or class differs")
    for pair in pairs:
        if pair["split"] != ("reference" if pair["replicate"] in (5, 6) else "test"):
            raise ValueError("reference/test identity differs")
    if Counter(
        next(p["broad_class"] for p in pairs if p["pairing_family"] == f)
        for f in families
    ) != {"geometry": 4, "stochastic": 4}:
        raise ValueError("four families per broad class required")
    ordinal, jobs = 408, []
    for split in ("reference", "test"):
        for pair in pairs:
            if pair["split"] == split:
                for shift in (0, 1, 2, 3):
                    for cell in CELLS:
                        ordinal += 1
                        jobs.append(
                            {
                                "ordinal": ordinal,
                                "split": split,
                                "pair_id": pair["pair_id"],
                                "shift": shift,
                                "cell": cell,
                            }
                        )
    return jobs


def seal_reference_metadata(rows, spec):
    jobs = ordered_jobs(spec)
    if len(rows) != 256:
        raise ValueError("all 256 reference cells required")
    for row, job in zip(rows, jobs[:256], strict=True):
        if any(row.get(k) != v for k, v in job.items()):
            raise ValueError("reference identity or order differs")
        for key in ("representation_sha256", "prepared_pixel_sha256"):
            digest = row.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(x not in "0123456789abcdef" for x in digest)
            ):
                raise ValueError("complete reference digest required")
    digest = hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "kind": "metadata_only_reference_seal",
        "cells": 256,
        "last_context": 664,
        "first_test_context": 665,
        "rows_sha256": digest,
        "artifact_bytes_independently_verified": False,
    }


def checked_norm(a):
    value = float(np.linalg.norm(a))
    if not math.isfinite(value) or (value == 0 and np.any(a)):
        raise ValueError("norm overflow or underflow")
    return value


def contrast_direction(cells, *, learned, relative_floor=1e-6):
    if relative_floor not in FLOORS:
        raise ValueError("unregistered sensitivity floor")
    z = np.ascontiguousarray(factorial_interaction(cells).ravel(), dtype="<f8")
    if not z.size or not np.isfinite(z).all():
        raise ValueError("finite nonempty interaction required")
    norm = checked_norm(z)
    scale = sum(
        checked_norm(np.asarray(cells[k], dtype=np.float64).ravel()) for k in CELLS
    )
    if not math.isfinite(scale):
        raise ValueError("cell scale overflow")
    envelope = 4e-10 * math.sqrt(z.size)
    threshold = relative_floor * scale if learned else 100 * envelope
    available = norm > threshold
    reason = (
        None if available else ("exact_zero" if norm == 0 else "below_amplitude_guard")
    )
    return (
        z,
        z / norm if available else None,
        {
            "dimension": z.size,
            "norm": norm,
            "cell_scale": scale,
            "threshold": threshold,
            "available": available,
            "reason": reason,
            "vector_sha256": word_sha(z),
            "baseline_error_envelope": None if learned else envelope,
        },
    )


def family_reference(units):
    if len(units) != 2:
        raise ValueError("two reference replicates required")
    if any(v is None for v in units):
        return None, "unavailable_reference_replicate"
    units = [np.asarray(v, dtype=np.float64) for v in units]
    if (
        units[0].shape != units[1].shape
        or units[0].ndim != 1
        or not units[0].size
        or any(not np.isfinite(v).all() for v in units)
    ):
        raise ValueError("reference direction shapes or finiteness differ")
    total = units[0] + units[1]
    norm = checked_norm(total)
    if norm <= 1e-12:
        return None, "cancelling_unit_references"
    return total / norm, None


def _nullable(values):
    return np.where(np.isfinite(values), values, None).tolist()


def analyze_view(load_cells, spec, view, *, relative_floor=1e-6):
    """Read complete cells lazily; no generation, fitting, file writes or inference."""
    ordered_jobs(spec)
    if view not in spec["analysis"]["views"]:
        raise ValueError("unregistered view")
    learned = view in ("encoder", "projector")
    expected_shape = (
        tuple(spec["observation"]["sites"][view]) if learned else BASELINE_SHAPES[view]
    )
    if not learned and relative_floor != 1e-6:
        raise ValueError("baseline does not have learned sensitivity floors")
    pairs = spec["sources"]["pairs"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    metadata = {p["pairing_family"]: p["broad_class"] for p in pairs}
    refs, reference_rows, records = {}, [], []

    def read(pair, shift):
        cells = load_cells(pair["pair_id"], shift, view)
        if any(np.shape(v) != expected_shape for v in cells.values()):
            label = "full learned site" if learned else "registered baseline"
            raise ValueError(f"{label} shape differs")
        z, unit, row = contrast_direction(
            cells, learned=learned, relative_floor=relative_floor
        )
        records.append({"pair_id": pair["pair_id"], "shift": shift, **row})
        return z, unit, row

    for family in families:
        units = []
        for pair in pairs:
            if pair["pairing_family"] == family and pair["split"] == "reference":
                for shift in range(4):
                    _, unit, _ = read(pair, shift)
                    if shift == 0:
                        units.append(unit)
        refs[family], reason = family_reference(units)
        reference_rows.append(
            {"family": family, "available": refs[family] is not None, "reason": reason}
        )
    tests = [p for p in pairs if p["split"] == "test"]
    scores = np.full((4, 16, 8), np.nan)
    diagnostics = []
    for pi, pair in enumerate(tests):
        z0, u0, row0 = read(pair, 0)
        for shift in range(4):
            z, unit, row = (z0, u0, row0) if shift == 0 else read(pair, shift)
            for fi, family in enumerate(families):
                if unit is not None and refs[family] is not None:
                    scores[shift, pi, fi] = np.dot(unit, refs[family])
            if shift:
                denominator = row["norm"] + row0["norm"]
                diagnostics.append(
                    {
                        "pair_id": pair["pair_id"],
                        "shift": shift,
                        "direction_cosine": float(np.dot(u0, unit))
                        if u0 is not None and unit is not None
                        else None,
                        "norm_ratio": row["norm"] / row0["norm"]
                        if row0["norm"]
                        else None,
                        "relative_change": checked_norm(z - z0) / denominator
                        if denominator
                        else None,
                    }
                )
    margins = np.full((4, 16), np.nan)
    matrices = np.full((4, 8, 8), np.nan)
    for fi, family in enumerate(families):
        indices = [i for i, p in enumerate(tests) if p["pairing_family"] == family]
        other = [
            j
            for j, f in enumerate(families)
            if f != family and metadata[f] == metadata[family]
        ]
        for shift in range(4):
            for pi in indices:
                required = scores[shift, pi, [fi, *other]]
                if np.isfinite(required).all():
                    margins[shift, pi] = required[0] - required[1:].mean()
            matrices[shift, fi] = scores[shift, indices].mean(axis=0)
    paired = margins[1:].mean(axis=0) - margins[0]
    family_deltas = [
        float(
            paired[[i for i, p in enumerate(tests) if p["pairing_family"] == f]].mean()
        )
        for f in families
    ]
    available = bool(np.isfinite(family_deltas).all())
    return {
        "view": view,
        "relative_floor": relative_floor if learned else None,
        "primary_learned_profile": learned and relative_floor == 1e-6,
        "available": available,
        "paired_margin_change": float(np.mean(family_deltas)) if available else None,
        "reason": None if available else "required_margin_unavailable_no_filtering",
        "families": families,
        "test_pair_ids": [p["pair_id"] for p in tests],
        "fixed_test_pairs": 16,
        "available_test_pairs": int(np.isfinite(paired).sum()),
        "family_deltas": _nullable(np.array(family_deltas)),
        "scores": _nullable(scores),
        "score_matrices": _nullable(matrices),
        "margins": _nullable(margins),
        "references": reference_rows,
        "interactions": records,
        "diagnostics": diagnostics,
        "p_values": None,
        "confidence_intervals": None,
        "equivalence_tested": False,
    }


def disk_budget(spec):
    """Planned uncompressed upper bounds; not a future runner's release receipt."""
    b = spec["budget"]
    scalars = sum(math.prod(s) for s in spec["observation"]["sites"].values())
    pixel_bytes = 3 * 1024 * 1024 * 4
    cache_bytes = 3 * 1 * 2 * 397 * 64 * 4
    parts = {
        "all_saved_native_pixels": (512 + 128 + 25) * pixel_bytes,
        "raw_source_candidate_synthetic_arrays": 64 * 240 * 320 * 3
        + (512 + 25) * 320 * 320 * 3,
        "experimental_packed_sites": 512 * scalars * 2,
        "historical_and_sham_ON_float32_sites": (128 + 8) * scalars * 4,
        "all_selected_caches": 920 * cache_bytes,
        "optional_complete_interactions": 128 * scalars * 8,
        "feature_oracle_and_residual_arrays": (512 * 2 + 384) * 2 * (6 + 5 * 192) * 8,
        "maximum_scratch": 64 * math.prod(spec["observation"]["sites"]["encoder"]) * 8,
    }
    total = sum(parts.values())
    if total > b["payload_allowance_bytes"]:
        raise ValueError("enumerated payload exceeds fixed allowance")
    return {
        "parts": parts,
        "enumerated_payload_bytes": total,
        "payload_allowance_bytes": b["payload_allowance_bytes"],
        "required_launch_free_bytes": b["required_launch_free_bytes"],
        "acquisition_released": False,
    }
