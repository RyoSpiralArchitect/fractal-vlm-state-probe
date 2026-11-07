"""Guarded-tile constructor and complete, unnormalized matching audits."""

from __future__ import annotations

import hashlib

import numpy as np

from .nonlearned_processing import FEATURES, STARTS, extract_features
from .nonlearned_processing_calibration import oracle_features


def word_sha(a):
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def coordinate_map(spec, shift):
    c = spec["construction"]
    if shift not in c["arrangement_shifts"]:
        raise ValueError("unregistered shift")
    h, w, _ = c["raw_shape_hwc"]
    source = np.arange(h * w, dtype="<u4").reshape(h, w)
    out = source.copy()
    ch, cw = c["cell_shape_hw"]
    th, tw = c["tile_shape_hw"]
    offsets = c["tile_offsets_yx"]
    for r in range(*c["active_grid_rows"]):
        for col in range(*c["active_grid_cols"]):
            for k, (dy, dx) in enumerate(offsets):
                sy, sx = offsets[(k + shift) % len(offsets)]
                y, x = r * ch, col * cw
                out[y + dy : y + dy + th, x + dx : x + dx + tw] = source[
                    y + sy : y + sy + th, x + sx : x + sx + tw
                ]
    return out


def tile(name, r, c, k, spec):
    shape = (*spec["construction"]["tile_shape_hw"], 3)
    u, v = np.indices(shape[:2])
    a = np.zeros(shape, dtype=np.uint8)
    if name == "constant":
        a[:] = spec["fixtures"]["background_rgb"][name]
    elif name == "ramp":
        a[:] = np.stack(
            (
                (41 * k + 11 * r + 3 * c + 5 * u + v) % 256,
                (17 * k + 7 * r + 13 * c + 3 * v) % 256,
                (29 * k + 5 * r + 19 * c + 7 * u) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
    elif name == "checkerboard":
        z = 15 + 32 * k + 128 * ((u + v + r + c) % 2)
        a[:] = np.stack((z, 255 - z, z), axis=-1)
    elif name == "impulse":
        a[k, (2 * k) % 6] = [255, 128, 64]
    elif name == "noise":
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([spec["fixtures"]["noise_seed"], r, c, k])
            )
        )
        a = rng.integers(0, 256, size=shape, dtype=np.uint8)
    elif name != "zero":
        raise ValueError("unregistered fixture")
    return a


def identity_fixture(spec, name):
    c = spec["construction"]
    a = np.broadcast_to(
        np.array(spec["fixtures"]["background_rgb"][name], dtype=np.uint8),
        c["raw_shape_hwc"],
    ).copy()
    ch, cw = c["cell_shape_hw"]
    th, tw = c["tile_shape_hw"]
    first_r, first_c = c["active_grid_rows"][0], c["active_grid_cols"][0]
    for r in range(*c["active_grid_rows"]):
        for col in range(*c["active_grid_cols"]):
            for k, (dy, dx) in enumerate(c["tile_offsets_yx"]):
                a[
                    r * ch + dy : r * ch + dy + th, col * cw + dx : col * cw + dx + tw
                ] = tile(name, r - first_r, col - first_c, k, spec)
    return a


def cases(spec):
    return [
        (f"{n}_s{s}", n, s)
        for n in spec["fixtures"]["order"]
        for s in spec["construction"]["arrangement_shifts"]
    ] + [("sentinel", "noise", 0)]


def fixture(spec, name, shift, mapping, sentinel=False):
    base = identity_fixture(spec, name)
    a = base.reshape(-1, 3)[mapping.ravel()].reshape(base.shape)
    if sentinel:
        s = spec["fixtures"]["sentinel"]
        a[s["raw_rows"][0] : s["raw_rows"][1], s["raw_cols"][0] : s["raw_cols"][1]] = s[
            "overwrite_rgb"
        ]
    return a


def map_audit(mapping, expected, spec, shift):
    c = spec["construction"]
    if (
        mapping.shape != tuple(c["map_shape"])
        or mapping.dtype != np.dtype("<u4")
        or mapping.tobytes() != expected.tobytes()
    ):
        raise ValueError("coordinate constructor/oracle mismatch")
    flat = mapping.ravel()
    identity = np.arange(flat.size, dtype="<u4")
    moved = int(np.count_nonzero(flat != identity))
    if not np.array_equal(np.sort(flat), identity) or moved != (
        c["moved_sites_nonidentity"] if shift else 0
    ):
        raise ValueError("map bijection or moved count differs")
    return {
        "shift": shift,
        "shape": list(mapping.shape),
        "dtype": str(mapping.dtype),
        "word_sha256": word_sha(mapping),
        "moved": moved,
        "fixed": flat.size - moved,
    }


def checked_pixels(a, spec):
    if (
        a.dtype != np.float32
        or list(a.shape) != spec["future_calibration"]["processor_shape_nchw"]
        or not np.isfinite(a).all()
    ):
        raise ValueError("native processor shape/dtype/finiteness mismatch")


def joint_multisets(a, start, grid):
    if start == "raw_rgb":
        if a.dtype != np.uint8:
            raise ValueError("raw RGB words must be uint8")
        hwc = a
    else:
        if a.dtype != np.float32 or a.shape[:2] != (1, 3):
            raise ValueError("native processor words must be float32 NCHW")
        hwc = np.ascontiguousarray(np.moveaxis(a[0], 0, -1), dtype="<f4").view("<u4")
    h, w, _ = hwc.shape
    nr, nc = grid
    if h % nr or w % nc:
        raise ValueError("grid does not divide image")

    def digest(x):
        rgb = x.reshape(-1, 3)
        order = np.lexsort((rgb[:, 2], rgb[:, 1], rgb[:, 0]))
        return word_sha(np.ascontiguousarray(rgb[order]))

    return [digest(hwc)] + [
        digest(hwc[r * h // nr : (r + 1) * h // nr, c * w // nc : (c + 1) * w // nc])
        for r in range(nr)
        for c in range(nc)
    ]


def feature_arrays(raw, pixels, config):
    result, errors = {}, []
    for start, values in zip(STARTS, (raw, pixels), strict=True):
        actual = extract_features(values, start, config)
        oracle = oracle_features(values, start, config)
        for f in FEATURES:
            if actual[f].shape != oracle[f].shape or not np.isfinite(oracle[f]).all():
                raise ValueError("feature oracle shape mismatch")
            for kind, arrays in (("actual", actual), ("oracle", oracle)):
                result[f"{kind}__{start}__{f}"] = arrays[f]
            errors.append(
                {
                    "start": start,
                    "feature": f,
                    "max_abs": float(np.max(np.abs(actual[f] - oracle[f]))),
                }
            )
    return result, errors


def case_audit(key, name, shift, raw, pixels, arrays, errors, reference, config, spec):
    tol = spec["feature_contract"]["absolute_tolerance"]
    failures = [
        f"oracle:{e['start']}:{e['feature']}" for e in errors if e["max_abs"] > tol
    ]
    row = {
        "key": key,
        "fixture": name,
        "shift": shift,
        "raw_word_sha256": word_sha(raw),
        "pixel_word_sha256": word_sha(pixels),
        "oracle_checks": errors,
        "comparisons": [],
    }
    residuals = {}
    for start, values in zip(STARTS, (raw, pixels), strict=True):
        groups = joint_multisets(values, start, config["grid"])
        row.setdefault("joint_multisets", {})[start] = groups
        if reference is None:
            continue
        b, ref_arrays, ref_groups = reference[start]
        same = values.tobytes() == b.tobytes()
        matched = [x == y for x, y in zip(groups, ref_groups, strict=True)]
        sentinel = key == "sentinel"
        expected_same = name in ("zero", "constant")
        if not sentinel and (not all(matched) or same != expected_same):
            failures.append(f"words_or_multiset:{start}")
        if sentinel and matched[0]:
            failures.append(f"sentinel_multiset:{start}")
        for f in FEATURES:
            k = f"actual__{start}__{f}"
            delta = arrays[k] - ref_arrays[k]
            residuals[f"{start}__{f}"] = delta
            maximum = float(np.max(np.abs(delta)))
            row["comparisons"].append(
                {
                    "start": start,
                    "feature": f,
                    "max_abs": maximum,
                    "l2": float(np.linalg.norm(delta)),
                    "norm": float(np.linalg.norm(arrays[k])),
                    "reference_norm": float(np.linalg.norm(ref_arrays[k])),
                    "residual_word_sha256": word_sha(delta),
                }
            )
            if not sentinel and maximum > tol:
                failures.append(f"feature_match:{start}:{f}")
            if sentinel and f == "grid_mean" and maximum <= 1e-3:
                failures.append(f"sentinel_grid:{start}")
        row.setdefault("word_checks", {})[start] = {
            "word_equal": same,
            "matched_groups": matched,
        }
    row["failures"] = failures
    return row, residuals
