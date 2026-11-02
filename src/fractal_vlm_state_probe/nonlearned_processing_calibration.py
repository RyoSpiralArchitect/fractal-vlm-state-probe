from __future__ import annotations

import numpy as np

from .nonlearned_processing import (
    FEATURES,
    STARTS,
    direction_status,
    extract_features,
    factorial_interaction,
    local_features,
)


def fixture(name, side, start, seed):
    generator = np.random.default_rng(np.random.SeedSequence([seed, side]))
    y, x = np.indices((side, side))
    a = np.zeros((side, side, 3), dtype=np.uint8)
    if name == "constant":
        a[:] = [51, 128, 204]
    elif name == "ramp":
        a = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    elif name == "checkerboard":
        a[:] = ((x + y) % 2 * 255)[..., None]
    elif name == "impulse":
        a[0, 0] = [255, 128, 64]
        a[side // 2, side // 2] = [32, 192, 240]
    elif name == "noise":
        a = generator.integers(0, 256, size=a.shape, dtype=np.uint8)
    elif name != "zero":
        raise ValueError("unregistered calibration fixture")
    if start == "raw_rgb":
        return a
    if start != "processor_pixel_values":
        raise ValueError("unregistered fixture start")
    # Synthetic normalized values exercise signed channels; this is not a processor simulation.
    b = (a.astype(np.float32) / 255 - np.float32(0.5)) / np.float32(0.25)
    return np.moveaxis(b, -1, 0)[None]


def oracle_features(values, start, config):
    if start == "raw_rgb":
        u = values.reshape(-1, 3).astype(np.uint64)
        moments = []
        for power in (1, 2):
            moments.extend(
                float(np.sum(u[:, c] ** power, dtype=np.uint64)) / (len(u) * 255**power)
                for c in range(3)
            )
        a = values.astype(np.float64) / 255
    else:
        a = np.transpose(values[0].astype(np.float64), (1, 2, 0))
        moments = [
            float(np.mean(a[:, :, c] ** power)) for power in (1, 2) for c in range(3)
        ]
    return {
        "global_moments": np.asarray(moments),
        **oracle_local_features(a, config["windows"][start], config["grid"]),
    }


def oracle_local_features(a, widths, grid):
    def pool(image):
        rows, cols = grid
        h, w, _ = image.shape
        return np.asarray(
            [
                float(
                    image[
                        i * h // rows : (i + 1) * h // rows,
                        j * w // cols : (j + 1) * w // cols,
                        c,
                    ].mean()
                )
                for i in range(rows)
                for j in range(cols)
                for c in range(3)
            ]
        )

    out = {"grid_mean": pool(a)}
    for scale in ("small", "large"):
        width = widths[scale]
        radius = width // 2
        kernel = np.ones(width, dtype=np.float64) / width
        padded = np.pad(a, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        mixed = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="valid"), 0, padded
        )
        padded = np.pad(mixed, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        mixed = np.apply_along_axis(
            lambda v: np.convolve(v, kernel, mode="valid"), 1, padded
        )
        out[f"{scale}_mean"] = pool(mixed)
        out[f"{scale}_energy"] = pool(mixed**2)
    return out


def synthetic_calibration(config):
    spec = config["calibration"]
    tolerance, seed = spec["oracle_absolute_tolerance"], spec["synthetic_seed"]
    checks, arrays = [], {}
    for start in STARTS:
        cases = [(name, spec["fixture_side"]) for name in spec["fixtures"]]
        cases.append((spec["full_size_fixture"], 320 if start == "raw_rgb" else 1024))
        for name, side in cases:
            values = fixture(name, side, start, seed)
            actual, expected = (
                extract_features(values, start, config),
                oracle_features(values, start, config),
            )
            for feature in FEATURES:
                key = f"{start}__{name}__{side}__{feature}"
                a, b = actual[feature], expected[feature]
                error = float(np.max(np.abs(a - b)))
                if a.shape != b.shape or not np.isfinite(a).all() or error > tolerance:
                    raise ValueError(
                        f"independent feature oracle differs: {key}: {error}"
                    )
                arrays[key] = a
                checks.append(
                    {
                        "start": start,
                        "fixture": name,
                        "side": side,
                        "feature": feature,
                        "array_key": key,
                        "shape": list(a.shape),
                        "max_abs_error": error,
                        "passed": True,
                    }
                )
    generator = np.random.default_rng(seed)
    cells = {
        k: generator.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        for k in ("mm", "jj")
    }
    cells["jm"] = generator.permutation(cells["mm"].reshape(-1, 3)).reshape(32, 32, 3)
    cells["mj"] = generator.permutation(cells["jj"].reshape(-1, 3)).reshape(32, 32, 3)
    features = {
        k: extract_features(v, "raw_rgb", config)["global_moments"]
        for k, v in cells.items()
    }
    negative = factorial_interaction(features)
    if np.any(negative) or direction_status(negative)["available"]:
        raise ValueError("raw multiset negative control did not cancel exactly")
    arrays["raw_multiset_zero_interaction"] = negative
    linear, nonlinear = [], []
    for start in STARTS:
        raw = {
            k: fixture("noise", 32, start, seed + i + 1)
            for i, k in enumerate(("mm", "jj", "mj", "jm"))
        }
        images = {
            k: (
                v.astype(np.float64) / 255
                if start == "raw_rgb"
                else np.moveaxis(v[0].astype(np.float64), 0, -1)
            )
            for k, v in raw.items()
        }
        z = factorial_interaction(images)
        transformed = {
            k: local_features(v, config["windows"][start], config["grid"])
            for k, v in images.items()
        }
        on_z = local_features(z, config["windows"][start], config["grid"])
        for feature in ("grid_mean", "small_mean", "large_mean"):
            contrast = factorial_interaction(
                {k: v[feature] for k, v in transformed.items()}
            )
            error = float(np.max(np.abs(contrast - on_z[feature])))
            if error > tolerance:
                raise ValueError("linear baseline does not commute within tolerance")
            linear.append({"start": start, "feature": feature, "max_abs_error": error})
        constants = {"mm": 0.0, "jj": 1.0, "mj": 0.25, "jm": 0.5}
        values = {k: np.full((32, 32, 3), value) for k, value in constants.items()}
        on_z = local_features(
            factorial_interaction(values), config["windows"][start], config["grid"]
        )
        transformed = {
            k: local_features(v, config["windows"][start], config["grid"])
            for k, v in values.items()
        }
        for feature in ("small_energy", "large_energy"):
            contrast = factorial_interaction(
                {k: v[feature] for k, v in transformed.items()}
            )
            error = float(np.max(np.abs(contrast - on_z[feature])))
            if error <= 0.1:
                raise ValueError("nonlinear order witness failed")
            nonlinear.append(
                {"start": start, "feature": feature, "max_abs_order_difference": error}
            )
    if (
        len(checks) != spec["feature_oracle_pairs"]
        or len(linear) != spec["linear_contrast_checks"]
        or len(nonlinear) != spec["nonlinear_order_checks"]
    ):
        raise ValueError("synthetic calibration denominator differs")
    return {
        "feature_checks": checks,
        "linear_contrast_checks": linear,
        "nonlinear_order_checks": nonlinear,
        "raw_multiset_negative_control": {
            "exact_zero": True,
            **direction_status(negative),
        },
        "maximum_feature_oracle_abs_error": max(c["max_abs_error"] for c in checks),
    }, arrays
