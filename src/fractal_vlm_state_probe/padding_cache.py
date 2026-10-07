from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .stimulus import write_json


def input_fingerprint(payload: dict) -> dict:
    result = {}
    for key, value in payload.items():
        if value is None:
            result[key] = {"kind": "none"}
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"unsupported or nonfinite prepared input: {key}")
        result[key] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
        if key != "pixel_values":
            result[key]["values"] = array.tolist()
    return result


def changed_input_keys(before: dict, after: dict) -> list[str]:
    return sorted(
        k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )


def override_image_sizes(payload: dict, sizes: list) -> dict:
    if "image_sizes" not in payload:
        raise ValueError("prepared inputs do not contain image_sizes")
    original = payload["image_sizes"]
    expected = np.asarray(original)
    desired = np.asarray(sizes, dtype=expected.dtype)
    if expected.shape != desired.shape:
        raise ValueError("image_sizes override shape differs")
    if isinstance(original, list):
        replacement = desired.tolist()
    elif isinstance(original, np.ndarray):
        replacement = desired
    else:
        import mlx.core as mx

        if not isinstance(original, mx.array):
            raise ValueError("unsupported image_sizes representation")
        replacement = mx.array(desired, dtype=original.dtype)
    result = dict(payload)
    result["image_sizes"] = replacement
    if changed_input_keys(input_fingerprint(payload), input_fingerprint(result)) != [
        "image_sizes"
    ]:
        raise ValueError("metadata intervention must change image_sizes only")
    return result


@contextmanager
def prepared_input_audit(
    generate_module,
    *,
    audit_path: Path,
    context: dict,
    expected_pixel_sha256: str,
    expected_sizes: list,
    override_sizes: list | None = None,
):
    original = generate_module.prepare_inputs
    calls = []

    def observed(*args, **kwargs):
        if calls:
            raise ValueError("source-only capture prepared more than one input")
        payload = original(*args, **kwargs)
        native = input_fingerprint(payload)
        if native["pixel_values"]["sha256"] != expected_pixel_sha256:
            raise ValueError(
                "actual generation pixels differ from accepted input receipt"
            )
        if native.get("image_sizes", {}).get("values") != expected_sizes:
            raise ValueError(
                "actual generation image_sizes differs from registered input"
            )
        effective = (
            payload
            if override_sizes is None
            else override_image_sizes(payload, override_sizes)
        )
        record = {
            "schema_version": 1,
            "analysis_kind": "actual_generation_prepared_input_audit",
            "context": context,
            "native": native,
            "effective": input_fingerprint(effective),
            "override_image_sizes": override_sizes,
        }
        calls.append(record)
        write_json(audit_path, record)
        return effective

    generate_module.prepare_inputs = observed
    try:
        yield calls
        if len(calls) != 1:
            raise ValueError("source-only capture did not prepare exactly one input")
    finally:
        generate_module.prepare_inputs = original


def tensor_difference(before: np.ndarray, after: np.ndarray) -> dict:
    equal_shape = before.shape == after.shape
    equal_dtype = before.dtype == after.dtype
    result = {
        "shape_equal": equal_shape,
        "dtype_equal": equal_dtype,
        "bitwise_equal": equal_shape
        and equal_dtype
        and before.tobytes() == after.tobytes(),
        "max_abs_difference": None,
        "rms_difference": None,
        "relative_l2_difference": None,
        "cosine": None,
    }
    if not equal_shape:
        return result
    a, b = before.astype(np.float64).ravel(), after.astype(np.float64).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite cache comparison")
    delta = b - a
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    result.update(
        max_abs_difference=float(np.max(np.abs(delta))),
        rms_difference=float(np.sqrt(np.mean(delta**2))),
        relative_l2_difference=float(np.linalg.norm(delta) / na) if na else None,
        cosine=float(np.dot(a, b) / (na * nb)) if na and nb else None,
    )
    return result


def require_calibration(checks: list[dict], *, expected_count: int) -> None:
    if len(checks) != expected_count or not all(c["bitwise_equal"] for c in checks):
        raise ValueError(
            "calibration is incomplete or nonidentical; stop before panel capture"
        )
