from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


CacheTensorName = Literal["keys", "values"]


@dataclass(frozen=True, order=True)
class CacheTensorCaptureSpec:
    layer_index: int
    tensor: CacheTensorName = "values"

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("cache tensor layer_index must be non-negative")
        if self.tensor not in {"keys", "values"}:
            raise ValueError("cache tensor must be keys or values")

    @property
    def identifier(self) -> str:
        return f"layer_{self.layer_index:03d}_{self.tensor}"


def parse_cache_tensor_capture_spec(raw: str) -> CacheTensorCaptureSpec:
    layer_text, separator, tensor = raw.partition(":")
    if not separator:
        tensor = "values"
    try:
        layer_index = int(layer_text)
    except ValueError as exc:
        raise ValueError(
            f"cache tensor capture must be LAYER[:keys|values]: {raw}"
        ) from exc
    return CacheTensorCaptureSpec(layer_index=layer_index, tensor=tensor)  # type: ignore[arg-type]


def cache_tensor_capture_policy(
    specs: tuple[CacheTensorCaptureSpec, ...] | list[CacheTensorCaptureSpec],
) -> list[dict[str, Any]]:
    return [
        {"layer_index": spec.layer_index, "tensor": spec.tensor}
        for spec in sorted(set(specs))
    ]


def capture_prompt_cache_tensors(
    prompt_cache_state: Any,
    *,
    specs: tuple[CacheTensorCaptureSpec, ...] | list[CacheTensorCaptureSpec],
    run_output_path: Path,
    relative_to: Path,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    cache = getattr(prompt_cache_state, "cache", None)
    if not isinstance(cache, (list, tuple)) or not cache:
        raise ValueError("prompt cache state has no layer cache entries")

    token_ids = getattr(prompt_cache_state, "token_ids", None)
    token_count = len(token_ids) if token_ids is not None else 0
    records = []
    for spec in sorted(set(specs)):
        if spec.layer_index >= len(cache):
            raise ValueError(
                f"cache tensor layer {spec.layer_index} is outside {len(cache)} layers"
            )
        entry = cache[spec.layer_index]
        tensor = getattr(entry, spec.tensor, None)
        if tensor is None or not hasattr(tensor, "shape"):
            raise ValueError(
                f"cache layer {spec.layer_index} has no {spec.tensor} tensor"
            )
        raw_shape = [int(value) for value in tensor.shape]
        if len(raw_shape) < 3:
            raise ValueError(
                f"cache tensor must expose a sequence axis at -2: {raw_shape}"
            )
        allocated_sequence_length = raw_shape[-2]
        cache_offset = int(getattr(entry, "offset", allocated_sequence_length))
        if cache_offset <= 0 or cache_offset > allocated_sequence_length:
            raise ValueError(
                f"invalid cache offset {cache_offset} for shape {raw_shape}"
            )
        trimmed = tensor[..., :cache_offset, :]
        values = _to_float32_numpy(trimmed)
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError(
                f"cache layer {spec.layer_index} {spec.tensor} contains non-finite values"
            )

        path = cache_tensor_sidecar_path(run_output_path, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, tensor=values)
        records.append(
            {
                "path": Path(os.path.relpath(path, relative_to)).as_posix(),
                "sha256": _sha256(path),
                "format": "numpy_npz",
                "array_key": "tensor",
                "value_kind": "prompt_cache_tensor",
                "layer_index": spec.layer_index,
                "tensor": spec.tensor,
                "source_dtype": str(getattr(tensor, "dtype", "unknown")),
                "dtype": str(values.dtype),
                "raw_shape": raw_shape,
                "shape": list(values.shape),
                "sequence_axis": -2,
                "allocated_sequence_length": allocated_sequence_length,
                "effective_sequence_length": cache_offset,
                "cache_offset": cache_offset,
                "token_count": token_count,
                "offset_matches_token_count": cache_offset == token_count,
                "uncompressed_nbytes": int(values.nbytes),
            }
        )
    return records


def cache_tensor_sidecar_path(
    run_output_path: Path,
    spec: CacheTensorCaptureSpec,
) -> Path:
    sidecar_dir = run_output_path.parent / f"{run_output_path.stem}_cache_tensors"
    return sidecar_dir / f"source__{spec.identifier}.npz"


def load_cache_tensor_artifact(
    run_path: Path,
    metadata: dict[str, Any],
    *,
    verify_hash: bool = True,
) -> np.ndarray:
    path = (run_path.parent / str(metadata["path"])).resolve()
    if verify_hash and _sha256(path) != metadata.get("sha256"):
        raise ValueError(f"cache tensor sidecar hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(
            archive[str(metadata.get("array_key", "tensor"))],
            dtype=np.float32,
        )
    expected_shape = tuple(int(value) for value in metadata.get("shape") or ())
    if expected_shape and values.shape != expected_shape:
        raise ValueError(
            f"cache tensor sidecar shape mismatch: {values.shape} != {expected_shape}"
        )
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError(f"cache tensor sidecar contains non-finite values: {path}")
    return values


def _to_float32_numpy(values: Any) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float32)
    except (RuntimeError, TypeError, ValueError):
        pass
    if hasattr(values, "astype"):
        try:
            import mlx.core as mx

            promoted = values.astype(mx.float32)
            mx.eval(promoted)
            return np.asarray(promoted, dtype=np.float32)
        except Exception:
            pass
    if hasattr(values, "tolist"):
        return np.asarray(values.tolist(), dtype=np.float32)
    raise TypeError(f"cannot convert {type(values).__name__} cache tensor to float32")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
