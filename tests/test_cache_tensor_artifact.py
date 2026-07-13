from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_tensor_artifact import (
    CacheTensorCaptureSpec,
    capture_prompt_cache_tensors,
    load_cache_tensor_artifact,
    parse_cache_tensor_capture_spec,
)


def test_cache_tensor_capture_trims_allocated_capacity_and_round_trips(
    tmp_path: Path,
) -> None:
    values = np.arange(24, dtype=np.float16).reshape(1, 1, 6, 4)
    state = SimpleNamespace(
        token_ids=[1, 2, 3, 4],
        cache=[SimpleNamespace(keys=-values, values=values, offset=4)],
    )
    run_path = tmp_path / "run.json"

    records = capture_prompt_cache_tensors(
        state,
        specs=[CacheTensorCaptureSpec(0, "values")],
        run_output_path=run_path,
        relative_to=tmp_path,
    )

    assert len(records) == 1
    metadata = records[0]
    assert metadata["raw_shape"] == [1, 1, 6, 4]
    assert metadata["shape"] == [1, 1, 4, 4]
    assert metadata["cache_offset"] == 4
    assert metadata["offset_matches_token_count"] is True
    loaded = load_cache_tensor_artifact(run_path, metadata)
    np.testing.assert_array_equal(loaded, values[..., :4, :].astype(np.float32))


def test_cache_tensor_artifact_rejects_hash_mismatch(tmp_path: Path) -> None:
    state = SimpleNamespace(
        token_ids=[1, 2],
        cache=[
            SimpleNamespace(
                keys=np.ones((1, 1, 2, 2), dtype=np.float32),
                values=np.ones((1, 1, 2, 2), dtype=np.float32),
                offset=2,
            )
        ],
    )
    run_path = tmp_path / "run.json"
    metadata = capture_prompt_cache_tensors(
        state,
        specs=[CacheTensorCaptureSpec(0)],
        run_output_path=run_path,
        relative_to=tmp_path,
    )[0]
    metadata["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="hash mismatch"):
        load_cache_tensor_artifact(run_path, metadata)


def test_parse_cache_tensor_capture_spec_defaults_to_values() -> None:
    assert parse_cache_tensor_capture_spec("33") == CacheTensorCaptureSpec(33, "values")
    assert parse_cache_tensor_capture_spec("2:keys") == CacheTensorCaptureSpec(
        2, "keys"
    )
