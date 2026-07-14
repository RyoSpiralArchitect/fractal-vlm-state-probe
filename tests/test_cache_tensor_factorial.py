from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fractal_vlm_state_probe.cache_tensor_artifact import (
    CacheTensorCaptureSpec,
    capture_prompt_cache_tensors,
)
from fractal_vlm_state_probe.cache_tensor_factorial import (
    analyze_cache_tensor_factorial,
    balanced_factorial_contrast_vectors,
    cache_tensor_regions,
    factorial_effect_vectors,
)


def test_cache_tensor_factorial_localizes_interaction_to_image_tokens(
    tmp_path: Path,
) -> None:
    arrays = {
        "mm": np.zeros((1, 1, 4, 2), dtype=np.float32),
        "jj": np.array(
            [[[[0.0, 0.0], [2.0, 2.0], [2.0, 2.0], [0.0, 0.0]]]],
            dtype=np.float32,
        ),
        "mj": np.zeros((1, 1, 4, 2), dtype=np.float32),
        "jm": np.zeros((1, 1, 4, 2), dtype=np.float32),
    }
    runs = {}
    paths = {}
    for cell, values in arrays.items():
        run_path = tmp_path / f"{cell}.json"
        metadata = capture_prompt_cache_tensors(
            SimpleNamespace(
                token_ids=[10, 11, 11, 12],
                cache=[SimpleNamespace(values=values, keys=-values, offset=4)],
            ),
            specs=[CacheTensorCaptureSpec(0)],
            run_output_path=run_path,
            relative_to=tmp_path,
        )[0]
        runs[cell] = _run(cell, metadata)
        paths[cell] = run_path

    result = analyze_cache_tensor_factorial(
        runs=runs,
        run_paths=paths,
        layer_index=0,
    )

    regions = {record["region"]: record for record in result["regions"]}
    image = regions["image_tokens"]["effects"]["interaction"]
    non_image = regions["non_image_tokens"]["effects"]["interaction"]
    assert image["rms"] == 2.0
    assert image["argmax_sequence_position"] == 1
    assert non_image["l2_norm"] == 0.0
    assert result["interaction_partition"]["image_energy_fraction"] == 1.0
    assert result["interaction_partition"]["partition_closes"] is True
    assert (
        regions["image_tokens"]["balanced_contrast_energy"]["energy_shares"][
            "interaction_contrast"
        ]
        == 1.0 / 3.0
    )
    json.dumps(result)


def test_factorial_effect_vector_formulas() -> None:
    cells = {
        "mm": np.array([0.0]),
        "jj": np.array([4.0]),
        "mj": np.array([1.0]),
        "jm": np.array([2.0]),
    }

    effects = factorial_effect_vectors(cells)

    assert effects["spatial_main_effect"].item() == 2.5
    assert effects["palette_main_effect"].item() == 1.5
    assert effects["interaction"].item() == 1.0

    contrasts = balanced_factorial_contrast_vectors(cells)
    assert contrasts["spatial_contrast"].item() == 5.0
    assert contrasts["palette_contrast"].item() == 3.0
    assert contrasts["interaction_contrast"].item() == 1.0


def test_factorial_interaction_is_invariant_to_simultaneous_source_role_swap() -> None:
    cells = {
        "mm": np.array([1.0, 2.0]),
        "jj": np.array([7.0, 11.0]),
        "mj": np.array([3.0, 5.0]),
        "jm": np.array([13.0, 17.0]),
    }
    swapped = {
        "mm": cells["jj"],
        "jj": cells["mm"],
        "mj": cells["jm"],
        "jm": cells["mj"],
    }

    original = factorial_effect_vectors(cells)["interaction"]
    role_reversed = factorial_effect_vectors(swapped)["interaction"]

    np.testing.assert_array_equal(role_reversed, original)


def test_cache_tensor_regions_partition_effective_sequence() -> None:
    regions = cache_tensor_regions(
        {"image_token_runs": [{"start": 2, "end": 4, "length": 3}]},
        sequence_length=7,
    )

    assert regions["image_tokens"] == [2, 3, 4]
    assert regions["non_image_tokens"] == [0, 1, 5, 6]
    assert regions["pre_image"] == [0, 1]
    assert regions["post_image"] == [5, 6]


def test_cache_tensor_regions_leave_roles_unassigned_without_image_positions() -> None:
    regions = cache_tensor_regions({}, sequence_length=4)

    assert regions["all_effective"] == [0, 1, 2, 3]
    assert regions["image_tokens"] == []
    assert regions["non_image_tokens"] == []
    assert regions["pre_image"] == []
    assert regions["post_image"] == []


def test_cache_tensor_regions_use_validated_cache_position_mapping() -> None:
    regions = cache_tensor_regions(
        {
            "token_count": 6,
            "image_token_runs": [{"start": 1, "end": 3, "length": 3}],
            "cache_position_layout": {
                "available": True,
                "cache_sequence_length": 9,
                "image_position_runs": [{"start": 1, "end": 6, "length": 6}],
            },
        },
        sequence_length=9,
    )

    assert regions["image_tokens"] == [1, 2, 3, 4, 5, 6]
    assert regions["pre_image"] == [0]
    assert regions["post_image"] == [7, 8]


def test_cache_tensor_regions_fail_closed_on_unmapped_length_mismatch() -> None:
    regions = cache_tensor_regions(
        {
            "token_count": 6,
            "image_token_runs": [{"start": 1, "end": 3, "length": 3}],
        },
        sequence_length=9,
    )

    assert regions["image_tokens"] == []
    assert regions["non_image_tokens"] == []


def _run(cell: str, metadata: dict) -> dict:
    return {
        "model_id": "example/model",
        "stimulus": {"condition": {"condition_id": cell}},
        "stream_events": [
            {
                "cache_tensor_artifacts": [metadata],
                "cache_token_layout": {
                    "token_count": 4,
                    "image_token_count": 2,
                    "image_token_runs": [{"start": 1, "end": 2, "length": 2}],
                },
            }
        ],
    }
