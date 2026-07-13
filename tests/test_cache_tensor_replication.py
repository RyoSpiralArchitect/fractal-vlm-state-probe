from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_tensor_artifact import (
    CacheTensorCaptureSpec,
    capture_prompt_cache_tensors,
)
from fractal_vlm_state_probe.cache_tensor_factorial import (
    analyze_cache_tensor_factorial,
)
from fractal_vlm_state_probe.cache_tensor_replication import (
    analyze_cache_tensor_replication,
)


def test_cache_tensor_replication_reports_shared_image_direction(
    tmp_path: Path,
) -> None:
    first = _factorial(tmp_path / "first", mm_id="b", jj_id="c", scale=2.0)
    second = _factorial(tmp_path / "second", mm_id="c", jj_id="d", scale=1.0)

    result = analyze_cache_tensor_replication(
        {"b_c": first, "c_d": second},
        analysis_paths={"b_c": Path("b_c.json"), "c_d": Path("c_d.json")},
    )

    assert result["group_count"] == 1
    group = result["groups"][0]
    assert group["source_pair_ids"] == ["b_c", "c_d"]
    assert group["interaction_argmax_in_image_count"] == 2
    assert group["pre_image_all_effects_zero_count"] == 2
    image = next(
        record for record in group["regions"] if record["region"] == "image_tokens"
    )
    assert image["direction_cosine_summary"]["median"] == 1.0
    json.dumps(result)


def test_cache_tensor_replication_rejects_unaligned_token_regions(
    tmp_path: Path,
) -> None:
    first = _factorial(tmp_path / "first", mm_id="b", jj_id="c", scale=2.0)
    second = _factorial(tmp_path / "second", mm_id="c", jj_id="d", scale=1.0)
    for cell in second["cells"].values():
        cell["cache_token_layout"]["image_token_runs"] = [
            {"start": 2, "end": 3, "length": 2}
        ]

    with pytest.raises(ValueError, match="token-region positions do not align"):
        analyze_cache_tensor_replication(
            {"b_c": first, "c_d": second},
            analysis_paths={"b_c": Path("b_c.json"), "c_d": Path("c_d.json")},
        )


def _factorial(
    root: Path,
    *,
    mm_id: str,
    jj_id: str,
    scale: float,
) -> dict:
    root.mkdir(parents=True)
    image_effect = np.array(
        [[[[0.0, 0.0], [scale, scale], [scale, scale], [0.0, 0.0]]]],
        dtype=np.float32,
    )
    arrays = {
        "mm": np.zeros_like(image_effect),
        "jj": image_effect,
        "mj": np.zeros_like(image_effect),
        "jm": np.zeros_like(image_effect),
    }
    condition_ids = {
        "mm": f"mandelbrot_zoom_{mm_id}_50f",
        "jj": f"julia_zoom_{jj_id}_50f",
        "mj": "hybrid_mj",
        "jm": "hybrid_jm",
    }
    runs = {}
    paths = {}
    for cell, values in arrays.items():
        path = root / f"{cell}.json"
        metadata = capture_prompt_cache_tensors(
            SimpleNamespace(
                token_ids=[10, 11, 11, 12],
                cache=[SimpleNamespace(values=values, keys=-values, offset=4)],
            ),
            specs=[CacheTensorCaptureSpec(0)],
            run_output_path=path,
            relative_to=root,
        )[0]
        runs[cell] = {
            "model_id": "example/model",
            "stimulus": {"condition": {"condition_id": condition_ids[cell]}},
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
        path.write_text(json.dumps(runs[cell]), encoding="utf-8")
        paths[cell] = path
    return analyze_cache_tensor_factorial(
        runs=runs,
        run_paths=paths,
        layer_index=0,
    )
