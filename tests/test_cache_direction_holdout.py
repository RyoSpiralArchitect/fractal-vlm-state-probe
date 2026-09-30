from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_direction_holdout import (
    analyze_pairing_seed_holdout,
    evaluate_pairing_holdout,
    holm_adjust,
)
from fractal_vlm_state_probe.cache_tensor_artifact import (
    CacheTensorCaptureSpec,
    capture_prompt_cache_tensors,
)


def _orthogonal_families() -> tuple[dict, dict]:
    vectors, metadata = {}, {}
    for index in range(8):
        family = f"family_{index}"
        for replicate in (1, 2, 3, 4):
            label = f"{family}_r{replicate}"
            vectors[label] = np.eye(8)[index] * replicate
            metadata[label] = {
                "pairing_family": family,
                "broad_class": "geometry" if index < 4 else "stochastic",
                "replicate": replicate,
            }
    return vectors, metadata


def test_frozen_references_predict_new_seeds_with_family_block_exact_test() -> None:
    vectors, metadata = _orthogonal_families()
    result = evaluate_pairing_holdout(vectors, metadata)

    assert result["available"] is True
    assert result["own_family_cosine_mean"] == pytest.approx(1)
    assert result["other_family_cosine_mean"] == pytest.approx(0)
    assert result["mean_family_margin"] == pytest.approx(1)
    assert result["positive_family_margin_count"] == 8
    assert result["retrieval_correct_count"] == 16
    assert result["exact_block_test"]["assignment_count"] == 576
    assert result["exact_block_test"]["p_greater"] == pytest.approx(1 / 576)


def test_test_seeds_cannot_redefine_the_reference() -> None:
    vectors, metadata = _orthogonal_families()
    for replicate in (3, 4):
        vectors[f"family_0_r{replicate}"] = np.eye(8)[1]
    result = evaluate_pairing_holdout(vectors, metadata)

    assert result["family_margins"]["family_0"] == pytest.approx(-1 / 3)
    rows = [r for r in result["test_seed_scores"] if r["pairing_family"] == "family_0"]
    assert all(r["retrieved_family"] == "family_1" for r in rows)
    assert all(r["own_family_cosine"] == pytest.approx(0) for r in rows)


def test_reference_seeds_receive_equal_direction_weight_despite_amplitude() -> None:
    vectors, metadata = _orthogonal_families()
    vectors["family_0_r1"] = np.eye(8)[0] * 1000
    vectors["family_0_r2"] = np.eye(8)[1]
    vectors["family_0_r3"] = np.eye(8)[0] + np.eye(8)[1]
    vectors["family_0_r4"] = np.eye(8)[0] + np.eye(8)[1]
    result = evaluate_pairing_holdout(vectors, metadata)

    rows = [r for r in result["test_seed_scores"] if r["pairing_family"] == "family_0"]
    assert all(r["own_family_cosine"] == pytest.approx(1) for r in rows)


def test_zero_and_cancelling_directions_are_explicitly_unavailable() -> None:
    vectors, metadata = _orthogonal_families()
    vectors["family_0_r3"] *= 0
    result = evaluate_pairing_holdout(vectors, metadata)
    assert result["available"] is False
    assert "zero interaction" in result["reason"]

    vectors, metadata = _orthogonal_families()
    vectors["family_0_r2"] *= -1
    result = evaluate_pairing_holdout(vectors, metadata)
    assert result["available"] is False
    assert "cancelling reference" in result["reason"]


def test_missing_replicate_and_cross_class_family_are_rejected() -> None:
    vectors, metadata = _orthogonal_families()
    metadata["family_0_r4"]["replicate"] = 3
    with pytest.raises(ValueError, match="unique replicates"):
        evaluate_pairing_holdout(vectors, metadata)
    metadata["family_0_r4"]["replicate"] = 4
    metadata["family_0_r4"]["broad_class"] = "stochastic"
    with pytest.raises(ValueError, match="crosses broad classes"):
        evaluate_pairing_holdout(vectors, metadata)


def test_holm_adjustment_preserves_original_order_and_monotonicity() -> None:
    assert holm_adjust([0.03, 0.001, 0.02, 0.9]) == pytest.approx(
        [0.06, 0.004, 0.06, 0.9]
    )
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust([float("nan")])


def test_head_and_band_views_preserve_tensor_energy_and_registered_split(
    tmp_path: Path,
) -> None:
    _, metadata = _orthogonal_families()
    analyses, paths = {}, {}
    layout = {
        "token_count": 8,
        "image_token_runs": [
            {"start": 2, "end": 3, "length": 2},
            {"start": 5, "end": 6, "length": 2},
        ],
    }
    for label, record in metadata.items():
        root = tmp_path / label
        root.mkdir()
        vector = np.zeros((1, 2, 8, 8), dtype=np.float32)
        feature = int(record["pairing_family"].rsplit("_", 1)[1])
        vector[:, 0, 2:, feature] = 1
        vector[:, 1, 2:, feature] = 2
        cells = {}
        for cell in ("mm", "jj", "mj", "jm"):
            path = root / f"{cell}.json"
            array = vector if cell == "jj" else np.zeros_like(vector)
            artifact = capture_prompt_cache_tensors(
                SimpleNamespace(
                    token_ids=list(range(8)),
                    cache=[SimpleNamespace(values=array, offset=8)],
                ),
                specs=[CacheTensorCaptureSpec(0)],
                run_output_path=path,
                relative_to=root,
            )[0]
            cells[cell] = {
                "source_path": str(path),
                "cache_tensor_artifact": artifact,
                "cache_token_layout": layout,
            }
        analyses[label] = {
            "analysis_kind": "source_cache_tensor_factorial_contrast",
            "model_id": "example/model",
            "layer_index": 0,
            "tensor": "values",
            "cells": cells,
        }
        paths[label] = root / "analysis.json"

    result = analyze_pairing_seed_holdout(
        analyses, analysis_paths=paths, metadata=metadata
    )

    views = {v["view"]: v for v in result["views"]}
    assert len(views) == 10
    assert result["within_image_span_non_image_positions"] == [4]
    assert result["token_region_position_counts"]["post_image"] == 1
    assert {k for k, v in views.items() if v["primary"]} == {
        "image_tokens",
        "post_image",
    }
    assert views["image_tokens"]["interaction_l2_norm_by_pair"][
        "family_0_r3"
    ] == pytest.approx(np.sqrt(20))
    assert views["post_image"]["interaction_l2_norm_by_pair"][
        "family_0_r3"
    ] == pytest.approx(np.sqrt(5))
    assert views["image_tokens/head_0"]["interaction_energy_fraction_by_pair"][
        "family_0_r3"
    ] == pytest.approx(0.2)
    assert views["image_tokens/head_1"]["interaction_energy_fraction_by_pair"][
        "family_0_r3"
    ] == pytest.approx(0.8)
    for band in range(4):
        view = views[f"image_tokens/band_{band}"]
        assert view["sequence_positions"] == [[2, 3, 5, 6][band]]
        assert view["interaction_energy_fraction_by_pair"][
            "family_0_r3"
        ] == pytest.approx(0.25)
    assert all(v["retrieval_correct_count"] == 16 for v in views.values())
