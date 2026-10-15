import copy
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
from fractal_vlm_state_probe.cli.summarize_map_seed_cache_study import (
    verify_aggregate,
    verify_factorial,
)
from fractal_vlm_state_probe.map_seed_cache import aggregate_map_scores
from test_map_seed_cache import select, toy_matrix


def test_aggregate_publication_independently_reproduces_scores_and_assignments():
    rows, metadata = toy_matrix(rotate=True)
    for regime, same in (("same_map", True), ("cross_map", False)):
        actual = {
            "regime": regime,
            **aggregate_map_scores(select(rows, same), metadata, regime),
        }
        assert verify_aggregate(actual, rows, metadata) == 128


@pytest.mark.parametrize(
    "corruption",
    ["cosine", "retrieval", "sample_count", "p", "margin", "duplicate_pair"],
)
def test_aggregate_publication_rejects_corruption(corruption):
    rows, metadata = toy_matrix(rotate=True)
    actual = {
        "regime": "cross_map",
        **aggregate_map_scores(select(rows, False), metadata, "cross_map"),
    }
    if corruption == "cosine":
        actual["test_seed_scores"][0]["reference_cosines"]["f0"] += 0.01
    elif corruption == "retrieval":
        actual["retrieval_correct_count"] += 1
    elif corruption == "sample_count":
        actual["test_seed_count"] = 96
    elif corruption == "p":
        actual["exact_block_test"]["p_greater"] = -1
    elif corruption == "margin":
        actual["mean_family_margin"] += 0.01
    else:
        rows[1] = copy.deepcopy(rows[2])
    with pytest.raises(ValueError):
        verify_aggregate(actual, rows, metadata)


def test_unavailable_aggregate_is_not_promoted_by_publication():
    rows, metadata = toy_matrix()
    selected = select(rows, False)
    selected[0]["available"] = False
    actual = {
        "regime": "cross_map",
        **aggregate_map_scores(selected, metadata, "cross_map"),
    }
    assert verify_aggregate(actual, rows, metadata) == 0
    with pytest.raises(ValueError):
        verify_aggregate({**actual, "available": True}, rows, metadata)


@pytest.fixture
def factorial(tmp_path):
    rng = np.random.default_rng(5)
    arrays = {
        c: rng.integers(-16, 17, size=(1, 2, 397, 4)).astype(np.float32) / 8
        for c in ("mm", "jj", "mj", "jm")
    }
    runs, paths = {}, {}
    for cell, values in arrays.items():
        values[:, :, :42] = 0
        path = tmp_path / f"{cell}.json"
        artifacts = capture_prompt_cache_tensors(
            SimpleNamespace(
                token_ids=list(range(397)),
                cache=[SimpleNamespace(values=values, keys=-values, offset=397)],
            ),
            specs=[CacheTensorCaptureSpec(0)],
            run_output_path=path,
            relative_to=tmp_path,
        )
        runs[cell] = {
            "model_id": "test/model",
            "stimulus": {"condition": {"condition_id": cell}},
            "stream_events": [
                {
                    "cache_tensor_artifacts": artifacts,
                    "cache_token_layout": {
                        "token_count": 397,
                        "image_token_runs": [{"start": 42, "end": 297, "length": 256}],
                    },
                }
            ],
        }
        paths[cell] = path
    analysis = analyze_cache_tensor_factorial(runs=runs, run_paths=paths, layer_index=0)
    return analysis, {c: a.astype(np.float64) for c, a in arrays.items()}


def test_factorial_publication_recomputes_effects_and_localization(factorial):
    analysis, arrays = factorial
    actual = verify_factorial(analysis, arrays)
    np.testing.assert_array_equal(
        actual, arrays["jj"] - arrays["jm"] - arrays["mj"] + arrays["mm"]
    )


@pytest.mark.parametrize(
    "corruption", ["norm", "position", "share", "partition", "region"]
)
def test_factorial_publication_rejects_corruption(factorial, corruption):
    analysis, arrays = factorial
    if corruption == "norm":
        analysis["regions"][0]["effects"]["interaction"]["l2_norm"] += 1
    elif corruption == "position":
        analysis["regions"][0]["effects"]["interaction"][
            "argmax_sequence_position"
        ] = -1
    elif corruption == "share":
        analysis["regions"][0]["balanced_contrast_energy"]["energy_shares"][
            "interaction_contrast"
        ] += 0.1
    elif corruption == "partition":
        analysis["interaction_partition"]["image_energy_fraction"] += 0.1
    else:
        analysis["regions"].pop()
    with pytest.raises(ValueError):
        verify_factorial(analysis, arrays)
