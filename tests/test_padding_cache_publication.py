import copy

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_direction_holdout import evaluate_pairing_holdout
from fractal_vlm_state_probe.cli.summarize_padding_cache_study import verify_exact_test


@pytest.fixture
def exact_result():
    metadata, vectors = {}, {}
    for family in range(8):
        for replicate in range(1, 5):
            label = f"f{family}_r{replicate}"
            metadata[label] = {
                "pairing_family": f"f{family}",
                "broad_class": "geometry" if family < 4 else "stochastic",
                "replicate": replicate,
            }
            vectors[label] = np.eye(8)[family]
    return evaluate_pairing_holdout(vectors, metadata), metadata


def test_publication_independently_reconstructs_exact_assignment_test(exact_result):
    result, metadata = exact_result
    assert result["exact_block_test"]["p_greater"] == 1 / 576
    verify_exact_test(result, metadata)


@pytest.mark.parametrize(
    "field", ["count", "margin", "seed", "label", "reference", "family"]
)
def test_publication_rejects_corrupted_exact_test(exact_result, field):
    result, metadata = copy.deepcopy(exact_result)
    if field == "count":
        result["exact_block_test"]["extreme_count"] += 1
    elif field == "margin":
        result["mean_family_margin"] += 0.01
    elif field == "seed":
        result["test_seed_scores"][0]["margin"] += 0.01
    elif field == "label":
        result["test_seed_scores"][0]["label"] = "f0_r1"
    elif field == "reference":
        result["reference_labels"][0] = "f0_r3"
    else:
        result["family_margins"]["f0"] += 0.01
    with pytest.raises(ValueError):
        verify_exact_test(result, metadata)


def test_publication_preserves_unavailable_view():
    verify_exact_test({"available": False, "reason": "zero vector"}, {})
