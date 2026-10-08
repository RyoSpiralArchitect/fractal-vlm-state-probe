import copy

import numpy as np
import pytest

from fractal_vlm_state_probe.cache_direction_holdout import evaluate_pairing_holdout
from fractal_vlm_state_probe.cli.summarize_graded_cache_study import (
    verify_decomposition,
    verify_transfer,
)
from fractal_vlm_state_probe.graded_cache import (
    directional_decomposition,
    endpoint_reference_vectors,
)


def test_independent_direction_decomposition_reconstructs_vectors():
    a, b = np.random.default_rng(74).normal(size=(2, 1024))
    verify_decomposition(directional_decomposition(a, b), a, b)
    for x, y in ((a, a), (a, np.zeros_like(a)), (np.zeros_like(a), a)):
        verify_decomposition(directional_decomposition(x, y), x, y)


@pytest.mark.parametrize(
    "field",
    [
        "projection_coefficient",
        "norm_ratio",
        "perpendicular_delta_energy_fraction",
        "parallel_delta_rms",
        "cosine",
    ],
)
def test_independent_direction_check_rejects_corrupt_metrics(field):
    a, b = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    row = directional_decomposition(a, b)
    row[field] += 0.1
    with pytest.raises(ValueError):
        verify_decomposition(row, a, b)


@pytest.fixture
def transfer_data():
    endpoint, compared, metadata = {}, {}, {}
    for family in range(8):
        for replicate in range(1, 5):
            label = f"f{family}_r{replicate}"
            metadata[label] = {
                "pairing_family": f"f{family}",
                "broad_class": "a" if family < 4 else "b",
                "replicate": replicate,
            }
            endpoint[label] = np.eye(8)[family]
            compared[label] = np.roll(endpoint[label], 1)
    result = evaluate_pairing_holdout(
        endpoint_reference_vectors(endpoint, compared, metadata), metadata
    )
    result.pop("exact_block_test")
    return result, metadata, endpoint, compared


def test_independent_transfer_reconstruction_checks_all_cosines(transfer_data):
    assert verify_transfer(*transfer_data) == 128


@pytest.mark.parametrize(
    "field", ["cosine", "margin", "retrieval", "reference", "inferential"]
)
def test_transfer_publication_rejects_corruption(transfer_data, field):
    row, metadata, endpoint, compared = copy.deepcopy(transfer_data)
    if field == "cosine":
        row["test_seed_scores"][0]["reference_cosines"]["f0"] += 0.1
    elif field == "margin":
        row["mean_family_margin"] += 0.1
    elif field == "retrieval":
        row["retrieval_correct_count"] += 1
    elif field == "reference":
        row["reference_labels"][0] = "f0_r3"
    else:
        row["exact_block_test"] = {}
    with pytest.raises(ValueError):
        verify_transfer(row, metadata, endpoint, compared)


@pytest.mark.parametrize("zero_case", ["test_zero", "cancelling_references"])
def test_unavailable_transfer_is_preserved(transfer_data, zero_case):
    _, metadata, endpoint, compared = transfer_data
    if zero_case == "test_zero":
        compared["f0_r3"] = np.zeros(8)
    else:
        endpoint["f0_r2"] = -endpoint["f0_r1"]
    row = evaluate_pairing_holdout(
        endpoint_reference_vectors(endpoint, compared, metadata), metadata
    )
    assert not row["available"]
    assert verify_transfer(row, metadata, endpoint, compared) == 0
