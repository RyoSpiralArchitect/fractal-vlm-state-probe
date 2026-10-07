import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.run_graded_cache_study import validate_config
from fractal_vlm_state_probe.graded_cache import (
    directional_decomposition,
    endpoint_reference_vectors,
)


def test_decomposition_distinguishes_scale_from_orthogonal_change():
    scale = directional_decomposition([1, 0], [2, 0])
    assert scale["cosine"] == 1
    assert scale["projection_coefficient"] == scale["norm_ratio"] == 2
    assert scale["perpendicular_delta_energy_fraction"] == 0
    rotation = directional_decomposition([1, 0], [1, 1])
    assert rotation["projection_coefficient"] == 1
    assert rotation["parallel_delta_rms"] == 0
    assert rotation["perpendicular_delta_energy_fraction"] == 1
    assert rotation["norm_ratio"] == pytest.approx(np.sqrt(2))


def test_decomposition_keeps_sign_and_zero_denominators():
    flipped = directional_decomposition([1, 0], [-1, 0])
    assert flipped["projection_coefficient"] == flipped["cosine"] == -1
    same = directional_decomposition([1, 0], [1, 0])
    assert same["delta_rms"] == 0
    assert same["perpendicular_delta_energy_fraction"] is None
    zero = directional_decomposition([1, 0], [0, 0])
    assert zero["cosine"] is None and zero["norm_ratio"] == 0
    assert not directional_decomposition([0, 0], [1, 0])["available"]


def test_random_vector_pythagorean_identity():
    rng = np.random.default_rng(17)
    for _ in range(20):
        a, b = rng.normal(size=(2, 64))
        r = directional_decomposition(a, b)
        assert r["relative_pythagorean_residual"] < 1e-12
        assert r["delta_rms"] ** 2 == pytest.approx(
            r["parallel_delta_rms"] ** 2 + r["perpendicular_delta_rms"] ** 2
        )
        assert r["projection_coefficient"] == pytest.approx(
            r["cosine"] * r["norm_ratio"]
        )


@pytest.mark.parametrize(
    "a,b", [([], []), ([1], [1, 2]), ([float("nan")], [1]), ([1], [float("inf")])]
)
def test_invalid_directional_inputs_fail_closed(a, b):
    with pytest.raises(ValueError):
        directional_decomposition(a, b)


def test_endpoint_transfer_uses_only_endpoint_reference_seeds():
    endpoint = {str(r): np.array([r, 0]) for r in range(1, 5)}
    compared = {str(r): np.array([0, r]) for r in range(1, 5)}
    metadata = {str(r): {"replicate": r} for r in range(1, 5)}
    mixed = endpoint_reference_vectors(endpoint, compared, metadata)
    for r in range(1, 5):
        assert mixed[str(r)] is (endpoint if r < 3 else compared)[str(r)]
    changed = {**endpoint, "3": np.array([999, 999]), "4": np.array([-999, 999])}
    other = endpoint_reference_vectors(changed, compared, metadata)
    assert all(np.array_equal(mixed[k], other[k]) for k in mixed)


def test_transfer_rejects_misaligned_labels_and_replicates():
    with pytest.raises(ValueError):
        endpoint_reference_vectors({"a": [1]}, {}, {"a": {"replicate": 1}})
    with pytest.raises(ValueError):
        endpoint_reference_vectors({"a": [1]}, {"a": [1]}, {"a": {"replicate": 5}})


def test_registered_config_rejects_posthoc_target_or_family_changes():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/graded_cache_fastvlm_v1.json").read_text())
    validate_config(config)
    for key, value in (
        ("targets", ["0:keys"]),
        ("primary_test_family_size", 12),
        ("levels", ["selected_08_of_08"]),
        ("test_replicates", [2, 3]),
        ("expected_calibration_cells", 4),
    ):
        with pytest.raises(ValueError):
            validate_config({**config, key: value})
