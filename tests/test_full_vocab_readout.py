from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.full_vocab_readout import (
    analyze_full_vocab_readout_contrast,
    load_full_vocab_logprobs,
    write_full_vocab_logprob_sidecar,
)


def test_full_vocab_sidecar_round_trip_and_hash_check(tmp_path: Path) -> None:
    path = tmp_path / "run_full_vocab" / "after__probe__step_000.npz"
    values = np.log(np.array([0.5, 0.3, 0.2], dtype=np.float32))

    metadata = write_full_vocab_logprob_sidecar(
        values,
        path=path,
        relative_to=tmp_path,
    )

    assert metadata["path"] == "run_full_vocab/after__probe__step_000.npz"
    assert metadata["vocab_size"] == 3
    assert math.isclose(metadata["probability_sum"], 1.0, rel_tol=1e-6)
    loaded = load_full_vocab_logprobs(tmp_path / "run.json", metadata)
    assert np.allclose(loaded, values)

    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_full_vocab_logprobs(tmp_path / "run.json", metadata)


def test_full_vocab_sidecar_promotes_mlx_bfloat16(tmp_path: Path) -> None:
    import mlx.core as mx

    metadata = write_full_vocab_logprob_sidecar(
        mx.array([-2.0, -1.0, -0.5], dtype=mx.bfloat16),
        path=tmp_path / "bfloat16.npz",
        relative_to=tmp_path,
    )

    loaded = load_full_vocab_logprobs(tmp_path / "run.json", metadata)
    assert loaded.dtype == np.float64
    assert np.allclose(loaded, [-2.0, -1.0, -0.5])


def test_full_vocab_factorial_uses_complete_probability_distributions(
    tmp_path: Path,
) -> None:
    probabilities = {
        "mm": [0.4, 0.3, 0.2, 0.1],
        "jj": [0.4, 0.1, 0.1, 0.4],
        "mj": [0.3, 0.3, 0.2, 0.2],
        "jm": [0.2, 0.2, 0.3, 0.3],
    }
    runs = {}
    run_paths = {}
    for cell, cell_probabilities in probabilities.items():
        run_path = tmp_path / f"{cell}.json"
        sidecar_path = (
            tmp_path
            / f"{cell}_full_vocab"
            / "after__forced_family_choice__step_000.npz"
        )
        metadata = write_full_vocab_logprob_sidecar(
            np.log(np.asarray(cell_probabilities)),
            path=sidecar_path,
            relative_to=tmp_path,
        )
        runs[cell] = _run(cell, metadata)
        run_paths[cell] = run_path

    analysis = analyze_full_vocab_readout_contrast(
        runs=runs,
        run_paths=run_paths,
        max_token_effects=4,
    )

    assert analysis["available_record_count"] == 1
    record = analysis["records"][0]
    assert record["vocab_size"] == 4
    assert len(record["pairwise_distances"]) == 6
    assert all(item["jensen_shannon"] >= 0 for item in record["pairwise_distances"])
    interaction = record["probability_contrasts"]["interaction"]
    assert math.isclose(interaction["l1_norm"], 0.6, rel_tol=1e-6)
    assert interaction["argmax_token_id"] == 0
    assert interaction["argmax_token"] == "A"
    assert record["top_probability_interaction_tokens"][0]["token_id"] == 0
    balanced = record["balanced_probability_contrasts"]
    assert math.isclose(
        balanced["spatial_contrast"]["l1_norm"],
        2.0 * record["probability_contrasts"]["spatial_main_effect"]["l1_norm"],
        rel_tol=1e-6,
    )
    assert math.isclose(
        balanced["palette_contrast"]["l1_norm"],
        2.0 * record["probability_contrasts"]["palette_main_effect"]["l1_norm"],
        rel_tol=1e-6,
    )
    shares = record["balanced_probability_contrast_energy"]["energy_shares"]
    assert math.isclose(sum(shares.values()), 1.0, rel_tol=1e-12)
    candidates = record["forced_choice_candidates"]
    assert candidates["available"] is True
    assert math.isclose(
        candidates["cells"]["mm"]["candidate_probability_mass"],
        0.9,
        rel_tol=1e-6,
    )


def test_full_vocab_factorial_handles_zero_probability_support(tmp_path: Path) -> None:
    probabilities = {
        "mm": [0.5, 0.5, 0.0, 0.0],
        "jj": [0.5, 0.0, 0.5, 0.0],
        "mj": [0.5, 0.25, 0.25, 0.0],
        "jm": [0.5, 0.25, 0.0, 0.25],
    }
    runs = {}
    run_paths = {}
    for cell, cell_probabilities in probabilities.items():
        run_path = tmp_path / f"{cell}.json"
        values = np.asarray(
            [math.log(value) if value else -math.inf for value in cell_probabilities]
        )
        metadata = write_full_vocab_logprob_sidecar(
            values,
            path=tmp_path
            / f"{cell}_full_vocab"
            / "after__forced_family_choice__step_000.npz",
            relative_to=tmp_path,
        )
        runs[cell] = _run(cell, metadata)
        run_paths[cell] = run_path

    analysis = analyze_full_vocab_readout_contrast(
        runs=runs,
        run_paths=run_paths,
    )

    record = analysis["records"][0]
    assert any(
        item["symmetric_kl_is_infinite"] for item in record["pairwise_distances"]
    )
    assert any(item["symmetric_kl"] is None for item in record["pairwise_distances"])
    assert record["logprob_interaction"]["excluded_nonfinite_token_count"] == 3
    _assert_finite_json_value(analysis)


def test_full_vocab_factorial_aligns_rotated_labels_to_semantics(
    tmp_path: Path,
) -> None:
    probabilities = {
        "mm": [0.7, 0.2, 0.1, 0.0],
        "jj": [0.2, 0.2, 0.6, 0.0],
        "mj": [0.3, 0.4, 0.3, 0.0],
        "jm": [0.4, 0.2, 0.4, 0.0],
    }
    prompt_metadata = {
        "probe_family": "family",
        "prompt_variant": "rotated_labels",
        "candidate_labels": ["A", "B", "C"],
        "candidate_order": ["A", "B", "C"],
        "candidate_semantics": {
            "A": "unclear",
            "B": "mandelbrot",
            "C": "julia",
        },
    }
    runs = {}
    run_paths = {}
    for cell, cell_probabilities in probabilities.items():
        run_path = tmp_path / f"{cell}.json"
        values = np.asarray(
            [math.log(value) if value else -math.inf for value in cell_probabilities]
        )
        metadata = write_full_vocab_logprob_sidecar(
            values,
            path=tmp_path
            / f"{cell}_full_vocab"
            / "after__forced_family_choice_rotated_labels__step_000.npz",
            relative_to=tmp_path,
        )
        runs[cell] = _run(
            cell,
            metadata,
            probe_id="forced_family_choice_rotated_labels",
            prompt_metadata=prompt_metadata,
        )
        run_paths[cell] = run_path

    analysis = analyze_full_vocab_readout_contrast(
        runs=runs,
        run_paths=run_paths,
    )

    record = analysis["records"][0]
    assert record["prompt_variant"] == "rotated_labels"
    assert record["generated_semantics"] == {
        cell: "unclear" for cell in ("mm", "jj", "mj", "jm")
    }
    semantic = record["forced_choice_candidates"]["cells"]["mm"][
        "semantic_conditional_probabilities"
    ]
    assert semantic == {
        "unclear": pytest.approx(0.7),
        "mandelbrot": pytest.approx(0.2),
        "julia": pytest.approx(0.1),
    }


def _assert_finite_json_value(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_finite_json_value(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_json_value(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def _run(
    cell: str,
    metadata: dict,
    *,
    probe_id: str = "forced_family_choice",
    prompt_metadata: dict | None = None,
) -> dict:
    return {
        "stimulus": {"condition": {"condition_id": cell}},
        "probes": {
            "after": [
                {
                    "probe_id": probe_id,
                    **(prompt_metadata or {}),
                    "generation": {
                        "steps": [
                            {
                                "step_index": 0,
                                "token": "A",
                                "top_logprobs": [
                                    {"token_id": 0, "token": "T0", "logprob": -1.0},
                                    {"token_id": 0, "token": "A", "logprob": -1.0},
                                    {"token_id": 1, "token": "B", "logprob": -1.2},
                                    {"token_id": 2, "token": "C", "logprob": -1.4},
                                ],
                                "full_vocab_sidecar": metadata,
                            }
                        ]
                    },
                }
            ]
        },
    }
