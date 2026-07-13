from __future__ import annotations

from fractal_vlm_state_probe.cache_intervention_analysis import (
    analyze_values_swap_interventions,
    format_values_swap_analysis_markdown,
)


def test_values_swap_analysis_reports_complete_donor_pull() -> None:
    source = _probe([1, 2], [{10: -1.0, 11: -2.0}, {20: -1.0}])
    donor = _probe([1, 3], [{10: -2.0, 11: -1.0}, {20: -2.0}])
    run = _run(
        source=source,
        donor=donor,
        intervention=donor,
        reciprocal=source,
    )

    analysis = analyze_values_swap_interventions([run], run_paths=["run.json"])

    trial = analysis["trials"][0]
    assert trial["available"] is True
    assert trial["baseline_distance"]["token"]["normalized_edit_distance"] == 0.5
    assert trial["source_intervention"]["donor_pull_index"]["token"] == 1.0
    assert trial["source_intervention"]["donor_pull_index"]["top_k"] == 1.0
    assert trial["source_intervention"]["effect_to_baseline_ratio"]["token"] == 1.0
    assert trial["source_intervention"]["effect_to_baseline_ratio"]["top_k"] == 1.0
    assert trial["reciprocal_intervention"]["source_pull_index"]["token"] == 1.0
    assert analysis["group_summaries"][0]["trial_count"] == 1
    markdown = format_values_swap_analysis_markdown(analysis)
    assert "Cache Values-Swap" in markdown
    assert "| 0/1 | n/a |" in markdown


def test_values_swap_analysis_marks_pull_unavailable_for_identical_baselines() -> None:
    source = _probe([1, 2], [{10: -1.0}])
    run = _run(source=source, donor=source, intervention=source)

    analysis = analyze_values_swap_interventions([run])

    intervention = analysis["trials"][0]["source_intervention"]
    assert intervention["donor_pull_index"]["token"] is None
    assert intervention["donor_pull_index"]["top_k"] is None
    assert intervention["effect_to_baseline_ratio"]["token"] is None
    assert intervention["effect_to_baseline_ratio"]["top_k"] is None


def test_values_swap_analysis_reports_missing_probe_records() -> None:
    run = _run(
        source=_probe([1], [{10: -1.0}]),
        donor=_probe([2], [{10: -2.0}]),
        intervention=None,
    )

    analysis = analyze_values_swap_interventions([run])

    assert analysis["available_trial_count"] == 0
    assert analysis["trials"][0]["missing_probe_labels"] == ["source_with_donor_values"]


def test_values_swap_analysis_expands_sweep_trials() -> None:
    source = _probe([1, 2], [{10: -1.0}])
    donor = _probe([1, 3], [{10: -2.0}])
    base = _run(source=source, donor=donor, intervention=donor)
    sweep = {
        **base,
        "run_kind": "mlx_cache_values_swap_probe_sweep",
        "intervention_policy": {"layer_indices": [0, 23], "tensor": "values"},
        "trials": [
            {
                "trial_index": 0,
                "probe_seed": 4,
                "layer_index": 0,
                "tensor": "values",
                "probes": base["probes"],
            },
            {
                "trial_index": 1,
                "probe_seed": 5,
                "layer_index": 23,
                "tensor": "values",
                "probes": base["probes"],
            },
        ],
    }

    analysis = analyze_values_swap_interventions([sweep], run_paths=["sweep.json"])

    assert analysis["input_run_count"] == 1
    assert analysis["trial_count"] == 2
    assert [trial["metadata"]["layer_index"] for trial in analysis["trials"]] == [0, 23]
    assert [trial["metadata"]["probe_seed"] for trial in analysis["trials"]] == [4, 5]


def test_values_swap_group_summaries_sort_layers_numerically() -> None:
    source = _probe([1], [{10: -1.0}])
    donor = _probe([2], [{10: -2.0}])
    base = _run(source=source, donor=donor, intervention=source)
    sweep = {
        **base,
        "run_kind": "mlx_cache_values_swap_probe_sweep",
        "trials": [
            {"probe_seed": 0, "layer_index": layer, "tensor": "values", "probes": base["probes"]}
            for layer in (10, 8, 9)
        ],
    }

    analysis = analyze_values_swap_interventions([sweep])

    assert [group["layer_index"] for group in analysis["group_summaries"]] == [8, 9, 10]


def _run(
    *,
    source: dict,
    donor: dict,
    intervention: dict | None,
    reciprocal: dict | None = None,
) -> dict:
    probes = {
        "source_baseline": source,
        "donor_baseline": donor,
    }
    if intervention is not None:
        probes["source_with_donor_values"] = intervention
    if reciprocal is not None:
        probes["donor_with_source_values"] = reciprocal
    return {
        "model_id": "test/model",
        "reproducibility": {"probe_seed": 7},
        "stimulus": {
            "source_condition": {"condition_id": "mm"},
            "donor_condition": {"condition_id": "jj"},
            "probe_phase": "mid",
            "frame_count_consumed": 25,
        },
        "context_policy": {"probe_temperature": 0.7},
        "intervention_policy": {"layer_index": 23, "tensor": "values"},
        "manifests": {"source": "source.json", "donor": "donor.json"},
        "probes": probes,
    }


def _probe(token_ids: list[int], top_k_steps: list[dict[int, float]]) -> dict:
    steps = []
    for index, token_id in enumerate(token_ids):
        top_map = top_k_steps[min(index, len(top_k_steps) - 1)]
        steps.append(
            {
                "step_index": index,
                "token_id": token_id,
                "token": str(token_id),
                "top_logprobs": [
                    {"token_id": key, "token": str(key), "logprob": value}
                    for key, value in top_map.items()
                ],
            }
        )
    return {
        "assistant_text": " ".join(str(value) for value in token_ids),
        "generation": {"steps": steps},
    }
