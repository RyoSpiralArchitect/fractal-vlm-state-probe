from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .cache_interventions import CacheTensorSwapSpec, swap_prompt_cache_tensor
from .delivery import frame_artifact_list, prepare_frame_deliveries, stimulus_delivery_record
from .mlx_cache_swap import (
    ValuesSwapRunConfig,
    _artifact_output_path,
    _build_stream_context,
    _context_record,
    _load_valid_manifest,
    _probe_frame_count,
    _run_creative_probe,
)
from .mlx_stream import (
    _load_mlx_runtime,
    _select_frames,
    clone_prompt_cache_state,
    summarize_prompt_cache_state,
)
from .seeding import set_global_seed
from .stimulus import write_json


@dataclass(frozen=True)
class ValuesSwapSweepConfig:
    run: ValuesSwapRunConfig
    layer_indices: tuple[int, ...] = (23,)
    probe_seeds: tuple[int, ...] = (0,)
    include_self_sham: bool = True


def run_values_swap_probe_sweep(config: ValuesSwapSweepConfig) -> dict[str, Any]:
    _validate_sweep_config(config)
    base = config.run
    seed_record = set_global_seed(base.seed, include_mlx=True)
    source_manifest = _load_valid_manifest(base.source_manifest_path)
    donor_manifest = _load_valid_manifest(base.donor_manifest_path)
    source_frames = _select_frames(
        source_manifest["frames"],
        frame_stride=base.frame_stride,
        max_frames=base.max_frames,
    )
    donor_frames = _select_frames(
        donor_manifest["frames"],
        frame_stride=base.frame_stride,
        max_frames=base.max_frames,
    )
    if len(source_frames) != len(donor_frames):
        raise ValueError(
            "source and donor selected frame counts differ: "
            f"{len(source_frames)} != {len(donor_frames)}"
        )
    frame_count_to_consume = _probe_frame_count(base.probe_phase, len(source_frames))
    source_frames = source_frames[:frame_count_to_consume]
    donor_frames = donor_frames[:frame_count_to_consume]

    mlx = _load_mlx_runtime(base.model_id)
    prompt_cache_prototype = mlx["prompt_cache_state"]
    if prompt_cache_prototype is None:
        raise RuntimeError("MLX PromptCacheState is unavailable; cache swap sweep cannot run")

    source_deliveries = prepare_frame_deliveries(
        output_path=_artifact_output_path(base.output_path, base.source_label),
        manifest_base=base.source_manifest_path.parent,
        frames=source_frames,
        mode="visual_stream",
        include_frame_artifacts=base.include_frame_artifacts,
    )
    donor_deliveries = prepare_frame_deliveries(
        output_path=_artifact_output_path(base.output_path, base.donor_label),
        manifest_base=base.donor_manifest_path.parent,
        frames=donor_frames,
        mode="visual_stream",
        include_frame_artifacts=base.include_frame_artifacts,
    )
    source_context = _build_stream_context(
        label=base.source_label,
        frames=source_frames,
        deliveries=source_deliveries,
        manifest_path=base.source_manifest_path,
        output_base=base.output_path.parent,
        prompt_cache_state=prompt_cache_prototype.__class__(),
        mlx=mlx,
        config=base,
    )
    donor_context = _build_stream_context(
        label=base.donor_label,
        frames=donor_frames,
        deliveries=donor_deliveries,
        manifest_path=base.donor_manifest_path,
        output_base=base.output_path.parent,
        prompt_cache_state=prompt_cache_prototype.__class__(),
        mlx=mlx,
        config=base,
    )
    source_summary = summarize_prompt_cache_state(
        source_context["prompt_cache_state"],
        max_layers=base.cache_summary_max_layers,
    )
    donor_summary = summarize_prompt_cache_state(
        donor_context["prompt_cache_state"],
        max_layers=base.cache_summary_max_layers,
    )

    trials = []
    for probe_seed in config.probe_seeds:
        seed_config = replace(base, probe_seed=probe_seed)
        source_baseline = _run_creative_probe(
            label="source_baseline",
            history=source_context["history"],
            prompt_cache_state=clone_prompt_cache_state(source_context["prompt_cache_state"]),
            source_cache_summary=source_summary,
            intervention=None,
            mlx=mlx,
            config=seed_config,
        )
        donor_baseline = _run_creative_probe(
            label="donor_baseline",
            history=donor_context["history"],
            prompt_cache_state=clone_prompt_cache_state(donor_context["prompt_cache_state"]),
            source_cache_summary=donor_summary,
            intervention=None,
            mlx=mlx,
            config=seed_config,
        )
        for layer_index in config.layer_indices:
            trial_config = replace(seed_config, layer_index=layer_index)
            spec = CacheTensorSwapSpec(layer_index=layer_index, tensor="values")
            source_with_donor, source_record = swap_prompt_cache_tensor(
                source_context["prompt_cache_state"],
                donor_context["prompt_cache_state"],
                spec,
            )
            probes = {
                "source_baseline": source_baseline,
                "donor_baseline": donor_baseline,
                "source_with_donor_values": _run_creative_probe(
                    label="source_with_donor_values",
                    history=source_context["history"],
                    prompt_cache_state=source_with_donor,
                    source_cache_summary=source_summary,
                    intervention={
                        **source_record,
                        "direction": f"{base.source_label}_with_{base.donor_label}_values",
                    },
                    mlx=mlx,
                    config=trial_config,
                ),
            }
            if base.include_reciprocal:
                donor_with_source, donor_record = swap_prompt_cache_tensor(
                    donor_context["prompt_cache_state"],
                    source_context["prompt_cache_state"],
                    spec,
                )
                probes["donor_with_source_values"] = _run_creative_probe(
                    label="donor_with_source_values",
                    history=donor_context["history"],
                    prompt_cache_state=donor_with_source,
                    source_cache_summary=donor_summary,
                    intervention={
                        **donor_record,
                        "direction": f"{base.donor_label}_with_{base.source_label}_values",
                    },
                    mlx=mlx,
                    config=trial_config,
                )
            if config.include_self_sham:
                source_with_source, sham_record = swap_prompt_cache_tensor(
                    source_context["prompt_cache_state"],
                    source_context["prompt_cache_state"],
                    spec,
                )
                probes["source_with_source_values"] = _run_creative_probe(
                    label="source_with_source_values",
                    history=source_context["history"],
                    prompt_cache_state=source_with_source,
                    source_cache_summary=source_summary,
                    intervention={
                        **sham_record,
                        "direction": f"{base.source_label}_with_self_values",
                        "sham": True,
                    },
                    mlx=mlx,
                    config=trial_config,
                )
            trials.append(
                {
                    "trial_index": len(trials),
                    "probe_seed": probe_seed,
                    "layer_index": layer_index,
                    "tensor": "values",
                    "probes": probes,
                }
            )

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "mlx_cache_values_swap_probe_sweep",
        "model_id": base.model_id,
        "runtime": mlx["runtime"],
        "reproducibility": {
            **seed_record,
            "probe_seeds": list(config.probe_seeds),
            "probe_seed_policy": "each probe seed is reset before every matched branch probe",
        },
        "manifests": {
            base.source_label: str(base.source_manifest_path),
            base.donor_label: str(base.donor_manifest_path),
        },
        "stimulus": {
            "source_condition": source_manifest.get("stimulus_condition"),
            "donor_condition": donor_manifest.get("stimulus_condition"),
            "source_frame_count_available": len(source_manifest.get("frames", [])),
            "donor_frame_count_available": len(donor_manifest.get("frames", [])),
            "frame_count_consumed": frame_count_to_consume,
            "probe_phase": base.probe_phase,
        },
        "context_policy": {
            "frame_delivery": "visual_stream",
            "stream_temperature": base.temperature,
            "probe_temperature": base.probe_temperature,
            "probe_id": base.probe_id,
            "probe_prompt": base.probe_prompt,
            "probe_max_tokens": base.probe_max_tokens,
            "generation_readout_top_k": base.generation_readout_top_k,
            "cache_summary_every": base.cache_summary_every,
            "cache_summary_max_layers": base.cache_summary_max_layers,
            "include_frame_artifacts": base.include_frame_artifacts,
        },
        "intervention_policy": {
            "kind": "locus_target_and_control_sweep",
            "tensor": "values",
            "layer_indices": list(config.layer_indices),
            "strict_token_ids": True,
            "strict_shape": True,
            "include_reciprocal": base.include_reciprocal,
            "include_self_sham": config.include_self_sham,
            "stream_context_reused_across_trials": True,
        },
        "non_claims": [
            "No consciousness or subjective-state claim is made.",
            "This is a causal cache intervention over computational traces, not a claim about experiences.",
            "A target-layer effect requires comparison with self-sham and off-locus controls.",
        ],
        "stimulus_delivery": stimulus_delivery_record(
            mode="visual_stream",
            include_frame_artifacts=base.include_frame_artifacts,
            blank_rgb=(0, 0, 0),
        ),
        "frame_artifacts": {
            base.source_label: frame_artifact_list(source_deliveries),
            base.donor_label: frame_artifact_list(donor_deliveries),
        },
        "stream_contexts": {
            base.source_label: _context_record(source_context, config=base),
            base.donor_label: _context_record(donor_context, config=base),
        },
        "trial_count": len(trials),
        "trials": trials,
    }
    write_json(base.output_path, result)
    return result


def _validate_sweep_config(config: ValuesSwapSweepConfig) -> None:
    if not config.layer_indices:
        raise ValueError("at least one layer index is required")
    if not config.probe_seeds:
        raise ValueError("at least one probe seed is required")
    if len(set(config.layer_indices)) != len(config.layer_indices):
        raise ValueError("layer indices must be unique")
    if len(set(config.probe_seeds)) != len(config.probe_seeds):
        raise ValueError("probe seeds must be unique")
    if any(layer_index < 0 for layer_index in config.layer_indices):
        raise ValueError("layer indices must be non-negative")
