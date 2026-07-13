from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .cache_interventions import CacheTensorSwapSpec, swap_prompt_cache_tensor
from .delivery import (
    frame_artifact_list,
    prepare_frame_deliveries,
    stimulus_delivery_record,
)
from .mlx_stream import (
    _collect_stream,
    _load_manifest,
    _load_mlx_runtime,
    _mid_probe_after_position,
    _run_frame_turn,
    _select_frames,
    clone_prompt_cache_state,
    summarize_prompt_cache_state,
)
from .prompts import CREATIVE_REFLECTION_PROMPT, SYSTEM_PROMPT
from .seeding import set_global_seed
from .stimulus import validate_manifest, write_json

ProbePhase = Literal["mid", "after"]

CREATIVE_IMPRESSION_PROMPT = CREATIVE_REFLECTION_PROMPT


@dataclass(frozen=True)
class ValuesSwapRunConfig:
    source_manifest_path: Path
    donor_manifest_path: Path
    output_path: Path
    model_id: str
    source_label: str = "source"
    donor_label: str = "donor"
    layer_index: int = 23
    probe_phase: ProbePhase = "mid"
    max_frames: int | None = None
    frame_stride: int = 1
    max_tokens: int = 2
    probe_max_tokens: int = 120
    temperature: float = 0.0
    probe_temperature: float = 0.7
    seed: int | None = 20260604
    probe_seed: int | None = 0
    probe_id: str = "creative_visual_impression"
    probe_prompt: str = CREATIVE_IMPRESSION_PROMPT
    cache_summary_every: int = 10
    cache_summary_max_layers: int | None = 4
    generation_readout_top_k: int = 20
    include_frame_artifacts: bool = True
    include_reciprocal: bool = True


def run_values_swap_probe(config: ValuesSwapRunConfig) -> dict[str, Any]:
    seed_record = set_global_seed(config.seed, include_mlx=True)
    source_manifest = _load_valid_manifest(config.source_manifest_path)
    donor_manifest = _load_valid_manifest(config.donor_manifest_path)
    source_frames = _select_frames(
        source_manifest["frames"],
        frame_stride=config.frame_stride,
        max_frames=config.max_frames,
    )
    donor_frames = _select_frames(
        donor_manifest["frames"],
        frame_stride=config.frame_stride,
        max_frames=config.max_frames,
    )
    if len(source_frames) != len(donor_frames):
        raise ValueError(
            "source and donor selected frame counts differ: "
            f"{len(source_frames)} != {len(donor_frames)}"
        )
    frame_count_to_consume = _probe_frame_count(config.probe_phase, len(source_frames))
    source_frames = source_frames[:frame_count_to_consume]
    donor_frames = donor_frames[:frame_count_to_consume]

    mlx = _load_mlx_runtime(config.model_id)
    prompt_cache_prototype = mlx["prompt_cache_state"]
    if prompt_cache_prototype is None:
        raise RuntimeError("MLX PromptCacheState is unavailable; cache swap intervention cannot run")

    source_deliveries = prepare_frame_deliveries(
        output_path=_artifact_output_path(config.output_path, config.source_label),
        manifest_base=config.source_manifest_path.parent,
        frames=source_frames,
        mode="visual_stream",
        include_frame_artifacts=config.include_frame_artifacts,
    )
    donor_deliveries = prepare_frame_deliveries(
        output_path=_artifact_output_path(config.output_path, config.donor_label),
        manifest_base=config.donor_manifest_path.parent,
        frames=donor_frames,
        mode="visual_stream",
        include_frame_artifacts=config.include_frame_artifacts,
    )

    source_context = _build_stream_context(
        label=config.source_label,
        frames=source_frames,
        deliveries=source_deliveries,
        manifest_path=config.source_manifest_path,
        output_base=config.output_path.parent,
        prompt_cache_state=prompt_cache_prototype.__class__(),
        mlx=mlx,
        config=config,
    )
    donor_context = _build_stream_context(
        label=config.donor_label,
        frames=donor_frames,
        deliveries=donor_deliveries,
        manifest_path=config.donor_manifest_path,
        output_base=config.output_path.parent,
        prompt_cache_state=prompt_cache_prototype.__class__(),
        mlx=mlx,
        config=config,
    )

    swap_spec = CacheTensorSwapSpec(layer_index=config.layer_index, tensor="values")
    source_with_donor, source_swap_record = swap_prompt_cache_tensor(
        source_context["prompt_cache_state"],
        donor_context["prompt_cache_state"],
        swap_spec,
    )
    probe_records = {
        "source_baseline": _run_creative_probe(
            label="source_baseline",
            history=source_context["history"],
            prompt_cache_state=clone_prompt_cache_state(source_context["prompt_cache_state"]),
            source_cache_summary=summarize_prompt_cache_state(
                source_context["prompt_cache_state"],
                max_layers=config.cache_summary_max_layers,
            ),
            intervention=None,
            mlx=mlx,
            config=config,
        ),
        "donor_baseline": _run_creative_probe(
            label="donor_baseline",
            history=donor_context["history"],
            prompt_cache_state=clone_prompt_cache_state(donor_context["prompt_cache_state"]),
            source_cache_summary=summarize_prompt_cache_state(
                donor_context["prompt_cache_state"],
                max_layers=config.cache_summary_max_layers,
            ),
            intervention=None,
            mlx=mlx,
            config=config,
        ),
        "source_with_donor_values": _run_creative_probe(
            label="source_with_donor_values",
            history=source_context["history"],
            prompt_cache_state=source_with_donor,
            source_cache_summary=summarize_prompt_cache_state(
                source_context["prompt_cache_state"],
                max_layers=config.cache_summary_max_layers,
            ),
            intervention={
                **source_swap_record,
                "direction": f"{config.source_label}_with_{config.donor_label}_values",
            },
            mlx=mlx,
            config=config,
        ),
    }

    if config.include_reciprocal:
        donor_with_source, donor_swap_record = swap_prompt_cache_tensor(
            donor_context["prompt_cache_state"],
            source_context["prompt_cache_state"],
            swap_spec,
        )
        probe_records["donor_with_source_values"] = _run_creative_probe(
            label="donor_with_source_values",
            history=donor_context["history"],
            prompt_cache_state=donor_with_source,
            source_cache_summary=summarize_prompt_cache_state(
                donor_context["prompt_cache_state"],
                max_layers=config.cache_summary_max_layers,
            ),
            intervention={
                **donor_swap_record,
                "direction": f"{config.donor_label}_with_{config.source_label}_values",
            },
            mlx=mlx,
            config=config,
        )

    result: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "mlx_cache_values_swap_probe",
        "model_id": config.model_id,
        "runtime": mlx["runtime"],
        "reproducibility": {
            **seed_record,
            "probe_seed": config.probe_seed,
            "probe_seed_policy": "same probe_seed is reset before each baseline and intervention probe",
        },
        "manifests": {
            config.source_label: str(config.source_manifest_path),
            config.donor_label: str(config.donor_manifest_path),
        },
        "stimulus": {
            "source_condition": source_manifest.get("stimulus_condition"),
            "donor_condition": donor_manifest.get("stimulus_condition"),
            "source_frame_count_available": len(source_manifest.get("frames", [])),
            "donor_frame_count_available": len(donor_manifest.get("frames", [])),
            "frame_count_consumed": frame_count_to_consume,
            "probe_phase": config.probe_phase,
        },
        "context_policy": {
            "frame_delivery": "visual_stream",
            "stream_temperature": config.temperature,
            "probe_temperature": config.probe_temperature,
            "probe_id": config.probe_id,
            "probe_prompt": config.probe_prompt,
            "probe_max_tokens": config.probe_max_tokens,
            "generation_readout_top_k": config.generation_readout_top_k,
            "cache_summary_every": config.cache_summary_every,
            "cache_summary_max_layers": config.cache_summary_max_layers,
            "include_frame_artifacts": config.include_frame_artifacts,
        },
        "intervention_policy": {
            "kind": "locus_target_swap",
            "tensor": "values",
            "layer_index": config.layer_index,
            "strict_token_ids": True,
            "strict_shape": True,
            "source_history_is_used_for_source_intervention": True,
            "donor_history_is_used_only_for_donor_baseline_or_reciprocal": True,
        },
        "non_claims": [
            "No consciousness or subjective-state claim is made.",
            "This is a causal cache intervention over computational traces, not a claim about experiences.",
            "Text drift under creative probing is exploratory until replicated across seeds, prompts, and controls.",
        ],
        "stimulus_delivery": stimulus_delivery_record(
            mode="visual_stream",
            include_frame_artifacts=config.include_frame_artifacts,
            blank_rgb=(0, 0, 0),
        ),
        "frame_artifacts": {
            config.source_label: frame_artifact_list(source_deliveries),
            config.donor_label: frame_artifact_list(donor_deliveries),
        },
        "stream_contexts": {
            config.source_label: _context_record(source_context, config=config),
            config.donor_label: _context_record(donor_context, config=config),
        },
        "probes": probe_records,
    }
    write_json(config.output_path, result)
    return result


def _build_stream_context(
    *,
    label: str,
    frames: list[dict[str, Any]],
    deliveries: dict[int, Any],
    manifest_path: Path,
    output_base: Path,
    prompt_cache_state: Any,
    mlx: dict[str, Any],
    config: ValuesSwapRunConfig,
) -> dict[str, Any]:
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    events = []
    for position, frame in enumerate(frames):
        event = _run_frame_turn(
            frame=frame,
            history=history,
            delivery=deliveries[int(frame["index"])],
            output_base=output_base,
            model=mlx["model"],
            processor=mlx["processor"],
            model_config=mlx["model_config"],
            stream_generate=mlx["stream_generate"],
            apply_chat_template=mlx["apply_chat_template"],
            prompt_cache_state=prompt_cache_state,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            capture_cache_summary=_should_capture_context_cache_summary(
                position,
                len(frames),
                config.cache_summary_every,
            ),
            cache_summary_max_layers=config.cache_summary_max_layers,
            generation_readout_top_k=config.generation_readout_top_k,
        )
        event["stream_label"] = label
        events.append(event)
    return {
        "label": label,
        "manifest_path": str(manifest_path),
        "history": history,
        "prompt_cache_state": prompt_cache_state,
        "events": events,
    }


def _run_creative_probe(
    *,
    label: str,
    history: list[dict[str, str]],
    prompt_cache_state: Any,
    source_cache_summary: dict[str, Any] | None,
    intervention: dict[str, Any] | None,
    mlx: dict[str, Any],
    config: ValuesSwapRunConfig,
) -> dict[str, Any]:
    probe_seed_record = set_global_seed(config.probe_seed, include_mlx=True)
    if prompt_cache_state is None:
        raise ValueError(f"{label} probe cache is unavailable")
    messages = history + [{"role": "user", "content": config.probe_prompt}]
    formatted_prompt = mlx["apply_chat_template"](
        mlx["processor"],
        mlx["model_config"],
        messages,
        num_images=0,
    )
    started = time.perf_counter()
    generation = _collect_stream(
        mlx["stream_generate"](
            mlx["model"],
            mlx["processor"],
            formatted_prompt,
            image=None,
            max_tokens=config.probe_max_tokens,
            temperature=config.probe_temperature,
            prompt_cache_state=prompt_cache_state,
        ),
        processor=mlx["processor"],
        top_k=config.generation_readout_top_k,
    )
    return {
        "label": label,
        "probe_id": config.probe_id,
        "prompt": config.probe_prompt,
        "assistant_text": generation["text"],
        "generation": generation["summary"],
        "probe_seed": config.probe_seed,
        "probe_seed_record": probe_seed_record,
        "wall_s": time.perf_counter() - started,
        "intervention": intervention,
        "source_cache_summary_before_probe": source_cache_summary,
        "cache_summary": summarize_prompt_cache_state(
            prompt_cache_state,
            max_layers=config.cache_summary_max_layers,
        ),
    }


def _context_record(context: dict[str, Any], *, config: ValuesSwapRunConfig) -> dict[str, Any]:
    return {
        "label": context["label"],
        "manifest_path": context["manifest_path"],
        "history_turns": len(context["history"]),
        "events": context["events"],
        "cache_summary": summarize_prompt_cache_state(
            context["prompt_cache_state"],
            max_layers=config.cache_summary_max_layers,
        ),
    }


def _load_valid_manifest(path: Path) -> dict[str, Any]:
    issues = validate_manifest(path)
    if issues:
        raise ValueError(f"manifest validation failed for {path}: " + "; ".join(issues))
    return _load_manifest(path)


def _probe_frame_count(phase: ProbePhase, frame_count: int) -> int:
    if frame_count < 1:
        raise ValueError("at least one selected frame is required")
    if phase == "after":
        return frame_count
    if phase == "mid":
        mid_position = _mid_probe_after_position(frame_count)
        if mid_position is None:
            raise ValueError("mid probe phase requires at least one selected frame")
        return mid_position + 1
    raise ValueError(f"unsupported probe phase: {phase}")


def _artifact_output_path(output_path: Path, label: str) -> Path:
    return output_path.with_name(f"{output_path.stem}.{label}{output_path.suffix}")


def _should_capture_context_cache_summary(position: int, frame_count: int, every: int) -> bool:
    if frame_count < 1 or every <= 0:
        return False
    return position == 0 or position == frame_count - 1 or position % every == 0
