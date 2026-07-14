from __future__ import annotations

import importlib
import json
import time
from copy import copy, deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Literal

from .delivery import (
    FrameDelivery,
    StimulusDeliveryMode,
    delivery_prompt_prefix,
    frame_artifact_list,
    prepare_frame_deliveries,
    stimulus_delivery_record,
)
from .full_vocab_readout import (
    full_vocab_sidecar_path,
    write_full_vocab_logprob_sidecar,
)
from .mlx_processor_compat import (
    ensure_mlx_chat_template_compat,
    ensure_mlx_processor_compat,
    load_mlx_vlm_with_compat,
)
from .prompts import SYNC_PROMPT, SYSTEM_PROMPT, probe_metadata, resolve_probe_preset
from .providers import get_capabilities
from .seeding import set_global_seed
from .stimulus import validate_manifest, write_json
from .timecode import format_timecode
from .trace import select_layer_indices, select_sequence_positions

ProbeCachePolicy = Literal["isolated", "shared_append", "no_cache"]


@dataclass(frozen=True)
class StreamRunConfig:
    manifest_path: Path
    output_path: Path
    model_id: str
    max_frames: int | None = None
    frame_stride: int = 1
    max_tokens: int = 2
    probe_max_tokens: int = 64
    temperature: float = 0.0
    probe_temperature: float | None = None
    dry_run: bool = False
    use_prompt_cache_state: bool = True
    probe_cache_policy: ProbeCachePolicy = "isolated"
    probe_preset: str = "default"
    cache_summary_every: int = 10
    cache_summary_max_layers: int | None = 4
    generation_readout_top_k: int = 10
    save_full_vocab_first_step: bool = False
    adapter_id: str = "mlx_vlm"
    include_frame_artifacts: bool = True
    seed: int | None = None
    probe_seed: int | None = None
    delivery_mode: StimulusDeliveryMode = "visual_stream"
    blank_rgb: tuple[int, int, int] = (0, 0, 0)


def run_stream_probe(
    config: StreamRunConfig,
    *,
    mlx_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_record = set_global_seed(config.seed, include_mlx=True)
    probe_phase_seeds = _probe_phase_seeds(config.probe_seed)
    probes = resolve_probe_preset(config.probe_preset)
    manifest = _load_manifest(config.manifest_path)
    issues = validate_manifest(config.manifest_path)
    if issues:
        raise ValueError("manifest validation failed: " + "; ".join(issues))

    source_frames = _select_frames(
        manifest["frames"],
        frame_stride=config.frame_stride,
        max_frames=config.max_frames,
    )
    frames = [] if config.delivery_mode == "probe_only" else source_frames
    frame_deliveries = prepare_frame_deliveries(
        output_path=config.output_path,
        manifest_base=config.manifest_path.parent,
        frames=frames,
        mode=config.delivery_mode,
        include_frame_artifacts=config.include_frame_artifacts,
        blank_rgb=config.blank_rgb,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "fractal_vlm_stream_probe",
        "model_id": config.model_id,
        "adapter_capabilities": get_capabilities(config.adapter_id).to_dict(),
        "manifest_path": str(config.manifest_path),
        "dry_run": config.dry_run,
        "reproducibility": {
            **seed_record,
            "probe_seed": config.probe_seed,
            "probe_phase_seeds": probe_phase_seeds,
        },
        "context_policy": {
            "frame_delivery": config.delivery_mode,
            "history": "explicit chat transcript stack",
            "stream_temperature": config.temperature,
            "probe_temperature": _probe_temperature(config),
            "probe_seed_policy": _probe_seed_policy(config.probe_seed),
            "probe_preset": config.probe_preset,
            "probe_count": len(probes),
            "prompt_cache_state_requested": config.use_prompt_cache_state,
            "probe_cache_policy": config.probe_cache_policy,
            "probe_history_policy": _probe_history_policy(config.probe_cache_policy),
            "mid_probe_policy": "after half of the selected frame stream has been consumed",
            "cache_summary_every": config.cache_summary_every,
            "cache_summary_max_layers": config.cache_summary_max_layers,
            "generation_readout_top_k": config.generation_readout_top_k,
            "save_full_vocab_first_step": config.save_full_vocab_first_step,
            "include_frame_artifacts": config.include_frame_artifacts,
        },
        "non_claims": [
            "No consciousness or subjective-state claim is made.",
            "KV-cache summaries are treated as computational traces, not experiences.",
            "Single-run observations are pilot evidence only.",
        ],
        "stimulus": {
            "condition": manifest.get("stimulus_condition"),
            "config_sha256": manifest.get("stimulus_config_sha256"),
            "frame_count_available": len(manifest.get("frames", [])),
            "frame_count_selected": len(frames),
            "source_frame_count_selected": len(source_frames),
        },
        "stimulus_delivery": stimulus_delivery_record(
            mode=config.delivery_mode,
            include_frame_artifacts=config.include_frame_artifacts,
            blank_rgb=config.blank_rgb,
        ),
        "probes": {},
        "stream_events": [],
    }
    result["frame_artifacts"] = frame_artifact_list(frame_deliveries)
    result["probe_schedule"] = {
        "mid_probe_after_position": _mid_probe_after_position(len(frames)),
        "position_is_zero_based": True,
    }

    if config.dry_run:
        result["probes"]["before"] = _dry_probe_records("before", probes=probes)
        for frame in frames:
            delivery = frame_deliveries[int(frame["index"])]
            event = _planned_frame_event(
                frame,
                delivery=delivery,
                output_base=config.output_path.parent,
            )
            result["stream_events"].append(event)
        result["probes"]["mid"] = _dry_probe_records("mid", probes=probes)
        result["probes"]["after"] = _dry_probe_records("after", probes=probes)
        write_json(config.output_path, result)
        return result

    mlx = mlx_runtime or _load_mlx_runtime(config.model_id)
    model = mlx["model"]
    processor = mlx["processor"]
    model_config = mlx["model_config"]
    stream_generate = mlx["stream_generate"]
    apply_chat_template = mlx["apply_chat_template"]
    prompt_cache_state = (
        _new_prompt_cache_state(mlx.get("prompt_cache_state"))
        if config.use_prompt_cache_state
        else None
    )

    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    result["runtime"] = mlx["runtime"]

    result["probes"]["before"] = _run_probe_batch(
        probes=probes,
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
        phase="before",
        model=model,
        processor=processor,
        model_config=model_config,
        stream_generate=stream_generate,
        apply_chat_template=apply_chat_template,
        max_tokens=config.probe_max_tokens,
        temperature=_probe_temperature(config),
        prompt_cache_state_source=None,
        probe_cache_policy="no_cache",
        cache_summary_max_layers=config.cache_summary_max_layers,
        generation_readout_top_k=config.generation_readout_top_k,
        probe_seed=probe_phase_seeds["before"],
        full_vocab_output_path=config.output_path
        if config.save_full_vocab_first_step
        else None,
    )

    mid_index = _mid_probe_after_position(len(frames))
    for position, frame in enumerate(frames):
        event = _run_frame_turn(
            frame=frame,
            history=history,
            delivery=frame_deliveries[int(frame["index"])],
            output_base=config.output_path.parent,
            model=model,
            processor=processor,
            model_config=model_config,
            stream_generate=stream_generate,
            apply_chat_template=apply_chat_template,
            prompt_cache_state=prompt_cache_state,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            capture_cache_summary=_should_summarize_cache(
                position,
                len(frames),
                config.cache_summary_every,
            ),
            cache_summary_max_layers=config.cache_summary_max_layers,
            generation_readout_top_k=config.generation_readout_top_k,
        )
        result["stream_events"].append(event)

        if mid_index is not None and position == mid_index:
            result["probes"]["mid"] = _run_probe_batch(
                probes=probes,
                history=history,
                phase="mid",
                model=model,
                processor=processor,
                model_config=model_config,
                stream_generate=stream_generate,
                apply_chat_template=apply_chat_template,
                max_tokens=config.probe_max_tokens,
                temperature=_probe_temperature(config),
                prompt_cache_state_source=prompt_cache_state,
                probe_cache_policy=config.probe_cache_policy,
                cache_summary_max_layers=config.cache_summary_max_layers,
                generation_readout_top_k=config.generation_readout_top_k,
                probe_seed=probe_phase_seeds["mid"],
                full_vocab_output_path=config.output_path
                if config.save_full_vocab_first_step
                else None,
            )

    result["probes"]["after"] = _run_probe_batch(
        probes=probes,
        history=history,
        phase="after",
        model=model,
        processor=processor,
        model_config=model_config,
        stream_generate=stream_generate,
        apply_chat_template=apply_chat_template,
        max_tokens=config.probe_max_tokens,
        temperature=_probe_temperature(config),
        prompt_cache_state_source=prompt_cache_state,
        probe_cache_policy=config.probe_cache_policy,
        cache_summary_max_layers=config.cache_summary_max_layers,
        generation_readout_top_k=config.generation_readout_top_k,
        probe_seed=probe_phase_seeds["after"],
        full_vocab_output_path=config.output_path
        if config.save_full_vocab_first_step
        else None,
    )

    write_json(config.output_path, result)
    return result


def _probe_temperature(config: StreamRunConfig) -> float:
    return (
        config.temperature
        if config.probe_temperature is None
        else config.probe_temperature
    )


def _probe_phase_seeds(base_seed: int | None) -> dict[str, int | None]:
    if base_seed is None:
        return {"before": None, "mid": None, "after": None}
    return {"before": base_seed, "mid": base_seed + 1, "after": base_seed + 2}


def _probe_seed_policy(base_seed: int | None) -> str:
    if base_seed is None:
        return "probe sampling uses the active global RNG state"
    return (
        "probe RNG is reset per phase from probe_seed, probe_seed+1, and probe_seed+2"
    )


def _run_frame_turn(
    *,
    frame: dict[str, Any],
    history: list[dict[str, str]],
    delivery: FrameDelivery,
    output_base: Path,
    model: Any,
    processor: Any,
    model_config: Any,
    stream_generate: Any,
    apply_chat_template: Any,
    prompt_cache_state: Any,
    max_tokens: int,
    temperature: float,
    capture_cache_summary: bool,
    cache_summary_max_layers: int | None,
    generation_readout_top_k: int,
) -> dict[str, Any]:
    timecode = format_timecode(float(frame["t_seconds"]))
    prompt = f"{delivery_prompt_prefix(frame, delivery.mode, timecode)}\n{SYNC_PROMPT}"
    messages = history + [{"role": "user", "content": prompt}]
    formatted_prompt = apply_chat_template(
        processor,
        model_config,
        messages,
        num_images=delivery.num_images,
    )
    cache_prefix_audit = audit_prompt_cache_prefix(
        prompt_cache_state,
        formatted_prompt=formatted_prompt,
        processor=processor,
        model_config=model_config,
    )
    image_arg = str(delivery.image_path) if delivery.image_path is not None else None
    started = time.perf_counter()
    generation = _collect_stream(
        stream_generate(
            model,
            processor,
            formatted_prompt,
            image=image_arg,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_cache_state=prompt_cache_state,
        ),
        processor=processor,
        top_k=generation_readout_top_k,
    )
    wall_s = time.perf_counter() - started

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": generation["text"]})

    return {
        "frame_index": frame["index"],
        "t_seconds": frame["t_seconds"],
        "timecode": timecode,
        "frame_path": frame["path"],
        "frame_artifact": delivery.frame_artifact,
        "frame_sha256": frame["sha256"],
        "delivery": delivery.to_event_record(output_base),
        "prompt": prompt,
        "assistant_text": generation["text"],
        "generation": generation["summary"],
        "cache_prefix_audit": cache_prefix_audit,
        "wall_s": wall_s,
        "cache_summary": summarize_prompt_cache_state(
            prompt_cache_state,
            max_layers=cache_summary_max_layers,
        )
        if capture_cache_summary
        else None,
        "cache_summary_captured": capture_cache_summary,
    }


def _run_probe_batch(
    *,
    probes: list[dict[str, Any]],
    history: list[dict[str, str]],
    phase: str,
    model: Any,
    processor: Any,
    model_config: Any,
    stream_generate: Any,
    apply_chat_template: Any,
    max_tokens: int,
    temperature: float,
    prompt_cache_state_source: Any,
    probe_cache_policy: ProbeCachePolicy,
    cache_summary_max_layers: int | None,
    generation_readout_top_k: int,
    probe_seed: int | None,
    cache_summary_sequence_positions: list[int] | None = None,
    full_vocab_output_path: Path | None = None,
) -> list[dict[str, Any]]:
    records = []
    probe_seed_record = set_global_seed(probe_seed, include_mlx=True)
    for probe in probes:
        source_cache_summary = summarize_prompt_cache_state(
            prompt_cache_state_source,
            max_layers=cache_summary_max_layers,
            sequence_positions=cache_summary_sequence_positions,
        )
        prompt_cache_state, cache_branch_status = _probe_prompt_cache_state(
            prompt_cache_state_source,
            probe_cache_policy,
        )
        prompt = probe["prompt"]
        messages = history + [{"role": "user", "content": prompt}]
        formatted_prompt = apply_chat_template(
            processor,
            model_config,
            messages,
            num_images=0,
        )
        cache_prefix_audit = audit_prompt_cache_prefix(
            prompt_cache_state_source,
            formatted_prompt=formatted_prompt,
            processor=processor,
            model_config=model_config,
        )
        started = time.perf_counter()
        full_vocab_writer = None
        if full_vocab_output_path is not None:
            sidecar_path = full_vocab_sidecar_path(
                full_vocab_output_path,
                phase=phase,
                probe_id=probe["id"],
            )

            def full_vocab_writer(
                logprobs: Any, *, path: Path = sidecar_path
            ) -> dict[str, Any]:
                return write_full_vocab_logprob_sidecar(
                    logprobs,
                    path=path,
                    relative_to=full_vocab_output_path.parent,
                )

        generation = _collect_stream(
            stream_generate(
                model,
                processor,
                formatted_prompt,
                image=None,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_cache_state=prompt_cache_state,
            ),
            processor=processor,
            top_k=generation_readout_top_k,
            first_step_full_vocab_writer=full_vocab_writer,
        )
        records.append(
            {
                "phase": phase,
                "probe_id": probe["id"],
                "prompt": prompt,
                **probe_metadata(probe),
                "assistant_text": generation["text"],
                "generation": generation["summary"],
                "probe_seed": probe_seed,
                "probe_seed_record": probe_seed_record,
                "wall_s": time.perf_counter() - started,
                "cache_branch_status": cache_branch_status,
                "cache_prefix_audit": cache_prefix_audit,
                "source_cache_summary_before_probe": source_cache_summary,
                "cache_summary": summarize_prompt_cache_state(
                    prompt_cache_state,
                    max_layers=cache_summary_max_layers,
                    sequence_positions=cache_summary_sequence_positions,
                ),
            }
        )
        if probe_cache_policy == "shared_append":
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": generation["text"]})
    return records


def summarize_prompt_cache_state(
    prompt_cache_state: Any,
    *,
    max_layers: int | None = 4,
    sequence_positions: list[int] | None = None,
) -> dict[str, Any] | None:
    if prompt_cache_state is None or getattr(prompt_cache_state, "cache", None) is None:
        return None
    if max_layers is not None and max_layers < 0:
        raise ValueError("max_layers must be non-negative or None")

    try:
        import mlx.core as mx
    except Exception:
        return {"available": False, "reason": "mlx import failed"}

    token_count = len(getattr(prompt_cache_state, "token_ids", []) or [])
    requested_positions = set(sequence_positions or [])
    if token_count > 0:
        requested_positions.update(
            {token_count // 2, max(0, token_count - 2), token_count - 1}
        )
    layers = []
    cache_entries = list(prompt_cache_state.cache)
    for layer_index in select_layer_indices(len(cache_entries), max_layers):
        entry = cache_entries[layer_index]
        layer_record: dict[str, Any] = {"layer_index": layer_index}
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is not None:
            layer_record["keys"] = _mlx_tensor_stats(
                mx,
                keys,
                sequence_positions=sorted(requested_positions),
            )
        if values is not None:
            layer_record["values"] = _mlx_tensor_stats(
                mx,
                values,
                sequence_positions=sorted(requested_positions),
            )
        if "keys" in layer_record or "values" in layer_record:
            layers.append(layer_record)
    return {
        "available": True,
        "token_count": token_count,
        "total_layers": len(cache_entries),
        "reported_layers": len(layers),
        "truncated": max_layers is not None and len(cache_entries) > max_layers,
        "sequence_position_sampling": {
            "policy": "cache-shape anchors plus token-count anchors and caller-requested positions",
            "requested_positions": sorted(set(sequence_positions or [])),
            "token_count_anchor_positions": sorted(
                requested_positions - set(sequence_positions or [])
            ),
        },
        "layers": layers,
    }


def summarize_prompt_cache_token_layout(
    prompt_cache_state: Any,
    model_config: Any,
) -> dict[str, Any]:
    token_ids = list(getattr(prompt_cache_state, "token_ids", []) or [])
    marker_names = (
        "vision_start_token_id",
        "vision_end_token_id",
        "vision_token_id",
        "image_token_id",
        "image_token_index",
    )
    marker_token_ids = {
        name: int(value)
        for name in marker_names
        if (value := _model_config_value(model_config, name)) is not None
    }
    marker_positions = {
        name: [
            index for index, token_id in enumerate(token_ids) if token_id == marker_id
        ]
        for name, marker_id in marker_token_ids.items()
    }
    image_token_id = marker_token_ids.get("image_token_id")
    if image_token_id is None:
        image_token_id = marker_token_ids.get("image_token_index")
    image_positions = (
        [
            index
            for index, token_id in enumerate(token_ids)
            if token_id == image_token_id
        ]
        if image_token_id is not None
        else []
    )
    image_runs = _contiguous_position_runs(image_positions)
    focus_roles: dict[int, set[str]] = {}

    def add_focus(position: int, role: str) -> None:
        if 0 <= position < len(token_ids):
            focus_roles.setdefault(position, set()).add(role)

    dense_image_markers = {"image_token_id", "image_token_index"}
    for name, positions in marker_positions.items():
        if name in dense_image_markers:
            continue
        for position in positions:
            add_focus(position, name)
    for run_index, run in enumerate(image_runs):
        add_focus(run["start"], f"image_run_{run_index}_start")
        add_focus((run["start"] + run["end"]) // 2, f"image_run_{run_index}_mid")
        add_focus(run["end"], f"image_run_{run_index}_end")
    if token_ids:
        add_focus(0, "first_token")
        add_focus(len(token_ids) // 2, "token_count_mid")
        add_focus(len(token_ids) - 1, "last_token")
    cache_position_layout = _resolve_cache_position_layout(
        token_count=len(token_ids),
        image_token_runs=image_runs,
        cache_sequence_lengths=_cache_effective_sequence_lengths(
            getattr(prompt_cache_state, "cache", None)
        ),
        model_type=_model_config_value(model_config, "model_type"),
    )
    return {
        "token_count": len(token_ids),
        "coordinate_space": "processor_token_sequence",
        "marker_token_ids": marker_token_ids,
        "marker_positions": marker_positions,
        "image_token_count": len(image_positions),
        "image_token_runs": image_runs,
        "sequence_position_plan": [
            {"position": position, "roles": sorted(roles)}
            for position, roles in sorted(focus_roles.items())
        ],
        "cache_position_layout": cache_position_layout,
    }


def cache_layout_sequence_positions(token_layout: dict[str, Any]) -> list[int]:
    cache_layout = token_layout.get("cache_position_layout")
    if cache_layout is None:
        plan = token_layout.get("sequence_position_plan")
    elif cache_layout.get("strategy") == "identity":
        plan = token_layout.get("sequence_position_plan")
    elif cache_layout.get("available"):
        plan = cache_layout.get("sequence_position_plan")
    else:
        plan = []
    return [int(record["position"]) for record in plan or []]


def _resolve_cache_position_layout(
    *,
    token_count: int,
    image_token_runs: list[dict[str, int]],
    cache_sequence_lengths: list[int],
    model_type: Any,
) -> dict[str, Any]:
    unique_lengths = sorted(set(cache_sequence_lengths))
    base = {
        "available": False,
        "coordinate_space": "effective_cache_sequence",
        "processor_token_count": token_count,
        "cache_sequence_lengths": unique_lengths,
        "processor_image_position_count": sum(
            int(run["length"]) for run in image_token_runs
        ),
    }
    if len(unique_lengths) != 1:
        return {
            **base,
            "reason": (
                "cache_effective_length_unavailable"
                if not unique_lengths
                else "cache_effective_lengths_disagree"
            ),
        }

    cache_length = unique_lengths[0]
    if cache_length == token_count:
        return _available_cache_position_layout(
            base,
            cache_length=cache_length,
            image_position_runs=image_token_runs,
            strategy="identity",
        )
    if model_type != "granite_vision":
        return {
            **base,
            "cache_sequence_length": cache_length,
            "reason": "processor_token_and_cache_lengths_differ",
        }
    if len(image_token_runs) != 1:
        return {
            **base,
            "cache_sequence_length": cache_length,
            "reason": "granite_vision_mapping_requires_one_image_run",
        }

    source_run = image_token_runs[0]
    source_start = int(source_run["start"])
    source_end = int(source_run["end"])
    processor_image_count = int(source_run["length"])
    processor_post_count = token_count - source_end - 1
    cache_image_count = cache_length - (token_count - processor_image_count)
    cache_end = cache_length - processor_post_count - 1
    if (
        cache_image_count <= 0
        or cache_end < source_start
        or cache_end - source_start + 1 != cache_image_count
    ):
        return {
            **base,
            "cache_sequence_length": cache_length,
            "reason": "granite_vision_image_expansion_is_inconsistent",
        }

    return _available_cache_position_layout(
        base,
        cache_length=cache_length,
        image_position_runs=[
            {
                "start": source_start,
                "end": cache_end,
                "length": cache_image_count,
            }
        ],
        strategy="granite_vision_single_image_run_replacement",
    )


def _available_cache_position_layout(
    base: dict[str, Any],
    *,
    cache_length: int,
    image_position_runs: list[dict[str, int]],
    strategy: str,
) -> dict[str, Any]:
    image_count = sum(int(run["length"]) for run in image_position_runs)
    first_image = int(image_position_runs[0]["start"]) if image_position_runs else None
    last_image = int(image_position_runs[-1]["end"]) if image_position_runs else None
    return {
        **base,
        "available": True,
        "reason": None,
        "strategy": strategy,
        "cache_sequence_length": cache_length,
        "image_partition_available": bool(image_position_runs),
        "image_position_count": image_count,
        "image_position_runs": image_position_runs,
        "non_image_position_count": cache_length - image_count,
        "pre_image_position_count": first_image or 0,
        "post_image_position_count": (
            cache_length - last_image - 1 if last_image is not None else 0
        ),
        "expansion_delta": cache_length - int(base["processor_token_count"]),
        "sequence_position_plan": _cache_sequence_position_plan(
            cache_length,
            image_position_runs=image_position_runs,
        ),
    }


def _cache_sequence_position_plan(
    sequence_length: int,
    *,
    image_position_runs: list[dict[str, int]],
) -> list[dict[str, Any]]:
    roles: dict[int, set[str]] = {}

    def add(position: int, role: str) -> None:
        if 0 <= position < sequence_length:
            roles.setdefault(position, set()).add(role)

    for run_index, run in enumerate(image_position_runs):
        start = int(run["start"])
        end = int(run["end"])
        add(start, f"cache_image_run_{run_index}_start")
        add((start + end) // 2, f"cache_image_run_{run_index}_mid")
        add(end, f"cache_image_run_{run_index}_end")
    if sequence_length:
        add(0, "first_cache_position")
        add(sequence_length // 2, "cache_sequence_mid")
        add(sequence_length - 1, "last_cache_position")
    return [
        {"position": position, "roles": sorted(position_roles)}
        for position, position_roles in sorted(roles.items())
    ]


def _cache_effective_sequence_lengths(cache: Any) -> list[int]:
    if cache is None:
        return []
    if isinstance(cache, (list, tuple)):
        return [
            length
            for entry in cache
            for length in _cache_effective_sequence_lengths(entry)
        ]
    nested = getattr(cache, "caches", None)
    if nested is not None:
        return _cache_effective_sequence_lengths(nested)
    offset = getattr(cache, "offset", None)
    if offset is not None:
        return [int(offset)]
    keys = getattr(cache, "keys", None)
    if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
        return [int(keys.shape[-2])]
    return []


def audit_prompt_cache_prefix(
    prompt_cache_state: Any,
    *,
    formatted_prompt: Any,
    processor: Any,
    model_config: Any,
) -> dict[str, Any]:
    if prompt_cache_state is None or getattr(prompt_cache_state, "cache", None) is None:
        return {
            "available": False,
            "reason": "source prompt cache is unavailable",
        }
    source_ids = list(getattr(prompt_cache_state, "token_ids", []) or [])
    try:
        formatted_ids = _tokenize_formatted_prompt(
            formatted_prompt,
            processor=processor,
            model_config=model_config,
        )
    except Exception as exc:
        return {
            "available": False,
            "reason": f"formatted prompt tokenization failed: {type(exc).__name__}: {exc}",
            "source_token_count": len(source_ids),
        }

    common_prefix = _common_prefix_length(source_ids, formatted_ids)
    cache_lengths = sorted(set(_cache_sequence_lengths(prompt_cache_state.cache)))
    token_cache_aligned = bool(cache_lengths) and all(
        length == len(source_ids) for length in cache_lengths
    )
    full_prefix = common_prefix == len(source_ids)
    return {
        "available": True,
        "source_token_count": len(source_ids),
        "formatted_prompt_token_count": len(formatted_ids),
        "common_prefix_token_count": common_prefix,
        "common_prefix_fraction": common_prefix / len(source_ids)
        if source_ids
        else 0.0,
        "full_source_prefix_match": full_prefix,
        "cache_sequence_lengths": cache_lengths,
        "token_cache_length_aligned": token_cache_aligned,
        "reuse_safe_under_token_prefix_contract": full_prefix and token_cache_aligned,
        "caveat": (
            "MLX-VLM 0.4.4 trims cache tensors by token-prefix length; multimodal "
            "embedding expansion can make token and cache lengths diverge."
        ),
    }


def _tokenize_formatted_prompt(
    formatted_prompt: Any,
    *,
    processor: Any,
    model_config: Any,
) -> list[int]:
    if not isinstance(formatted_prompt, str):
        raise TypeError(
            f"expected formatted prompt string, got {type(formatted_prompt).__name__}"
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    model_type = _model_config_value(model_config, "model_type")
    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model_type in {"gemma3", "gemma3n", "gemma4"}
        else True
    )
    encoded = tokenizer(
        formatted_prompt,
        add_special_tokens=add_special_tokens,
        padding=True,
        padding_side="left",
        return_tensors="np",
    )
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise ValueError("tokenizer did not return input_ids")
    return [int(value) for value in input_ids.reshape(-1).tolist()]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def _cache_sequence_lengths(cache: Any) -> list[int]:
    lengths = []
    if cache is None:
        return lengths
    if isinstance(cache, (list, tuple)):
        for entry in cache:
            lengths.extend(_cache_sequence_lengths(entry))
        return lengths
    nested = getattr(cache, "caches", None)
    if nested is not None:
        lengths.extend(_cache_sequence_lengths(nested))
    keys = getattr(cache, "keys", None)
    if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
        lengths.append(int(keys.shape[-2]))
    return lengths


def _probe_prompt_cache_state(
    prompt_cache_state_source: Any,
    probe_cache_policy: ProbeCachePolicy,
) -> tuple[Any, str]:
    if probe_cache_policy == "no_cache":
        return None, "disabled"
    if prompt_cache_state_source is None:
        return None, "unavailable"
    if probe_cache_policy == "shared_append":
        return prompt_cache_state_source, "shared_append"
    if probe_cache_policy == "isolated":
        cloned = clone_prompt_cache_state(prompt_cache_state_source)
        if cloned is None:
            return None, "clone_failed_no_cache_fallback"
        return cloned, "isolated_clone"
    raise ValueError(f"unsupported probe_cache_policy: {probe_cache_policy}")


def clone_prompt_cache_state(prompt_cache_state: Any) -> Any | None:
    """Branch a PromptCacheState so probes cannot mutate the stream cache."""
    if prompt_cache_state is None or getattr(prompt_cache_state, "cache", None) is None:
        return None

    try:
        clone = prompt_cache_state.__class__()
        token_ids = getattr(prompt_cache_state, "token_ids", None)
        clone.token_ids = list(token_ids) if token_ids is not None else None
        clone.cache = [_clone_cache_entry(entry) for entry in prompt_cache_state.cache]
        return clone
    except Exception:
        return None


def _clone_cache_entry(entry: Any) -> Any:
    try:
        shallow = copy(entry)
    except Exception:
        return deepcopy(entry)
    if hasattr(entry, "caches"):
        shallow.caches = [_clone_cache_entry(child) for child in entry.caches]
    return shallow


def _mlx_tensor_stats(
    mx: Any,
    tensor: Any,
    *,
    sequence_positions: list[int] | None = None,
) -> dict[str, Any]:
    # Cache tensors are commonly float16; promote before reductions so
    # variance and norm do not overflow on otherwise finite activations.
    stats_tensor = tensor.astype(mx.float32)
    mean = mx.mean(stats_tensor)
    var = mx.var(stats_tensor)
    std = mx.sqrt(var)
    abs_mean = mx.mean(mx.abs(stats_tensor))
    min_val = mx.min(stats_tensor)
    max_val = mx.max(stats_tensor)
    l2 = mx.linalg.norm(stats_tensor)
    mx.eval(mean, var, std, abs_mean, min_val, max_val, l2)
    stats = {
        "shape": list(tensor.shape),
        "mean": float(mean.item()),
        "variance": float(var.item()),
        "std": float(std.item()),
        "abs_mean": float(abs_mean.item()),
        "min": float(min_val.item()),
        "max": float(max_val.item()),
        "l2_norm": float(l2.item()),
    }
    slices = _mlx_sequence_position_stats(
        mx,
        tensor,
        sequence_positions=sequence_positions,
    )
    if slices:
        stats["sequence_position_stats"] = slices
    return stats


def _mlx_sequence_position_stats(
    mx: Any,
    tensor: Any,
    *,
    sequence_positions: list[int] | None = None,
) -> list[dict[str, Any]]:
    if not hasattr(tensor, "shape") or len(tensor.shape) < 3:
        return []
    sequence_length = int(tensor.shape[-2])
    selected_positions = set(select_sequence_positions(sequence_length))
    selected_positions.update(
        position
        for position in sequence_positions or []
        if 0 <= position < sequence_length
    )
    records = []
    for position in sorted(selected_positions):
        sliced = tensor[..., position, :].astype(mx.float32)
        mean = mx.mean(sliced)
        var = mx.var(sliced)
        abs_mean = mx.mean(mx.abs(sliced))
        l2 = mx.linalg.norm(sliced)
        mx.eval(mean, var, abs_mean, l2)
        records.append(
            {
                "position": position,
                "mean": float(mean.item()),
                "variance": float(var.item()),
                "abs_mean": float(abs_mean.item()),
                "l2_norm": float(l2.item()),
            }
        )
    return records


def _model_config_value(model_config: Any, name: str) -> Any:
    if isinstance(model_config, dict):
        return model_config.get(name)
    return getattr(model_config, name, None)


def _contiguous_position_runs(positions: list[int]) -> list[dict[str, int]]:
    if not positions:
        return []
    runs = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append(
                {"start": start, "end": previous, "length": previous - start + 1}
            )
            start = position
        previous = position
    runs.append({"start": start, "end": previous, "length": previous - start + 1})
    return runs


def _collect_stream(
    chunks: Any,
    *,
    processor: Any | None = None,
    top_k: int = 10,
    first_step_full_vocab_writer: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text_parts = []
    last = None
    step_records = []
    for step_index, chunk in enumerate(chunks):
        last = chunk
        text_parts.append(getattr(chunk, "text", str(chunk)))
        step_record = _summarize_mlx_generation_step(
            chunk,
            processor=processor,
            step_index=step_index,
            top_k=top_k,
        )
        if step_record is not None:
            if (
                step_index == 0
                and first_step_full_vocab_writer is not None
                and getattr(chunk, "logprobs", None) is not None
            ):
                step_record["full_vocab_sidecar"] = first_step_full_vocab_writer(
                    chunk.logprobs
                )
            step_records.append(step_record)

    summary = {}
    if last is not None:
        for key in (
            "prompt_tokens",
            "generation_tokens",
            "total_tokens",
            "prompt_tps",
            "generation_tps",
            "peak_memory",
        ):
            value = getattr(last, key, None)
            if value is not None:
                summary[key] = value
    if step_records:
        summary["score_steps"] = len(step_records)
        summary["steps"] = step_records
    return {"text": "".join(text_parts).strip(), "summary": summary}


def _summarize_mlx_generation_step(
    chunk: Any,
    *,
    processor: Any | None,
    step_index: int,
    top_k: int,
) -> dict[str, Any] | None:
    logprobs = getattr(chunk, "logprobs", None)
    token = getattr(chunk, "token", None)
    if logprobs is None and token is None:
        return None

    record: dict[str, Any] = {"step_index": step_index}
    token_id = _coerce_token_id(token)
    if token_id is not None:
        record["token_id"] = token_id
        record["token"] = _decode_token(processor, token_id)
    if logprobs is None:
        return record
    record["top_logprobs"] = summarize_mlx_logprobs(
        logprobs,
        processor=processor,
        top_k=top_k,
    )
    if token_id is not None:
        token_logprob = _mlx_logprob_at(logprobs, token_id)
        if token_logprob is not None:
            record["token_logprob"] = token_logprob
    return record


def summarize_mlx_logprobs(
    logprobs: Any,
    *,
    processor: Any | None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    try:
        import mlx.core as mx
    except Exception:
        return []
    if not hasattr(logprobs, "shape") or len(logprobs.shape) < 1:
        return []
    vocab_size = int(logprobs.shape[-1])
    if vocab_size < 1:
        return []
    k = min(top_k, vocab_size)
    indices = mx.argsort(logprobs)[-k:][::-1]
    values = logprobs[indices]
    mx.eval(indices, values)
    return [
        {
            "token_id": int(token_id),
            "token": _decode_token(processor, int(token_id)),
            "logprob": float(logprob),
        }
        for token_id, logprob in zip(indices.tolist(), values.tolist())
    ]


def _mlx_logprob_at(logprobs: Any, token_id: int) -> float | None:
    try:
        import mlx.core as mx
    except Exception:
        return None
    if (
        token_id < 0
        or not hasattr(logprobs, "shape")
        or token_id >= int(logprobs.shape[-1])
    ):
        return None
    value = logprobs[token_id]
    mx.eval(value)
    return float(value.item())


def _coerce_token_id(token: Any) -> int | None:
    if token is None:
        return None
    try:
        return int(token.item()) if hasattr(token, "item") else int(token)
    except Exception:
        return None


def _decode_token(processor: Any | None, token_id: int) -> str:
    if processor is None:
        return ""
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        try:
            return tokenizer.decode([token_id])
        except Exception:
            return ""
    except Exception:
        return ""


def _load_mlx_runtime(model_id: str) -> dict[str, Any]:
    try:
        import mlx.core as mx
        from mlx_vlm import load, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except Exception as exc:
        raise RuntimeError(
            "MLX runtime is unavailable. Install optional dependencies or use --dry-run."
        ) from exc

    generate_module = importlib.import_module("mlx_vlm.generate")
    prompt_cache_cls = getattr(generate_module, "PromptCacheState", None)
    prompt_cache_state = prompt_cache_cls() if prompt_cache_cls is not None else None

    # mlx_vlm.load forwards **kwargs to model, image processor, and processor
    # loaders. Passing mlx dtype objects here breaks recent Transformers
    # processor deepcopy paths, so keep loading kwargs processor-safe.
    model, processor, model_load_compatibility = load_mlx_vlm_with_compat(
        model_id,
        default_load=load,
    )
    model_config = getattr(model, "config", None)
    if model_config is None:
        try:
            from mlx_vlm.utils import load_config

            model_config = load_config(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"could not resolve model config for {model_id}"
            ) from exc
    processor, processor_compatibility = ensure_mlx_processor_compat(
        processor,
        model_config,
    )
    apply_chat_template, chat_template_compatibility = ensure_mlx_chat_template_compat(
        apply_chat_template, model_config
    )

    return {
        "model": model,
        "processor": processor,
        "model_config": model_config,
        "stream_generate": stream_generate,
        "apply_chat_template": apply_chat_template,
        "prompt_cache_state": prompt_cache_state,
        "runtime": {
            "mlx_available": True,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "mlx_vlm_version": _package_version("mlx-vlm"),
            "mlx_vlm_load_kwargs": {},
            "model_load_compatibility": model_load_compatibility,
            "prompt_cache_state_available": prompt_cache_cls is not None,
            "processor_compatibility": processor_compatibility,
            "chat_template_compatibility": chat_template_compatibility,
        },
    }


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _new_prompt_cache_state(prototype: Any) -> Any | None:
    if prototype is None:
        return None
    try:
        return prototype.__class__()
    except Exception as exc:
        raise RuntimeError("could not create a fresh MLX PromptCacheState") from exc


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_frames(
    frames: list[dict[str, Any]],
    *,
    frame_stride: int,
    max_frames: int | None,
) -> list[dict[str, Any]]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    selected = frames[::frame_stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def _mid_probe_after_position(frame_count: int) -> int | None:
    if frame_count < 1:
        return None
    return (frame_count - 1) // 2


def _should_summarize_cache(position: int, frame_count: int, every: int) -> bool:
    if every <= 0 or frame_count < 1:
        return False
    key_positions = {0, frame_count - 1}
    mid_position = _mid_probe_after_position(frame_count)
    if mid_position is not None:
        key_positions.add(mid_position)
    return position in key_positions or position % every == 0


def _probe_history_policy(probe_cache_policy: ProbeCachePolicy) -> str:
    if probe_cache_policy == "shared_append":
        return "probe turns are appended to the active stream transcript"
    return "probe turns are run as branch reads and do not mutate stream history"


def _planned_frame_event(
    frame: dict[str, Any],
    *,
    delivery: FrameDelivery | None = None,
    output_base: Path | None = None,
) -> dict[str, Any]:
    timecode = format_timecode(float(frame["t_seconds"]))
    mode = delivery.mode if delivery is not None else "visual_stream"
    return {
        "frame_index": frame["index"],
        "t_seconds": frame["t_seconds"],
        "timecode": timecode,
        "frame_path": frame["path"],
        "frame_artifact": delivery.frame_artifact if delivery is not None else None,
        "frame_sha256": frame["sha256"],
        "delivery": delivery.to_event_record(output_base or Path("."))
        if delivery is not None
        else None,
        "planned_prompt": f"{delivery_prompt_prefix(frame, mode, timecode)}\n{SYNC_PROMPT}",
    }


def _dry_probe_records(
    phase: str,
    *,
    probes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "probe_id": probe["id"],
            "prompt": probe["prompt"],
            **probe_metadata(probe),
            "assistant_text": "<dry-run>",
        }
        for probe in probes
    ]
