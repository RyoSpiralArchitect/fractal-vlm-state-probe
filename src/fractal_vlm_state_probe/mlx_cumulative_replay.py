from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .cache_tensor_artifact import (
    CacheTensorCaptureSpec,
    cache_tensor_capture_policy,
    capture_prompt_cache_tensors,
)
from .delivery import (
    frame_artifact_list,
    prepare_frame_deliveries,
    stimulus_delivery_record,
)
from .full_vocab_readout import (
    full_vocab_sidecar_path,
    write_full_vocab_logprob_sidecar,
)
from .mlx_stream import (
    _collect_stream,
    _dry_probe_records,
    _load_manifest,
    _load_mlx_runtime,
    _new_prompt_cache_state,
    _probe_phase_seeds,
    _probe_seed_policy,
    _run_probe_batch,
    _select_frames,
    cache_layout_sequence_positions,
    summarize_prompt_cache_state,
    summarize_prompt_cache_token_layout,
)
from .prompts import SYNC_PROMPT, SYSTEM_PROMPT, probe_metadata, resolve_probe_preset
from .providers import get_capabilities
from .seeding import set_global_seed
from .stimulus import validate_manifest, write_json
from .timecode import format_timecode


@dataclass(frozen=True)
class CumulativeReplayRunConfig:
    manifest_path: Path
    output_path: Path
    model_id: str
    max_frames: int | None = None
    frame_stride: int = 1
    max_tokens: int = 2
    probe_max_tokens: int = 64
    temperature: float = 0.0
    probe_temperature: float = 0.0
    probe_preset: str = "forced_choice"
    cache_summary_max_layers: int | None = 4
    generation_readout_top_k: int = 10
    save_full_vocab_first_step: bool = False
    dry_run: bool = False
    include_frame_artifacts: bool = True
    seed: int | None = 20260604
    probe_seed: int | None = 0
    adapter_id: str = "mlx_vlm"
    after_probe_protocol: Literal["direct_multimodal_replay", "legacy_cache_branch"] = (
        "direct_multimodal_replay"
    )
    capture_direct_probe_cache: bool = False
    source_cache_only: bool = False
    cache_tensor_captures: tuple[CacheTensorCaptureSpec, ...] = ()


def run_cumulative_replay_probe(
    config: CumulativeReplayRunConfig,
    *,
    mlx_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config.source_cache_only and config.save_full_vocab_first_step:
        raise ValueError(
            "source-cache-only mode cannot save probe first-step vocabulary scores"
        )
    if config.source_cache_only and config.capture_direct_probe_cache:
        raise ValueError("source-cache-only mode cannot capture a direct-probe cache")
    seed_record = set_global_seed(config.seed, include_mlx=True)
    probe_phase_seeds = _probe_phase_seeds(config.probe_seed)
    probes = (
        [] if config.source_cache_only else resolve_probe_preset(config.probe_preset)
    )
    manifest = _load_manifest(config.manifest_path)
    issues = validate_manifest(config.manifest_path)
    if issues:
        raise ValueError("manifest validation failed: " + "; ".join(issues))

    frames = _select_frames(
        manifest["frames"],
        frame_stride=config.frame_stride,
        max_frames=config.max_frames,
    )
    if not frames:
        raise ValueError("cumulative replay requires at least one selected frame")
    deliveries = prepare_frame_deliveries(
        output_path=config.output_path,
        manifest_base=config.manifest_path.parent,
        frames=frames,
        mode="visual_stream",
        include_frame_artifacts=config.include_frame_artifacts,
    )
    prompt = cumulative_replay_prompt(frames)
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "mlx_vlm_cumulative_visual_replay_probe",
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
            "visual_context_protocol": "single_turn_ordered_multi_image_replay",
            "history": "fresh prompt cache per replay condition",
            "stream_temperature": config.temperature,
            "probe_temperature": config.probe_temperature,
            "probe_seed_policy": _probe_seed_policy(config.probe_seed),
            "probe_preset": config.probe_preset,
            "probe_count": len(probes),
            "source_cache_only": config.source_cache_only,
            "probe_cache_policy": "isolated",
            "after_probe_protocol": config.after_probe_protocol,
            "capture_direct_probe_cache": config.capture_direct_probe_cache,
            "probe_history_policy": (
                "after probes use fresh direct multimodal forwards over the same ordered images"
                if config.after_probe_protocol == "direct_multimodal_replay"
                else "after probes use the legacy branch read from the replay cache"
            ),
            "cache_summary_max_layers": config.cache_summary_max_layers,
            "generation_readout_top_k": config.generation_readout_top_k,
            "save_full_vocab_first_step": config.save_full_vocab_first_step,
            "cache_tensor_captures": cache_tensor_capture_policy(
                config.cache_tensor_captures
            ),
            "include_frame_artifacts": config.include_frame_artifacts,
        },
        "non_claims": [
            "Cumulative replay is a distinct protocol from incremental multi-turn cache persistence.",
            "Ordered multi-image batching does not prove temporal processing equivalent to a video stream.",
            "Direct multimodal after probes measure immediate image-conditioned readout, not persistence after the ACK turn.",
            "KV-cache summaries are computational traces, not subjective states.",
        ],
        "stimulus": {
            "condition": manifest.get("stimulus_condition"),
            "config_sha256": manifest.get("stimulus_config_sha256"),
            "frame_count_available": len(manifest.get("frames", [])),
            "frame_count_selected": len(frames),
            "source_frame_count_selected": len(frames),
        },
        "stimulus_delivery": {
            **stimulus_delivery_record(
                mode="visual_stream",
                include_frame_artifacts=config.include_frame_artifacts,
                blank_rgb=(0, 0, 0),
            ),
            "batching": "all selected images delivered in one ordered model turn",
        },
        "frame_artifacts": frame_artifact_list(deliveries),
        "probe_schedule": {
            "before": (
                "omitted in source-cache-only mode"
                if config.source_cache_only
                else "clean text-only branch"
            ),
            "after": (
                "omitted in source-cache-only mode"
                if config.source_cache_only
                else (
                    "fresh direct multimodal read using the same ordered image batch"
                    if config.after_probe_protocol == "direct_multimodal_replay"
                    else "legacy branch read after one ordered multi-image replay turn"
                )
            ),
            "mid": None,
        },
        "probes": {},
        "stream_events": [],
    }

    if config.dry_run:
        result["probes"]["before"] = _dry_probe_records("before", probes=probes)
        result["stream_events"].append(
            _planned_replay_event(
                frames=frames,
                deliveries=deliveries,
                output_base=config.output_path.parent,
                prompt=prompt,
            )
        )
        result["probes"]["after"] = _dry_probe_records("after", probes=probes)
        write_json(config.output_path, result)
        return result

    mlx = mlx_runtime or _load_mlx_runtime(config.model_id)
    prompt_cache_state = _new_prompt_cache_state(mlx.get("prompt_cache_state"))
    if prompt_cache_state is None:
        raise RuntimeError(
            "MLX PromptCacheState is unavailable; cumulative replay cannot run"
        )
    result["runtime"] = mlx["runtime"]
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    result["probes"]["before"] = (
        []
        if config.source_cache_only
        else _run_probe_batch(
            probes=probes,
            history=history,
            phase="before",
            model=mlx["model"],
            processor=mlx["processor"],
            model_config=mlx["model_config"],
            stream_generate=mlx["stream_generate"],
            apply_chat_template=mlx["apply_chat_template"],
            max_tokens=config.probe_max_tokens,
            temperature=config.probe_temperature,
            prompt_cache_state_source=None,
            probe_cache_policy="no_cache",
            cache_summary_max_layers=config.cache_summary_max_layers,
            generation_readout_top_k=config.generation_readout_top_k,
            probe_seed=probe_phase_seeds["before"],
            full_vocab_output_path=config.output_path
            if config.save_full_vocab_first_step
            else None,
        )
    )

    set_global_seed(config.seed, include_mlx=True)
    replay_event = _run_replay_turn(
        frames=frames,
        deliveries=deliveries,
        output_base=config.output_path.parent,
        prompt=prompt,
        history=history,
        prompt_cache_state=prompt_cache_state,
        mlx=mlx,
        config=config,
    )
    result["stream_events"].append(replay_event)
    replay_positions = [
        record["position"]
        for record in replay_event["cache_token_layout"]["sequence_position_plan"]
    ]
    if config.source_cache_only:
        result["probes"]["after"] = []
    elif config.after_probe_protocol == "direct_multimodal_replay":
        result["probes"]["after"] = _run_direct_multimodal_probe_batch(
            probes=probes,
            frames=frames,
            deliveries=deliveries,
            prompt_cache_state_reference=prompt_cache_state,
            replay_positions=replay_positions,
            mlx=mlx,
            config=config,
            probe_seed=probe_phase_seeds["after"],
        )
    else:
        result["probes"]["after"] = _run_probe_batch(
            probes=probes,
            history=history,
            phase="after",
            model=mlx["model"],
            processor=mlx["processor"],
            model_config=mlx["model_config"],
            stream_generate=mlx["stream_generate"],
            apply_chat_template=mlx["apply_chat_template"],
            max_tokens=config.probe_max_tokens,
            temperature=config.probe_temperature,
            prompt_cache_state_source=prompt_cache_state,
            probe_cache_policy="isolated",
            cache_summary_max_layers=config.cache_summary_max_layers,
            generation_readout_top_k=config.generation_readout_top_k,
            probe_seed=probe_phase_seeds["after"],
            cache_summary_sequence_positions=replay_positions,
            full_vocab_output_path=config.output_path
            if config.save_full_vocab_first_step
            else None,
        )
    write_json(config.output_path, result)
    return result


def cumulative_replay_prompt(frames: list[dict[str, Any]]) -> str:
    lines = cumulative_replay_context_lines(frames)
    lines.extend(
        [
            "Treat the ordered image batch as the cumulative visual context. Do not classify or describe it yet.",
            SYNC_PROMPT,
        ]
    )
    return "\n".join(lines)


def cumulative_replay_probe_prompt(
    frames: list[dict[str, Any]],
    probe_prompt: str,
) -> str:
    lines = cumulative_replay_context_lines(frames)
    lines.extend(
        [
            "Use the ordered image batch directly as context for this measurement question:",
            probe_prompt,
        ]
    )
    return "\n".join(lines)


def cumulative_replay_context_lines(frames: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"This is an ordered cumulative replay of {len(frames)} frames from one deterministic visual stream.",
        "Read the attached images in exactly this order:",
    ]
    for ordinal, frame in enumerate(frames):
        lines.append(
            f"{ordinal + 1}. frame {int(frame['index']):06d} at "
            f"{format_timecode(float(frame['t_seconds']))} ({float(frame['t_seconds']):.3f} seconds)"
        )
    return lines


def _run_direct_multimodal_probe_batch(
    *,
    probes: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    deliveries: dict[int, Any],
    prompt_cache_state_reference: Any,
    replay_positions: list[int],
    mlx: dict[str, Any],
    config: CumulativeReplayRunConfig,
    probe_seed: int | None,
) -> list[dict[str, Any]]:
    probe_seed_record = set_global_seed(probe_seed, include_mlx=True)
    image_paths = [str(deliveries[int(frame["index"])].image_path) for frame in frames]
    reference_summary = summarize_prompt_cache_state(
        prompt_cache_state_reference,
        max_layers=config.cache_summary_max_layers,
        sequence_positions=replay_positions,
    )
    records = []
    for probe in probes:
        prompt_cache_state = (
            _new_prompt_cache_state(mlx.get("prompt_cache_state"))
            if config.capture_direct_probe_cache
            else None
        )
        prompt = cumulative_replay_probe_prompt(frames, probe["prompt"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = mlx["apply_chat_template"](
            mlx["processor"],
            mlx["model_config"],
            messages,
            num_images=len(image_paths),
        )
        full_vocab_writer = None
        if config.save_full_vocab_first_step:
            sidecar_path = full_vocab_sidecar_path(
                config.output_path,
                phase="after",
                probe_id=probe["id"],
            )

            def full_vocab_writer(
                logprobs: Any, *, path: Path = sidecar_path
            ) -> dict[str, Any]:
                return write_full_vocab_logprob_sidecar(
                    logprobs,
                    path=path,
                    relative_to=config.output_path.parent,
                )

        started = time.perf_counter()
        generation = _collect_stream(
            mlx["stream_generate"](
                mlx["model"],
                mlx["processor"],
                formatted_prompt,
                image=image_paths,
                max_tokens=config.probe_max_tokens,
                temperature=config.probe_temperature,
                prompt_cache_state=prompt_cache_state,
            ),
            processor=mlx["processor"],
            top_k=config.generation_readout_top_k,
            first_step_full_vocab_writer=full_vocab_writer,
        )
        token_layout = (
            summarize_prompt_cache_token_layout(prompt_cache_state, mlx["model_config"])
            if prompt_cache_state is not None
            else None
        )
        direct_positions = (
            cache_layout_sequence_positions(token_layout)
            if token_layout is not None
            else []
        )
        records.append(
            {
                "phase": "after",
                "probe_id": probe["id"],
                "prompt": probe["prompt"],
                **probe_metadata(probe),
                "direct_multimodal_prompt": prompt,
                "assistant_text": generation["text"],
                "generation": generation["summary"],
                "probe_seed": probe_seed,
                "probe_seed_record": probe_seed_record,
                "wall_s": time.perf_counter() - started,
                "after_probe_protocol": "direct_multimodal_replay",
                "cache_branch_status": "not_used_direct_multimodal_replay",
                "cache_prefix_audit": {
                    "available": False,
                    "reason": "fresh direct multimodal forward; no source cache reuse attempted",
                },
                "source_cache_summary_before_probe": reference_summary,
                "cache_token_layout": token_layout,
                "cache_summary": summarize_prompt_cache_state(
                    prompt_cache_state,
                    max_layers=config.cache_summary_max_layers,
                    sequence_positions=direct_positions,
                )
                if config.capture_direct_probe_cache
                else None,
                "cache_summary_captured": config.capture_direct_probe_cache,
            }
        )
    return records


def _run_replay_turn(
    *,
    frames: list[dict[str, Any]],
    deliveries: dict[int, Any],
    output_base: Path,
    prompt: str,
    history: list[dict[str, str]],
    prompt_cache_state: Any,
    mlx: dict[str, Any],
    config: CumulativeReplayRunConfig,
) -> dict[str, Any]:
    image_paths = [str(deliveries[int(frame["index"])].image_path) for frame in frames]
    messages = history + [{"role": "user", "content": prompt}]
    formatted_prompt = mlx["apply_chat_template"](
        mlx["processor"],
        mlx["model_config"],
        messages,
        num_images=len(image_paths),
    )
    started = time.perf_counter()
    generation = _collect_stream(
        mlx["stream_generate"](
            mlx["model"],
            mlx["processor"],
            formatted_prompt,
            image=image_paths,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            prompt_cache_state=prompt_cache_state,
        ),
        processor=mlx["processor"],
        top_k=config.generation_readout_top_k,
    )
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": generation["text"]})
    token_layout = summarize_prompt_cache_token_layout(
        prompt_cache_state,
        mlx["model_config"],
    )
    cache_tensor_artifacts = capture_prompt_cache_tensors(
        prompt_cache_state,
        specs=config.cache_tensor_captures,
        run_output_path=config.output_path,
        relative_to=config.output_path.parent,
    )
    replay_positions = cache_layout_sequence_positions(token_layout)
    return {
        "event_kind": "ordered_multi_image_replay",
        "frame_index": int(frames[-1]["index"]),
        "frame_path": str(frames[-1]["path"]),
        "frame_artifact": deliveries[int(frames[-1]["index"])].frame_artifact,
        "frame_sha256": str(frames[-1]["sha256"]),
        "frame_indices": [int(frame["index"]) for frame in frames],
        "frame_paths": [str(frame["path"]) for frame in frames],
        "frame_sha256s": [str(frame["sha256"]) for frame in frames],
        "deliveries": [
            deliveries[int(frame["index"])].to_event_record(output_base)
            for frame in frames
        ],
        "image_count": len(image_paths),
        "prompt": prompt,
        "assistant_text": generation["text"],
        "generation": generation["summary"],
        "wall_s": time.perf_counter() - started,
        "cache_token_layout": token_layout,
        "cache_tensor_artifacts": cache_tensor_artifacts,
        "cache_summary": summarize_prompt_cache_state(
            prompt_cache_state,
            max_layers=config.cache_summary_max_layers,
            sequence_positions=replay_positions,
        ),
        "cache_summary_captured": True,
    }


def _planned_replay_event(
    *,
    frames: list[dict[str, Any]],
    deliveries: dict[int, Any],
    output_base: Path,
    prompt: str,
) -> dict[str, Any]:
    return {
        "event_kind": "ordered_multi_image_replay",
        "frame_index": int(frames[-1]["index"]),
        "frame_path": str(frames[-1]["path"]),
        "frame_artifact": deliveries[int(frames[-1]["index"])].frame_artifact,
        "frame_sha256": str(frames[-1]["sha256"]),
        "frame_indices": [int(frame["index"]) for frame in frames],
        "frame_paths": [str(frame["path"]) for frame in frames],
        "frame_sha256s": [str(frame["sha256"]) for frame in frames],
        "deliveries": [
            deliveries[int(frame["index"])].to_event_record(output_base)
            for frame in frames
        ],
        "image_count": len(frames),
        "planned_prompt": prompt,
        "cache_summary": None,
        "cache_summary_captured": False,
    }
