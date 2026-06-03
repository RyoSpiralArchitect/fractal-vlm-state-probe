from __future__ import annotations

import importlib
import json
import time
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .frame_artifacts import prepare_frame_artifacts
from .prompts import DEFAULT_PROBES, SYNC_PROMPT, SYSTEM_PROMPT
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
    dry_run: bool = False
    use_prompt_cache_state: bool = True
    probe_cache_policy: ProbeCachePolicy = "isolated"
    cache_summary_every: int = 10
    cache_summary_max_layers: int | None = 4
    adapter_id: str = "mlx_vlm"
    include_frame_artifacts: bool = True
    seed: int | None = None


def run_stream_probe(config: StreamRunConfig) -> dict[str, Any]:
    seed_record = set_global_seed(config.seed, include_mlx=True)
    manifest = _load_manifest(config.manifest_path)
    issues = validate_manifest(config.manifest_path)
    if issues:
        raise ValueError("manifest validation failed: " + "; ".join(issues))

    frames = _select_frames(
        manifest["frames"],
        frame_stride=config.frame_stride,
        max_frames=config.max_frames,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_kind": "fractal_vlm_stream_probe",
        "model_id": config.model_id,
        "adapter_capabilities": get_capabilities(config.adapter_id).to_dict(),
        "manifest_path": str(config.manifest_path),
        "dry_run": config.dry_run,
        "reproducibility": seed_record,
        "context_policy": {
            "frame_delivery": "one frame path per stream turn",
            "history": "explicit chat transcript stack",
            "prompt_cache_state_requested": config.use_prompt_cache_state,
            "probe_cache_policy": config.probe_cache_policy,
            "probe_history_policy": _probe_history_policy(config.probe_cache_policy),
            "mid_probe_policy": "after half of the selected frame stream has been consumed",
            "cache_summary_every": config.cache_summary_every,
            "cache_summary_max_layers": config.cache_summary_max_layers,
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
        },
        "probes": {},
        "stream_events": [],
    }
    frame_artifacts = prepare_frame_artifacts(
        output_path=config.output_path,
        manifest_base=config.manifest_path.parent,
        frames=frames,
        enabled=config.include_frame_artifacts,
    )
    result["frame_artifacts"] = list(frame_artifacts.values())
    result["probe_schedule"] = {
        "mid_probe_after_position": _mid_probe_after_position(len(frames)),
        "position_is_zero_based": True,
    }

    if config.dry_run:
        result["probes"]["before"] = _dry_probe_records("before")
        for frame in frames:
            event = _planned_frame_event(frame)
            event["frame_artifact"] = frame_artifacts.get(int(frame["index"]))
            result["stream_events"].append(event)
        result["probes"]["mid"] = _dry_probe_records("mid")
        result["probes"]["after"] = _dry_probe_records("after")
        write_json(config.output_path, result)
        return result

    mlx = _load_mlx_runtime(config.model_id)
    model = mlx["model"]
    processor = mlx["processor"]
    model_config = mlx["model_config"]
    stream_generate = mlx["stream_generate"]
    apply_chat_template = mlx["apply_chat_template"]
    prompt_cache_state = mlx["prompt_cache_state"] if config.use_prompt_cache_state else None

    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    result["runtime"] = mlx["runtime"]

    result["probes"]["before"] = _run_probe_batch(
        probes=DEFAULT_PROBES,
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
        phase="before",
        model=model,
        processor=processor,
        model_config=model_config,
        stream_generate=stream_generate,
        apply_chat_template=apply_chat_template,
        max_tokens=config.probe_max_tokens,
        temperature=config.temperature,
        prompt_cache_state_source=None,
        probe_cache_policy="no_cache",
        cache_summary_max_layers=config.cache_summary_max_layers,
    )

    mid_index = _mid_probe_after_position(len(frames))
    for position, frame in enumerate(frames):
        event = _run_frame_turn(
            frame=frame,
            manifest_base=config.manifest_path.parent,
            history=history,
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
            frame_artifact=frame_artifacts.get(int(frame["index"])),
        )
        result["stream_events"].append(event)

        if mid_index is not None and position == mid_index:
            result["probes"]["mid"] = _run_probe_batch(
                probes=DEFAULT_PROBES,
                history=history,
                phase="mid",
                model=model,
                processor=processor,
                model_config=model_config,
                stream_generate=stream_generate,
                apply_chat_template=apply_chat_template,
                max_tokens=config.probe_max_tokens,
                temperature=config.temperature,
                prompt_cache_state_source=prompt_cache_state,
                probe_cache_policy=config.probe_cache_policy,
                cache_summary_max_layers=config.cache_summary_max_layers,
            )

    result["probes"]["after"] = _run_probe_batch(
        probes=DEFAULT_PROBES,
        history=history,
        phase="after",
        model=model,
        processor=processor,
        model_config=model_config,
        stream_generate=stream_generate,
        apply_chat_template=apply_chat_template,
        max_tokens=config.probe_max_tokens,
        temperature=config.temperature,
        prompt_cache_state_source=prompt_cache_state,
        probe_cache_policy=config.probe_cache_policy,
        cache_summary_max_layers=config.cache_summary_max_layers,
    )

    write_json(config.output_path, result)
    return result


def _run_frame_turn(
    *,
    frame: dict[str, Any],
    manifest_base: Path,
    history: list[dict[str, str]],
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
    frame_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    timecode = format_timecode(float(frame["t_seconds"]))
    prompt = (
        f"This is frame {frame['index']:06d} at {timecode} "
        f"({frame['t_seconds']:.3f} seconds) in the deterministic visual stream.\n"
        f"{SYNC_PROMPT}"
    )
    messages = history + [{"role": "user", "content": prompt}]
    formatted_prompt = apply_chat_template(
        processor,
        model_config,
        messages,
        num_images=1,
    )
    frame_path = manifest_base / frame["path"]
    started = time.perf_counter()
    generation = _collect_stream(
        stream_generate(
            model,
            processor,
            formatted_prompt,
            image=str(frame_path),
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_cache_state=prompt_cache_state,
        )
    )
    wall_s = time.perf_counter() - started

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": generation["text"]})

    return {
        "frame_index": frame["index"],
        "t_seconds": frame["t_seconds"],
        "timecode": timecode,
        "frame_path": frame["path"],
        "frame_artifact": frame_artifact,
        "frame_sha256": frame["sha256"],
        "prompt": prompt,
        "assistant_text": generation["text"],
        "generation": generation["summary"],
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
    probes: list[dict[str, str]],
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
) -> list[dict[str, Any]]:
    records = []
    for probe in probes:
        source_cache_summary = summarize_prompt_cache_state(
            prompt_cache_state_source,
            max_layers=cache_summary_max_layers,
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
        started = time.perf_counter()
        generation = _collect_stream(
            stream_generate(
                model,
                processor,
                formatted_prompt,
                image=None,
                max_tokens=max_tokens,
                temperature=temperature,
                prompt_cache_state=prompt_cache_state,
            )
        )
        records.append(
            {
                "phase": phase,
                "probe_id": probe["id"],
                "prompt": prompt,
                "assistant_text": generation["text"],
                "generation": generation["summary"],
                "wall_s": time.perf_counter() - started,
                "cache_branch_status": cache_branch_status,
                "source_cache_summary_before_probe": source_cache_summary,
                "cache_summary": summarize_prompt_cache_state(
                    prompt_cache_state,
                    max_layers=cache_summary_max_layers,
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
) -> dict[str, Any] | None:
    if prompt_cache_state is None or getattr(prompt_cache_state, "cache", None) is None:
        return None
    if max_layers is not None and max_layers < 0:
        raise ValueError("max_layers must be non-negative or None")

    try:
        import mlx.core as mx
    except Exception:
        return {"available": False, "reason": "mlx import failed"}

    layers = []
    cache_entries = list(prompt_cache_state.cache)
    for layer_index in select_layer_indices(len(cache_entries), max_layers):
        entry = cache_entries[layer_index]
        layer_record: dict[str, Any] = {"layer_index": layer_index}
        keys = getattr(entry, "keys", None)
        values = getattr(entry, "values", None)
        if keys is not None:
            layer_record["keys"] = _mlx_tensor_stats(mx, keys)
        if values is not None:
            layer_record["values"] = _mlx_tensor_stats(mx, values)
        if "keys" in layer_record or "values" in layer_record:
            layers.append(layer_record)
    return {
        "available": True,
        "token_count": len(getattr(prompt_cache_state, "token_ids", []) or []),
        "total_layers": len(cache_entries),
        "reported_layers": len(layers),
        "truncated": max_layers is not None and len(cache_entries) > max_layers,
        "layers": layers,
    }


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


def _mlx_tensor_stats(mx: Any, tensor: Any) -> dict[str, Any]:
    mean = mx.mean(tensor)
    var = mx.var(tensor)
    std = mx.sqrt(var)
    abs_mean = mx.mean(mx.abs(tensor))
    min_val = mx.min(tensor)
    max_val = mx.max(tensor)
    l2 = mx.linalg.norm(tensor)
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
    slices = _mlx_sequence_position_stats(mx, tensor)
    if slices:
        stats["sequence_position_stats"] = slices
    return stats


def _mlx_sequence_position_stats(mx: Any, tensor: Any) -> list[dict[str, Any]]:
    if not hasattr(tensor, "shape") or len(tensor.shape) < 3:
        return []
    sequence_length = int(tensor.shape[-2])
    records = []
    for position in select_sequence_positions(sequence_length):
        sliced = tensor[..., position, :]
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


def _collect_stream(chunks: Any) -> dict[str, Any]:
    text_parts = []
    last = None
    for chunk in chunks:
        last = chunk
        text_parts.append(getattr(chunk, "text", str(chunk)))

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
    return {"text": "".join(text_parts).strip(), "summary": summary}


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
    model, processor = load(model_id)
    model_config = getattr(model, "config", None)
    if model_config is None:
        try:
            from mlx_vlm.utils import load_config

            model_config = load_config(model_id)
        except Exception as exc:
            raise RuntimeError(f"could not resolve model config for {model_id}") from exc

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
            "mlx_vlm_load_kwargs": {},
            "prompt_cache_state_available": prompt_cache_cls is not None,
        },
    }


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


def _planned_frame_event(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": frame["index"],
        "t_seconds": frame["t_seconds"],
        "timecode": format_timecode(float(frame["t_seconds"])),
        "frame_path": frame["path"],
        "frame_sha256": frame["sha256"],
        "planned_prompt": (
            f"This is frame {frame['index']:06d} at "
            f"{format_timecode(float(frame['t_seconds']))}. {SYNC_PROMPT}"
        ),
    }


def _dry_probe_records(phase: str) -> list[dict[str, str]]:
    return [
        {
            "phase": phase,
            "probe_id": probe["id"],
            "prompt": probe["prompt"],
            "assistant_text": "<dry-run>",
        }
        for probe in DEFAULT_PROBES
    ]
