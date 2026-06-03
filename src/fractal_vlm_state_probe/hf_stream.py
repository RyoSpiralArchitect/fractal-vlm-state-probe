from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .delivery import (
    FrameDelivery,
    StimulusDeliveryMode,
    delivery_prompt_prefix,
    frame_artifact_list,
    prepare_frame_deliveries,
    stimulus_delivery_record,
)
from .mlx_stream import (
    ProbeCachePolicy,
    _dry_probe_records,
    _mid_probe_after_position,
    _planned_frame_event,
    _probe_history_policy,
    _select_frames,
    _should_summarize_cache,
)
from .prompts import DEFAULT_PROBES, SYNC_PROMPT, SYSTEM_PROMPT
from .providers import get_capabilities
from .seeding import set_global_seed
from .stimulus import validate_manifest, write_json
from .timecode import format_timecode
from .trace import select_layer_indices, select_sequence_positions

TorchDTypeName = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class HFStreamRunConfig:
    manifest_path: Path
    output_path: Path
    model_ref: str
    max_frames: int | None = None
    frame_stride: int = 1
    max_new_tokens: int = 2
    probe_max_new_tokens: int = 64
    temperature: float = 0.0
    dry_run: bool = False
    probe_cache_policy: ProbeCachePolicy = "isolated"
    trace_every: int = 10
    trace_max_layers: int | None = 4
    torch_dtype: TorchDTypeName = "auto"
    device: str = "auto"
    trust_remote_code: bool = False
    local_files_only: bool = False
    adapter_id: str = "hf_transformers"
    include_frame_artifacts: bool = True
    seed: int | None = None
    delivery_mode: StimulusDeliveryMode = "visual_stream"
    blank_rgb: tuple[int, int, int] = (0, 0, 0)


def run_hf_stream_probe(config: HFStreamRunConfig) -> dict[str, Any]:
    seed_record = set_global_seed(config.seed, include_torch=True)
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
        "run_kind": "hf_vlm_stream_probe",
        "model_ref": config.model_ref,
        "adapter_capabilities": get_capabilities(config.adapter_id).to_dict(),
        "manifest_path": str(config.manifest_path),
        "dry_run": config.dry_run,
        "reproducibility": seed_record,
        "context_policy": {
            "frame_delivery": config.delivery_mode,
            "history": "full transcript replay",
            "probe_cache_policy": config.probe_cache_policy,
            "probe_history_policy": _probe_history_policy(config.probe_cache_policy),
            "mid_probe_policy": "after half of the selected frame stream has been consumed",
            "trace_every": config.trace_every,
            "trace_max_layers": config.trace_max_layers,
            "include_frame_artifacts": config.include_frame_artifacts,
        },
        "non_claims": [
            "No consciousness or subjective-state claim is made.",
            "Hidden-state summaries are computational traces, not experiences.",
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
        "probe_schedule": {
            "mid_probe_after_position": _mid_probe_after_position(len(frames)),
            "position_is_zero_based": True,
        },
        "probes": {},
        "stream_events": [],
    }
    result["frame_artifacts"] = frame_artifact_list(frame_deliveries)

    if config.dry_run:
        result["probes"]["before"] = _dry_probe_records("before")
        for frame in frames:
            delivery = frame_deliveries[int(frame["index"])]
            event = _planned_frame_event(
                frame,
                delivery=delivery,
                output_base=config.output_path.parent,
            )
            result["stream_events"].append(event)
        result["probes"]["mid"] = _dry_probe_records("mid")
        result["probes"]["after"] = _dry_probe_records("after")
        write_json(config.output_path, result)
        return result

    runtime = _load_hf_runtime(config)
    result["runtime"] = runtime["runtime"]
    history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    result["probes"]["before"] = _run_probe_batch(
        probes=DEFAULT_PROBES,
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
        phase="before",
        runtime=runtime,
        max_new_tokens=config.probe_max_new_tokens,
        temperature=config.temperature,
        trace_max_layers=config.trace_max_layers,
        append_to_history=False,
    )

    mid_index = _mid_probe_after_position(len(frames))
    for position, frame in enumerate(frames):
        event = _run_frame_turn(
            frame=frame,
            history=history,
            delivery=frame_deliveries[int(frame["index"])],
            output_base=config.output_path.parent,
            runtime=runtime,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            capture_trace=_should_summarize_cache(position, len(frames), config.trace_every),
            trace_max_layers=config.trace_max_layers,
        )
        result["stream_events"].append(event)

        if mid_index is not None and position == mid_index:
            result["probes"]["mid"] = _run_probe_batch(
                probes=DEFAULT_PROBES,
                history=history,
                phase="mid",
                runtime=runtime,
                max_new_tokens=config.probe_max_new_tokens,
                temperature=config.temperature,
                trace_max_layers=config.trace_max_layers,
                append_to_history=config.probe_cache_policy == "shared_append",
            )

    result["probes"]["after"] = _run_probe_batch(
        probes=DEFAULT_PROBES,
        history=history,
        phase="after",
        runtime=runtime,
        max_new_tokens=config.probe_max_new_tokens,
        temperature=config.temperature,
        trace_max_layers=config.trace_max_layers,
        append_to_history=config.probe_cache_policy == "shared_append",
    )

    write_json(config.output_path, result)
    return result


def _run_frame_turn(
    *,
    frame: dict[str, Any],
    history: list[dict[str, str]],
    delivery: FrameDelivery,
    output_base: Path,
    runtime: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    capture_trace: bool,
    trace_max_layers: int | None,
) -> dict[str, Any]:
    timecode = format_timecode(float(frame["t_seconds"]))
    prompt = f"{delivery_prompt_prefix(frame, delivery.mode, timecode)}\n{SYNC_PROMPT}"
    started = time.perf_counter()
    turn = _run_hf_turn(
        runtime=runtime,
        history=history,
        prompt=prompt,
        image_path=delivery.image_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        capture_trace=capture_trace,
        trace_max_layers=trace_max_layers,
    )
    wall_s = time.perf_counter() - started

    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": turn["assistant_text"]})

    return {
        "frame_index": frame["index"],
        "t_seconds": frame["t_seconds"],
        "timecode": timecode,
        "frame_path": frame["path"],
        "frame_artifact": delivery.frame_artifact,
        "frame_sha256": frame["sha256"],
        "delivery": delivery.to_event_record(output_base),
        "prompt": prompt,
        "assistant_text": turn["assistant_text"],
        "generation": turn["generation"],
        "trace": turn["trace"],
        "trace_captured": capture_trace,
        "wall_s": wall_s,
    }


def _run_probe_batch(
    *,
    probes: list[dict[str, str]],
    history: list[dict[str, str]],
    phase: str,
    runtime: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    trace_max_layers: int | None,
    append_to_history: bool,
) -> list[dict[str, Any]]:
    records = []
    for probe in probes:
        started = time.perf_counter()
        turn = _run_hf_turn(
            runtime=runtime,
            history=history,
            prompt=probe["prompt"],
            image_path=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            capture_trace=True,
            trace_max_layers=trace_max_layers,
        )
        records.append(
            {
                "phase": phase,
                "probe_id": probe["id"],
                "prompt": probe["prompt"],
                "assistant_text": turn["assistant_text"],
                "generation": turn["generation"],
                "trace": turn["trace"],
                "history_mutated": append_to_history,
                "wall_s": time.perf_counter() - started,
            }
        )
        if append_to_history:
            history.append({"role": "user", "content": probe["prompt"]})
            history.append({"role": "assistant", "content": turn["assistant_text"]})
    return records


def _run_hf_turn(
    *,
    runtime: dict[str, Any],
    history: list[dict[str, str]],
    prompt: str,
    image_path: Path | None,
    max_new_tokens: int,
    temperature: float,
    capture_trace: bool,
    trace_max_layers: int | None,
) -> dict[str, Any]:
    torch = runtime["torch"]
    model = runtime["model"]
    processor = runtime["processor"]
    device = runtime["device"]
    image = Image.open(image_path).convert("RGB") if image_path is not None else None

    messages = _build_hf_messages(history, prompt, has_image=image is not None)
    prompt_text = _apply_hf_chat_template(processor, messages)
    inputs = _prepare_hf_inputs(processor, prompt_text, image)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    trace = None
    if capture_trace:
        with torch.no_grad():
            forward_output = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )
        trace = summarize_forward_output(
            torch=torch,
            output=forward_output,
            processor=processor,
            max_layers=trace_max_layers,
        )

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "return_dict_in_generate": True,
        "output_scores": True,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)

    input_token_count = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else None
    new_tokens = generated.sequences
    if input_token_count is not None and new_tokens.shape[-1] >= input_token_count:
        new_tokens = new_tokens[:, input_token_count:]
    assistant_text = _decode_tokens(processor, new_tokens)
    return {
        "assistant_text": assistant_text,
        "generation": summarize_generation(
            torch=torch,
            generated=generated,
            processor=processor,
            input_token_count=input_token_count,
        ),
        "trace": trace,
    }


def summarize_forward_output(
    *,
    torch: Any,
    output: Any,
    processor: Any,
    max_layers: int | None,
) -> dict[str, Any]:
    hidden_states = getattr(output, "hidden_states", None)
    logits = getattr(output, "logits", None)
    past_key_values = getattr(output, "past_key_values", None)
    return {
        "hidden_states": summarize_hidden_states(
            torch=torch,
            hidden_states=hidden_states,
            max_layers=max_layers,
        ),
        "last_token": summarize_last_token_logits(
            torch=torch,
            logits=logits,
            processor=processor,
        ),
        "kv_cache": summarize_past_key_values(
            torch=torch,
            past_key_values=past_key_values,
            max_layers=max_layers,
        ),
    }


def summarize_hidden_states(
    *,
    torch: Any,
    hidden_states: Any,
    max_layers: int | None,
) -> dict[str, Any] | None:
    if hidden_states is None:
        return None
    total_layers = len(hidden_states)
    records = []
    for layer_index in select_layer_indices(total_layers, max_layers):
        tensor = hidden_states[layer_index].detach().float()
        records.append({"layer_index": layer_index, **_torch_tensor_stats(torch, tensor)})
    return {
        "available": True,
        "total_layers": total_layers,
        "reported_layers": len(records),
        "truncated": max_layers is not None and total_layers > max_layers,
        "layers": records,
    }


def summarize_last_token_logits(
    *,
    torch: Any,
    logits: Any,
    processor: Any,
    top_k: int = 10,
) -> dict[str, Any] | None:
    if logits is None:
        return None
    last_logits = logits[:, -1, :].detach().float()
    logprobs = torch.log_softmax(last_logits, dim=-1)
    values, indices = torch.topk(logprobs[0], k=min(top_k, logprobs.shape[-1]))
    return {
        "available": True,
        "top_logprobs": [
            {
                "token_id": int(token_id),
                "token": _decode_token(processor, int(token_id)),
                "logprob": float(logprob),
            }
            for token_id, logprob in zip(indices.tolist(), values.tolist())
        ],
    }


def summarize_generation(
    *,
    torch: Any,
    generated: Any,
    processor: Any,
    input_token_count: int | None,
    top_k: int = 10,
) -> dict[str, Any]:
    sequences = getattr(generated, "sequences", None)
    scores = getattr(generated, "scores", None) or []
    new_token_count = None
    if sequences is not None and input_token_count is not None:
        new_token_count = max(0, int(sequences.shape[-1]) - input_token_count)

    step_records = []
    for step_index, score in enumerate(scores):
        logprobs = torch.log_softmax(score.detach().float(), dim=-1)
        values, indices = torch.topk(logprobs[0], k=min(top_k, logprobs.shape[-1]))
        step_records.append(
            {
                "step_index": step_index,
                "top_logprobs": [
                    {
                        "token_id": int(token_id),
                        "token": _decode_token(processor, int(token_id)),
                        "logprob": float(logprob),
                    }
                    for token_id, logprob in zip(indices.tolist(), values.tolist())
                ],
            }
        )
    return {
        "input_tokens": input_token_count,
        "generation_tokens": new_token_count,
        "score_steps": len(step_records),
        "steps": step_records,
    }


def summarize_past_key_values(
    *,
    torch: Any,
    past_key_values: Any,
    max_layers: int | None,
) -> dict[str, Any] | None:
    legacy_cache = _as_legacy_cache(past_key_values)
    if legacy_cache is None:
        return None
    total_layers = len(legacy_cache)
    records = []
    for layer_index in select_layer_indices(total_layers, max_layers):
        layer = legacy_cache[layer_index]
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            continue
        records.append(
            {
                "layer_index": layer_index,
                "keys": _torch_tensor_stats(torch, layer[0].detach().float()),
                "values": _torch_tensor_stats(torch, layer[1].detach().float()),
            }
        )
    return {
        "available": True,
        "total_layers": total_layers,
        "reported_layers": len(records),
        "truncated": max_layers is not None and total_layers > max_layers,
        "layers": records,
    }


def _torch_tensor_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    mean = torch.mean(tensor)
    variance = torch.var(tensor)
    std = torch.sqrt(variance)
    abs_mean = torch.mean(torch.abs(tensor))
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    l2_norm = torch.linalg.vector_norm(tensor)
    stats = {
        "shape": list(tensor.shape),
        "mean": float(mean.detach().cpu().item()),
        "variance": float(variance.detach().cpu().item()),
        "std": float(std.detach().cpu().item()),
        "abs_mean": float(abs_mean.detach().cpu().item()),
        "min": float(min_val.detach().cpu().item()),
        "max": float(max_val.detach().cpu().item()),
        "l2_norm": float(l2_norm.detach().cpu().item()),
    }
    slices = _torch_sequence_position_stats(torch, tensor)
    if slices:
        stats["sequence_position_stats"] = slices
    return stats


def _torch_sequence_position_stats(torch: Any, tensor: Any) -> list[dict[str, Any]]:
    if not hasattr(tensor, "shape") or len(tensor.shape) < 3:
        return []
    sequence_length = int(tensor.shape[-2])
    records = []
    for position in select_sequence_positions(sequence_length):
        sliced = tensor[..., position, :]
        mean = torch.mean(sliced)
        variance = torch.var(sliced)
        abs_mean = torch.mean(torch.abs(sliced))
        l2_norm = torch.linalg.vector_norm(sliced)
        records.append(
            {
                "position": position,
                "mean": float(mean.detach().cpu().item()),
                "variance": float(variance.detach().cpu().item()),
                "abs_mean": float(abs_mean.detach().cpu().item()),
                "l2_norm": float(l2_norm.detach().cpu().item()),
            }
        )
    return records


def _as_legacy_cache(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return past_key_values.to_legacy_cache()
        except Exception:
            return None
    if isinstance(past_key_values, (tuple, list)):
        return past_key_values
    return None


def _load_hf_runtime(config: HFStreamRunConfig) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq, AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face runtime is unavailable. Install optional dependencies or use --dry-run."
        ) from exc

    device = _resolve_device(torch, config.device)
    dtype = _resolve_dtype(torch, config.torch_dtype, device)
    processor = AutoProcessor.from_pretrained(
        config.model_ref,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    model = _load_hf_model(
        model_ref=config.model_ref,
        dtype=dtype,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    model.to(device)
    model.eval()
    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": device,
        "runtime": {
            "torch_version": torch.__version__,
            "transformers_loader": "AutoModelForImageTextToText/AutoModelForVision2Seq",
            "device": str(device),
            "torch_dtype": str(dtype).replace("torch.", ""),
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
        },
    }


def _load_hf_model(
    *,
    model_ref: str,
    dtype: Any,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq

    errors = []
    for auto_cls in (AutoModelForImageTextToText, AutoModelForVision2Seq):
        try:
            return auto_cls.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            errors.append(f"{auto_cls.__name__}: {type(exc).__name__}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"could not load {model_ref!r} as a HF multimodal model:\n{joined}")


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch: Any, requested: TorchDTypeName, device: str) -> Any:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def _build_hf_messages(
    history: list[dict[str, str]],
    prompt: str,
    *,
    has_image: bool,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    content: list[dict[str, Any]] = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _apply_hf_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
    return "\n".join(
        f"{message['role'].capitalize()}: {_message_text(message['content'])}"
        for message in messages
    ) + "\nAssistant:"


def _prepare_hf_inputs(processor: Any, prompt_text: str, image: Image.Image | None) -> dict[str, Any]:
    if image is None:
        try:
            return dict(processor(text=[prompt_text], return_tensors="pt"))
        except TypeError:
            tokenizer = getattr(processor, "tokenizer", processor)
            return dict(tokenizer([prompt_text], return_tensors="pt"))
    try:
        return dict(processor(text=[prompt_text], images=[image], return_tensors="pt"))
    except TypeError:
        return dict(processor(prompt_text, image, return_tensors="pt"))


def _decode_tokens(processor: Any, tokens: Any) -> str:
    decoder = getattr(processor, "batch_decode", None)
    if decoder is None:
        decoder = getattr(getattr(processor, "tokenizer", None), "batch_decode", None)
    if decoder is None:
        return ""
    return decoder(tokens, skip_special_tokens=True)[0].strip()


def _decode_token(processor: Any, token_id: int) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return str(token_id)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return " ".join(parts)
    return str(content)


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
