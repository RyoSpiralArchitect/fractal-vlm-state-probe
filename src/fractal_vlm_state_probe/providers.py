from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderTier = Literal[
    "T1_LOCAL_INTERNAL_TRACE",
    "T2_REMOTE_LOGPROB_PROBE",
    "T3_INSTRUMENTED_SERVING",
]


@dataclass(frozen=True)
class ProviderCapabilities:
    adapter_id: str
    tier: ProviderTier
    supports_images: bool
    supports_output_logprobs: bool
    supports_top_logprobs: bool
    supports_prompt_logprobs: bool
    supports_stateful_thread: bool
    supports_hidden_states: bool
    supports_kv_cache_summary: bool
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "tier": self.tier,
            "supports_images": self.supports_images,
            "supports_output_logprobs": self.supports_output_logprobs,
            "supports_top_logprobs": self.supports_top_logprobs,
            "supports_prompt_logprobs": self.supports_prompt_logprobs,
            "supports_stateful_thread": self.supports_stateful_thread,
            "supports_hidden_states": self.supports_hidden_states,
            "supports_kv_cache_summary": self.supports_kv_cache_summary,
            "notes": self.notes,
        }


CAPABILITY_REGISTRY: dict[str, ProviderCapabilities] = {
    "mlx_vlm": ProviderCapabilities(
        adapter_id="mlx_vlm",
        tier="T1_LOCAL_INTERNAL_TRACE",
        supports_images=True,
        supports_output_logprobs=True,
        supports_top_logprobs=True,
        supports_prompt_logprobs=False,
        supports_stateful_thread=True,
        supports_hidden_states=False,
        supports_kv_cache_summary=True,
        notes="Current adapter records generated-token logprobs, configurable top-k readouts, public generation metrics, and KV-cache summaries; hidden-state hooks are planned.",
    ),
    "hf_transformers": ProviderCapabilities(
        adapter_id="hf_transformers",
        tier="T1_LOCAL_INTERNAL_TRACE",
        supports_images=True,
        supports_output_logprobs=True,
        supports_top_logprobs=True,
        supports_prompt_logprobs=True,
        supports_stateful_thread=False,
        supports_hidden_states=True,
        supports_kv_cache_summary=True,
        notes="Local adapter for model_id and local_path workflows; records forward-pass hidden-state/logit/KV summaries where model outputs support them.",
    ),
    "openai_logprob": ProviderCapabilities(
        adapter_id="openai_logprob",
        tier="T2_REMOTE_LOGPROB_PROBE",
        supports_images=True,
        supports_output_logprobs=True,
        supports_top_logprobs=True,
        supports_prompt_logprobs=False,
        supports_stateful_thread=False,
        supports_hidden_states=False,
        supports_kv_cache_summary=False,
        notes="Remote logprob proxy only; exact model capability must be checked at runtime.",
    ),
    "google_logprob": ProviderCapabilities(
        adapter_id="google_logprob",
        tier="T2_REMOTE_LOGPROB_PROBE",
        supports_images=True,
        supports_output_logprobs=True,
        supports_top_logprobs=True,
        supports_prompt_logprobs=False,
        supports_stateful_thread=False,
        supports_hidden_states=False,
        supports_kv_cache_summary=False,
        notes="Remote logprob proxy only; exact model capability must be checked at runtime.",
    ),
    "xai_logprob": ProviderCapabilities(
        adapter_id="xai_logprob",
        tier="T2_REMOTE_LOGPROB_PROBE",
        supports_images=True,
        supports_output_logprobs=True,
        supports_top_logprobs=True,
        supports_prompt_logprobs=False,
        supports_stateful_thread=False,
        supports_hidden_states=False,
        supports_kv_cache_summary=False,
        notes="Remote logprob proxy only; exact model capability must be checked at runtime.",
    ),
    "instrumented_openai_compatible": ProviderCapabilities(
        adapter_id="instrumented_openai_compatible",
        tier="T3_INSTRUMENTED_SERVING",
        supports_images=True,
        supports_output_logprobs=True,
        supports_top_logprobs=True,
        supports_prompt_logprobs=True,
        supports_stateful_thread=True,
        supports_hidden_states=True,
        supports_kv_cache_summary=True,
        notes="Target shape for modified vLLM/SGLang/llama.cpp/custom servers.",
    ),
}


def get_capabilities(adapter_id: str) -> ProviderCapabilities:
    try:
        return CAPABILITY_REGISTRY[adapter_id]
    except KeyError as exc:
        known = ", ".join(sorted(CAPABILITY_REGISTRY))
        raise ValueError(f"unknown adapter_id {adapter_id!r}; known: {known}") from exc
