# Provider Tiers

The project separates measurement depth from model access. Bigger remote models
may give stronger behavioral signals, while local models expose richer traces.

## T1 Local Internal Trace

Primary adapters:

- MLX / MLX-VLM
- Hugging Face Transformers

Target measurements:

- hidden states,
- attention summaries,
- logits and logprobs,
- KV/cache summaries,
- vision/text fusion traces where model internals permit.

Adapters should accept both `model_id` and `local_path`. Local paths make it
possible to freeze snapshots, run offline, and reproduce exact checkpoints.
The first HF adapter uses full transcript replay rather than a mutable stateful
thread, which keeps probe isolation straightforward while still allowing
`output_hidden_states=True` traces.

## T2 Remote Logprob Probe

Primary adapters:

- OpenAI-compatible remote logprob adapters,
- Google remote logprob adapters,
- xAI remote logprob adapters.

Remote APIs are black boxes. This tier should not claim hidden-state access.
Use it for:

- output logprob drift,
- top-token entropy,
- forced-choice token probes,
- text-output differences across stimulus conditions.

Provider and model capabilities must be checked at runtime. A provider can be
supported while a specific model is behavior-only.

Claude/Anthropic-style comparisons are intentionally not part of the first
logprob tier unless the selected model exposes comparable logprob signals.

## T3 Instrumented Serving

Target systems:

- vLLM,
- SGLang,
- llama.cpp,
- custom Hugging Face servers,
- other OpenAI-compatible serving layers.

Goal: keep a serving-like interface while exposing approximate internal traces.
This is not the cleanest science path, but it can answer engineering questions:

- What can be measured without leaving serving infrastructure?
- How much overhead does instrumentation add?
- Do serving-time approximations agree with T1 local traces?

## Capability Contract

The code scaffold uses:

```python
supports_images
supports_output_logprobs
supports_top_logprobs
supports_prompt_logprobs
supports_stateful_thread
supports_hidden_states
supports_kv_cache_summary
```

Run reports should include the adapter capability record so downstream analysis
does not accidentally compare a hidden-state trace against a behavior-only
result.
