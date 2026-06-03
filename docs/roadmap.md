# Roadmap

## Phase 0: Reproducible Stimuli

- Deterministic Mandelbrot and Julia streams.
- Manifest validation with hashes and timestamps.
- External frame manifests for natural and geometric controls.
- Condition metadata for comparison axes.

Status: scaffolded.

## Phase 1: MLX-VLM Stream Probe

- Frame-by-frame chat streaming.
- Isolated before/mid/after probes.
- Sampled KV-cache summaries.
- Honest run metadata and non-claims.

Status: scaffolded.

## Phase 2: Hugging Face Internal Trace

- `model_id` and `local_path` loading.
- Hidden-state extraction via supported model outputs or hooks.
- Token/logit metrics for probe prompts.
- Matched output schema with MLX runs.

Status: first adapter scaffolded. Model-specific hooks and broader compatibility
passes are next.

## Phase 3: Remote Logprob Adapters

- Provider capability probes.
- OpenAI/Google/xAI adapter boundaries.
- Forced-choice and open-ended logprob probes.
- Behavior-only fallback labeling.

Status: planned.

## Phase 4: Comparison Runs

Run matched conditions inside the same model:

- Mandelbrot ordered,
- Mandelbrot shuffled,
- Julia ordered,
- non-fractal geometry,
- natural city/street,
- natural animal,
- deterministic noise.

Status: design-ready after Phase 2/3 adapters.

## Phase 5: Research Report

- Analysis notebook or script.
- Tables for condition metadata and run capabilities.
- Probe output diffs.
- Logprob/trace plots.
- Strict separation between observed effects and interpretation.

Status: future.
