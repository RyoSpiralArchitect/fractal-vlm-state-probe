# Experiment Design

## Question

When the same multimodal model receives different visual streams one turn at a
time, do its unrelated probe responses, logprob distributions, or local internal
traces shift in condition-specific ways?

The first comparison is not between providers. It is within a single model under
matched stimulus conditions.

The first control question is also not "which fractal differs?" It is whether a
stream of frame-indexed context changes later probes beyond a clean probe-only
baseline.

## Unit Of Analysis

Each run binds together:

- one model or provider adapter,
- one stimulus manifest,
- one stimulus delivery mode,
- one probe set,
- one probe cache policy,
- one frame sampling policy,
- one output file.

The manifest is part of the measurement. A run without frame hashes, timestamps,
and condition metadata is not considered analysis-ready.

## Delivery Modes

The same manifest can be delivered in multiple ways:

- `probe_only`: no stream turns; clean before/after probes only.
- `text_only_stream`: same frame index and timecode schedule, no image.
- `blank_visual_stream`: generated blank images at the manifest frame
  dimensions.
- `visual_stream`: the manifest frames are delivered to the VLM.

These modes make Null-vs-Stream comparisons possible before interpreting
Mandelbrot-vs-Julia or other stimulus-family differences.

## Phase Schedule

1. **Before**: run the probe prompt on a clean context.
2. **During**: stream frames one turn at a time with frame index and timecode.
3. **Mid**: run the same probe after half of the selected frame stream has been
   consumed. For an even frame count, this is after the earlier middle frame.
4. **After**: run the same probe after the final frame.

By default, probes are branch reads: they can inspect the current transcript and
cache state, but they do not mutate the main stream history or main stream
cache. The `shared_append` policy is reserved for deliberate contamination
studies.

## Default Probe

The initial probe is intentionally narrow:

```text
Write a single, abstract, creative poem titled 'Blue Silence'.
```

It is creative enough to expose style drift, but simple enough to run repeatedly
across providers. Later probe batteries can add logic, metaphor, visual-free
classification, and forced-choice token tests.

## Metrics By Tier

**T1 Local Internal Trace**

- hidden-state norms and cosine drift,
- attention summaries where available,
- KV-cache shape, variance, mean, and norm by sampled layer,
- probe output text and local token probabilities,
- timing and memory summaries.

MLX runs can reuse an explicit prompt cache where supported. HF runs currently
replay the full transcript and capture forward-pass traces on selected turns.

**T2 Remote Logprob Probe**

- output token logprobs,
- top-token entropy and rank shifts,
- forced-choice probe deltas,
- behavior-only text diffs when logprobs are unavailable.

**T3 Instrumented Serving**

- serving-compatible outputs plus selected hidden/logit/cache taps,
- parity checks against the unmodified serving interface,
- overhead measurements for instrumentation.

## Non-Claims

Do not report the model as experiencing, entering, or simulating a mental state.
Use language like:

- "context-conditioned output shift",
- "probe-logprob drift",
- "KV-cache summary change",
- "single-run pilot observation",
- "condition-specific trace under this adapter."

## Analysis-Ready Checklist

- The stimulus manifest validates.
- The condition metadata identifies family, temporal policy, semantic load, and
  source kind.
- The run output records adapter capabilities.
- The run output records stimulus delivery mode.
- The probe cache policy is explicit.
- The mid-point schedule is explicit.
- The report separates observations from interpretations.
