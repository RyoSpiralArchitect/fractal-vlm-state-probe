# Fractal VLM State Probe

Deterministic visual streams for probing how multimodal language models change
their observable behavior, token distributions, and accessible internal traces
under sustained visual context.

This project starts from a simple spiral-shaped hunch: a visual context is not
just an image, and a model's response is not just a final string. If we deliver a
carefully controlled stream one frame at a time, keep the transcript explicit,
and preserve enough instrumentation, we can ask a cleaner question:

> How does a multimodal model's non-visual reasoning and generation shift after
> exposure to different classes of visual motion?

The repo is deliberately cautious. It does not claim that a model has subjective
experience, enters a mental state, or undergoes adaptation in the human sense.
It measures reproducible stimuli, observable outputs, logprob proxies, and local
tensor/cache traces.

## Research Shape

The first program compares stimulus conditions within the same model, not
providers against each other. That keeps the early question sharp:

1. Fractal A vs the same fractal shuffled or repeated.
2. Fractal A vs Fractal B.
3. Fractal streams vs non-fractal geometry.
4. Fractal streams vs natural texture or landscape sequences.
5. Fractal streams vs semantic-rich video such as streets, rooms, or animals.

Every condition is represented as a manifest with frame timestamps, relative
paths, hashes, and `stimulus_condition` metadata. Runs then use the same model,
probe, frame count, cache policy, and timing where possible.

## Infrastructure Tiers

- **T1 Local Internal Trace**: MLX-VLM now, Hugging Face Transformers next.
  This is the main path for hidden states, attention hooks, and KV/cache
  summaries. Adapters should support both `model_id` and `local_path`.
- **T2 Remote Logprob Probe**: provider APIs only when the selected multimodal
  model returns useful logprobs. This tier measures output-distribution proxies,
  not internal state.
- **T3 Instrumented Serving**: modified OpenAI-compatible serving stacks such as
  vLLM, SGLang, llama.cpp, or custom HF servers that expose extra traces while
  preserving a serving-like interface.

Claude/Anthropic-style behavior-only comparisons are intentionally outside the
first logprob-focused pass unless a selected model exposes the needed signal.

## Current Capabilities

- Generate deterministic Mandelbrot and Julia frame streams.
- Attach condition metadata for fractal, geometry, natural, and control
  comparisons.
- Build manifests for external frame directories, so natural videos can be
  sampled elsewhere and brought into the same run format.
- Stream frames into MLX-VLM one turn at a time with an explicit chat transcript.
- Run a Hugging Face Transformers stream probe with `model_id` or `local_path`,
  full transcript replay, hidden-state summaries, KV-cache summaries, and
  generation logprob summaries where the selected model exposes them.
- Use isolated probe branches by default so before/mid/after probes do not
  contaminate the main stream cache/history.
- Record probe outputs, generation metrics, and sampled KV-cache summaries.
- Provide provider capability scaffolding for T1/T2/T3 adapters.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python3 scripts/generate_fractal_frames.py \
  --config configs/mandelbrot_smoke.json \
  --output runs/smoke/mandelbrot_a \
  --overwrite

python3 scripts/run_mlx_stream_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --dry-run \
  --output runs/smoke/mandelbrot_a_dry_run.json
```

On a machine with MLX and MLX-VLM installed:

```bash
python3 scripts/run_mlx_stream_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-frames 4 \
  --max-tokens 2 \
  --probe-max-tokens 48 \
  --probe-cache-policy isolated \
  --cache-summary-every 10 \
  --cache-summary-max-layers 4 \
  --output runs/smoke/mandelbrot_a_mlx.json
```

With Hugging Face Transformers:

```bash
python3 scripts/run_hf_stream_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-frames 4 \
  --max-new-tokens 2 \
  --probe-max-new-tokens 48 \
  --probe-cache-policy isolated \
  --trace-every 10 \
  --trace-max-layers 4 \
  --output runs/smoke/mandelbrot_a_hf.json
```

By default, selected frames are copied next to the run JSON under
`<output>.frames/` and referenced from each stream event. Add
`--no-frame-artifacts` for metadata-only runs.

For a frozen local snapshot, replace `--model` with:

```bash
--local-path /path/to/local/model
```

To bring in an external natural or geometric frame sequence:

```bash
python3 scripts/build_external_frame_manifest.py \
  --frames-dir /path/to/extracted_frames \
  --condition configs/conditions/external_natural_city.json \
  --fps 1 \
  --output runs/stimuli/natural_city/manifest.json
```

To create the first Julia comparison stimulus:

```bash
python3 scripts/run_julia_smoke.py --overwrite
```

To generate Julia frames and immediately run the same MLX smoke probe:

```bash
python3 scripts/run_julia_smoke.py --overwrite --run-mlx
```

## Documentation

- [Experiment Design](docs/experiment_design.md)
- [Comparison Axes](docs/comparison_axes.md)
- [Provider Tiers](docs/provider_tiers.md)
- [Roadmap](docs/roadmap.md)

## Claim Boundary

A single run can support "observed in this run" only. Repeated seeds, models,
stimulus families, probes, and controls are required before describing an effect
as stable. The strongest early result is not a grand claim; it is a clean
artifact trail that makes the next run easier to trust.
