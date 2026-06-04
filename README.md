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
providers against each other. But the first question is even simpler than
"which fractal is different?":

> Does sustained visual streaming change the later non-visual probe at all,
> beyond ordinary transcript length and cache growth?

That makes the early ladder:

1. Probe-only baseline.
2. Text-only stream with the same timecodes and synchronization prompt.
3. Blank, static, repeated, or shuffled visual controls.
4. Fractal A vs the same fractal repeated or shuffled.
5. Fractal A vs Fractal B.
6. Fractal streams vs non-fractal geometry.
7. Fractal streams vs natural texture or landscape sequences.
8. Fractal streams vs semantic-rich video such as streets, rooms, or animals.

Every condition is represented as a manifest with frame timestamps, relative
paths, hashes, and `stimulus_condition` metadata. Runs then use the same model,
probe, frame count, cache policy, and timing where possible.

## Current Pilot Reading

In the first seeded Mandelbrot-vs-Julia smoke comparison, the before/mid/after
creative probe stayed inside the same broad generative attractor and the paired
Mandelbrot/Julia probe text matched exactly. At the same time, the probe wording
shifted across phases within each run, and sampled source-cache summaries showed
nonzero deltas after visual streaming.

That is not evidence that one fractal family has a stable effect. It is a more
modest and more useful signal: the apparatus can capture cases where surface
text remains similar while the traced context state moves. The next priority is
therefore `Null vs Stream`, before stronger fractal-family claims.

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
- Generate deterministic visual controls: blank, white/blue noise, random dots,
  checkerboards, square/triangle/hex tilings, Voronoi fields, and
  quasicrystal-like patterns.
- Transform existing manifests into phase-scrambled, static-repeat, shuffled,
  and reversed-order controls.
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
- Seed MLX/HF smoke runs for reproducible paired comparisons.
- Run paired stochastic-probe MLX batches where stream generation stays greedy
  but probe sampling is repeated across matched probe seeds.
- Compare paired run JSONs with probe text, frame artifacts, stream-cache
  deltas, and probe-source-cache deltas.
- Train a small nearest-centroid classifier on saved cache-summary features to
  test whether measured traces retain condition information when probe text is
  unchanged.
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
  --seed 20260604 \
  --output runs/smoke/mandelbrot_a_mlx.json
```

To run the first Null-vs-Stream ladder from the same manifest, keep the seed,
probe, frame count, and model fixed while changing only `--delivery-mode`:

```bash
python3 scripts/run_mlx_stream_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --delivery-mode probe_only \
  --max-frames 4 \
  --seed 20260604 \
  --output runs/null_vs_stream/probe_only.json

python3 scripts/run_mlx_stream_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --delivery-mode text_only_stream \
  --max-frames 4 \
  --seed 20260604 \
  --output runs/null_vs_stream/text_only.json

python3 scripts/run_mlx_stream_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --delivery-mode blank_visual_stream \
  --max-frames 4 \
  --seed 20260604 \
  --output runs/null_vs_stream/blank_visual.json

python3 scripts/run_mlx_stream_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --delivery-mode visual_stream \
  --max-frames 4 \
  --seed 20260604 \
  --output runs/null_vs_stream/visual.json
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
  --delivery-mode visual_stream \
  --seed 20260604 \
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

To generate low-level visual controls:

```bash
python3 scripts/generate_control_frames.py \
  --kind voronoi \
  --output runs/controls/voronoi_seed_7 \
  --width 320 \
  --height 240 \
  --total-frames 50 \
  --fps 1 \
  --seed 7 \
  --overwrite

python3 scripts/generate_control_frames.py \
  --kind phase_scrambled \
  --source-manifest runs/smoke/mandelbrot_a/manifest.json \
  --output runs/controls/mandelbrot_phase_scrambled_seed_7 \
  --seed 7 \
  --overwrite
```

To create the first Julia comparison stimulus:

```bash
python3 scripts/run_julia_smoke.py --overwrite
```

To generate Julia frames and immediately run the same MLX smoke probe:

```bash
python3 scripts/run_julia_smoke.py --overwrite --run-mlx
```

To compare matched Mandelbrot and Julia run JSONs:

```bash
python3 scripts/compare_runs.py \
  runs/compare_seed_20260604/mandelbrot_mlx.json \
  runs/compare_seed_20260604/julia_mlx.json \
  --output-md runs/compare_seed_20260604/mandelbrot_vs_julia.md \
  --output-json runs/compare_seed_20260604/mandelbrot_vs_julia.json
```

The same comparison command can compare `probe_only` vs `visual_stream`, or
`text_only_stream` vs `visual_stream`, to keep the first claim boundary focused
on Null-vs-Stream before fractal-family effects.

To run the first seeded 50-frame null/fractal batch, use one generated blank
visual null plus Mandelbrot and Julia, three seeds each:

```bash
python3 scripts/run_mlx_null_fractal_batch.py \
  --output-root runs/null_fractal_50_seed_batch \
  --seeds 20260604 20260605 20260606 \
  --frames 50 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --cache-summary-every 10 \
  --cache-summary-max-layers 4 \
  --overwrite
```

To test whether saved cache-summary features separate those conditions:

```bash
python3 scripts/analyze_cache_classifier.py \
  --batch-root runs/null_fractal_50_seed_batch \
  --output-md runs/null_fractal_50_seed_batch/cache_classifier.md \
  --output-json runs/null_fractal_50_seed_batch/cache_classifier.json
```

For an unseeded output-drift smoke, keep stream turns greedy while sampling only
the probes:

```bash
python3 scripts/run_mlx_stream_probe.py \
  --manifest runs/null_fractal_50_seed_batch/stimuli/julia/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-frames 12 \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-max-tokens 80 \
  --output runs/unseeded_output_smoke/julia_probe_temp_0_7.json
```

To move from unpaired sampled text toward a paired stochastic design:

```bash
python3 scripts/run_mlx_paired_stochastic_probe_batch.py \
  --output-root runs/paired_stochastic_probe_smoke \
  --probe-seeds 0 1 2 \
  --frames 12 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-max-tokens 80 \
  --probe-cache-policy isolated \
  --overwrite
```

## Documentation

- [Experiment Design](docs/experiment_design.md)
- [Comparison Axes](docs/comparison_axes.md)
- [Control Stimuli](docs/control_stimuli.md)
- [Provider Tiers](docs/provider_tiers.md)
- [Roadmap](docs/roadmap.md)
- [Research Note 0002: Null vs Stream Smoke](docs/research_notes/0002_null_vs_stream_smoke.md)
- [Research Note 0003: Cache Summary Condition Classifier](docs/research_notes/0003_cache_summary_condition_classifier.md)
- [Research Note 0004: Unseeded Probe-Temperature Smoke](docs/research_notes/0004_unseeded_probe_temperature_smoke.md)
- [Research Note 0005: Paired Stochastic Probe Design](docs/research_notes/0005_paired_stochastic_probe_design.md)
- [Research Note 0006: Paired Stochastic Probe Smoke](docs/research_notes/0006_paired_stochastic_probe_smoke.md)

## Claim Boundary

A single run can support "observed in this run" only. Repeated seeds, models,
stimulus families, probes, and controls are required before describing an effect
as stable. The strongest early result is not a grand claim; it is a clean
artifact trail that makes the next run easier to trust.
