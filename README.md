# Fractal VLM State Probe

Deterministic visual streams, factorial controls, and cache interventions for
separating visible multimodal readout from the internal state geometry that
precedes it.

This project starts from a simple spiral-shaped hunch: a visual context is not
just an image, and a model's response is not just a final string. The project
began by asking whether controlled visual streams change later non-visual
generation. The evidence has made the current question more precise:

> When visible answers remain unchanged, which input-conditioned differences
> persist in traced multimodal state, where do they appear, and which of them
> can affect later readout under direct intervention?

The repo is deliberately cautious. It does not claim that a model has subjective
experience, enters a mental state, or undergoes adaptation in the human sense.
It measures reproducible stimuli, observable outputs, logprob proxies, and local
tensor/cache traces.

## Research Shape

The first program compares stimulus conditions within the same model, not
providers against each other. The current program keeps four measured objects
separate:

1. the transformed input and processor-space perturbation,
2. the visible label or generated-token readout,
3. the geometry of saved cache summaries,
4. the readout effect of direct cache intervention.

The foundational question is still simpler than "which fractal is different?":

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

The visible readout and the measured object have separated. In the paired
SmolVLM runs, creative probes and forced-choice labels can remain identical
across conditions while sampled source-cache summaries still move. The current
signal is therefore not "the model says Mandelbrot instead of Julia"; it is
"the readout label is fixed, but the traced context-state geometry changes."

The cross-family palette controls make the interpretation stricter. Swapping
frame-level RGB pixel multisets by luminance rank did not simply remove a
palette confound. It exposed a nonlinear interaction among palette donor,
spatial luminance-rank field, and processor-space frequency structure. Macro
geometry alone is now too coarse as the main story; the working hypothesis is
distribution-coupled visual perturbation of persistent multimodal state.

The intervention and cross-model results sharpen that story. In two SmolVLM
source pairs, single-layer `values` swaps did not steer generated tokens toward
the donor. Dense seed-0 scans over layers 8-23 produced highly similar
susceptibility profiles across the two pairs (`r=0.964`, `rho=0.982`), with the
same peak at layer 10 and the same top three layers: 10, 13, and 12. The repeated
layer 23 summary locus remained weak under direct replacement. In one-frame
Qwen2.5-VL pilots, the surface and first-token top-k10 readouts again stayed
fixed while all-layer cache contrasts repeated at late layer 33 `values`.
Summary-stat salience, readout sensitivity, and causal leverage are now treated
as separate objects.

## Current Restart Point

For a fresh read, start with `docs/experiment_design.md`, then
`docs/research_notes/0002_null_vs_stream_smoke.md`, then
`docs/research_notes/0006_paired_stochastic_probe_smoke.md`, then
`docs/research_notes/0007_paired_stochastic_probe_50_seed.md`, then
`docs/research_notes/0008_pattern_probe_smoke.md`, then
`docs/research_notes/0009_stimulus_seed_8_variant_smoke.md`, then
`docs/research_notes/0010_phase_scramble_image_stats.md`, then
`docs/research_notes/0011_quantile_matched_phase_scramble.md`, then
`docs/research_notes/0012_frequency_ablation_smoke.md`, then
`docs/research_notes/0013_frequency_cutoff_sweep.md`, then
`docs/research_notes/0014_image_cache_correlation.md`, then
`docs/research_notes/0015_source_variant_followups.md`, then
`docs/research_notes/0016_five_step_instrumentation_kickoff.md`, then
`docs/research_notes/0017_cross_palette_control_smoke.md`, then
`docs/research_notes/0018_cross_palette_replication_path.md`, then
`docs/research_notes/0019_cross_palette_replication_readout.md`, then
`docs/research_notes/0020_true_50_frame_cross_palette_replication.md`, then
`docs/research_notes/0021_layer23_values_swap_intervention_scaffold.md`, then
`docs/research_notes/0022_two_pair_values_swap_intervention.md`, then
`docs/research_notes/0023_qwen_cross_model_factorial_pilot.md`, then
`docs/research_notes/0024_dense_mid_layer_values_swap_profile.md`. The compact
paper-facing evidence index is `docs/paper_evidence_matrix.md`.

The first true 50-frame cross-palette replication kept all `MM/JJ/MJ/JM` cells
at 50 frames for two source pairs. Surface forced-choice labels stayed fixed,
while scalar cache-summary interaction argmax locations repeated at mid layer 23
`values` and after layer 0 `keys`. Processor-space image-stat interactions
remained pair-dependent, so the current interpretation is a replicated
state-geometry locus, not a single scalar image-stat mechanism. A top-k20
readout rerun then found identical first-token top-20 sets across `MM/JJ/MJ/JM`
for every phase/probe record, with max common-token interaction around `0.004`
logprob.

The targeted intervention is now a two-pair, four-layer, three-seed result.
Every source/donor swap preserved the origin's generated token sequence. Layer
12 produced the largest tested top-k perturbation in both pairs, about `22%` of
source/donor baseline separation, while layer 23 produced only `3.6-5.4%` and
remained strongly origin-like. The matched self-swap sham effect was zero. A
follow-up seed-0 screen over layers 8-23 repeated the same profile shape across
both source pairs (`r=0.964`, `rho=0.982`), peaking at layer 10 with layers 13
and 12 next. Strict multi-seed reciprocal and sham confirmation of those three
layers is the next intervention step.

The first separate-architecture lane is also live. Two one-frame Qwen2.5-VL
four-cell runs kept labels and first-token top-k10 identical while the full
36-layer cache interaction argmax repeated at layer 33 `values`, position 128.
Qwen's current MLX-VLM `PromptCacheState` path fails on a second image turn, so
this remains a one-frame cross-model trace pilot, not a persistent-stream
replication. Full-vocabulary scoring, confirmed mid-layer interventions, and an
explicit cumulative-replay lane are next.

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
- Run paired stochastic-probe batches over selected fractal, geometric, and
  low-level control patterns.
- Transform existing manifests into phase-scrambled, quantile-matched
  phase-scrambled, low/high-pass frequency ablations, static-repeat, shuffled,
  and reversed-order controls.
- Transform one manifest into the frame-aligned RGB pixel distribution of
  another manifest for cross-family palette controls.
- Analyze image statistics after a model processor converts frames into
  `pixel_values`, including cycles-per-patch spectral summaries.
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
- Analyze 2x2 cache-summary factorial contrasts for spatial main effect,
  palette main effect, and spatial-by-palette interaction.
- Prepare replicated 2x2 cross-palette batches and analyze the same factorial
  contrast over raw and processor-space image statistics.
- Compare saved HF first-step top-k logprobs for probe readout deltas when
  generation score summaries are available.
- Run a targeted MLX `PromptCacheState` values-swap probe that replaces only one
  source cache layer's `values` tensor with the donor tensor before a creative
  branch probe.
- Sweep targeted values swaps over layers and matched probe seeds with
  reciprocal branches, self-swap shams, token/edit comparisons, top-k RMSE, and
  normalized donor-pull summaries.
- Compare dense intervention layer profiles across source pairs, including
  Pearson/Spearman similarity, peak layers, and top-k overlap.
- Reuse one loaded MLX model across manifest-batch conditions while creating a
  fresh prompt-cache state for every run.
- Promote MLX cache tensors to float32 for summary reductions, avoiding
  float16 variance/L2 overflow on Qwen-class caches.
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

python3 scripts/generate_control_frames.py \
  --kind phase_scrambled_luminance_quantile_matched \
  --source-manifest runs/smoke/mandelbrot_a/manifest.json \
  --output runs/controls/mandelbrot_phase_luminance_quantile_seed_7 \
  --seed 7 \
  --overwrite

python3 scripts/generate_control_frames.py \
  --kind cross_palette_luminance_matched \
  --source-manifest runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --palette-manifest runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json \
  --condition-id julia_d_spatial_mandelbrot_c_palette \
  --output runs/controls/julia_d_spatial_mandelbrot_c_palette \
  --max-frames 50 \
  --overwrite
```

To audit what reaches the VLM after processor resize/normalization:

```bash
python3 scripts/analyze_processor_image_stats.py \
  --manifest runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json \
  --manifest runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --patch-size 14 \
  --max-frames 50 \
  --output-json runs/processor_stats/mandelbrot_c_julia_d.json \
  --output-md runs/processor_stats/mandelbrot_c_julia_d.md
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

To move into non-fractal pattern controls with the same paired probe design:

```bash
python3 scripts/run_mlx_pattern_probe_batch.py \
  --output-root runs/pattern_probe_smoke \
  --conditions null_blank mandelbrot julia checkerboard voronoi quasicrystal white_noise blue_noise \
  --probe-seeds 0 1 2 \
  --frames 12 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-max-tokens 80 \
  --probe-cache-policy isolated \
  --overwrite
```

To run the same paired probe design over arbitrary existing manifests:

```bash
python3 scripts/run_mlx_manifest_probe_batch.py \
  --output-root runs/frequency_cutoff_sweep_probe_seed_0 \
  --manifest mandelbrot_low_018=runs/frequency_ablation_smoke/mandelbrot_low_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest mandelbrot_high_018=runs/frequency_ablation_smoke/mandelbrot_high_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest julia_low_018=runs/frequency_ablation_smoke/julia_low_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest julia_high_018=runs/frequency_ablation_smoke/julia_high_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --probe-seeds 0 \
  --max-frames 12 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-cache-policy isolated
```

To prepare replicated cross-palette factorial source pairs before launching MLX
inference:

```bash
python3 scripts/prepare_cross_palette_factorial_batch.py \
  --output-root runs/cross_palette_replications \
  --pair c_d=runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json,runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --pair b_c=runs/source_variant_smoke/stimuli/mandelbrot_b/manifest.json,runs/source_variant_smoke/stimuli/julia_c/manifest.json \
  --max-frames 50 \
  --processor-model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --patch-size 14 \
  --overwrite
```

To analyze a saved raw or processor image-stat batch in the same 2x2 coordinate
system as cache summaries:

```bash
python3 scripts/analyze_factorial_image_contrast.py \
  --stats-json runs/cross_palette_controls/cross_palette_processor_image_stats.json \
  --mm mandelbrot_zoom_c \
  --jj julia_zoom_d \
  --mj mandelbrot_c_spatial_julia_d_palette \
  --jm julia_d_spatial_mandelbrot_c_palette \
  --output-json runs/cross_palette_controls/processor_factorial_image_contrast.json \
  --output-md runs/cross_palette_controls/processor_factorial_image_contrast.md
```

To analyze saved first generated-token top-k readouts from a rerun that includes
`generation.steps`:

```bash
python3 scripts/analyze_probe_readout_contrast.py \
  --mm runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/mm_mlx.json \
  --jj runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/jj_mlx.json \
  --mj runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/mj_mlx.json \
  --jm runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/jm_mlx.json \
  --output-json runs/cross_palette_replication_50_v1/c_d_50f/probe_readout_contrast.json \
  --output-md runs/cross_palette_replication_50_v1/c_d_50f/probe_readout_contrast.md
```

To run the first pinpoint cache intervention scaffold, swap donor layer 23
`values` into a source stream cache before a creative probe:

```bash
python3 scripts/run_mlx_cache_values_swap_probe.py \
  --source-manifest runs/cross_palette_replication_50_v1/stimuli/mandelbrot_zoom_c_50f/manifest.json \
  --donor-manifest runs/cross_palette_replication_50_v1/stimuli/julia_zoom_d_50f/manifest.json \
  --source-label mandelbrot_c \
  --donor-label julia_d \
  --output runs/cache_values_swap/c_d_layer23_values_mid.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-frames 50 \
  --probe-phase mid \
  --layer-index 23 \
  --probe-temperature 0.7 \
  --probe-seed 0 \
  --generation-readout-top-k 20
```

To sweep target and control layers over matched seeds, including reciprocal and
self-swap sham branches, then aggregate token and top-k effects:

```bash
python3 scripts/run_mlx_cache_values_swap_sweep.py \
  --source-manifest runs/cross_palette_replication_50_v1/stimuli/mandelbrot_zoom_c_50f/manifest.json \
  --donor-manifest runs/cross_palette_replication_50_v1/stimuli/julia_zoom_d_50f/manifest.json \
  --source-label mandelbrot_c \
  --donor-label julia_d \
  --output runs/cache_values_swap/c_d_mid_values_sweep.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-frames 50 \
  --probe-phase mid \
  --layer-index 0 \
  --layer-index 12 \
  --layer-index 22 \
  --layer-index 23 \
  --probe-seeds 0 1 2 \
  --generation-readout-top-k 50

python3 scripts/analyze_cache_interventions.py \
  --run runs/cache_values_swap/c_d_mid_values_sweep.json \
  --output-json runs/cache_values_swap/c_d_mid_values_sweep_analysis.json \
  --output-md runs/cache_values_swap/c_d_mid_values_sweep_analysis.md

python3 scripts/analyze_cache_intervention_profiles.py \
  --analysis c_d=runs/cache_values_swap/c_d_mid_values_dense_l8_23_seed0_screen_analysis.json \
  --analysis b_c=runs/cache_values_swap/b_c_mid_values_dense_l8_23_seed0_screen_analysis.json \
  --output-json runs/cache_values_swap/dense_l8_23_seed0_profile_comparison.json \
  --output-md runs/cache_values_swap/dense_l8_23_seed0_profile_comparison.md
```

## Documentation

- [Experiment Design](docs/experiment_design.md)
- [Comparison Axes](docs/comparison_axes.md)
- [Control Stimuli](docs/control_stimuli.md)
- [Provider Tiers](docs/provider_tiers.md)
- [Roadmap](docs/roadmap.md)
- [Paper Evidence Matrix](docs/paper_evidence_matrix.md)
- [Examples](examples/README.md)
- [Research Note 0002: Null vs Stream Smoke](docs/research_notes/0002_null_vs_stream_smoke.md)
- [Research Note 0003: Cache Summary Condition Classifier](docs/research_notes/0003_cache_summary_condition_classifier.md)
- [Research Note 0004: Unseeded Probe-Temperature Smoke](docs/research_notes/0004_unseeded_probe_temperature_smoke.md)
- [Research Note 0005: Paired Stochastic Probe Design](docs/research_notes/0005_paired_stochastic_probe_design.md)
- [Research Note 0006: Paired Stochastic Probe Smoke](docs/research_notes/0006_paired_stochastic_probe_smoke.md)
- [Research Note 0007: Paired Stochastic Probe 50-Seed Batch](docs/research_notes/0007_paired_stochastic_probe_50_seed.md)
- [Research Note 0008: Pattern Probe Smoke](docs/research_notes/0008_pattern_probe_smoke.md)
- [Research Note 0009: Stimulus Seed 8 Variant Smoke](docs/research_notes/0009_stimulus_seed_8_variant_smoke.md)
- [Research Note 0010: Phase Scramble and Image Statistics Smoke](docs/research_notes/0010_phase_scramble_image_stats.md)
- [Research Note 0011: Quantile-Matched Phase Scramble Controls](docs/research_notes/0011_quantile_matched_phase_scramble.md)
- [Research Note 0012: Frequency Ablation Smoke](docs/research_notes/0012_frequency_ablation_smoke.md)
- [Research Note 0013: Frequency Cutoff Sweep](docs/research_notes/0013_frequency_cutoff_sweep.md)
- [Research Note 0014: Image-Stat / Cache-Distance Correlation](docs/research_notes/0014_image_cache_correlation.md)
- [Research Note 0015: Source Variant Follow-Up Probes](docs/research_notes/0015_source_variant_followups.md)
- [Research Note 0016: Five-Step Instrumentation Kickoff](docs/research_notes/0016_five_step_instrumentation_kickoff.md)
- [Research Note 0017: Cross-Palette Control Smoke](docs/research_notes/0017_cross_palette_control_smoke.md)
- [Research Note 0018: Cross-Palette Replication Path](docs/research_notes/0018_cross_palette_replication_path.md)
- [Research Note 0019: Cross-Palette Replication Readout](docs/research_notes/0019_cross_palette_replication_readout.md)
- [Research Note 0020: True 50-Frame Cross-Palette Replication](docs/research_notes/0020_true_50_frame_cross_palette_replication.md)
- [Research Note 0021: Layer 23 Values-Swap Intervention Scaffold](docs/research_notes/0021_layer23_values_swap_intervention_scaffold.md)
- [Research Note 0022: Two-Pair Values-Swap Intervention](docs/research_notes/0022_two_pair_values_swap_intervention.md)
- [Research Note 0023: Qwen Cross-Model Factorial Pilot](docs/research_notes/0023_qwen_cross_model_factorial_pilot.md)
- [Research Note 0024: Dense Mid-Layer Values-Swap Profile](docs/research_notes/0024_dense_mid_layer_values_swap_profile.md)

## Claim Boundary

A single run can support "observed in this run" only. Repeated seeds, models,
stimulus families, probes, and controls are required before describing an effect
as stable. The strongest early result is not a grand claim; it is a clean
artifact trail that makes the next run easier to trust.
