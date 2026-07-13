# Fractal VLM State Probe

Deterministic visual controls, audited multimodal protocols, full-vocabulary
factorial readouts, and full-vector source-cache contrasts for asking what a
VLM changes before a generated label makes that change visible.

This project began by asking whether controlled visual streams change later
non-visual generation. A cache-prefix audit changed both the protocol and the
question. The current target is narrower and directly measurable:

> Under fresh multimodal forwards, how do controlled input transformations
> decompose into spatial, palette, and interaction effects across source-cache
> token regions and prompt-conditioned full-vocabulary readouts, and which
> locations, magnitudes, or directions repeat across models, source pairs, and
> probe formulations?

The main measured object is no longer a generated label or one scalar cache
distance. It is now the aligned 2x2 contrast itself: first over complete
readout distributions, and, at selected cache targets, over the full tensor
split into image, pre-image, and post-image token regions.

The repo is deliberately cautious. It does not claim that a model has subjective
experience, enters a mental state, or undergoes adaptation in the human sense.
It measures reproducible stimuli, observable outputs, logprob proxies, and local
tensor/cache traces.

## Research Shape

The program compares controlled stimulus conditions within the same model and
keeps four objects separate:

1. the transformed input and processor-space perturbation,
2. the complete first-step readout distribution from a fresh direct probe,
3. the cache summary or selected full tensor from a separate fresh multimodal
   ACK forward,
4. whether any cache reuse path preserves the full multimodal prefix and has a
   cache sequence length compatible with its token history.

Every condition is represented as a manifest with frame timestamps, relative
paths, hashes, and `stimulus_condition` metadata. A run contributes to the
current evidence set only when its protocol boundary is explicit.

## Current Evidence

The July 2026 audit found that MLX-VLM `0.4.4` text-history reconstruction does
not retain processor-inserted image tokens. In the audited incremental and
text-only branch paths, the common token prefix was incomplete and cache tensor
length did not match token-history length. The earlier 25/50-frame persistence,
branched readout, and cache-swap intervention claims are therefore withdrawn
pending a verified multimodal suffix path.

The replacement protocol uses no image-conditioned cache reuse:

1. Run one fresh ordered multi-image ACK forward and summarize its source cache.
2. Run each forced-choice probe as a separate fresh multimodal forward over the
   same ordered images.
3. Save the complete first-step vocabulary distribution, not only a generated
   letter or top-k slice.

The current valid standard matrix contains 30 `MM/JJ/MJ/JM` factorial points,
120 cell runs, and 480 compressed full-vocabulary sidecars across SmolVLM2,
Qwen2.5-VL, Gemma 3, and InternVL3. Every direct after-factorial contains
non-identical cell distributions. At one frame, the independent replication
surface is now four source pairs x four models.

- Qwen repeats a late layer 33 `values` source-cache summary locus with a
  negative interaction at all four independent one-frame source pairs and all
  12 tested pair-by-length points. Its full-vocabulary interaction still
  changes non-monotonically over 1/2/4/8/16 images.
- InternVL3 places all four one-frame maxima in a late layer 25-27 `values`
  band with a negative interaction. The component, sign, and normalized depth
  repeat, while the exact layer does not.
- SmolVLM's one-frame source-cache argmax spans layers 1/21/22 and both keys and
  values across four pairs; the old persistent layer-23 story does not
  replicate.
- Gemma 3 stays in early `values` components across all four one-frame pairs,
  but its exact layer and interaction sign are pair-dependent. Its standard
  family readout is nearly saturated while its frequency interaction is much
  larger and more variable.

A separate three-model prompt audit adds 192 sidecars over four semantically
aligned probe variants. All 48 baseline sidecars reproduce byte-for-byte, but
candidate order, paraphrase, and label remapping change generated semantics and
candidate distributions in architecture-specific ways. The direct readout is
therefore a measurement of visual evidence combined with prompt calibration,
not a prompt-invariant extraction of what the model "sees."

The first full-vector follow-up adds 32 source-only ACK runs and 64 target
tensor sidecars over Qwen layer 33 `values` and InternVL layers 25-27 `values`.
Across all 16 layer-by-pair factorial analyses, the interaction argmax lies in
the image-token region, the shared pre-image prefix is exactly unchanged, and
more than 99.1% of interaction energy lies in image tokens. Those high-energy
image directions are only weakly aligned across source pairs, while the much
smaller post-image effect is more directionally aligned. A repeatable scalar
locus therefore does not imply one shared underlying vector direction.

The cross-palette input result remains intact: luminance-rank palette transfer
creates a nonlinear interaction among palette donor, spatial rank field, and
processor-space frequency structure. The current reading is therefore
distribution-coupled visual perturbation under fresh multimodal inference, not
yet persistent latent-state steering.

The repo does **not** currently support a universal layer, persistent
multi-turn state, a valid cache intervention effect, causal mediation from the
ACK cache to the direct probe, a prompt-invariant categorical visual readout,
or full-distribution equality inferred from an unchanged generated label.

## Start Here

1. [Note 0030](docs/research_notes/0030_full_vector_cache_factorials.md)
   for the newest full-vector 2x2 cache result and updated research object.
2. [Note 0029](docs/research_notes/0029_cross_model_prompt_and_internvl_expansion.md)
   for the three-model prompt audit and InternVL expansion.
3. [Note 0028](docs/research_notes/0028_source_pair_replication_and_prompt_robustness.md)
   for the four-pair replication and first prompt audit.
4. [Note 0027](docs/research_notes/0027_cache_prefix_audit_and_direct_full_vocab.md)
   for the protocol audit that defines the valid fresh-forward boundary.
5. [Paper Evidence Matrix](docs/paper_evidence_matrix.md) for the compact
   supported/provisional/withdrawn map.
6. [Experiment Design](docs/experiment_design.md) for the control ladder.
7. [Note 0020](docs/research_notes/0020_true_50_frame_cross_palette_replication.md)
   for the still-valid input and processor-space cross-palette analysis.
8. [Examples](examples/README.md) for tracked summaries and the historical note
   sequence.

Relevant historical cross-palette and intervention notes now carry
protocol-audit banners where their cache or branched readout interpretations
were superseded.

## Infrastructure Tiers

- **T1 Local Internal Trace**: MLX-VLM is the primary measured path; a Hugging
  Face Transformers adapter is also available. This tier covers hidden-state,
  attention, and KV/cache summaries where the model exposes them, using either
  `model_id` or `local_path`.
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
- Attach condition metadata and build manifests for external natural or
  geometric frame directories.
- Run a fresh ordered multi-image ACK forward plus fresh direct multimodal
  probes, without reusing image-conditioned cache state.
- Save compressed full-vocabulary first-step logprobs with SHA-256, vocabulary
  size, dtype, and log-normalization diagnostics.
- Compare complete readout distributions with KL, Jensen-Shannon, total
  variation, Hellinger, conditional candidate probabilities, and
  probability-space 2x2 factorial contrasts.
- Run forced-choice paraphrase, candidate-order, and label-remapping controls;
  align candidate probabilities by declared semantics before comparing prompt
  variants.
- Aggregate semantic prompt-robustness audits across models while keeping
  generated-label stability and probability-surface stability separate.
- Record sampled KV-cache summaries over all layers and map local argmaxes to
  image-token runs and vision markers.
- Save selected source-cache tensors as offset-trimmed float32 sidecars with
  shape, token-count, and SHA-256 integrity metadata.
- Compute full-vector spatial, palette, and interaction contrasts over image,
  pre-image, post-image, and complete effective-cache regions; aggregate
  model-local directions across independent source pairs.
- Analyze cache-summary spatial, palette, and interaction contrasts; track
  loci by model, source pair, replay length, tensor, and normalized depth.
- Aggregate independent source-pair replications across models while reporting
  exact-layer, K/V-component, and interaction-sign stability separately.
- Audit reconstructed prompt prefixes and cache sequence lengths before any
  `PromptCacheState` reuse is interpreted.
- Prepare replicated 2x2 cross-palette batches and analyze the same factorial
  contrast over raw and processor-space image statistics.
- Reuse one loaded MLX model across manifest-batch conditions while creating a
  fresh prompt-cache state for every run.
- Adapt InternVL's custom MLX image expansion and string-oriented chat template
  when the installed Transformers processor path cannot construct them.
- Promote MLX cache tensors to float32 for summary reductions, avoiding
  low-precision variance/L2 overflow, and promote unsupported bfloat16
  logprobs before saving.
- Run paired stochastic probes and compare paired run JSONs for earlier
  behavior-level experiments.
- Run Hugging Face Transformers probes with `model_id` or `local_path` where a
  selected model exposes compatible hidden-state, cache, or logprob outputs.
- Train a small nearest-centroid classifier on saved cache-summary features to
  test whether measured traces retain condition information when probe text is
  unchanged.
- Retain legacy incremental-stream, cache-branch, and values-swap CLIs for
  protocol forensics. Under MLX-VLM `0.4.4`, their image-conditioned reuse path
  is audit-only and does not support persistence or intervention claims.

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
python3 scripts/run_mlx_cumulative_replay_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --output runs/smoke/mandelbrot_a_direct_4f.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-frames 4 \
  --probe-preset forced_choice \
  --after-probe-protocol direct_multimodal_replay \
  --save-full-vocab-first-step \
  --cache-summary-max-layers -1 \
  --no-frame-artifacts
```

This writes a fresh ACK source-cache summary and fresh direct multimodal probe
records. It does not claim that state persisted from the ACK into the probes.

To capture selected source-cache tensors without running any probes:

```bash
python3 scripts/run_mlx_cumulative_replay_probe.py \
  --manifest runs/smoke/mandelbrot_a/manifest.json \
  --output runs/full_vector/mandelbrot_a_mlx.json \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --max-frames 1 \
  --source-cache-only \
  --capture-cache-tensor 33:values \
  --cache-summary-max-layers -1 \
  --no-frame-artifacts
```

After capturing aligned `MM/JJ/MJ/JM` cells, compute one full-vector factorial:

```bash
python3 scripts/analyze_cache_tensor_factorial.py \
  --mm runs/full_vector/mm_mlx.json \
  --jj runs/full_vector/jj_mlx.json \
  --mj runs/full_vector/mj_mlx.json \
  --jm runs/full_vector/jm_mlx.json \
  --layer 33 \
  --tensor values \
  --output-json runs/full_vector/layer33_factorial.json \
  --output-md runs/full_vector/layer33_factorial.md
```

Use `scripts/analyze_cache_tensor_replication.py` with repeated
`--analysis KEY=PATH` arguments to compare same-target directions across
independent source pairs. Raw directions are intentionally not compared across
models or layers.

The original Null-vs-Stream ladder remains available for protocol development.
Keep the seed, probe, frame count, and model fixed while changing only
`--delivery-mode`; treat incremental visual-cache reuse as audit-only until its
prefix and cache-length invariants pass:

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

The following null/fractal, unseeded, and paired-pattern runners reproduce the
historical behavior-level ladder. Their incremental image-cache traces are not
part of the current evidence set.

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

To run the current fresh cumulative/direct protocol over arbitrary existing
manifests:

```bash
python3 scripts/run_mlx_manifest_probe_batch.py \
  --output-root runs/frequency_cutoff_sweep_direct_seed_0 \
  --manifest mandelbrot_low_018=runs/frequency_ablation_smoke/mandelbrot_low_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest mandelbrot_high_018=runs/frequency_ablation_smoke/mandelbrot_high_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest julia_low_018=runs/frequency_ablation_smoke/julia_low_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest julia_high_018=runs/frequency_ablation_smoke/julia_high_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --probe-seeds 0 \
  --max-frames 12 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --context-protocol cumulative_replay \
  --after-probe-protocol direct_multimodal_replay \
  --save-full-vocab-first-step \
  --cache-summary-max-layers -1 \
  --temperature 0 \
  --probe-temperature 0.7 \
  --no-frame-artifacts \
  --overwrite
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

To compare complete first-step distributions from four fresh direct cell runs:

```bash
python3 scripts/analyze_full_vocab_readout_contrast.py \
  --mm runs/direct_factorial/mm_mlx.json \
  --jj runs/direct_factorial/jj_mlx.json \
  --mj runs/direct_factorial/mj_mlx.json \
  --jm runs/direct_factorial/jm_mlx.json \
  --output-json runs/direct_factorial/full_vocab_readout_contrast.json \
  --output-md runs/direct_factorial/full_vocab_readout_contrast.md
```

To aggregate prefix/cache-length invariants from audit runs:

```bash
python3 scripts/analyze_cache_prefix_audits.py \
  --run qwen=runs/cache_prefix_audit/qwen_legacy_branch.json \
  --run smol=runs/cache_prefix_audit/smol_incremental_two_frame.json \
  --output-json runs/cache_prefix_audit/prefix_audit_analysis.json \
  --output-md runs/cache_prefix_audit/prefix_audit_analysis.md
```

To compare semantic-aligned prompt robustness across models:

```bash
python3 scripts/analyze_prompt_robustness_aggregate.py \
  --analysis gemma3=runs/prompt_controls/gemma3/prompt_robustness.json \
  --analysis qwen=runs/prompt_controls/qwen/prompt_robustness.json \
  --analysis smol=runs/prompt_controls/smol/prompt_robustness.json \
  --output-json runs/prompt_controls/cross_model/prompt_robustness_aggregate.json \
  --output-md runs/prompt_controls/cross_model/prompt_robustness_aggregate.md
```

The legacy cache-swap and cache-branch CLIs remain in the repository so the
withdrawn path can be reproduced and audited. Do not treat their current
MLX-VLM `0.4.4` outputs as persistence or intervention evidence.

To run a fresh multi-image ACK plus direct probes and summarize a frame-count
trajectory:

```bash
python3 scripts/run_mlx_cumulative_replay_probe.py \
  --manifest runs/cross_palette_replication_50_v1/stimuli/mandelbrot_zoom_c_50f/manifest.json \
  --output runs/direct_replay/mandelbrot_c_4f.json \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --max-frames 4 \
  --probe-preset forced_choice \
  --after-probe-protocol direct_multimodal_replay \
  --save-full-vocab-first-step \
  --cache-summary-max-layers -1 \
  --no-frame-artifacts

python3 scripts/analyze_factorial_cache_trajectory.py \
  --analysis c_d_1f=runs/full_vocab_factorials/qwen_c_d_1f_direct_all_layers_seed0/factorial_cache_contrast.json \
  --analysis c_d_2f=runs/full_vocab_factorials/qwen_c_d_2f_direct_all_layers_seed0/factorial_cache_contrast.json \
  --analysis c_d_4f=runs/full_vocab_factorials/qwen_c_d_4f_direct_all_layers_seed0/factorial_cache_contrast.json \
  --output-json runs/full_vocab_factorials/qwen_c_d_trajectory.json \
  --output-md runs/full_vocab_factorials/qwen_c_d_trajectory.md
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
- [Research Note 0025: Controlled Mid-Layer Values-Swap Confirmation](docs/research_notes/0025_controlled_mid_layer_values_swap_confirmation.md)
- [Research Note 0026: Qwen Cumulative-Replay Trajectory](docs/research_notes/0026_qwen_cumulative_replay_trajectory.md)
- [Research Note 0027: Cache-Prefix Audit and Direct Full-Vocabulary Replay](docs/research_notes/0027_cache_prefix_audit_and_direct_full_vocab.md)
- [Research Note 0028: Source-Pair Replication and Prompt Robustness](docs/research_notes/0028_source_pair_replication_and_prompt_robustness.md)
- [Research Note 0029: Cross-Model Prompt Audit and InternVL Expansion](docs/research_notes/0029_cross_model_prompt_and_internvl_expansion.md)
- [Research Note 0030: Full-Vector Source-Cache Factorials](docs/research_notes/0030_full_vector_cache_factorials.md)

## Claim Boundary

A single run can support "observed in this run" only. Nested frame lengths are
not independent stimulus replications, and four factorial cells do not turn
their six pairwise distances into six independent samples. An unchanged letter
does not establish equality of output distributions. No persistence or cache
intervention claim is promoted unless full multimodal prefix and cache-length
invariants pass on the exact execution path being interpreted.
