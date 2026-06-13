# Frequency Cutoff Sweep

Date: 2026-06-08

## Setup

- Source streams:
  - `runs/pattern_probe_smoke/stimuli/mandelbrot/manifest.json`
  - `runs/pattern_probe_smoke/stimuli/julia/manifest.json`
- Controls:
  - `low_pass_luminance_quantile_matched`
  - `high_pass_luminance_quantile_matched`
- Cutoffs: `0.10`, `0.18`, `0.28`, `0.40`
- Frames per manifest: `12`
- Image-stat output: `runs/image_stats/frequency_cutoff_sweep_stats.md/json`
- MLX batch output: `runs/frequency_cutoff_sweep_probe_seed_0`
- Runner: `scripts/run_mlx_manifest_probe_batch.py`
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_seed=0`
- Probe cache policy: `isolated`

This note extends
[0012 Frequency Ablation Smoke](0012_frequency_ablation_smoke.md). The goal is
to test whether the low/high frequency read at cutoff `0.18` is a local accident
or a broader pattern across cutoffs.

## Manifest Batch Runner

This run uses a new arbitrary manifest batch runner. It accepts repeated
`--manifest KEY=PATH` arguments, runs the same MLX stream probe for each key,
writes all pairwise comparisons per probe seed, and produces
`manifest_batch_summary.md/json` plus `paired_stochastic_analysis.md/json`.

The important operational detail is resume behavior: if a run JSON already
exists, the runner reuses it unless `--overwrite` is passed. That makes long
transformed-control sweeps much less brittle.

## Image-Side Sweep

Each transformed condition keeps the source luminance mean, luminance std,
entropy, and colorfulness fixed. The cutoff sweep changes the rank map used to
reassign the source pixels.

Key aggregate rows:

| Condition | Edge density | HF ratio | Spectral centroid |
| --- | ---: | ---: | ---: |
| `mandelbrot_zoom_a` | `0.057` | `0.048` | `0.044` |
| `julia_zoom_b` | `0.278` | `0.428` | `0.321` |
| `mandelbrot_low_lq_cutoff_010` | `0.020` | `0.002` | `0.014` |
| `mandelbrot_high_lq_cutoff_010` | `0.263` | `0.143` | `0.176` |
| `julia_low_lq_cutoff_010` | `0.146` | `0.027` | `0.070` |
| `julia_high_lq_cutoff_010` | `0.357` | `0.513` | `0.390` |
| `mandelbrot_low_lq_cutoff_018` | `0.028` | `0.003` | `0.016` |
| `mandelbrot_high_lq_cutoff_018` | `0.408` | `0.220` | `0.246` |
| `julia_low_lq_cutoff_018` | `0.239` | `0.043` | `0.107` |
| `julia_high_lq_cutoff_018` | `0.443` | `0.577` | `0.432` |
| `mandelbrot_low_lq_cutoff_028` | `0.038` | `0.005` | `0.020` |
| `mandelbrot_high_lq_cutoff_028` | `0.560` | `0.301` | `0.322` |
| `julia_low_lq_cutoff_028` | `0.284` | `0.057` | `0.143` |
| `julia_high_lq_cutoff_028` | `0.522` | `0.698` | `0.483` |
| `mandelbrot_low_lq_cutoff_040` | `0.050` | `0.014` | `0.025` |
| `mandelbrot_high_lq_cutoff_040` | `0.639` | `0.764` | `0.396` |
| `julia_low_lq_cutoff_040` | `0.302` | `0.147` | `0.182` |
| `julia_high_lq_cutoff_040` | `0.559` | `0.905` | `0.548` |

The controls move in the expected direction: low-pass controls stay low on
high-frequency energy, while high-pass controls increase HF ratio and edge
density as the cutoff rises.

## MLX Core Comparisons

Surface probe text stayed identical across all core comparisons. The table is
again a sampled source-cache summary read.

| Cutoff | Comparison | Mid max L2 | After max L2 | Mid mean L2 | After mean L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `0.10` | Mandelbrot low vs Julia low | `10.939` | `10.081` | `4.649` | `3.652` |
| `0.10` | Mandelbrot high vs Julia high | `5.639` | `9.677` | `3.017` | `2.449` |
| `0.10` | Mandelbrot low vs high | `14.414` | `18.436` | `5.296` | `5.734` |
| `0.10` | Julia low vs high | `6.496` | `2.944` | `2.160` | `1.218` |
| `0.18` | Mandelbrot low vs Julia low | `8.665` | `11.409` | `3.266` | `3.703` |
| `0.18` | Mandelbrot high vs Julia high | `2.488` | `3.444` | `1.073` | `1.260` |
| `0.18` | Mandelbrot low vs high | `11.809` | `11.113` | `4.449` | `3.602` |
| `0.18` | Julia low vs high | `8.814` | `2.093` | `2.726` | `0.750` |
| `0.28` | Mandelbrot low vs Julia low | `5.580` | `12.341` | `2.854` | `3.737` |
| `0.28` | Mandelbrot high vs Julia high | `5.356` | `4.068` | `2.038` | `1.406` |
| `0.28` | Mandelbrot low vs high | `10.378` | `7.937` | `5.302` | `3.417` |
| `0.28` | Julia low vs high | `3.153` | `4.942` | `1.321` | `1.668` |
| `0.40` | Mandelbrot low vs Julia low | `11.939` | `9.994` | `4.294` | `3.380` |
| `0.40` | Mandelbrot high vs Julia high | `7.662` | `4.206` | `2.375` | `1.030` |
| `0.40` | Mandelbrot low vs high | `17.228` | `7.574` | `6.542` | `3.127` |
| `0.40` | Julia low vs high | `3.761` | `4.106` | `1.444` | `1.687` |

## First Reading

The earlier `0.18` observation mostly survives, but with an important cutoff
dependence:

- Cross-family low-pass controls stay separated after streaming across all
  cutoffs tested: after max L2 is about `10.0` to `12.3`.
- Cross-family high-pass controls are much closer around `0.18` and `0.28`,
  especially after streaming: `3.444` and `4.068`.
- The high-pass collapse is not monotonic. At `0.10`, the high-pass condition
  retains more mid/low structure and the after distance is `9.677`; at `0.40`,
  it becomes very high-frequency and the mid distance rises to `7.662`.
- Mandelbrot low-vs-high remains large across cutoffs, while Julia low-vs-high
  is smaller after streaming, especially at `0.18`.

The careful claim is:

> In this single-probe-seed cutoff sweep, entropy-fixed high-pass controls make
> Mandelbrot and Julia substantially closer than entropy-fixed low-pass
> controls near the middle cutoffs, while low-pass controls preserve stronger
> cross-family separation. The effect is cutoff-dependent rather than a simple
> monotonic high-frequency story.

This supports a more precise version of the previous hypothesis: the
Mandelbrot/Julia separation appears to be carried more by low/mid-frequency
spatial organization than by high-frequency detail alone, but the phase-control
divergence is still likely a compound effect.

## Caveats

- One probe seed only.
- One model only.
- One source stream per family.
- Pairwise surface text is seed-locked; the signal here is in sampled cache
  summaries.
- The cutoff transform is deterministic, so transform-seed aggregation does not
  apply to low/high LQ directly; replication should vary source streams,
  cutoffs, probe seeds, and models.

## Next Steps

1. Add a compact analysis script that correlates cache-summary distances with
   image-stat deltas across all comparisons.
2. Repeat the cutoff sweep with more probe seeds.
3. Add source-stream variants for Mandelbrot and Julia, not only transform
   variants of one stream.
4. Use the manifest batch runner for phase-scramble seeds and natural controls.

## Local Artifacts

- `runs/image_stats/frequency_cutoff_sweep_stats.md`
- `runs/image_stats/frequency_cutoff_sweep_stats.json`
- `runs/frequency_cutoff_sweep_probe_seed_0/manifest_batch_summary.md`
- `runs/frequency_cutoff_sweep_probe_seed_0/manifest_batch_summary.json`
- `runs/frequency_cutoff_sweep_probe_seed_0/comparisons/cutoff_sweep_core_comparison_summary.md`
- `runs/frequency_cutoff_sweep_probe_seed_0/comparisons/cutoff_sweep_core_comparison_summary.json`

## Tracked Examples

- [Frequency cutoff sweep image statistics JSON](../../examples/research_notes/0013_frequency_cutoff_sweep/frequency_cutoff_sweep_stats.json)
- [Frequency cutoff sweep core comparison JSON](../../examples/research_notes/0013_frequency_cutoff_sweep/cutoff_sweep_core_comparison_summary.json)
- [Manifest batch summary JSON](../../examples/research_notes/0013_frequency_cutoff_sweep/manifest_batch_summary.json)
