# Paired Stochastic Probe 50-Seed Batch

Date: 2026-06-13

## Setup

- Source batch: `runs/paired_stochastic_probe_50_seed_batch`
- Conditions: `null_blank`, `mandelbrot`, `julia`
- Probe seeds: `0..49`
- Frames per run: `12`
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_max_tokens=80`
- Probe cache policy: `isolated`
- Probe phase seed policy: `before=probe_seed`, `mid=probe_seed+1`,
  `after=probe_seed+2`

This run extends the small paired stochastic-probe smoke from three probe seeds
to fifty probe seeds. It is a probe-seed expansion, not a fifty-frame stream
run.

## Result

The paired seed policy held across the full batch:

| Check | Result |
| --- | ---: |
| Before probes identical across conditions | `50/50` seeds |
| Pairwise lexical condition distance at `mid` | `0.000` |
| Pairwise lexical condition distance at `after` | `0.000` |
| Before-adjusted pairwise lexical distance at `mid` | `0.000` |
| Before-adjusted pairwise lexical distance at `after` | `0.000` |

The sampled probe text still moved strongly from `before` to `mid` and `after`,
but that movement was shared across conditions:

| Condition | Phase | Mean lexical distance from before | Std | Samples |
| --- | --- | ---: | ---: | ---: |
| `julia` | `mid` | `0.826` | `0.047` | `50` |
| `julia` | `after` | `0.827` | `0.046` | `50` |
| `mandelbrot` | `mid` | `0.826` | `0.047` | `50` |
| `mandelbrot` | `after` | `0.827` | `0.046` | `50` |
| `null_blank` | `mid` | `0.826` | `0.047` | `50` |
| `null_blank` | `after` | `0.827` | `0.046` | `50` |

At the same time, source-cache summaries still separated visual conditions:

| Comparison | Mid max L2 delta | After max L2 delta |
| --- | ---: | ---: |
| `null_vs_mandelbrot` | `1.457` | `3.219` |
| `null_vs_julia` | `11.970` | `9.311` |
| `mandelbrot_vs_julia` | `11.407` | `9.084` |

The cache-summary standard deviations are `0.000` across probe seeds because
the visual streams are deterministic and probe reads are isolated. The fifty
probe seeds are useful for checking sampled probe-text behavior; they are not
fifty independent source-cache replications.

## First Reading

This batch strengthens the narrow reading from Note 0006:

> Under matched stochastic probe seeds, open-ended sampled surface text follows
> the paired probe RNG and phase policy rather than the visual condition in this
> 12-frame batch, while measured source-cache summaries retain
> condition-specific separation.

This is not a claim that Julia has a stable semantic effect on model outputs.
It is a cleaner boundary: unpaired sampled text can look vivid, but when probe
seeds are paired, the surface text collapses to the same outputs across
conditions. The remaining condition-sensitive signal is in the trace summary,
not in the open-ended poem text.

The Julia-related cache-summary distances remain larger than the
`null_vs_mandelbrot` distances in this setup, but this should still be read as a
pilot trace-summary observation. Low-level image statistics, stimulus order,
frame reuse, and model-specific preprocessing are not yet controlled.

## Implications

This is a useful non-spectacular result. It says that the current open-ended
sampled probe is not sensitive enough to expose condition differences once RNG
is paired. That makes the trace side more important, and it also points toward
more constrained probes.

The next design should separate three sources of variation:

1. Probe RNG variation, already tested here with matched probe seeds.
2. Visual-condition variation, currently visible in deterministic cache
   summaries.
3. Independent stream variation, still needed through multiple stream seeds,
   stimulus variants, or controlled image transformations.

## Next Steps

1. Add forced-choice or logprob-style probes so small condition tilts can be
   measured without relying on open-ended poem divergence.
2. Add independent visual stream variants for each condition before treating
   cache-summary separability as a distributional result.
3. Extend controls: blank, static repeat, shuffled order, reversed order,
   phase-scrambled stimuli, histogram-matched stimuli, Voronoi, tiling,
   cellular automata, random noise, and natural video.
4. Repeat with longer streams, for example fifty frames, after the control
   generator path is in place.

## Local Artifacts

- `runs/paired_stochastic_probe_50_seed_batch/paired_stochastic_analysis.md`
- `runs/paired_stochastic_probe_50_seed_batch/paired_stochastic_analysis.json`
- `runs/paired_stochastic_probe_50_seed_batch/paired_stochastic_batch_summary.md`
- `runs/paired_stochastic_probe_50_seed_batch/paired_stochastic_batch_summary.json`

## Tracked Examples

- [Paired stochastic analysis JSON](../../examples/research_notes/0007_paired_stochastic_probe_50_seed/paired_stochastic_analysis.json)
- [Paired stochastic batch summary JSON](../../examples/research_notes/0007_paired_stochastic_probe_50_seed/paired_stochastic_batch_summary.json)
