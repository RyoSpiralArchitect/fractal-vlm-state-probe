# Paired Stochastic Probe Smoke

Date: 2026-06-05

## Setup

- Source batch: `runs/paired_stochastic_probe_smoke`
- Conditions: `null_blank`, `mandelbrot`, `julia`
- Probe seeds: `0`, `1`, `2`
- Frames per run: `12`
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`, `max_tokens=2`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_max_tokens=80`
- Probe cache policy: `isolated`
- Probe phase seed policy: `before=probe_seed`, `mid=probe_seed+1`,
  `after=probe_seed+2`

This is a small smoke of the paired stochastic-probe design. It is not yet the
full 50-probe-seed distributional pass.

## Result

The paired seed policy worked as intended:

| Check | Result |
| --- | ---: |
| Before probes identical across conditions | `3/3` seeds |
| Mid pairwise lexical condition distance | `0.000` |
| After pairwise lexical condition distance | `0.000` |

Within each seed, the sampled poem changed strongly from `before` to `mid` and
`after`, but the same text appeared for `null_blank`, `mandelbrot`, and `julia`.
In this smoke, paired sampling absorbed the surface-text mobility that appeared
in the earlier unseeded run.

At the same time, source-cache summaries still separated conditions:

| Comparison | Mid max L2 delta | After max L2 delta |
| --- | ---: | ---: |
| `null_vs_mandelbrot` | `1.457` | `3.219` |
| `null_vs_julia` | `11.970` | `9.311` |
| `mandelbrot_vs_julia` | `11.407` | `9.084` |

The cache-summary distances are identical across the three probe seeds because
the stream is deterministic and probes are isolated branch reads. That is
expected, not independent replication.

## First Reading

This smoke supports a narrow but useful read:

> Under matched stochastic probe seeds, the sampled surface text follows the
> probe RNG rather than the visual condition in this 12-frame batch, while the
> measured source-cache summaries still retain condition-specific separation.

This strengthens the earlier caution. Unseeded sampled text can look vivid, but
without pairing it mostly measures sampling mobility. The condition-sensitive
signal remains clearer in the trace summary than in open-ended surface text.

## Next Steps

1. Run the same design with more probe seeds, for example `0..49`.
2. Add embedding-distance and lexical-category summaries instead of relying only
   on set-based lexical distance.
3. Repeat with visual controls: phase-scrambled Julia, static repeat, shuffled
   order, reversed order, Voronoi, tiling, and noise.
4. Test whether forced-choice/logprob probes reveal condition tilt where
   open-ended text remains seed-locked.

## Local Artifacts

- `runs/paired_stochastic_probe_smoke/paired_stochastic_analysis.md`
- `runs/paired_stochastic_probe_smoke/paired_stochastic_analysis.json`
- `runs/paired_stochastic_probe_smoke/paired_stochastic_batch_summary.md`
- `runs/paired_stochastic_probe_smoke/paired_stochastic_batch_summary.json`
