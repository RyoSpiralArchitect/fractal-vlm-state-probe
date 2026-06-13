# Pattern Probe Smoke

Date: 2026-06-07

## Setup

- Source batch: `runs/pattern_probe_smoke`
- Conditions: `null_blank`, `mandelbrot`, `julia`, `checkerboard`,
  `voronoi`, `quasicrystal`, `white_noise`, `blue_noise`
- Probe seeds: `0`, `1`, `2`
- Frames per run: `12`
- Stimulus seed for generated pattern controls: `7`
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_max_tokens=80`
- Probe cache policy: `isolated`

This is the first run using the pattern batch runner. It places non-fractal
geometric and low-level controls into the same paired stochastic-probe design as
the earlier null/Mandelbrot/Julia batches.

## Surface Probe Result

The open-ended sampled probe stayed fully paired by seed:

| Check | Result |
| --- | ---: |
| Before probes identical across conditions | `3/3` seeds |
| Pairwise lexical condition distance at `mid` | `0.000` |
| Pairwise lexical condition distance at `after` | `0.000` |
| Before-adjusted pairwise lexical distance at `mid` | `0.000` |
| Before-adjusted pairwise lexical distance at `after` | `0.000` |

Within each condition, probe text still moved from `before` to `mid` and
`after`, but that motion was identical across all visual conditions:

| Phase | Mean lexical distance from before | Std | Samples per condition |
| --- | ---: | ---: | ---: |
| `mid` | `0.847` | `0.039` | `3` |
| `after` | `0.843` | `0.073` | `3` |

This reproduces the paired-probe lesson from Notes 0006 and 0007: under matched
probe seeds, open-ended poem text is not a sensitive condition readout in this
setup.

## Source-Cache Summary Geometry

The source-cache summary distances formed a nontrivial condition map even while
surface text stayed locked.

Nearest pairs by `mean_max_abs_l2_delta`:

| Phase | Comparison | Max L2 | Mean L2 |
| --- | --- | ---: | ---: |
| `mid` | `null_blank_vs_mandelbrot` | `1.457` | `0.543` |
| `mid` | `white_noise_vs_blue_noise` | `2.140` | `1.201` |
| `mid` | `julia_vs_voronoi` | `3.630` | `1.771` |
| `mid` | `voronoi_vs_quasicrystal` | `3.787` | `1.420` |
| `mid` | `julia_vs_quasicrystal` | `4.284` | `1.992` |
| `after` | `voronoi_vs_quasicrystal` | `3.045` | `1.185` |
| `after` | `null_blank_vs_mandelbrot` | `3.219` | `1.437` |
| `after` | `null_blank_vs_blue_noise` | `3.383` | `1.497` |
| `after` | `julia_vs_checkerboard` | `4.155` | `1.947` |
| `after` | `julia_vs_quasicrystal` | `4.221` | `1.374` |
| `after` | `julia_vs_voronoi` | `4.288` | `1.673` |

Key distances involving Julia:

| Comparison | Mid max L2 | After max L2 |
| --- | ---: | ---: |
| `julia_vs_voronoi` | `3.630` | `4.288` |
| `julia_vs_quasicrystal` | `4.284` | `4.221` |
| `julia_vs_checkerboard` | `4.989` | `4.155` |
| `null_blank_vs_julia` | `11.970` | `9.311` |
| `mandelbrot_vs_julia` | `11.407` | `9.084` |
| `julia_vs_white_noise` | `13.653` | `9.286` |
| `julia_vs_blue_noise` | `12.191` | `11.450` |

## First Reading

This run changes the shape of the earlier Julia observation. The narrow read is
not "Julia is uniquely distant." In this pilot, Julia is far from null and
Mandelbrot in the sampled source-cache summaries, but relatively close to
generated geometric controls such as Voronoi, quasicrystal, and checkerboard.

That suggests a more grounded hypothesis:

> The earlier Julia separation may partly reflect geometric or boundary-structure
> features, rather than a fractal-family effect by itself.

This is still only a pilot trace-summary observation. It does not establish a
stable condition family, and it does not say what the model perceives. The
stronger result is methodological: matched surface probes remain insensitive,
while the trace summaries expose a condition geometry worth testing with
independent stimulus variants.

## Caveats

- The `3` samples are matched probe seeds. They are useful for checking surface
  text pairing, but they are not independent source-cache replications.
- The generated geometric controls used one stimulus seed, `7`.
- Low-level image statistics such as brightness, contrast, edge density,
  spectrum, and entropy are not yet measured.
- All observations are for one model, one frame count, one stimulus size, and
  one cache-summary extraction policy.
- The cache-summary metrics are sampled summaries of computational traces, not
  raw full-state geometry.

## Next Steps

1. Run independent `stimulus_seed` variants for the generated controls,
   especially `checkerboard`, `voronoi`, and `quasicrystal`.
2. Keep `julia` and `mandelbrot` fixed while varying generated control seeds to
   ask whether Julia remains close to the geometric family.
3. Add image-statistic reports for each manifest before interpreting geometric
   clusters.
4. Add phase-scrambled and static-repeat transforms for Julia and Mandelbrot.
5. Add forced-choice or logprob-style probes, because open-ended poem text is
   currently too seed-locked to expose condition tilt.

## Local Artifacts

- `runs/pattern_probe_smoke/pattern_batch_summary.md`
- `runs/pattern_probe_smoke/pattern_batch_summary.json`
- `runs/pattern_probe_smoke/paired_stochastic_analysis.md`
- `runs/pattern_probe_smoke/paired_stochastic_analysis.json`

## Tracked Examples

- [Pattern batch summary JSON](../../examples/research_notes/0008_pattern_probe_smoke/pattern_batch_summary.json)
- [Paired stochastic analysis JSON](../../examples/research_notes/0008_pattern_probe_smoke/paired_stochastic_analysis.json)
