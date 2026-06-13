# Stimulus Seed 8 Variant Smoke

Date: 2026-06-07

## Setup

- Source batch: `runs/pattern_geometry_variants_seed_8`
- Conditions: `null_blank`, `mandelbrot`, `julia`, `checkerboard`,
  `voronoi`, `quasicrystal`
- Probe seed: `0`
- Generated-control stimulus seed: `8`
- Frames per run: `12`
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_max_tokens=80`
- Probe cache policy: `isolated`

This is the first independent generated-control variant after the pattern
smoke. Mandelbrot and Julia remain fixed; checkerboard, Voronoi, and
quasicrystal are regenerated with `stimulus_seed=8`.

## Surface Probe Result

Surface text remained paired:

| Check | Result |
| --- | ---: |
| Before probes identical across conditions | `1/1` seed |
| Pairwise lexical condition distance at `mid` | `0.000` |
| Pairwise lexical condition distance at `after` | `0.000` |

This is a single probe seed, so it should be read only as a continuity check
for the seed-paired probe apparatus.

## Source-Cache Summary Distances

Nearest pairs by `mean_max_abs_l2_delta`:

| Phase | Comparison | Max L2 | Mean L2 |
| --- | --- | ---: | ---: |
| `mid` | `null_blank_vs_mandelbrot` | `1.457` | `0.543` |
| `mid` | `julia_vs_checkerboard` | `4.222` | `1.833` |
| `mid` | `julia_vs_quasicrystal` | `6.129` | `2.732` |
| `mid` | `checkerboard_vs_quasicrystal` | `6.479` | `3.357` |
| `mid` | `checkerboard_vs_voronoi` | `6.525` | `3.051` |
| `mid` | `julia_vs_voronoi` | `7.603` | `3.697` |
| `after` | `julia_vs_checkerboard` | `2.949` | `1.439` |
| `after` | `null_blank_vs_mandelbrot` | `3.219` | `1.437` |
| `after` | `checkerboard_vs_voronoi` | `4.402` | `1.914` |
| `after` | `voronoi_vs_quasicrystal` | `4.410` | `1.605` |
| `after` | `julia_vs_voronoi` | `6.163` | `2.044` |

Julia-specific distances:

| Comparison | Mid max L2 | After max L2 |
| --- | ---: | ---: |
| `julia_vs_checkerboard` | `4.222` | `2.949` |
| `julia_vs_quasicrystal` | `6.129` | `8.762` |
| `julia_vs_voronoi` | `7.603` | `6.163` |
| `mandelbrot_vs_julia` | `11.407` | `9.084` |
| `null_blank_vs_julia` | `11.970` | `9.311` |

## First Reading

The broad direction from Note 0008 survives, but with a useful wrinkle. Julia
remains closer to generated geometry than to null or Mandelbrot in this
stimulus-seed-8 run. The closest relation is now `julia_vs_checkerboard`, not
`julia_vs_voronoi`.

This suggests that the earlier "Julia near geometric controls" observation is
not just one Voronoi/quasicrystal seed accident, but the internal neighborhood
is not stable enough to name a single geometric family yet. The honest read is:

> Julia's trace-summary neighborhood remains geometric-leaning under one
> generated-control seed variant, while the nearest geometric condition changes.

## Caveats

- This is one generated-control seed and one probe seed.
- Mandelbrot and Julia were not regenerated; only generated controls changed.
- `null_blank_vs_mandelbrot`, `null_blank_vs_julia`, and
  `mandelbrot_vs_julia` are expected to match earlier deterministic runs.
- No image-statistic matching has been performed.

## Next Steps

1. Run at least one more generated-control seed, for example `stimulus_seed=9`.
2. Aggregate geometry-control variants before interpreting Julia's nearest
   neighbors.
3. Add image-statistic reports for each generated manifest.
4. Add Julia/Mandelbrot phase-scrambled and static-repeat transforms to separate
   geometry from temporal evolution.

## Local Artifacts

- `runs/pattern_geometry_variants_seed_8/pattern_batch_summary.md`
- `runs/pattern_geometry_variants_seed_8/pattern_batch_summary.json`
- `runs/pattern_geometry_variants_seed_8/paired_stochastic_analysis.md`
- `runs/pattern_geometry_variants_seed_8/paired_stochastic_analysis.json`

## Tracked Examples

- [Pattern batch summary JSON](../../examples/research_notes/0009_stimulus_seed_8_variant/pattern_batch_summary.json)
- [Paired stochastic analysis JSON](../../examples/research_notes/0009_stimulus_seed_8_variant/paired_stochastic_analysis.json)
