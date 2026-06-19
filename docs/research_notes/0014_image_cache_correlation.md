# Image-Stat / Cache-Distance Correlation

Date: 2026-06-19

## Setup

- Image-stat sources:
  - `examples/research_notes/0010_phase_scramble_image_stats/pattern_phase_scramble_stats.json`
  - `examples/research_notes/0011_quantile_matched_phase_scramble/phase_scramble_quantile_stats.json`
  - `examples/research_notes/0012_frequency_ablation_smoke/frequency_ablation_stats.json`
  - `examples/research_notes/0013_frequency_cutoff_sweep/frequency_cutoff_sweep_stats.json`
- Cache-distance sources:
  - paired pattern and generated-control variant analyses from Notes 0008 and
    0009,
  - phase-scramble comparisons from Note 0010,
  - quantile, frequency-ablation, and cutoff-sweep comparison summaries from
    Notes 0011 through 0013.
- Join key: pairwise condition identity, with batch summaries used to map run
  keys such as `julia_high_018` onto image-stat condition IDs.
- Joined rows: `174`
- Unmatched rows: `0`

This is a report over saved pilot artifacts. It does not rerun MLX, and it does
not estimate a causal model. It asks a narrower question: across the comparisons
already run, which low-level image-stat deltas co-move most with sampled
source-cache summary distances?

## Result

The strongest correlations were modest rather than decisive:

| Phase | Cache metric | Image-stat delta | N | Pearson | Spearman |
| --- | --- | --- | ---: | ---: | ---: |
| `mid` | `max_abs_l2_delta` | `mean_abs_luminance_delta_from_previous` | `87` | `0.344` | `0.320` |
| `after` | `mean_abs_l2_delta` | `mean_abs_luminance_delta_from_previous` | `87` | `0.283` | `0.316` |
| `mid` | `mean_abs_l2_delta` | `mean_abs_luminance_delta_from_previous` | `87` | `0.276` | `0.323` |
| `all` | `mean_abs_l2_delta` | `mean_abs_luminance_delta_from_previous` | `174` | `0.274` | `0.298` |
| `all` | `max_abs_l2_delta` | `mean_abs_luminance_delta_from_previous` | `174` | `0.270` | `0.266` |
| `mid` | `mean_abs_l2_delta` | `high_frequency_energy_ratio` | `87` | `-0.227` | `-0.295` |

The top positive image-stat correlate is not entropy, edge density, or raw
high-frequency ratio. It is the adjacent-frame luminance change summary:
`mean_abs_luminance_delta_from_previous`.

The high-frequency energy ratio appears with a negative sign in the stronger
entries. That does not mean high-frequency content is unimportant. It means the
current joined artifact set does not support a simple "larger HF delta means
larger cache-distance" read.

## First Reading

This report weakens any one-scalar explanation. The sampled cache-summary
distances are not well explained by luminance entropy, edge density, spectral
centroid, or high-frequency ratio alone.

The most useful signal is instead methodological:

> Across the current pilot artifacts, temporal image change is the clearest
> low-level co-mover with sampled source-cache distance, while high-frequency
> energy by itself is not a monotonic explanation.

That fits the cutoff-sweep result from Note 0013. The current evidence points
toward a compound axis: low/mid-frequency spatial organization, temporal frame
change, and transform-specific edge redistribution all matter. The trace
summary is responding to more than a static image histogram.

## Caveats

- The rows are not independent experimental replications. Many come from
  deterministic transforms of the same source streams.
- Correlations are descriptive over selected comparisons, not causal effects.
- The image statistics are aggregate summaries; they can miss spatial
  organization that matters to the model.
- The cache metric is a sampled source-cache summary, not full hidden-state
  geometry.
- Because surface text remains seed-locked in these runs, this report speaks
  only to trace-summary behavior.

## Next Steps

1. Repeat the correlation report after adding independent Mandelbrot and Julia
   source-stream variants.
2. Add per-frame or per-phase image-stat/cache reads instead of only aggregate
   manifest-level deltas.
3. Add forced-choice or logprob probes to see whether the trace-summary axes
   have any output-distribution counterpart.
4. Move the same analysis onto HF hidden-state or logit summaries when the T1 HF
   path is ready.

## Local Artifacts

- `runs/image_cache_correlation/image_cache_correlation.md`
- `runs/image_cache_correlation/image_cache_correlation.json`

## Tracked Examples

- [Image/cache correlation JSON](../../examples/research_notes/0014_image_cache_correlation/image_cache_correlation.json)
