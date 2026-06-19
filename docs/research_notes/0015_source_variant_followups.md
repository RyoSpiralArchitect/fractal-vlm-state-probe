# Source Variant Follow-Up Probes

Date: 2026-06-19

This note records the next pass after
[0014 Image-Stat / Cache-Distance Correlation](0014_image_cache_correlation.md):
independent Mandelbrot/Julia source streams, a tiny source-variant cutoff
micro-sweep, a forced-choice probe preset, and a first deeper trace comparison.

## Setup

- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream temperature: `0`
- Probe seed: `0`
- Probe cache policy: `isolated`
- Source variant batch: `runs/source_variant_smoke/manifest_probe_seed_0`
- Cutoff micro-sweep batch: `runs/source_variant_cutoff_sweep/manifest_probe_seed_0`
- Forced-choice batch: `runs/forced_choice_source_variant_smoke/manifest_probe_seed_0`

The source-variant batch used six 12-frame streams:

- `mandelbrot_a`, `mandelbrot_b`, `mandelbrot_c`
- `julia_b`, `julia_c`, `julia_d`

The cutoff micro-sweep used two source variants, `mandelbrot_c` and `julia_d`,
and generated low/high luminance-quantile controls at cutoffs `0.18` and
`0.28`.

## Source Variants

The source variants broaden the earlier single Mandelbrot/single Julia setup.
As in prior paired runs, surface probe text stayed identical across conditions:
pairwise lexical distance was `0.000` for every before/mid/after condition pair.

The cache-summary distances did move:

| Comparison | Phase | Max L2 | Mean L2 |
| --- | --- | ---: | ---: |
| `mandelbrot_c_vs_julia_d` | `mid` | `8.599` | `4.776` |
| `mandelbrot_c_vs_julia_d` | `after` | `6.807` | `2.704` |
| `mandelbrot_a_vs_julia_d` | `mid` | `12.984` | `6.146` |
| `mandelbrot_a_vs_julia_d` | `after` | `10.849` | `3.631` |
| `mandelbrot_c_vs_julia_c` | `mid` | `3.144` | `1.418` |
| `mandelbrot_c_vs_julia_c` | `after` | `7.837` | `2.597` |
| `mandelbrot_a_vs_mandelbrot_b` | `mid` | `10.564` | `4.718` |
| `mandelbrot_a_vs_mandelbrot_b` | `after` | `2.625` | `1.212` |

This argues against a simple "Mandelbrot family versus Julia family" reading.
Within-family source changes can be large, and cross-family changes can be small
or large depending on which stream pair is chosen.

## Image/Cache Join

The source-variant image/cache correlation joined all `30` mid/after rows with
`0` unmatched cache rows. The strongest correlations were still descriptive, not
decisive:

| Phase | Cache metric | Image-stat delta | N | Pearson | Spearman |
| --- | --- | --- | ---: | ---: | ---: |
| `after` | `mean_abs_l2_delta` | `luminance_std` | 15 | `0.535` | `0.521` |
| `mid` | `max_abs_l2_delta` | `mean_abs_luminance_delta_from_previous` | 15 | `-0.523` | `-0.364` |
| `mid` | `mean_abs_l2_delta` | `mean_abs_luminance_delta_from_previous` | 15 | `-0.484` | `-0.300` |
| `after` | `max_abs_l2_delta` | `luminance_std` | 15 | `0.418` | `0.282` |

This continues the 0014 reading: scalar image stats help audit confounds, but
they do not reduce the cache-distance behavior to one low-level visual scalar.

## Cutoff Micro-Sweep

For `mandelbrot_c` and `julia_d`, low-pass controls remained more separated
cross-source than matched high-pass controls in this tiny sweep:

| Comparison | Phase | Max L2 | Mean L2 |
| --- | --- | ---: | ---: |
| `mandelbrot_c_low_018_vs_julia_d_low_018` | `mid` | `7.855` | `2.919` |
| `mandelbrot_c_low_018_vs_julia_d_low_018` | `after` | `10.161` | `3.384` |
| `mandelbrot_c_high_018_vs_julia_d_high_018` | `mid` | `4.544` | `1.898` |
| `mandelbrot_c_high_018_vs_julia_d_high_018` | `after` | `6.214` | `2.579` |
| `mandelbrot_c_low_028_vs_julia_d_low_028` | `mid` | `7.130` | `2.617` |
| `mandelbrot_c_low_028_vs_julia_d_low_028` | `after` | `10.463` | `3.550` |
| `mandelbrot_c_high_028_vs_julia_d_high_028` | `mid` | `6.855` | `2.218` |
| `mandelbrot_c_high_028_vs_julia_d_high_028` | `after` | `4.225` | `1.736` |

Same-source low/high separation was also nontrivial, especially for
`mandelbrot_c`:

| Comparison | Phase | Max L2 | Mean L2 |
| --- | --- | ---: | ---: |
| `mandelbrot_c_low_018_vs_mandelbrot_c_high_018` | `mid` | `7.373` | `2.194` |
| `mandelbrot_c_low_018_vs_mandelbrot_c_high_018` | `after` | `8.540` | `2.656` |
| `julia_d_low_018_vs_julia_d_high_018` | `mid` | `4.847` | `2.254` |
| `julia_d_low_018_vs_julia_d_high_018` | `after` | `3.036` | `1.117` |

Caveat: this micro-sweep generated transform manifests without explicit
cutoff-specific `condition_id` values. The manifest-batch keys are unambiguous,
so cache comparisons are readable, but image-stat joins should be rerun with
cutoff-specific condition IDs before treating the image-stat side as a clean
example artifact.

## Forced Choice

The new `forced_choice` probe preset adds two short-answer probes:

- `forced_family_choice`: answer `A`, `B`, or `C`
- `forced_frequency_choice`: answer `L`, `H`, or `C`

In the `mandelbrot_c` versus `julia_d` run, both conditions produced identical
forced-choice outputs:

| Condition | Phase | Family choice | Frequency choice |
| --- | --- | --- | --- |
| `mandelbrot_c` | `before` | `C` | `L` |
| `mandelbrot_c` | `mid` | `A` | `L` |
| `mandelbrot_c` | `after` | `A` | `L` |
| `julia_d` | `before` | `C` | `L` |
| `julia_d` | `mid` | `A` | `L` |
| `julia_d` | `after` | `A` | `L` |

So the first forced-choice pass does not expose the internal cache difference
at the surface answer level. It is still useful because it gives a compact,
deterministic prompt surface for future logprob-capable runs.

## Deeper Trace Comparison

`compare_runs.py` now reports top sequence-position L2 deltas in addition to
layer-level L2 deltas when `sequence_position_stats` are present.

For `mandelbrot_c` versus `julia_d`, the deeper comparison reproduces the
layer-level split:

| Phase | Probe | Max layer L2 | Mean layer L2 |
| --- | --- | ---: | ---: |
| `mid` | `blue_silence_poem` | `8.599` | `4.776` |
| `after` | `blue_silence_poem` | `6.807` | `2.704` |
| `mid` | `forced_family_choice` | `8.599` | `4.776` |
| `mid` | `forced_frequency_choice` | `8.599` | `4.776` |
| `after` | `forced_family_choice` | `6.807` | `2.704` |
| `after` | `forced_frequency_choice` | `6.807` | `2.704` |

The current sampled sequence positions are not yet informative at mid/after:
the top reported position-level L2 delta is `0` there. At frame `0`, a
nonzero position-level delta appears (`layer 23`, `values`, position `128`,
abs delta about `2.331`). This suggests that the trace reporter needs a richer
position-selection policy before sequence-position analysis can carry much
interpretive weight.

## Reading

The strongest reading after this pass:

> The apparatus continues to find cases where surface probe text, and even
> forced-choice surface labels, are identical while cache-summary distances
> differ. The difference is not cleanly explained by fractal family alone, nor
> by a single scalar image statistic. Source identity, frequency transform, and
> cutoff all matter, and the current trace summaries are still too coarse to
> localize the effect.

## Next

1. Rerun the source-variant cutoff sweep with explicit cutoff-specific
   `condition_id` values.
2. Repeat source variants across probe seeds and include at least one additional
   model.
3. Use a logprob-capable route for the forced-choice probes so identical labels
   can still expose probability tilt.
4. Improve `select_sequence_positions` or add a trace analyzer that samples
   nonzero positions around image-token and recent-text regions.
5. Separate source-family, source-instance, low/high frequency, and scalar
   image-stat effects in one small regression-style report.

## Artifacts

- [Source-variant image statistics JSON](../../examples/research_notes/0015_source_variant_followups/source_variant_image_stats.json)
- [Source-variant paired stochastic analysis JSON](../../examples/research_notes/0015_source_variant_followups/source_variant_paired_stochastic_analysis.json)
- [Source-variant image/cache correlation JSON](../../examples/research_notes/0015_source_variant_followups/source_variant_image_cache_correlation.json)
- [Source-variant cutoff paired stochastic analysis JSON](../../examples/research_notes/0015_source_variant_followups/source_variant_cutoff_paired_stochastic_analysis.json)
- [Forced-choice paired stochastic analysis JSON](../../examples/research_notes/0015_source_variant_followups/forced_choice_paired_stochastic_analysis.json)
- [Forced-choice deep trace comparison JSON](../../examples/research_notes/0015_source_variant_followups/forced_choice_deep_trace_comparison.json)
