# Cross-Family Palette Control Smoke

Date: 2026-06-19

This note records the first cross-family palette-control run following
[0016 Five-Step Instrumentation Kickoff](0016_five_step_instrumentation_kickoff.md).

The question was narrow:

> If Mandelbrot/Julia cache-summary distances are partly driven by simple
> brightness, color, or histogram differences, what happens when each stream is
> forced onto the other stream's frame-level RGB pixel multiset?

This is still a smoke run. It uses the existing 12-frame source-variant
manifests, not a fresh 50-frame replication.

## Setup

- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Runtime: MLX-VLM
- Stream temperature: `0`
- Probe temperature: `0.7`
- Stream seed: `20260604`
- Probe seed: `0`
- Probe preset: `forced_choice`
- Probe cache policy: `isolated`
- Frames: `12`, because the source-variant manifests used here contain 12
  frames.

Conditions:

1. `mandelbrot_c`
2. `julia_d`
3. `mandelbrot_c_spatial_julia_d_palette`
4. `julia_d_spatial_mandelbrot_c_palette`

The two hybrid controls were generated with:

```bash
python3 scripts/generate_control_frames.py \
  --kind cross_palette_luminance_matched \
  --source-manifest runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json \
  --palette-manifest runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --condition-id mandelbrot_c_spatial_julia_d_palette \
  --output runs/cross_palette_controls/mandelbrot_c_spatial_julia_d_palette \
  --max-frames 50 \
  --overwrite

python3 scripts/generate_control_frames.py \
  --kind cross_palette_luminance_matched \
  --source-manifest runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --palette-manifest runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json \
  --condition-id julia_d_spatial_mandelbrot_c_palette \
  --output runs/cross_palette_controls/julia_d_spatial_mandelbrot_c_palette \
  --max-frames 50 \
  --overwrite
```

`--max-frames 50` was harmless here: each source manifest only had 12 frames.

## Image Statistics

The frame-level RGB multiset swap worked as intended. The hybrid with Julia as
palette donor inherits Julia's luminance mean/std/entropy, while the hybrid
with Mandelbrot as palette donor inherits Mandelbrot's corresponding values.

| Condition | Lum mean | Lum std | Entropy | HF ratio | Spectral centroid |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mandelbrot_c` | `0.388` | `0.095` | `5.015` | `0.068` | `0.079` |
| `julia_d` | `0.616` | `0.154` | `5.380` | `0.094` | `0.090` |
| `mandelbrot_c_spatial_julia_d_palette` | `0.616` | `0.154` | `5.380` | `0.017` | `0.029` |
| `julia_d_spatial_mandelbrot_c_palette` | `0.388` | `0.095` | `5.015` | `0.165` | `0.133` |

This also shows why the control is not a pure "same structure, only palette
changed" intervention. Reassigning the exact RGB pixel multiset by luminance
rank changes spectral and edge statistics, and it does so asymmetrically.

## Processor-Space Statistics

After the SmolVLM processor produced `pixel_values`, palette donor still largely
controlled tensor mean/std, but the spectral side exposed the asymmetry more
strongly:

| Condition | Tensor mean | Tensor std | HF ratio | Centroid cpp | Energy >0.5 cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mandelbrot_c` | `0.050` | `0.431` | `0.002` | `0.153` | `0.070` |
| `julia_d` | `0.237` | `0.492` | `0.006` | `0.278` | `0.168` |
| `mandelbrot_c_spatial_julia_d_palette` | `0.237` | `0.495` | `0.004` | `0.134` | `0.065` |
| `julia_d_spatial_mandelbrot_c_palette` | `0.051` | `0.427` | `0.015` | `0.525` | `0.358` |

The Julia-spatial/Mandelbrot-palette hybrid becomes much higher-frequency in
processor space than either original stream. The Mandelbrot-spatial/Julia-
palette hybrid keeps Julia-like tensor mean/std while remaining low in
processor-space spectral centroid.

## Probe Surface

The forced-choice surface labels were identical across all four conditions:

| Phase | `forced_family_choice` | `forced_frequency_choice` |
| --- | --- | --- |
| before | `C` | `L` |
| mid | `A` | `H` |
| after | `A` | `L` |

So this run again confirms the earlier pattern: the visible labels can remain
locked while source-cache summaries separate.

## Source-Cache Distances

| Comparison | Phase | Max L2 | Mean L2 |
| --- | --- | ---: | ---: |
| `mandelbrot_c_vs_julia_d` | mid | `8.599` | `4.776` |
| `mandelbrot_c_vs_julia_d` | after | `6.807` | `2.704` |
| `mandelbrot_c_vs_mandelbrot_c_spatial_julia_d_palette` | mid | `10.281` | `4.373` |
| `mandelbrot_c_vs_mandelbrot_c_spatial_julia_d_palette` | after | `9.710` | `2.642` |
| `julia_d_vs_julia_d_spatial_mandelbrot_c_palette` | mid | `4.698` | `1.690` |
| `julia_d_vs_julia_d_spatial_mandelbrot_c_palette` | after | `4.370` | `1.564` |
| `julia_d_vs_mandelbrot_c_spatial_julia_d_palette` | mid | `3.866` | `1.993` |
| `julia_d_vs_mandelbrot_c_spatial_julia_d_palette` | after | `4.765` | `1.459` |
| `mandelbrot_c_vs_julia_d_spatial_mandelbrot_c_palette` | mid | `6.868` | `4.052` |
| `mandelbrot_c_vs_julia_d_spatial_mandelbrot_c_palette` | after | `4.191` | `1.974` |
| `mandelbrot_c_spatial_julia_d_palette_vs_julia_d_spatial_mandelbrot_c_palette` | mid | `6.437` | `2.192` |
| `mandelbrot_c_spatial_julia_d_palette_vs_julia_d_spatial_mandelbrot_c_palette` | after | `5.519` | `1.561` |

The important negative result:

> The hybrid controls do not support a simple "spatial structure alone explains
> the Mandelbrot/Julia split" reading.

The stronger reading is more interesting:

> Frame-level palette/statistic swaps remain visible in source-cache summaries,
> and the effect is asymmetric. The current apparatus is seeing an interaction
> among palette donor, luminance-rank spatial ordering, and post-processor
> frequency content.

In particular, `mandelbrot_c_spatial_julia_d_palette` moves farther away from
`mandelbrot_c` than the original `mandelbrot_c` vs `julia_d` comparison at max
L2. By contrast, `julia_d_spatial_mandelbrot_c_palette` remains closer to
`julia_d` than the original cross-family distance. That asymmetry lines up with
the processor-space statistics: the Julia-spatial/Mandelbrot-palette hybrid
becomes a high-frequency tensor, while the Mandelbrot-spatial/Julia-palette
hybrid becomes low-frequency but Julia-like in mean/std.

## Image/Cache Join

The image/cache join matched all `12` rows with `0` unmatched cache rows. In
this tiny four-condition set, the strongest descriptive correlations were:

| Phase | Cache metric | Image-stat delta | N | Pearson | Spearman |
| --- | --- | --- | ---: | ---: | ---: |
| after | `mean_abs_l2_delta` | `high_frequency_energy_ratio` | 6 | `-0.693` | `-0.714` |
| after | `mean_abs_l2_delta` | `spectral_centroid` | 6 | `-0.638` | `-0.771` |
| mid | `mean_abs_l2_delta` | `spectral_centroid` | 6 | `-0.560` | `-0.371` |
| after | `max_abs_l2_delta` | `luminance_mean` | 6 | `0.522` | `0.765` |

This is too small for a stable correlation claim. It is useful as a confound
audit: raw scalar image stats are not inert under the cross-palette transform,
and they should be carried forward in any later regression-style report.

## Reading

This smoke makes the research direction stricter:

- It weakens a naive "macro geometry only" interpretation.
- It strengthens the need for processor-space controls, because raw and
  post-processor frequency behavior can diverge.
- It keeps the core phenomenon alive: surface forced-choice labels are fixed,
  while source-cache summaries move by condition.
- It suggests that cross-family palette controls are not merely nuisance
  controls; they are informative perturbations of the persistent multimodal
  state.

The cautious claim after this run:

> In this 12-frame forced-choice smoke, cross-family RGB-palette swaps did not
> erase source-cache separation. Instead, they exposed a nontrivial interaction
> between palette donor, spatial luminance rank, and processor-space spectral
> content, while surface forced-choice labels remained identical.

## Next

1. Repeat the same four-condition design with fresh 50-frame source variants.
2. Add processor-stat deltas to the image/cache correlation reporter, not only
   raw image-stat deltas.
3. Run the same four conditions through HF forced-choice probes so first-step
   top-k logprob readout can be compared.
4. Repeat the control on at least one more Mandelbrot/Julia source pair to
   check whether the asymmetry is source-specific.
5. Only after that, consider cache-swap intervention on the strongest matched
   pairs.

## Artifacts

- [Cross-palette raw image statistics JSON](../../examples/research_notes/0017_cross_palette_control_smoke/cross_palette_image_stats.json)
- [Cross-palette processor image statistics JSON](../../examples/research_notes/0017_cross_palette_control_smoke/cross_palette_processor_image_stats.json)
- [Cross-palette paired stochastic analysis JSON](../../examples/research_notes/0017_cross_palette_control_smoke/cross_palette_paired_stochastic_analysis.json)
- [Cross-palette image/cache correlation JSON](../../examples/research_notes/0017_cross_palette_control_smoke/cross_palette_image_cache_correlation.json)
