# Cross-Palette Replication Readout

Date: 2026-06-22

This note records the first run of the replicated cross-palette factorial path
introduced in [0018](0018_cross_palette_replication_path.md).

The run used two independently paired Mandelbrot/Julia source variants:

| Pair | MM | JJ | Frames used |
| --- | --- | --- | ---: |
| `c_d` | `mandelbrot_zoom_c` | `julia_zoom_d` | 12 |
| `b_c` | `mandelbrot_zoom_b` | `julia_zoom_c` | 12 |

The commands used `--max-frames 50`, but the source-variant manifests currently
contain 12 frames. This is therefore a two-pair 12-frame replication smoke, not
a true 50-frame replication.

## Local Artifacts

The full local output is under:

- `runs/cross_palette_replications/cross_palette_factorial_batch_summary.md`
- `runs/cross_palette_replications/c_d/raw_factorial_image_contrast.md`
- `runs/cross_palette_replications/c_d/processor_factorial_image_contrast.md`
- `runs/cross_palette_replications/c_d/factorial_cache_contrast.md`
- `runs/cross_palette_replications/c_d/manifest_probe_seed_0/paired_stochastic_analysis.md`
- `runs/cross_palette_replications/b_c/raw_factorial_image_contrast.md`
- `runs/cross_palette_replications/b_c/processor_factorial_image_contrast.md`
- `runs/cross_palette_replications/b_c/factorial_cache_contrast.md`
- `runs/cross_palette_replications/b_c/manifest_probe_seed_0/paired_stochastic_analysis.md`

## Surface Readout

The forced-choice probe surface did not separate conditions.

For both `c_d` and `b_c`:

- before probes were identical across conditions,
- pairwise lexical condition distances were `0.000` at before, mid, and after,
- condition drift from before was identical across `mm`, `jj`, `mj`, and `jm`.

This is useful precisely because it keeps the visible readout quiet. The run is
not evidence that a generated label changed. It asks whether saved source-cache
summaries still carry a factorial trace when the surface answer does not.

## Image-Statistic Contrast

Raw image statistics showed interaction terms in edge and frequency summaries,
but the strength was source-pair dependent.

| Pair | Top raw interaction | Value | Relative |
| --- | --- | ---: | ---: |
| `c_d` | `edge_density` | 0.028 | 0.637 |
| `b_c` | `edge_density` | 0.155 | 1.522 |

Processor-space statistics were more revealing and more cautious. The
architecture-aware frequency fields were prominent for `c_d`, but much weaker
as interaction terms for `b_c`.

| Pair | Top processor interaction | Value | Relative |
| --- | --- | ---: | ---: |
| `c_d` | `spectral_centroid_cycles_per_patch` | -0.229 | 0.839 |
| `b_c` | `mean_abs_tensor_delta_from_previous` | -0.036 | 0.173 |

The spatial main effect in processor-space frequency remained visible in both
pairs:

| Pair | `spectral_centroid_cycles_per_patch` spatial effect |
| --- | ---: |
| `c_d` | 0.258 |
| `b_c` | 0.190 |

So the replicated image-stat layer does not support a simple claim that one
scalar processor interaction explains the cache behavior. It does support the
weaker and more useful claim that the hybrid controls perturb processor-space
frequency organization in a measurable, pair-sensitive way.

## Cache-Summary Contrast

The strongest result is that the cache-summary interaction argmax repeated in
location across both source pairs.

| Pair | Mid interaction argmax | Value | After interaction argmax | Value |
| --- | --- | ---: | --- | ---: |
| `c_d` | layer 23 `values` | 8.653 | layer 0 `keys` | -8.956 |
| `b_c` | layer 23 `values` | 10.167 | layer 0 `keys` | 16.400 |

The sign does not repeat at after, and the saved summary loses vector direction,
so the sign should not be over-read. The repeatable part is the locus:

- mid phase: late-layer value-cache norm,
- after phase: first-layer key-cache norm.

Both forced-choice probes reported the same cache-summary values because the
source cache was shared and the probe policy was isolated.

## Current Reading

This run strengthens the cross-palette design but does not close the causal
story.

The cautious reading is:

> In two 12-frame source-pair replications, cross-palette factorial controls
> left the surface forced-choice probe unchanged while producing repeatable
> cache-summary interaction peaks at the same mid and after locations.

The important negative result is equally useful:

> The repeated cache locus did not map cleanly onto a repeated top
> processor-space image-stat interaction.

That means the next version should not claim that a single image-stat scalar is
the mechanism. The better hypothesis is that the cache-summary interaction
reflects a distribution-coupled visual perturbation whose processor-space
frequency summaries are only partial observables.

## Next Steps

1. Generate true 50-frame source variants so `--max-frames 50` is no longer only
   a ceiling.
2. Add an aggregation report over many source pairs that tracks cache-location
   persistence, sign consistency, and top image-stat rank stability.
3. Add first-token logit, top-k, and teacher-forced probe readouts to test
   whether the quiet surface label hides a shifted output distribution.
4. Use the repeated loci, especially mid layer 23 `values` and after layer 0
   `keys`, as the first cache-swap intervention targets.
