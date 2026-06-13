# Quantile-Matched Phase Scramble Controls

Date: 2026-06-08

## Setup

- Source manifests:
  - `runs/pattern_probe_smoke/stimuli/mandelbrot/manifest.json`
  - `runs/pattern_probe_smoke/stimuli/julia/manifest.json`
- Plain phase-scrambled controls:
  - `runs/phase_scramble_smoke/mandelbrot_phase_scrambled_seed_7/manifest.json`
  - `runs/phase_scramble_smoke/julia_phase_scrambled_seed_7/manifest.json`
- RGB-quantile controls:
  - `runs/phase_scramble_quantile_smoke/mandelbrot_phase_quantile_seed_7/manifest.json`
  - `runs/phase_scramble_quantile_smoke/julia_phase_quantile_seed_7/manifest.json`
- Luminance-rank controls:
  - `runs/phase_scramble_luminance_quantile_smoke/mandelbrot_phase_luminance_quantile_seed_7/manifest.json`
  - `runs/phase_scramble_luminance_quantile_smoke/julia_phase_luminance_quantile_seed_7/manifest.json`
- Image-stat output: `runs/image_stats/phase_scramble_quantile_stats.md/json`
- Transform seed: `7`
- Frames per manifest: `12`

This note follows directly from
[0010 Phase Scramble and Image Statistics](0010_phase_scramble_image_stats.md).
The previous smoke showed that plain phase scrambling was not a clean
histogram-matched control: luminance entropy jumped from the original fractal
range into a high-entropy/noise-like range.

## Transform Variants

Three phase-based controls now exist:

| Kind | Intended control | Key limitation |
| --- | --- | --- |
| `phase_scrambled` | Break spatial phase while roughly preserving spectral amplitude per RGB channel. | It inflates luminance entropy in this pilot. |
| `phase_scrambled_quantile_matched` | Match each RGB channel's source quantiles after phase scrambling. | Per-channel matching does not guarantee source luminance entropy. |
| `phase_scrambled_luminance_quantile_matched` | Reassign the source RGB pixel multiset by scrambled luminance rank. | It preserves source color/luminance distribution but no longer promises the same spectrum as plain phase scrambling. |

The luminance-rank version is the stricter entropy control. It keeps the source
RGB pixel set and moves those pixels into a phase-scrambled rank order, so the
image-stat reporter sees the same luminance mean, luminance std, entropy, and
colorfulness as the source stream.

## Image Statistics

Key aggregate rows:

| Condition | Lum mean | Lum std | Entropy | Edge density | HF ratio | Spectral centroid | Colorfulness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mandelbrot_zoom_a` | `0.236` | `0.238` | `2.978` | `0.057` | `0.048` | `0.044` | `0.344` |
| `mandelbrot_zoom_a_phase_scrambled` | `0.263` | `0.156` | `7.095` | `0.024` | `0.083` | `0.068` | `0.460` |
| `mandelbrot_zoom_a_phase_scrambled_quantile_matched` | `0.236` | `0.186` | `5.231` | `0.175` | `0.144` | `0.112` | `0.556` |
| `mandelbrot_zoom_a_phase_scrambled_luminance_quantile_matched` | `0.236` | `0.238` | `2.978` | `0.204` | `0.157` | `0.120` | `0.344` |
| `julia_zoom_b` | `0.251` | `0.184` | `3.456` | `0.278` | `0.428` | `0.321` | `0.567` |
| `julia_zoom_b_phase_scrambled` | `0.270` | `0.146` | `7.145` | `0.300` | `0.358` | `0.272` | `0.517` |
| `julia_zoom_b_phase_scrambled_quantile_matched` | `0.251` | `0.175` | `6.253` | `0.344` | `0.446` | `0.326` | `0.602` |
| `julia_zoom_b_phase_scrambled_luminance_quantile_matched` | `0.251` | `0.184` | `3.456` | `0.583` | `0.428` | `0.315` | `0.567` |

The RGB-quantile version helps with luminance mean but does not fix entropy:
Julia still lands at `6.253` bits instead of the source `3.456`. The
luminance-rank version fixes this aggregate exactly in the current reporter:
Julia returns to `3.456`, and Mandelbrot returns to `2.978`.

## First Reading

This creates a cleaner next control for the "structure-only" question. If the
model trace still separates original Julia from the luminance-rank matched
phase control, the difference is less likely to be explained by a simple
luminance-entropy jump.

The careful claim is:

> In this image-stat pilot, luminance-rank quantile matching removes the large
> entropy confound introduced by plain phase scrambling while still disrupting
> the source stream's spatial arrangement.

That claim is image-side only. It says the control is better matched, not that a
model effect has been isolated.

## MLX First Pass

After generating the matched controls, a first MLX probe pass was run with the
same `probe_seed=0`, `temperature=0`, `probe_temperature=0.7`, `12` frames, and
isolated probe cache policy used in the prior phase-scramble smoke.

Surface probe text stayed identical for every comparison below. The signal is
only in sampled source-cache summaries:

| Comparison | Mid max L2 | After max L2 | Mid mean L2 | After mean L2 |
| --- | ---: | ---: | ---: | ---: |
| Mandelbrot vs plain phase | `8.409` | `7.905` | `3.421` | `2.776` |
| Mandelbrot vs RGB-quantile phase | `11.347` | `11.125` | `4.954` | `4.553` |
| Mandelbrot vs luminance-rank phase | `17.584` | `10.041` | `5.357` | `5.495` |
| Julia vs plain phase | `13.665` | `11.487` | `2.951` | `1.746` |
| Julia vs RGB-quantile phase | `8.957` | `8.769` | `3.196` | `1.667` |
| Julia vs luminance-rank phase | `6.238` | `6.448` | `2.524` | `1.424` |
| Plain phase Mandelbrot vs plain phase Julia | `3.170` | `3.980` | `1.644` | `0.958` |
| RGB-quantile Mandelbrot vs RGB-quantile Julia | `4.593` | `3.561` | `1.684` | `1.386` |
| Luminance-rank Mandelbrot vs luminance-rank Julia | `12.415` | `8.638` | `3.915` | `1.984` |

The pilot read is subtle:

- Julia moves closer to its luminance-rank matched control than to its plain
  phase-scrambled control.
- Mandelbrot moves farther from its luminance-rank matched control than from
  its plain phase-scrambled control.
- The two plain phase-scrambled fractals remain close to each other, but the
  two luminance-rank matched controls separate again.

This does not prove a structure-only cause. It does suggest that the large
entropy jump in plain phase scrambling was not a neutral nuisance: once source
pixel distributions are restored, the trace geometry changes direction, and the
Julia/Mandelbrot behavior is no longer symmetric.

## Caveats

- This is one transform seed and one 12-frame source stream per fractal.
- Luminance-rank matching preserves the source pixel multiset, so it is not the
  same spectral-control promise as plain phase scrambling.
- Edge density can change sharply after luminance-rank reassignment; for Julia
  it rises from `0.278` to `0.583`.
- The transform controls low-level image statistics only. It does not establish
  causal structure effects by itself.
- The MLX trace read uses sampled cache-summary features, not full hidden-state
  geometry, and should be repeated across transform seeds.

## Next Steps

1. Repeat original, plain phase-scrambled, RGB-quantile, and luminance-rank
   controls over multiple transform seeds.
2. Add a manifest-batch runner for arbitrary source/control manifests so
   transformed-control batches are not hand-wired.
3. Compare trace-summary distances against image-stat deltas, especially
   entropy, edge density, high-frequency ratio, and temporal delta.

## Local Artifacts

- `runs/image_stats/phase_scramble_quantile_stats.md`
- `runs/image_stats/phase_scramble_quantile_stats.json`
- `runs/phase_scramble_quantile_probe_seed_0/comparisons/quantile_comparison_summary.md`
- `runs/phase_scramble_quantile_probe_seed_0/comparisons/quantile_comparison_summary.json`
- `runs/phase_scramble_quantile_smoke/*/manifest.json`
- `runs/phase_scramble_luminance_quantile_smoke/*/manifest.json`

## Tracked Examples

- [Quantile phase-scramble image statistics JSON](../../examples/research_notes/0011_quantile_matched_phase_scramble/phase_scramble_quantile_stats.json)
- [Quantile phase-scramble comparison summary JSON](../../examples/research_notes/0011_quantile_matched_phase_scramble/quantile_comparison_summary.json)
