# Frequency Ablation Smoke

Date: 2026-06-08

## Setup

- Source manifests:
  - `runs/pattern_probe_smoke/stimuli/mandelbrot/manifest.json`
  - `runs/pattern_probe_smoke/stimuli/julia/manifest.json`
- Frequency controls:
  - `low_pass`
  - `high_pass`
  - `low_pass_luminance_quantile_matched`
  - `high_pass_luminance_quantile_matched`
- FFT cutoff: `0.18`
- Frames per manifest: `12`
- Image-stat output: `runs/image_stats/frequency_ablation_stats.md/json`
- MLX run output: `runs/frequency_ablation_probe_seed_0`
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_seed=0`
- Probe cache policy: `isolated`

This note follows
[0011 Quantile-Matched Phase Scramble Controls](0011_quantile_matched_phase_scramble.md).
That run showed an asymmetric result: Mandelbrot moved very far from its
luminance-rank phase control, while Julia moved closer to its corresponding
control. The question here is whether that asymmetry is better explained by
low-frequency macro-structure, high-frequency local detail, or their interaction.

## Image-Side Split

The luminance-rank versions keep each source stream's luminance mean, luminance
std, entropy, and colorfulness fixed while using a low-pass or high-pass rank
map.

Key aggregate rows:

| Condition | Entropy | Edge density | HF ratio | Spectral centroid | Temporal delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mandelbrot_zoom_a` | `2.978` | `0.057` | `0.048` | `0.044` | `0.076` |
| `mandelbrot_low_pass_luminance_quantile_matched` | `2.978` | `0.028` | `0.003` | `0.016` | `0.070` |
| `mandelbrot_high_pass_luminance_quantile_matched` | `2.978` | `0.408` | `0.220` | `0.246` | `0.187` |
| `mandelbrot_phase_scrambled_luminance_quantile_matched` | `2.978` | `0.204` | `0.157` | `0.120` | `0.259` |
| `julia_zoom_b` | `3.456` | `0.278` | `0.428` | `0.321` | `0.128` |
| `julia_low_pass_luminance_quantile_matched` | `3.456` | `0.239` | `0.043` | `0.107` | `0.130` |
| `julia_high_pass_luminance_quantile_matched` | `3.456` | `0.443` | `0.577` | `0.432` | `0.164` |
| `julia_phase_scrambled_luminance_quantile_matched` | `3.456` | `0.583` | `0.428` | `0.315` | `0.191` |

Image-side reading:

- Mandelbrot is strongly low-frequency in the source statistics; its
  high-pass rank control is a much more violent visual rearrangement.
- Julia already has substantial high-frequency energy, so the high-pass rank
  control is less alien on HF ratio and spectral centroid than it is for
  Mandelbrot.
- The luminance-rank phase controls are not equivalent to pure high-pass
  controls. They combine phase disruption, edge redistribution, and changed
  temporal delta.

## MLX First Pass

Surface probe text stayed identical across every comparison below. The table is
therefore only a sampled source-cache summary read.

| Comparison | Mid max L2 | After max L2 | Mid mean L2 | After mean L2 |
| --- | ---: | ---: | ---: | ---: |
| Mandelbrot vs luminance-rank phase | `17.584` | `10.041` | `5.357` | `5.495` |
| Mandelbrot vs low-pass LQ | `5.388` | `11.023` | `2.265` | `5.391` |
| Mandelbrot vs high-pass LQ | `10.140` | `7.296` | `5.320` | `2.663` |
| Mandelbrot low-pass LQ vs high-pass LQ | `11.809` | `11.113` | `4.449` | `3.602` |
| Julia vs luminance-rank phase | `6.238` | `6.448` | `2.524` | `1.424` |
| Julia vs low-pass LQ | `12.071` | `7.714` | `2.630` | `2.005` |
| Julia vs high-pass LQ | `4.509` | `8.340` | `2.122` | `2.132` |
| Julia low-pass LQ vs high-pass LQ | `8.814` | `2.093` | `2.726` | `0.750` |
| Mandelbrot phase LQ vs Julia phase LQ | `12.415` | `8.638` | `3.915` | `1.984` |
| Mandelbrot low-pass LQ vs Julia low-pass LQ | `8.665` | `11.409` | `3.266` | `3.703` |
| Mandelbrot high-pass LQ vs Julia high-pass LQ | `2.488` | `3.444` | `1.073` | `1.260` |

## First Reading

The cleanest signal is cross-family:

> The high-pass luminance-quantile controls for Mandelbrot and Julia are close
> to each other, while the low-pass luminance-quantile controls remain far
> apart.

That points toward the Mandelbrot/Julia family separation living more in
low-frequency macro-structure than in high-frequency local detail, at least
under this `0.18` cutoff and sampled cache-summary read.

The original-to-control distances are more nuanced:

- Mandelbrot's `17.584` mid-distance to luminance-rank phase is not reproduced
  by high-pass alone (`10.140`) or low-pass alone (`5.388`). The phase control
  may be combining macro phase disruption, edge redistribution, and temporal
  change.
- Julia is closest to the high-pass LQ control at mid (`4.509`), but its after
  distance is closer for luminance-rank phase (`6.448`) than high-pass LQ
  (`8.340`). This suggests Julia's phase control is not simply "the high-pass
  version"; it may preserve enough of Julia's spectral character while
  redistributing edge structure.
- Low/high ablations separate strongly within each family, so the band split is
  a useful axis rather than a cosmetic transform.

The cautious claim is:

> In this single-cutoff pilot, entropy-fixed high-frequency controls collapse
> Mandelbrot and Julia closer together, while entropy-fixed low-frequency
> controls keep them separated. The earlier Mandelbrot phase-control jump is
> likely a compound effect, not a pure high-frequency or pure low-frequency
> manipulation.

## Caveats

- This is one cutoff (`0.18`), one probe seed, and one source stream per family.
- The cache metrics are sampled cache summaries, not full hidden-state geometry.
- Luminance-quantile matching fixes source pixel distributions but can change
  edge density and temporal deltas.
- Frequency filters are descriptive controls; they do not establish causal
  macro-structure effects without cutoff sweeps, transform seeds, and model
  replication.

## Next Steps

1. Sweep cutoffs such as `0.10`, `0.18`, `0.28`, and `0.40` for low/high LQ
   controls. First pass: [0013 Frequency Cutoff Sweep](0013_frequency_cutoff_sweep.md).
2. Repeat over multiple transform seeds where the transform uses stochastic
   rank sources such as phase maps.
3. Add a manifest-batch runner that accepts arbitrary manifest groups, so
   transformed controls can be run and compared without custom shell loops.
4. Correlate source-cache summary distances with image-stat deltas, especially
   HF ratio, spectral centroid, edge density, and temporal delta.

## Local Artifacts

- `runs/image_stats/frequency_ablation_stats.md`
- `runs/image_stats/frequency_ablation_stats.json`
- `runs/frequency_ablation_probe_seed_0/comparisons/frequency_ablation_comparison_summary.md`
- `runs/frequency_ablation_probe_seed_0/comparisons/frequency_ablation_comparison_summary.json`

## Tracked Examples

- [Frequency ablation image statistics JSON](../../examples/research_notes/0012_frequency_ablation_smoke/frequency_ablation_stats.json)
- [Frequency ablation comparison summary JSON](../../examples/research_notes/0012_frequency_ablation_smoke/frequency_ablation_comparison_summary.json)
