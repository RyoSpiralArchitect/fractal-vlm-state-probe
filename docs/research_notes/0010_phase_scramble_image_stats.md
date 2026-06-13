# Phase Scramble And Image Statistics Smoke

Date: 2026-06-07

## Setup

- Source manifests: `runs/pattern_probe_smoke/stimuli/*/manifest.json`
- Phase-scrambled manifests:
  - `runs/phase_scramble_smoke/julia_phase_scrambled_seed_7/manifest.json`
  - `runs/phase_scramble_smoke/mandelbrot_phase_scrambled_seed_7/manifest.json`
- Image-stat output: `runs/image_stats/pattern_phase_scramble_stats.md/json`
- MLX phase-scramble runs: `runs/phase_scramble_smoke/probe_seed_0`
- Probe seed: `0`
- Frames per run: `12`
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_max_tokens=80`
- Probe cache policy: `isolated`

This smoke adds two controls to the pattern path:

1. Low-level image statistics for the original pattern manifests and
   phase-scrambled Julia/Mandelbrot.
2. A first MLX run comparing original vs phase-scrambled Julia/Mandelbrot.

## Image Statistics

The image-stat report measures luminance mean/std, luminance entropy, edge
density, edge strength, spectral centroid, high-frequency energy ratio,
colorfulness, and adjacent-frame luminance delta. These are image-side controls,
not model-state measurements.

Key aggregate rows:

| Condition | Lum std | Entropy | Edge density | HF ratio | Spectral centroid | Temporal delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mandelbrot_zoom_a` | `0.238` | `2.978` | `0.057` | `0.048` | `0.044` | `0.076` |
| `julia_zoom_b` | `0.184` | `3.456` | `0.278` | `0.428` | `0.321` | `0.128` |
| `mandelbrot_zoom_a_phase_scrambled` | `0.156` | `7.095` | `0.024` | `0.083` | `0.068` | `0.196` |
| `julia_zoom_b_phase_scrambled` | `0.146` | `7.145` | `0.300` | `0.358` | `0.272` | `0.167` |
| `checkerboard_seed_7` | `0.181` | `1.000` | `0.157` | `0.055` | `0.099` | `0.029` |
| `voronoi_seed_7` | `0.141` | `4.584` | `0.023` | `0.015` | `0.033` | `0.052` |
| `quasicrystal_seed_7` | `0.189` | `6.557` | `0.000` | `0.001` | `0.014` | `0.130` |
| `white_noise_seed_7` | `0.217` | `7.744` | `0.761` | `0.808` | `0.541` | `0.250` |
| `blue_noise_seed_7` | `0.140` | `7.156` | `0.496` | `0.968` | `0.622` | `0.161` |

The important surprise is that phase scrambling does not behave like a simple
"same low-level statistics, broken geometry" control under these aggregate
metrics. It preserves some spectral shape more than raw geometry, but it also
raises luminance entropy sharply:

| Pair | Entropy delta | Edge density delta | HF-ratio delta | Spectral-centroid delta |
| --- | ---: | ---: | ---: | ---: |
| Mandelbrot vs phase-scrambled Mandelbrot | `4.117` | `0.033` | `0.034` | `0.024` |
| Julia vs phase-scrambled Julia | `3.689` | `0.022` | `0.070` | `0.048` |

So this transform is useful, but it should not be described as histogram
matched. A later control should explicitly histogram-match or quantile-match
frames if that is the desired claim.

## MLX Phase-Scramble Comparison

Surface probe text again stayed identical across paired comparisons. The trace
summary moved strongly:

| Comparison | Mid max L2 | After max L2 | Mid mean L2 | After mean L2 |
| --- | ---: | ---: | ---: | ---: |
| Original Julia vs phase-scrambled Julia | `13.665` | `11.487` | `2.951` | `1.746` |
| Original Mandelbrot vs phase-scrambled Mandelbrot | `8.409` | `7.905` | `3.421` | `2.776` |
| Phase-scrambled Mandelbrot vs phase-scrambled Julia | `3.170` | `3.980` | `1.644` | `0.958` |
| Original Mandelbrot vs original Julia | `11.407` | `9.084` | `3.645` | `3.999` |

This is a striking pilot result: original Mandelbrot and Julia are far apart in
the trace summary, but their phase-scrambled versions are much closer to each
other.

Nearest-neighbor checks against existing noise controls make the phase-scramble
direction clearer:

| Source | Comparison | Mid max L2 | After max L2 |
| --- | --- | ---: | ---: |
| `julia_phase_scrambled` | vs `blue_noise` | `2.171` | `5.519` |
| `julia_phase_scrambled` | vs `white_noise` | `2.920` | `2.532` |
| `julia_phase_scrambled` | vs `mandelbrot_phase_scrambled` | `3.170` | `3.980` |
| `julia_phase_scrambled` | vs original `julia` | `13.665` | `11.487` |
| `mandelbrot_phase_scrambled` | vs `julia_phase_scrambled` | `3.170` | `3.980` |
| `mandelbrot_phase_scrambled` | vs `blue_noise` | `3.664` | `5.447` |
| `mandelbrot_phase_scrambled` | vs `white_noise` | `5.191` | `1.779` |
| `mandelbrot_phase_scrambled` | vs original `mandelbrot` | `8.409` | `7.905` |

## First Reading

The phase-scrambled controls appear to move both fractals away from their
original trace neighborhoods and toward a shared high-entropy/noise-like region.
That does not erase the earlier geometric-control observation, but it adds a
new axis: low-level phase disruption and entropy inflation may dominate the
trace summary more strongly than fractal-family identity.

The careful claim is:

> In this single-seed pilot, phase scrambling strongly changes source-cache
> summaries while leaving paired surface probe text unchanged; the scrambled
> fractals become closer to each other and to noise controls than to their
> original streams.

## Caveats

- This is one phase-scramble seed and one probe seed.
- The phase-scramble implementation is not histogram matching.
- Aggregate image statistics are descriptive; they are not causal explanations.
- The cache metrics are sampled source-cache summaries, not full hidden-state
  geometry.

## Next Steps

1. Add a manifest-batch runner for arbitrary manifest paths so transformed
   stimuli can be run without manual commands.
2. Repeat phase-scrambled Julia/Mandelbrot over multiple transform seeds.
3. Add histogram-matched and quantile-matched transforms.
4. Correlate image-stat deltas with trace-summary deltas across generated
   controls and transforms.

## Local Artifacts

- `runs/image_stats/pattern_phase_scramble_stats.md`
- `runs/image_stats/pattern_phase_scramble_stats.json`
- `runs/phase_scramble_smoke/probe_seed_0/comparisons/julia_vs_julia_phase_scrambled.md`
- `runs/phase_scramble_smoke/probe_seed_0/comparisons/mandelbrot_vs_mandelbrot_phase_scrambled.md`
- `runs/phase_scramble_smoke/probe_seed_0/comparisons/mandelbrot_phase_vs_julia_phase.md`
- `runs/phase_scramble_smoke/probe_seed_0/comparisons/julia_phase_vs_blue_noise.md`
- `runs/phase_scramble_smoke/probe_seed_0/comparisons/julia_phase_vs_white_noise.md`

## Tracked Examples

- [Image statistics JSON](../../examples/research_notes/0010_phase_scramble_image_stats/pattern_phase_scramble_stats.json)
- [Julia vs phase-scrambled Julia JSON](../../examples/research_notes/0010_phase_scramble_image_stats/julia_vs_julia_phase_scrambled.json)
- [Mandelbrot vs phase-scrambled Mandelbrot JSON](../../examples/research_notes/0010_phase_scramble_image_stats/mandelbrot_vs_mandelbrot_phase_scrambled.json)
- [Phase-scrambled Mandelbrot vs phase-scrambled Julia JSON](../../examples/research_notes/0010_phase_scramble_image_stats/mandelbrot_phase_vs_julia_phase.json)
- [Phase-scrambled Julia vs blue noise JSON](../../examples/research_notes/0010_phase_scramble_image_stats/julia_phase_vs_blue_noise.json)
- [Phase-scrambled Julia vs white noise JSON](../../examples/research_notes/0010_phase_scramble_image_stats/julia_phase_vs_white_noise.json)
