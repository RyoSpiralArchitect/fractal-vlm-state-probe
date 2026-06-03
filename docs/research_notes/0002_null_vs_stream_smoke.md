# Null vs Stream Smoke Observation

Date: 2026-06-04

## Setup

- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Adapter: `mlx_vlm`
- Seed: `20260604`
- Source manifest: `runs/null_vs_stream_seed_20260604/mandelbrot_stimulus/manifest.json`
- Selected source frames: 4
- Generation: greedy, `max_tokens=2` for stream turns, `probe_max_tokens=48`
- Probe policy: `isolated`
- Cache summary: sampled layers, first/mid/last stream captures

The four delivery modes used the same source manifest and seed:

1. `probe_only`: no stream turns.
2. `text_only_stream`: frame index and timecode text only, no image.
3. `blank_visual_stream`: generated black image frames at the manifest
   dimensions.
4. `visual_stream`: the original Mandelbrot frames.

## First Reading

The clean `probe_only` run produced the same `before` and `after` probe text.
This is useful: at this seed and probe length, the repeated clean probe did not
drift by itself.

All three stream modes changed the later probe text relative to the clean
`before` response. This is the important ordering correction: before asking
whether Mandelbrot differs from Julia, the apparatus now shows that even
frame-indexed stream context can move the probe.

The `text_only_stream` condition also shifted the probe strongly. Therefore the
current result cannot be read as a visual-content effect. Transcript growth,
timecode text, synchronization wording, cache growth, and visual tokens remain
separable variables.

## Paired Comparisons

| Comparison | Probe text | Source-cache delta |
| --- | --- | --- |
| `probe_only` vs `visual_stream` | `before` matched, `after` differed | not directly comparable because `probe_only` has no stream cache |
| `text_only_stream` vs `visual_stream` | `mid` and `after` differed | mid max abs L2 `249.391`; after max abs L2 `171.775` |
| `blank_visual_stream` vs `visual_stream` | `mid` and `after` differed | mid max abs L2 `49.782`; after max abs L2 `76.708` |

The blank-vs-visual cache deltas were much smaller than text-only-vs-visual
deltas in this single run, but they were not zero. This is descriptive only; it
should be repeated across seeds and prompts before interpretation.

## Interpretation Boundary

This note does not claim that fractal content caused a stable semantic effect.
The stronger pilot observation is narrower:

> Under one deterministic seed and a short stream, the clean probe-only baseline
> stayed fixed, while frame-indexed stream conditions changed the later creative
> probe and produced measurable sampled cache traces.

The next useful step is repeated-seed Null-vs-Stream runs, with prompt wording
kept as matched as possible across text-only, blank-image, and visual-image
conditions.

## Local Artifacts

- `runs/null_vs_stream_seed_20260604/probe_only_mlx.json`
- `runs/null_vs_stream_seed_20260604/text_only_mlx.json`
- `runs/null_vs_stream_seed_20260604/blank_visual_mlx.json`
- `runs/null_vs_stream_seed_20260604/visual_mlx.json`
- `runs/null_vs_stream_seed_20260604/probe_only_vs_visual.md`
- `runs/null_vs_stream_seed_20260604/text_only_vs_visual.md`
- `runs/null_vs_stream_seed_20260604/blank_visual_vs_visual.md`
