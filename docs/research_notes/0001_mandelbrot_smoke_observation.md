# Research Note 0001: Mandelbrot Smoke Observation

Date: 2026-06-04

Run artifact: `runs/smoke/mandelbrot_a_mlx.json`

## Status

This is a smoke observation, not evidence of a stable effect. The run is useful
because it proves the first measurement loop can execute end to end:

- deterministic Mandelbrot frames,
- MLX-VLM frame-by-frame delivery,
- isolated before/mid/after probe reads,
- sampled KV-cache summaries,
- explicit run metadata.

## Setup

- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Adapter: `mlx_vlm`
- Condition: `mandelbrot_zoom_a`
- Selected frames: 4 of 12
- Probe: `Blue Silence`
- Probe policy: `isolated`
- Sync response target: `ACK`

The mid probe was taken after frame index 1, meaning after half of the selected
four-frame stream had been consumed.

## Observed Probe Text

Before:

> In the heart of the night, where stars do shine, A blue silence whispers, a
> gentle rhyme. The world is hushed, the city sleeps, As the moon's silver glow,
> a gentle heats.

Mid:

> In the heart of the night, where stars do shine, A blue silence whispers, a
> gentle rhyme. A world of dreams, where thoughts do sway, A canvas of the soul,
> in a different way.

After:

> In the heart of the night, where stars do shine, A blue silence whispers, in
> the mind's design. A canvas of dreams, where thoughts take flight, A world of
> blue, where shadows take flight.

## What The Smell Is

The visible drift is not large, and the opening template is stable. Still, the
mid/after outputs move from a quiet night/city image toward dreams, canvas,
thoughts, mind, and flight. That is exactly the kind of small probe-output
movement this repo is meant to catch before it becomes a claim.

The current interpretation should be:

- "The first smoke run produced a detectable wording shift in one creative
  probe."
- "The run path is instrumented enough to compare against Julia, geometry,
  shuffled, and natural-video controls."
- "No condition-specific effect is established."

## Trace Notes

The stream returned `ACK` for each visual turn, which is expected with
`max_tokens=2`.

Cache token counts grew across frame turns because the transcript grew. This is
not itself an activation result. The next useful signal is layer-spread cache
statistics and, for HF runs, hidden-state/logit traces under matched conditions.

## Next Runs

1. Repeat `mandelbrot_zoom_a` with a longer frame stream and fixed generation
   settings.
2. Run `julia_zoom_b` with the same settings.
3. Add a non-fractal geometric stream.
4. Add one natural-video frame sequence, preferably low-motion city/street
   first, then semantic-rich animal/cat.
5. Compare probe text, token logprob proxies, and trace summaries across
   conditions.

## Follow-Up Smoke Path

After this note, the Julia generation and MLX smoke path was also exercised with
`scripts/run_julia_smoke.py`. The short verification used only two frames and a
12-token probe budget, so it is a wiring check rather than a comparison result.
The important point is that `julia_zoom_b` now has the same manifest/run shape as
`mandelbrot_zoom_a`.

## Tracked Example

- [Mandelbrot vs Julia paired comparison JSON](../../examples/research_notes/0001_mandelbrot_julia_smoke/mandelbrot_vs_julia.json)
