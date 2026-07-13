# Research Note 0021: Layer 23 Values-Swap Intervention Scaffold

Date: 2026-06-25

## Status

Implemented as an experiment scaffold. No result claim is made in this note.

## Motivation

The true 50-frame cross-palette replication kept forced-choice first-token
readout nearly invariant while scalar cache-summary interaction argmax locations
repeated at mid layer 23 `values` and after layer 0 `keys`.

That makes a targeted causal intervention more appropriate than another broad
surface probe. The first intervention should be the lower-risk one: replace only
one layer's `values` tensor and keep the source stream transcript and all other
cache tensors fixed.

## Implemented Shape

New CLI:

```bash
python3 scripts/run_mlx_cache_values_swap_probe.py \
  --source-manifest runs/cross_palette_replication_50_v1/stimuli/mandelbrot_zoom_c_50f/manifest.json \
  --donor-manifest runs/cross_palette_replication_50_v1/stimuli/julia_zoom_d_50f/manifest.json \
  --source-label mandelbrot_c \
  --donor-label julia_d \
  --output runs/cache_values_swap/c_d_layer23_values_mid.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --max-frames 50 \
  --max-tokens 2 \
  --probe-max-tokens 120 \
  --probe-temperature 0.7 \
  --probe-seed 0 \
  --layer-index 23 \
  --probe-phase mid \
  --generation-readout-top-k 20
```

The runner builds two independent MLX `PromptCacheState` objects:

- source stream cache,
- donor stream cache.

It then runs matched creative probes over:

- source baseline,
- donor baseline,
- source transcript plus source cache with donor layer 23 `values`,
- reciprocal donor transcript plus donor cache with source layer 23 `values`
  unless `--no-reciprocal` is used.

The default creative probe is:

```text
Describe the visual impression this evolving pattern leaves behind. Write a short abstract reflection in 3-5 sentences.
```

## Guardrails

The intervention is strict by default:

- source and donor `PromptCacheState.token_ids` must match;
- source and donor tensor shapes must match;
- probes run on cloned branch caches so the original stream caches are not
  mutated by baseline or intervention reads;
- the source intervention uses the source transcript, not the donor transcript.

This means a failed run can be informative. If token histories diverge, the
stream outputs or prompt templates are no longer aligned enough for a clean
single-tensor swap.

## Intended Reading

The first question is not whether the generated text becomes visibly dramatic.
It is narrower:

> Holding the source transcript and all non-target cache tensors fixed, does
> replacing layer 23 `values` move the creative probe trajectory toward the donor
> baseline?

Useful first metrics:

- exact generated token sequence under matched `probe_seed`,
- early-token top-k logprob overlap with source vs donor baselines,
- text embedding or lexical distance to source and donor baselines,
- a simple donor-pull index over multiple probe seeds.

## Non-Claims

- This is a causal cache intervention over computational traces, not a claim
  about subjective experience.
- A text shift under creative probing would be exploratory until replicated
  across seeds, prompts, source pairs, and sham/off-locus swaps.
- No conclusion about layer 0 `keys` follows from this values-only scaffold.
