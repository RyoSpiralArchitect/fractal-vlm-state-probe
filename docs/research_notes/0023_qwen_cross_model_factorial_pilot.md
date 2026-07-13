# Research Note 0023: Qwen Cross-Model Factorial Pilot

Date: 2026-07-13

## Status

Completed as a one-frame, two-pair cross-architecture pilot. This note does not
claim a persistent multi-frame replication.

## Motivation

The SmolVLM experiments repeatedly separated a quiet forced-choice readout from
condition-dependent cache-summary geometry. A useful next test is whether that
separation is specific to SmolVLM or can also be observed in a different local
vision-language architecture.

The second model was:

```text
mlx-community/Qwen2.5-VL-3B-Instruct-4bit
```

The local snapshot revision was
`46d4cf06a06ffc1a766c214174f9cbed2f45bcab`, with MLX `0.31.1` and MLX-VLM
`0.4.4`.

The run used the same cross-palette `MM/JJ/MJ/JM` cells and forced-choice probes
as the SmolVLM path. Two independent source pairs were tested:

- `mandelbrot c` x `julia d`,
- `mandelbrot b` x `julia c`.

Local run roots:

```text
runs/qwen_cross_palette_smoke/c_d_1f_all_layers_seed0/
runs/qwen_cross_palette_smoke/b_c_1f_all_layers_seed0/
```

Each cell used one frame, greedy stream/probe generation, first-token top-k10
capture, and all 36 language-model cache layers. Cache-summary reductions were
promoted to float32 before this run because Qwen's float16 key tensors
overflowed in variance and L2 reductions despite finite activations. All eight
full-layer run JSONs were checked recursively and contained zero non-finite
numeric values.

## Surface And First-Token Readout

The readout was exactly quiet in both source pairs.

For every `MM/JJ/MJ/JM` cell at before, mid, and after:

- family probe output: `A`,
- frequency probe output: `L`,
- mean first-token top-k10 Jaccard across cells: `1.000`,
- common-token factorial interaction: `0.000` at saved precision.

This is stronger than label equality alone: the saved first-token top-k10
logprobs were also identical across the four cells.

## Cache-Summary Separation

The source-cache summaries were not identical.

| Pair | Pairwise max-L2 range | Scalar interaction argmax | Local interaction argmax |
| --- | ---: | --- | --- |
| `c_d` | `11.623` to `62.940` | layer 33 `values`, `-71.090` | layer 33 `values`, position 128, `-51.026` |
| `b_c` | `5.209` to `36.433` | layer 33 `values`, `-34.139` | layer 33 `values`, position 128, `19.341` |

The local interaction magnitude was `130.4%` and `42.3%` of its four-cell grand
mean for `c_d` and `b_c`, respectively. Both the scalar and sequence-position
argmax therefore repeat at late layer 33 `values`; position 128 also repeats.
The local interaction sign flips across source pairs, so the replicated object
is the locus and large non-additivity, not a universal direction.

Mid and after are numerically identical in this one-frame design because the
mid probe occurs immediately after the only frame and no additional stream turn
separates it from after. Their agreement is not temporal persistence evidence.

## Multi-Frame Compatibility Boundary

A two-frame run was attempted with the same `PromptCacheState` path. The second
image turn failed inside MLX-VLM's Qwen2.5-VL RoPE preparation:

```text
ValueError: [broadcast_shapes] Shapes (263) and (221) cannot be broadcast.
```

The cached common prefix trims `input_ids`, while the Qwen attention mask still
describes the full prompt. SmolVLM tolerates the current reuse path; Qwen2.5-VL
0.4.4 does not for a second image turn.

This boundary matters for interpretation:

- the one-frame cache contrast is valid as a cross-architecture trace pilot;
- it is not a replication of the SmolVLM 50-frame persistent-state result;
- a Qwen multi-frame lane needs either an upstream-compatible cache-reuse fix or
  an explicitly labeled cumulative visual-context replay design.

## Current Reading

The narrow result is:

> In two one-frame Qwen2.5-VL cross-palette source pairs, generated labels and
> saved first-token top-k10 readouts were cell-invariant while cache
> summaries retained nonzero spatial-by-palette interaction structure.

The most interesting cross-pair candidate is the layer 33 `values`, position
128 interaction. It should not yet be compared directly with SmolVLM's
50-frame layer 23 `values` locus because model depth, tokenization, image-token
layout, and temporal protocol differ.

## Next Steps

1. Add an explicit cumulative-replay lane for models whose image-turn cache
   reuse is incompatible with the incremental stream path.
2. Repeat the Qwen factorial at early, middle, and late individual frame
   indices before attempting a full replay stream.
3. Store full-vocabulary or teacher-forced forced-choice scores so equality of
   top-k10 is not mistaken for equality of the complete output distribution.
4. Map Qwen sequence positions to text/image token regions before treating
   position 128 as a mechanistic location.

## Tracked Summary

The compact summary is stored at:

```text
examples/research_notes/0023_qwen_cross_model_factorial_pilot/summary.json
```
