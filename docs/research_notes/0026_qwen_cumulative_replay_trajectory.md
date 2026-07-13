# Research Note 0026: Qwen Cumulative-Replay Trajectory

Date: 2026-07-13

> **Protocol audit update (2026-07-13):** [Note 0027](0027_cache_prefix_audit_and_direct_full_vocab.md)
> retains the fresh multi-image ACK cache trajectory but withdraws the
> text-only branched readout. The legacy branch reused only `42/241` source
> tokens in the one-frame audit and produced byte-identical full-vocabulary
> sidecars; valid direct probes produce non-identical distributions.

## Status

Partially retained: the fresh 1/2/4-frame ACK cache trajectory over all 36
layers remains valid. The branched readout and its equality claim are
superseded by Note 0027.

## Motivation

[Note 0023](0023_qwen_cross_model_factorial_pilot.md) found a one-frame
cache/readout dissociation in Qwen2.5-VL but could not add a second incremental
image turn because MLX-VLM 0.4.4 produced an attention-mask broadcast error.
The planned alternative was an explicitly separate protocol: deliver all
selected frames in one ordered multi-image turn, then branch a text-only probe
from the resulting cache.

This note asks three narrower questions:

1. Can Qwen consume more than one frame under that explicit protocol?
2. Does the late layer-33 scalar interaction locus repeat over 1, 2, and 4
   frames and across both source pairs?
3. Does the earlier position-128 result survive image-token-aware sampling?

## Protocol

- model: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`, revision
  `46d4cf06a06ffc1a766c214174f9cbed2f45bcab`,
- runtime: MLX `0.31.1`, MLX-VLM `0.4.4`,
- source pairs: `c_d` and `b_c`,
- cells per pair: `MM/JJ/MJ/JM`,
- ordered replay lengths: 1, 2, and 4 frames,
- 24 cell runs total,
- greedy stream and forced-choice probes, probe seed `0`, saved top-k10,
- all 36 cache layers, float32 summary reductions,
- fresh prompt cache for every replay condition.

The replay prompt lists frame index and timestamp in order. MLX-VLM receives the
corresponding image paths as one ordered image list. A short `ACK` turn creates
the cache; family and frequency probes then run as isolated text-only branches.

Local artifacts:

```text
runs/qwen_cumulative_replay/c_d_1f_all_layers_seed0/
runs/qwen_cumulative_replay/c_d_2f_all_layers_seed0/
runs/qwen_cumulative_replay/c_d_4f_all_layers_seed0/
runs/qwen_cumulative_replay/b_c_1f_all_layers_seed0/
runs/qwen_cumulative_replay/b_c_2f_all_layers_seed0/
runs/qwen_cumulative_replay/b_c_4f_all_layers_seed0/
runs/qwen_cumulative_replay/frame_1_2_4_trajectory.json
```

All 24 run JSONs and all six factorial/readout analyses were checked
recursively and contained no non-finite numeric values.

## Readout

Every cumulative-replay cell produced:

- family choice: `C`,
- frequency choice: `L`,
- mean cross-cell first-token top-k10 Jaccard: `1.000`,
- saved common-token factorial interaction: `0.000`.

The original incremental one-frame pilot produced `A/L`, whereas matched
cumulative replay produces `C/L` even at one frame. That is a protocol/prompt
main effect, not a cross-palette condition effect: within each protocol the
four factorial cells remain identical on the saved readout.

## Scalar Cache Trajectory

The scalar interaction argmax was layer 33 `values` in all six pair-by-length
factorials.

| Pair | Frames | Cache token count | Image token count | Interaction | Relative to grand mean | Absolute / frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c_d` | 1 | 241 | 99 | `-65.126` | `0.078` | `65.126` |
| `c_d` | 2 | 371 | 198 | `-100.393` | `0.096` | `50.197` |
| `c_d` | 4 | 631 | 396 | `-194.989` | `0.141` | `48.747` |
| `b_c` | 1 | 241 | 99 | `-34.472` | `0.041` | `34.472` |
| `b_c` | 2 | 371 | 198 | `-94.036` | `0.090` | `47.018` |
| `b_c` | 4 | 631 | 396 | `-141.064` | `0.101` | `35.266` |

Absolute scalar interaction grows with replay length in both source pairs. The
descriptive frame-count/effect correlations are `0.998` for `c_d` and `0.967`
for `b_c`, but each series has only three nested frame-count points. The effect
is not simply proportional to frame count, as the per-frame values show.

The matched cumulative one-frame scalar results are close to the original
incremental one-frame pilot (`c_d -65.126` vs `-71.090`; `b_c -34.472` vs
`-34.139`) and retain the same layer/tensor locus despite different prompts and
visible family labels.

## Position Audit

The original position summary sampled only cache positions 0, midpoint,
penultimate, and final. Qwen cache sequence shapes grew from 256 to 512 to 768,
so the apparent `128 -> 256 -> 384` movement was exactly the moving midpoint of
that sparse grid.

The rerun records Qwen image-token runs and samples each run's start, midpoint,
and end, plus vision markers and token-count anchors. The local argmax became:

| Pair | Frames | Layer/tensor | Position | Token-region role |
| --- | ---: | --- | ---: | --- |
| `c_d` | 1 | layer 33 `values` | 128 | first image-token run interior |
| `c_d` | 2 | layer 33 `values` | 144 | second image-token run start |
| `c_d` | 4 | layer 33 `values` | 242 | second image-token run end |
| `b_c` | 1 | layer 33 `values` | 141 | first image-token run end |
| `b_c` | 2 | layer 33 `values` | 141 | first image-token run end |
| `b_c` | 4 | layer 33 `values` | 315 | third image-token run interior and token-count midpoint |

The local layer remains 33 `values` under the enriched sampler, but the exact
position is not stable across pair and replay length. Position 128 is therefore
withdrawn as a fixed mechanistic candidate. The supported locus is currently
the scalar late-layer interaction, with local concentration somewhere in the
image-token region.

## Current Reading

> Under an explicitly labeled cumulative multi-image replay protocol,
> Qwen2.5-VL preserves a cell-invariant forced-choice/top-k10 readout while the
> scalar cross-palette interaction repeatedly peaks at layer 33 `values` across
> two source pairs and 1/2/4-frame contexts.

This advances the cross-model lane beyond a single image, but it does not close
the persistent-stream claim. All frames are encoded in one model turn; there is
no evidence that this is computationally equivalent to incremental temporal
delivery. It also does not establish full-vocabulary equality.

## Next Steps

1. Extend the same cumulative trajectory to 8 and 16 frames.
2. Add full-vocabulary first-step or teacher-forced forced-choice scoring.
3. Compare image-run-local interaction profiles rather than one sampled argmax.
4. Test an upstream-compatible Qwen incremental-cache fix separately from the
   cumulative protocol.
5. Reuse the replay/trajectory contract on a third local VLM architecture.

## Tracked Summary

```text
examples/research_notes/0026_qwen_cumulative_replay_trajectory/summary.json
```
