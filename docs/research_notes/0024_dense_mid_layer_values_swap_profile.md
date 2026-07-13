# Research Note 0024: Dense Mid-Layer Values-Swap Profile

Date: 2026-07-13

> **Protocol audit update (2026-07-13):** [Note 0027](0027_cache_prefix_audit_and_direct_full_vocab.md)
> invalidated the multimodal cache-branch assumptions used here. The saved
> profile is historical diagnostic output, not current evidence for a causal
> layers 10-13 band.

## Status

Historical two-pair, seed-0 screening scan over layers 8-23. Its numerical
profile is reproducible, but its intervention interpretation is withdrawn.

## Motivation

The controlled four-layer intervention in
[Note 0022](0022_two_pair_values_swap_intervention.md) found a larger top-k
effect at layer 12 than at the repeated layer 23 cache-summary locus. A denser
scan was needed to determine whether layer 12 was an isolated point or part of
a broader intervention-sensitive band.

## Design

The scan reused the same SmolVLM model revision, two 25-frame mid-stream source
pairs, creative probe, 64-token limit, temperature `0.7`, and top-k50 readout as
Note 0022. It changed only the intervention grid:

- layers `8-23`, inclusive,
- `values` tensor only,
- probe seed `0`,
- source-with-donor branch plus source and donor baselines,
- no reciprocal or self-sham branches during screening.

The earlier controlled sweep already showed exact zero self-sham effects at
layers `0`, `12`, `22`, and `23`. That validates the branch machinery at anchor
layers, but it does not replace sham measurements for every newly screened
layer.

Local artifacts:

```text
runs/cache_values_swap/c_d_mid_values_dense_l8_23_seed0_screen.json
runs/cache_values_swap/c_d_mid_values_dense_l8_23_seed0_screen_analysis.json
runs/cache_values_swap/b_c_mid_values_dense_l8_23_seed0_screen.json
runs/cache_values_swap/b_c_mid_values_dense_l8_23_seed0_screen_analysis.json
runs/cache_values_swap/dense_l8_23_seed0_profile_comparison.json
```

## Layer Profiles

The metric is intervention-to-origin top-k RMSE divided by the matched
source-donor baseline RMSE.

| Layer | `c_d` effect / baseline | `b_c` effect / baseline |
| ---: | ---: | ---: |
| 8 | `0.166` | `0.214` |
| 9 | `0.165` | `0.149` |
| 10 | `0.279` | `0.243` |
| 11 | `0.097` | `0.114` |
| 12 | `0.203` | `0.214` |
| 13 | `0.238` | `0.233` |
| 14 | `0.129` | `0.156` |
| 15 | `0.141` | `0.175` |
| 16 | `0.090` | `0.127` |
| 17 | `0.083` | `0.098` |
| 18 | `0.074` | `0.090` |
| 19 | `0.069` | `0.094` |
| 20 | `0.056` | `0.064` |
| 21 | `0.055` | `0.074` |
| 22 | `0.027` | `0.042` |
| 23 | `0.033` | `0.050` |

The cross-pair profile comparison was:

| Common layers | Pearson | Spearman | Mean absolute difference | Argmax | Top-3 overlap |
| ---: | ---: | ---: | ---: | --- | ---: |
| 16 | `0.964` | `0.982` | `0.022` | layer `10` / layer `10` | `3/3` |

Both profiles ranked layers `10`, `13`, and `12` as the top three, in that
order. Both then showed a broad decline after layer 15, with layers 22-23 among
the weakest tested interventions.

## Readout Direction

All 32 source-intervention token sequences remained exactly equal to their
source baseline. The largest perturbation was still origin-like:

- `c_d` layer 10 donor-pull: `-0.565`,
- `b_c` layer 10 donor-pull: `-0.849`.

The profile therefore measures readout susceptibility, not donor-directed
steering. The different donor-pull values also show why perturbation magnitude
and movement direction must remain separate metrics.

## Current Reading

The narrow result is:

> In two seed-0 source-pair screens, whole-layer values-swap readout effects form
> a highly similar mid-layer profile, peaking at layer 10 with the same top-three
> layers, while the late layer 23 summary-stat locus remains weak under direct
> replacement.

This strengthens the dissociation between summary-stat salience and
intervention leverage. It does not establish that layer 10 controls generation,
because token outputs did not change, donor-pull remained negative, and the
dense scan omitted per-layer reciprocal and sham branches.

Pearson and Spearman values here are descriptive profile comparisons. The 16
layers are not independent samples and are not used for inferential p-values.

## Next Confirmation

1. Run layers `10`, `12`, and `13` over at least three matched probe seeds with
   reciprocal and self-sham branches in both source pairs.
2. Compare `keys`, `values`, and matched key-value swaps at the same layers.
3. Test contiguous windows such as `10-13` against individual layers.
4. Add full-vocabulary first-step divergence or teacher-forced candidate scores.

## Tracked Summary

The compact summary is stored at:

```text
examples/research_notes/0024_dense_mid_layer_values_swap_profile/summary.json
```
