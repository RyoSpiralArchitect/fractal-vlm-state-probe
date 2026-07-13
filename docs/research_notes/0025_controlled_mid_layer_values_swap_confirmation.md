# Research Note 0025: Controlled Mid-Layer Values-Swap Confirmation

Date: 2026-07-13

> **Protocol audit update (2026-07-13):** [Note 0027](0027_cache_prefix_audit_and_direct_full_vocab.md)
> found that reciprocal and sham controls did not repair the underlying
> multimodal prefix/cache-length mismatch. The layer-band claim is withdrawn
> until the intervention is rebuilt on a verified suffix path.

## Status

Historical two-pair, three-layer, three-seed run with reciprocal and self-sham
branches. It is no longer treated as a valid causal confirmation.

## Motivation

[Note 0024](0024_dense_mid_layer_values_swap_profile.md) found a reproducible
seed-0 intervention profile over layers 8-23, with layers 10, 13, and 12 as the
top three in both source pairs. The dense scan omitted per-layer reciprocal and
self-sham branches. This run tests whether that mid-layer band survives the
strict controls used in Note 0022.

## Design

- model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, revision
  `482adb537c021c86670beed01cd58990d01e72e4`,
- source pairs: `mandelbrot c -> julia d` and `mandelbrot b -> julia c`,
- context: first 25 frames of the 50-frame stream, probed at `mid`,
- layers: `10`, `12`, and `13`, `values` tensor only,
- probe seeds: `0`, `1`, and `2`,
- creative probe: 64-token limit, temperature `0.7`, saved top-k50,
- branches: source baseline, donor baseline, source-with-donor, reciprocal
  donor-with-source, and source-with-source self-sham.

This gives 18 layer-seed trial cells across the two source pairs. Strict token
history and tensor-shape checks passed for every swap.

Local artifacts:

```text
runs/cache_values_swap/c_d_mid_values_l10_12_13_confirm_seed0_2.json
runs/cache_values_swap/c_d_mid_values_l10_12_13_confirm_seed0_2_analysis.json
runs/cache_values_swap/b_c_mid_values_l10_12_13_confirm_seed0_2.json
runs/cache_values_swap/b_c_mid_values_l10_12_13_confirm_seed0_2_analysis.json
runs/cache_values_swap/l10_12_13_confirm_seed0_2_profile_comparison.json
```

## Controlled Effects

Effect ratios are intervention-to-origin top-k RMSE divided by the matched
source-donor baseline RMSE. Pull remains negative when the intervention stays
closer to its origin than to its target.

| Pair | Layer | Source effect / baseline | Source donor pull | Reciprocal effect / baseline | Reciprocal source pull | Self-sham RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c_d` | 10 | `0.269` | `-0.585` | `0.277` | `-0.577` | `0.000` |
| `c_d` | 12 | `0.224` | `-0.685` | `0.246` | `-0.664` | `0.000` |
| `c_d` | 13 | `0.241` | `-0.638` | `0.258` | `-0.617` | `0.000` |
| `b_c` | 10 | `0.247` | `-0.844` | `0.253` | `-0.841` | `0.000` |
| `b_c` | 12 | `0.217` | `-0.749` | `0.228` | `-0.734` | `0.000` |
| `b_c` | 13 | `0.249` | `-0.822` | `0.254` | `-0.819` | `0.000` |

All 18 source interventions, all 18 reciprocal interventions, and all 18
self-shams preserved the generated token sequence of their respective origin.
The intervention therefore changes the saved top-k distribution without
crossing the sampled decoding boundary.

## Directional Replication

Within each pair, the source and reciprocal layer profiles were almost exact
scaled copies:

| Pair | Pearson | Spearman | Mean absolute difference | Source argmax | Reciprocal argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c_d` | `0.999999` | `1.000` | `0.0156` | 10 | 10 |
| `b_c` | `0.999954` | `1.000` | `0.0078` | 13 | 13 |

The exact maximum is pair-dependent. `c_d` peaks at layer 10; `b_c` peaks at
layer 13, although its layer-10 and layer-13 source ratios differ by only
`0.0019`. Across pairs, the three-point source profiles have Pearson `0.756`,
Spearman `0.500`, and mean absolute difference `0.012`. With only three
selected layers, these correlations are descriptive and are not inferential
statistics.

## Current Reading

The controlled result supports a band, not a universal single-layer point:

> Whole-layer values replacement has reproducible but origin-like readout
> leverage in SmolVLM's layers 10-13. The rank order repeats almost perfectly
> under reciprocal direction within each source pair, while the exact peak
> shifts between layers 10 and 13 across source pairs.

This strengthens the dissociation from the late layer-23 cache-summary locus.
Layer 23 can be statistically salient in a saved summary without being the
strongest single-layer intervention point for this creative top-k readout.

It still does not show donor-directed generation. Every pull is negative, every
token sequence remains origin-identical, and the measurement is top-k50 rather
than full-vocabulary divergence.

## Next Steps

1. Compare `keys`, `values`, and matched key-value swaps at layers 10-13.
2. Test contiguous windows such as `10-13` against the three single layers.
3. Add position-local intervention within the mid-layer band.
4. Save full-vocabulary first-step sidecars or teacher-forced candidate scores.

## Tracked Summary

```text
examples/research_notes/0025_controlled_mid_layer_values_swap_confirmation/summary.json
```
