# Research Note 0022: Two-Pair Values-Swap Intervention

Date: 2026-07-13

> **Protocol audit update (2026-07-13):** [Note 0027](0027_cache_prefix_audit_and_direct_full_vocab.md)
> showed that the multimodal source path and creative probe branch did not
> satisfy full-prefix or token/cache-length reuse invariants. All causal and
> layer-susceptibility interpretations in this note are withdrawn pending a
> verified multimodal intervention path.

## Status

Historical run completed for two source pairs, four layers, and three matched
probe seeds. The numerical artifacts are retained, but the intervention path
is not considered technically valid after the Note 0027 audit.

## Question

The true 50-frame cross-palette replication repeatedly placed the mid-phase
scalar interaction argmax at layer 23 `values`. The causal follow-up asked:

> Holding the source transcript, cache token history, and all non-target cache
> tensors fixed, does replacing one layer's `values` tensor move the creative
> probe toward the donor baseline?

## Design

Model:

```text
HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

The cached model revision was
`482adb537c021c86670beed01cd58990d01e72e4`.

Source pairs:

- `mandelbrot c` -> `julia d`,
- `mandelbrot b` -> `julia c`.

Each source and donor stream consumed the first 25 frames, corresponding to the
mid point of a selected 50-frame stream. The intervention sweep used:

- layers `0`, `12`, `22`, and `23`,
- `values` only,
- probe seeds `0`, `1`, and `2`,
- creative probe temperature `0.7`,
- up to 64 generated tokens,
- first-token trajectory top-k50 capture,
- reciprocal donor-with-source swaps,
- a source-with-source self-swap sham at every layer and seed.

Strict token-id equality and tensor-shape equality were required before every
swap. The source and donor contexts passed both checks.

Local artifacts:

```text
runs/cache_values_swap/c_d_mid_values_sweep_seed0_2.json
runs/cache_values_swap/c_d_mid_values_sweep_seed0_2_analysis.md
runs/cache_values_swap/b_c_mid_values_sweep_seed0_2.json
runs/cache_values_swap/b_c_mid_values_sweep_seed0_2_analysis.md
```

## Token-Level Result

Generated creative token sequences did not change.

| Branch comparison | `c_d` | `b_c` |
| --- | ---: | ---: |
| source-with-donor vs source exact | `12/12` | `12/12` |
| donor-with-source vs donor exact | `12/12` | `12/12` |
| source-with-source sham vs source exact | `12/12` | `12/12` |

The source and donor baseline token sequences were also identical within each
matched seed, so a token-distance donor-pull index is undefined rather than
zero.

## Top-K Readout Result

The saved top-k logprobs did move under non-sham swaps, but not toward the donor
baseline.

| Pair | Layer | Baseline RMSE | Swap effect vs origin | Effect / baseline | Donor-pull mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c_d` | 0 | `4.67e-4` | `0` | `0.0%` | `-1.000` |
| `c_d` | 12 | `4.67e-4` | `1.04e-4` | `22.4%` | `-0.685` |
| `c_d` | 22 | `4.67e-4` | `1.22e-5` | `2.6%` | `-0.974` |
| `c_d` | 23 | `4.67e-4` | `1.66e-5` | `3.6%` | `-0.968` |
| `b_c` | 0 | `2.25e-4` | `0` | `0.0%` | `-1.000` |
| `b_c` | 12 | `2.25e-4` | `4.88e-5` | `21.7%` | `-0.749` |
| `b_c` | 22 | `2.25e-4` | `9.15e-6` | `4.1%` | `-0.962` |
| `b_c` | 23 | `2.25e-4` | `1.22e-5` | `5.4%` | `-0.937` |

The donor-pull convention is `-1 = origin-like` and `+1 = target-like`. Layer
12 produced the largest readout perturbation among the four tested layers in
both source pairs, but it remained origin-like. Layer 23 produced only a small
fraction of baseline separation and was strongly origin-like.

All self-swap sham top-k effects were exactly zero. Reciprocal swaps showed the
same ordering: layer 12 moved most, while layers 22 and 23 remained close to the
reciprocal origin.

## Interpretation

The cleanest reading is a dissociation:

> A cache-summary interaction argmax identifies where a collapsed norm statistic
> is most non-additive across `MM/JJ/MJ/JM`; it does not by itself identify a
> single-layer tensor that is sufficient to steer generation.

Several mechanisms remain compatible with the result:

- the layer 23 signal may be descriptive but not causally controlling;
- causal information may be distributed across layers or require coordinated
  key and value changes;
- whole-tensor replacement may miss a position-local effect;
- the creative branch may be insensitive while another readout is not;
- top-k50 can miss full-vocabulary changes.

The repeated layer 12 effect is now the stronger intervention lead, but only
among the four tested layers. It is not yet a global layer scan.

## Claim Boundary

This experiment supports a technically validated negative result for direct
single-layer donor steering under one creative probe. It does not show that
layer 23 is causally irrelevant, and it does not show that the full output
distribution is unchanged.

## Next Steps

1. Scan all layers or a denser layer `8-23` band with shorter fixed readouts.
2. Compare single-layer swaps with contiguous layer-window and `12+23` swaps.
3. Swap keys, values, and matched key-value pairs separately.
4. Swap only aligned sequence-position bands after mapping image-token regions.
5. Compute full-vocabulary first-step divergence or teacher-forced candidate
   scores instead of relying only on saved top-k sets.

## Tracked Summary

The compact summary is stored at:

```text
examples/research_notes/0022_two_pair_values_swap_intervention/summary.json
```
