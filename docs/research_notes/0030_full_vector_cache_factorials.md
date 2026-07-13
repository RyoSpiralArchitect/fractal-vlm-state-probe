# Research Note 0030: Full-Vector Source-Cache Factorials

Date: 2026-07-13

> **Balanced-contrast update (2026-07-14):** [Note 0031](0031_balanced_contrasts_five_model_expansion.md)
> expands this surface to five models and places spatial, palette, and
> interaction contrasts on the same cell-coefficient scale. The localization
> and direction results below remain valid; the raw interaction/pairwise RMS
> ratio should not be read as evidence that interaction is the dominant
> factorial axis.

## Status

Completed for the four independent one-frame source pairs in Qwen2.5-VL and
InternVL3. This is the first current-protocol result that retains the target
source-cache tensor coordinates instead of reducing each layer to scalar
summary statistics.

The measured surface contains:

- 2 models,
- 4 independent source pairs,
- 8 model-by-pair `MM/JJ/MJ/JM` factorials,
- 32 fresh source-only ACK forwards,
- 64 compressed tensor sidecars, and
- 16 target-layer factorial-vector analyses.

Qwen contributes layer 33 `values`. InternVL contributes layers 25, 26, and 27
`values`, the late band identified by the scalar screen in Note 0029.

## What Changed

Earlier current-protocol notes could identify a repeatable scalar locus, but a
scalar L2 contrast discards direction and token localization. The primary
question is now:

> Within one fixed model, layer, tensor, and token region, does the 2x2
> spatial-palette interaction carry substantial energy, where is that energy,
> and does its vector direction repeat across independent source pairs?

This is a narrower object than persistent state and a richer object than a
generated label or layer-level norm.

## Protocol

Each cell uses one selected image, stream seed 20260604, temperature 0, and a
fresh ordered multimodal ACK forward. `source-cache-only` mode omits all
before/after probes. The run therefore measures the source cache directly and
does not reuse image-conditioned cache state.

The capture path:

1. selects an explicit layer and `keys` or `values` tensor,
2. reads the cache entry's effective `offset`,
3. trims block-allocation padding on the sequence axis,
4. promotes the saved array to float32,
5. writes a compressed NumPy sidecar, and
6. records shape, dtype, offset, token count, and SHA-256 in the run JSON.

## Factorial Vectors

For spatial donor `S` and palette donor `P`, the four tensor cells are
`z_MM`, `z_JJ`, `z_MJ`, and `z_JM`. Within each aligned tensor coordinate the
analysis computes:

```text
spatial main effect = ((z_JM - z_MM) + (z_JJ - z_MJ)) / 2
palette main effect = ((z_MJ - z_MM) + (z_JJ - z_JM)) / 2
interaction         = z_JJ - z_JM - z_MJ + z_MM
```

Metrics are reported over the full effective tensor and four sequence
partitions: image tokens, all non-image tokens, the pre-image prefix, and the
post-image suffix. RMS permits comparison across differently sized regions;
L2-squared defines the interaction-energy partition. Cross-pair cosines are
computed only in the same model, layer, tensor, and region.

## Integrity Audit

All 32 run JSONs and 64 sidecars are present. Every sidecar passes its recorded
hash, shape, finite-value, and offset/token-count checks.

The source-only reruns also preserve the older scalar evidence:

- Qwen layer 33: 16/16 target-layer scalar records exactly match the prior
  standard runs.
- InternVL layers 25-27: 48/48 target-layer scalar records exactly match the
  prior standard runs.

An independent `e_f` MM smoke was repeated before the four-pair batch. Qwen
layer 33 and InternVL layers 25-27 are byte-identical to their batch copies:
4/4 SHA-256 matches, `np.array_equal=True`, and maximum absolute difference 0.

This establishes deterministic artifact capture for these four target
tensors. It does not establish statistical generalization.

## Token Layout

MLX cache tensors use block allocation, so raw capacity is larger than the
effective prompt cache:

| Model | Raw sequence capacity | Effective sequence | Image positions | Image tokens | Pre / post |
| --- | ---: | ---: | --- | ---: | ---: |
| Qwen2.5-VL | 256 | 241 | 43-141 | 99 | 43 / 99 |
| InternVL3 | 3,584 | 3,471 | 43-3,370 | 3,328 | 43 / 100 |

All analyses use the effective sequence only. The image-token counts are
processor-specific and are not treated as comparable replay lengths.

## Full-Vector Result

| Model / target | Image energy median (range) | Image interaction RMS | Interaction / pairwise RMS | Image cross-pair cosine | Post-image RMS | Post-image cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen L33 `values` | 0.993279 (0.991147-0.995227) | 6.0176 | 1.3522 | 0.06045 | 0.4751 | 0.68659 |
| InternVL L25 `values` | 0.999897 (0.999885-0.999920) | 3.3085 | 1.3364 | 0.09665 | 0.1895 | 0.36319 |
| InternVL L26 `values` | 0.999880 (0.999864-0.999904) | 4.1029 | 1.3458 | 0.08798 | 0.2551 | 0.36674 |
| InternVL L27 `values` | 0.999838 (0.999822-0.999874) | 4.0806 | 1.3474 | 0.08802 | 0.2904 | 0.35012 |

Three exact localization checks repeat across all 16 layer-by-pair analyses:

- 16/16 interaction maxima fall inside the image-token region.
- 16/16 pre-image prefixes have exactly zero spatial, palette, and interaction
  effect.
- 16/16 interaction contrasts have image-token energy fractions above 0.991.

The zero pre-image result is an expected causal-attention and alignment
control: a cell-specific image later in the sequence cannot alter an earlier
shared prefix. Its exact repetition is evidence that the four cell tensors and
their position partitions are aligned correctly.

## Energy And Direction Are Different Results

The image-token region carries almost all interaction energy, but its
cross-source-pair direction is only weakly aligned. Median image-region
interaction cosines range from 0.060 to 0.097. All 24 same-target pairwise
interaction cosines are positive, yet their small magnitude means the four
source pairs do not instantiate one nearly shared image-space vector.

The post-image suffix shows the opposite profile. Its interaction RMS is much
smaller, but its median direction cosine rises to 0.350-0.367 in InternVL and
0.687 in Qwen. The fixed suffix may transform diverse image-token
perturbations into a more shared downstream coordinate. That is a testable
interpretation, not a demonstrated mechanism.

The image interaction is also directionally more repeatable than either image
main effect in all four target groups. Median interaction cosines are
0.060-0.097, versus 0.038-0.057 for spatial effects and 0.037-0.043 for palette
effects. These differences are descriptive over four source pairs and have no
null calibration yet.

## Updated Reading

The scalar result and the vector result answer different questions.

Qwen's repeated layer 33 `values` scalar argmax says that the largest
layer-level norm interaction occurs at the same target. It does not say that
the underlying image-token interaction points in the same direction for every
source pair. InternVL's late scalar band likewise localizes a repeated region
without establishing a shared vector mechanism.

The strongest current reading is therefore:

> In targeted late `values` tensors of Qwen and InternVL, fresh multimodal 2x2
> spatial-palette interactions are concentrated almost entirely in
> processor-expanded image-token positions. Their high-energy image-space
> directions are source-pair-specific, while the lower-energy fixed
> post-image suffix carries more cross-pair directional alignment.

This sharpens the project from scalar latent-state steering toward localized,
distribution-coupled tensor geometry under fresh inference.

## Claim Boundaries

- The targets were selected from prior scalar screens; this is targeted
  follow-up, not an unbiased all-layer vector search.
- Four source pairs give six descriptive cosine comparisons per target group,
  not six independent experimental replicates.
- Positive cosine values have no matched-null or permutation calibration yet.
- Raw vector directions are not compared across models or layers because their
  cache bases and token layouts are not assumed to align.
- A full-vector interaction norm has no sign. It must not be conflated with
  the negative interaction previously computed from scalar L2 summaries.
- Source-cache tensors and fresh direct-probe distributions remain separate
  condition-level observations, not a causal chain.
- No result establishes persistence, intervention validity, causal mediation,
  semantic steering, a universal layer, or subjective state.

## Next Experiments

1. Extend the same capture to Gemma's early `values` band and SmolVLM's
   pair-dependent maxima, while preserving model-local direction comparisons.
2. Resolve the interaction by KV head and sequence position, then map image
   positions back to processor tiles or patches where the processor exposes a
   reliable layout.
3. Add matched natural, geometric, noise, and processor-frequency controls to
   test whether the image-energy concentration is fractal-specific.
4. Add source pairs and matched null factorials so cosine, energy fraction,
   and post-image alignment can be calibrated by source-level resampling and
   label-preserving permutations.
5. Rebuild a causal intervention only on a multimodal path that first proves
   exact prefix and cache-length compatibility, testing image and post-image
   regions separately.

## Local Artifacts

```text
runs/full_vector_smoke/
runs/full_vector_factorials/qwen_four_pair_1f_source_cache_seed0/
runs/full_vector_factorials/internvl3_four_pair_1f_source_cache_seed0/
runs/full_vector_factorials/analyses/
runs/full_vector_factorials/cross_pair_replication/
```

## Tracked Summary

```text
examples/research_notes/0030_full_vector_cache_factorials/summary.json
```
