# Research Note 0031: Balanced Contrasts and Five-Model Expansion

Date: 2026-07-14

## Status

Completed for five local VLMs over the four independent one-frame source
pairs. This note adds Gemma 3, SmolVLM2, and LFM2-VL to the full-vector cache
surface from Note 0030, completes the LFM2-VL direct readout matrix, and places
all three factorial axes on one coefficient scale.

The current one-frame surface contains:

- 5 models and 4 source pairs,
- 20 direct `MM/JJ/MJ/JM` factorials, 80 cell runs, and 320 first-step
  full-vocabulary sidecars,
- 80 source-only ACK runs and 416 selected cache-tensor sidecars,
- 104 layer-by-pair full-vector factorial analyses in 26 model-local target
  groups, and
- 156 within-target cross-pair interaction-direction cosines.

Including the earlier replay-length trajectories, the valid direct standard
matrix now contains 34 factorial points, 136 cell runs, and 544 compressed
full-vocabulary sidecars.

## Why The Calibration Changed The Reading

The original main-effect formulas divide spatial and palette contrasts by two,
while the interaction formula does not:

```text
spatial main effect = ((JM - MM) + (JJ - MJ)) / 2
palette main effect = ((MJ - MM) + (JJ - JM)) / 2
interaction         = JJ - JM - MJ + MM
```

Those are valid factorial estimands, but their raw RMS or L2 values are not on
the same cell-coefficient scale. Directly comparing them gives interaction a
factor-of-two amplitude advantage and a factor-of-four energy advantage.

The calibrated analysis therefore also records the three Hadamard contrasts:

```text
spatial contrast     = JJ + JM - MM - MJ
palette contrast     = JJ + MJ - MM - JM
interaction contrast = JJ + MM - JM - MJ
```

Every contrast now uses two `+1` and two `-1` coefficients. Squared L2 norms
are normalized across the three axes to produce energy shares. `1/3` is the
exchangeable isotropic reference across these balanced axes; it is a geometric
reference, not a fitted null distribution or significance threshold.

The interaction vectors and their localization from Note 0030 remain real.
What changes is the magnitude interpretation: a raw interaction-to-pairwise
RMS ratio near 1.3 is not, by itself, evidence that interaction is the dominant
factorial axis.

## LFM2-VL Compatibility And Integrity

The fifth model is
[`mlx-community/LFM2-VL-1.6B-4bit`](https://huggingface.co/mlx-community/LFM2-VL-1.6B-4bit)
under MLX `0.31.1` and MLX-VLM `0.4.4`. Its source cache exposes six attention
entries at global layers 2, 5, 8, 10, 12, and 14. Both keys and values were
captured at every entry.

The one-frame effective cache has 216 positions. The processor inserts 80
image tokens at positions 44-123, leaving 44 pre-image and 92 post-image
positions.

Integrity checks:

- 16/16 source-only runs and 192/192 LFM2-VL tensor sidecars are present and
  pass hash, shape, and finite-value checks.
- The independent `e_f/MM` full-vector smoke and batch copy are bitwise exact
  for all 12 tensors.
- 16/16 direct runs and 64/64 full-vocabulary sidecars pass hash, shape, and
  finite-value checks.
- Before each direct probe, all 16 cells share one byte-identical sidecar.
  After each probe, all 16 cell sidecars are distinct.
- The independent direct `e_f/MM` smoke and matrix copy are bitwise exact for
  all four before/after probe distributions.

A four-cell direct batch initially hit one Metal command-buffer timeout after
the first cell. Re-running each direct cell in its own process completed all
remaining cells deterministically. The 16-cell source-only tensor batch ran in
one loaded process without failure. This is an execution-lifetime caveat, not a
failed model-compatibility result.

## Five-Model Full-Vector Result

The table uses the image-token region. Dominance counts report which balanced
axis has the largest energy share in each layer-by-pair analysis.

| Model | Target groups | Analyses | Dominant S / P / I | Interaction share median (range) | Image energy median | Image argmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL | 1 | 4 | 4 / 0 / 0 | 0.3043 (0.2970-0.3331) | 0.9933 | 4/4 |
| InternVL3 | 3 | 12 | 12 / 0 / 0 | 0.2999 (0.2844-0.3403) | 0.9999 | 12/12 |
| Gemma 3 | 4 | 16 | 16 / 0 / 0 | 0.2884 (0.2569-0.3166) | 0.9999 | 16/16 |
| SmolVLM2 | 6 | 24 | 4 / 14 / 6 | 0.2769 (0.2418-0.4892) | 0.9535 | 20/24 |
| LFM2-VL | 12 | 48 | 48 / 0 / 0 | 0.2848 (0.2540-0.3175) | 0.9706 | 48/48 |

Across all 104 analyses:

- the dominant balanced axis is spatial in 84, palette in 14, and interaction
  in 6;
- interaction exceeds the `1/3` reference in 9/104 and is dominant in 6/104;
- 104/104 pre-image regions have exactly zero spatial, palette, and interaction
  effects;
- 100/104 interaction argmax positions are image tokens;
- 102/104 interaction energy fractions are at least 0.9, with a global median
  of 0.9795 and minimum of 0.8622; and
- all 156 same-target cross-pair interaction cosines are positive, but they are
  generally small (`0.0248-0.5623`).

The strongest architecture split is no longer only a layer index. Qwen,
InternVL, Gemma, and LFM2 are spatial-dominant at every captured target and
source pair. SmolVLM is qualitatively different: all six `b_c` targets are
interaction-dominant, most later pairs are palette-dominant, and only four of
its 24 target-by-pair points are spatial-dominant.

This does not make the interaction unimportant. It says that nonzero nonlinear
interaction, image-token localization, and factorial-axis dominance are three
different properties.

## Direct Full-Vocabulary Result

Every one of the 20 one-frame direct factorials has non-identical after-probe
cell distributions for both probe families. All 40 before-probe records have
zero balanced contrast energy; all 40 after-probe records have nonzero energy.

Across the 40 after-probe records, the dominant balanced axis is spatial in 13,
palette in 12, and interaction in 15. Interaction exceeds `1/3` in 18/40.
Only 6/40 records change the generated semantic label across cells, so 34/40
show a changed complete distribution behind one cell-invariant visible label.

The following values are componentwise medians over four source pairs. Because
each component is median-aggregated separately, the three displayed values do
not have to sum exactly to one.

| Model | Family S / P / I | Frequency S / P / I | Cell-varying labels |
| --- | --- | --- | ---: |
| Qwen2.5-VL | 0.206 / 0.160 / 0.680 | 0.215 / 0.411 / 0.394 | 0/8 |
| SmolVLM2 | 0.272 / 0.447 / 0.280 | 0.248 / 0.391 / 0.303 | 0/8 |
| Gemma 3 | 0.321 / 0.323 / 0.356 | 0.265 / 0.414 / 0.294 | 2/8 |
| InternVL3 | 0.653 / 0.092 / 0.273 | 0.363 / 0.064 / 0.408 | 1/8 |
| LFM2-VL | 0.739 / 0.104 / 0.161 | 0.367 / 0.377 / 0.258 | 3/8 |

LFM2-VL itself is source-pair dependent. Its family probe is spatial-dominant
for `c_d`, `d_e`, and `e_f`, but palette-dominant for `b_c`. Its frequency
probe is spatial-dominant for `b_c` and `c_d`, interaction-dominant for `d_e`,
and palette-dominant for `e_f`. The frequency label remains `unclear` in every
cell of all four pairs even though every full-vocabulary distribution changes.

## Two Measurement Surfaces

The source ACK cache and direct readout do not expose the same factorial profile:

| Surface | Records | Dominant S / P / I | Interaction above 1/3 |
| --- | ---: | ---: | ---: |
| Source-cache full vectors | 104 | 84 / 14 / 6 | 9/104 |
| Prompt-conditioned direct readout | 40 | 13 / 12 / 15 | 18/40 |

This difference is descriptive. The ACK cache and direct probe are separate
fresh forwards with different prompts, sequence lengths, and cache states. The
current design cannot identify a causal projection from one to the other.

What it does establish is that "the interaction" is not one scalar property of
an image factorial. Its apparent prominence depends on the measured object:
source-cache coordinates, token region, layer/component, and the
prompt-conditioned probability readout.

## Current Reading

The strongest manuscript-safe reading is now:

> Across five local VLMs and four independent source pairs, controlled
> cross-palette cells produce nonzero, image-localized full-vector cache
> contrasts and non-identical complete first-token readouts. Balanced
> coefficient calibration shows that source-cache interaction is usually not
> the dominant factorial axis, while prompt-conditioned readouts can amplify or
> suppress different axes in model-, pair-, and probe-dependent ways.

This supports a study of distribution-coupled visual perturbation and
measurement-conditioned latent/readout geometry. It does not support a
universal interaction mechanism, a prompt-invariant semantic decoder,
persistent multimodal state, or causal mediation from the source ACK cache to
the direct probe.

## Highest-Value Next Experiments

1. Repeat the four semantically aligned prompt variants on a second source pair
   and add LFM2-VL and InternVL3.
2. Build coefficient-balanced direction nulls by permuting cell assignments
   within each 2x2 design and by adding matched natural, geometric, noise, and
   processor-frequency controls.
3. Resolve the dominant spatial/palette/interaction vectors by KV head and
   image-token position instead of only layer and tensor.
4. Repeat selected source pairs over multiple independently generated fractal
   trajectories; probe seeds alone do not create visual replication.
5. Test readouts that are not forced-choice and compare semantic candidate mass
   with full-distribution geometry.

## Artifacts

```text
runs/full_vector_factorials/lfm2_vl_four_pair_1f_source_cache_seed0/
runs/full_vector_factorials/analyses/lfm2_vl_*/
runs/full_vector_factorials/five_model_replication/
runs/lfm2_vl_factorials/{b_c,c_d,d_e,e_f}_1f_direct_seed0/
examples/research_notes/0031_balanced_contrasts_five_model_expansion/summary.json
```
