# Research Note 0035: Six-Model Four-Pair Completion, Phi Full Vectors, and Granite Pilot

Date: 2026-07-14

## Status

Completed for the balanced six-model, four-source-pair direct and prompt
surface. Phi-3.5 Vision now covers `b_c`, `c_d`, `d_e`, and `e_f`, rather than
the single `c_d` pilot reported in Note 0034. Three fixed Phi source-cache
`keys` targets are also complete over all four pairs.

The selected evidence surface now contains:

- 24 independent one-frame direct factorials across 6 models and 4 source
  pairs,
- 24 model/source-pair prompt audits, 96 visual-cell run records, and 1,536
  complete first-step vocabulary sidecars,
- 384/384 prompt-baseline sidecars reproduced bitwise against the standard
  direct runs,
- 16 Phi source-only ACK cells, 48 selected tensor sidecars, and 12 Phi
  full-vector factorial analyses, and
- 96 source-only ACK cells, 464 tensor sidecars, and 116 selected full-vector
  analyses when the Phi expansion is joined to the existing five-model core.

A bounded seventh-architecture Granite Vision pilot adds one `c_d` direct
factorial and one `c_d` prompt audit. Including that deliberately unbalanced
pilot, the wider direct surface contains 39 factorials, 156 cells, and 624
standard sidecars; prompt artifact accounting contains 25 audits, 100 cells,
1,600 sidecars, and 400/400 bitwise-exact baseline sidecars. The balanced
cross-pair matrix remains six models by four pairs.

The earlier two-target Phi `c_d` pilot is superseded by the fixed three-target
four-pair set in these counts; it is not counted twice.

## Protocol Boundary

Every direct and prompt run uses one selected image, temperature 0, stream seed
20260604, probe seed 0, and a fresh multimodal forward. Source-cache tensors
come from separate fresh ACK forwards and are not reused by the direct probes.

The prompt audit retains baseline, paraphrase, reversed-order, and
rotated-label variants for family and frequency. Candidate probabilities are
aligned by declared semantics before comparison. Complete first-step
vocabulary distributions use equal-coefficient Hadamard contrasts:

```text
spatial     = JJ + JM - MM - MJ
palette     = JJ + MJ - MM - JM
interaction = JJ + MM - JM - MJ
```

The three squared L2 norms are normalized into spatial, palette, and
interaction energy shares. The `1/3` value is an exchangeable reference among
these equally scaled axes, not an empirical null distribution.

## Six-Model Prompt Matrix

The balanced matrix has 48 fixed model-family-prompt records. Each record is
observed over the same four source pairs.

| Criterion | All four pairs | Six dependent pairwise comparisons |
| --- | ---: | ---: |
| Generated semantic pattern agrees | 29/48 | 206/288 |
| Dominant balanced axis agrees | 3/48 | 120/288 |
| Both agree | 2/48 | 88/288 |

The pairwise agreement contingency is:

| Generated pattern | Balanced axis | Comparisons |
| --- | --- | ---: |
| same | same | 88 |
| same | different | 118 |
| different | same | 32 |
| different | different | 50 |

Thus the largest cell remains the categorical/distributional split. In
118/288 dependent comparisons, the visible four-cell semantic pattern repeats
while the dominant full-vocabulary spatial, palette, or interaction axis
changes. The median balanced-share L1 distance is 0.753, with range
0.015-1.983 on a simplex whose maximum L1 distance is 2.

| Model | All-pair generated / axis / both | Pairwise generated / axis / both | Share-L1 median |
| --- | --- | --- | ---: |
| Gemma 3 | 4/8 / 1/8 / 1/8 | 28/48 / 23/48 / 18/48 | 0.713 |
| InternVL3 | 7/8 / 0/8 / 0/8 | 45/48 / 20/48 / 18/48 | 0.821 |
| LFM2-VL | 5/8 / 0/8 / 0/8 | 35/48 / 16/48 / 11/48 | 0.854 |
| Phi-3.5 Vision | 2/8 / 1/8 / 0/8 | 21/48 / 20/48 / 8/48 | 0.681 |
| Qwen2.5-VL | 5/8 / 1/8 / 1/8 | 37/48 / 17/48 / 14/48 | 0.772 |
| SmolVLM2 | 6/8 / 0/8 / 0/8 | 40/48 / 24/48 / 19/48 | 0.744 |

Adding Phi increases all-pair axis agreement from 2/40 to 3/48, but the new
Phi axis-stable frequency/rotated-label record does not preserve one generated
pattern. Both-criterion agreement therefore remains 2 records, not 3.

Across all 192 model-pair-family-prompt records, balanced dominance is spatial,
palette, and interaction in 75, 59, and 58 records. Interaction exceeds `1/3`
in 72/192. No source pair has one prompt-variant semantic pattern shared by all
six models.

## Source-Pair Structure

`b_c` remains the most distributionally separated source pair in the prompt
surface. Its comparisons with `c_d`, `d_e`, and `e_f` have median share-L1
0.953, 0.861, and 0.928. Comparisons among the latter three pairs have medians
0.588, 0.622, and 0.610.

This is a descriptive structure over four source pairs. It is not a
population-level cluster or a significance test.

## Phi Direct and Scalar Cache Profile

Phi adds three direct factorials beyond Note 0034, bringing the balanced
six-model surface, before the Granite pilot, to 38 factorial points, 152
visual-cell runs, and 608 standard full-vocabulary sidecars. Every
after-factorial contains non-identical cell distributions.

Phi's standard direct dominant axes are:

| Source pair | Family | Frequency |
| --- | --- | --- |
| `b_c` | spatial | palette |
| `c_d` | palette | palette |
| `d_e` | palette | spatial |
| `e_f` | spatial | interaction |

Its fresh ACK scalar summaries show a compact early-`keys` regularity:

| Source pair | Scalar argmax | Interaction | Relative interaction |
| --- | --- | ---: | ---: |
| `b_c` | layer 1 `keys` | +155.948 | 0.097 |
| `c_d` | layer 3 `keys` | +229.393 | 0.077 |
| `d_e` | layer 3 `keys` | +169.687 | 0.057 |
| `e_f` | layer 3 `keys` | +192.696 | 0.065 |

The component is `keys` in 4/4 pairs, the sign is positive in 4/4, and the
exact layer is 3 in 3/4. Normalized depth spans 0.032-0.097. This is a
repeatable scalar-summary profile within Phi, not evidence that the same full
vector repeats.

## Phi Full-Vector Factorials

Layer 1, layer 3, and layer 27 `keys` were fixed before cross-pair comparison.
Each target has four aligned `MM/JJ/MJ/JM` tensor sidecars per source pair with
shape `[1, 32, 917, 96]`. Hash, dtype, shape, finite-value, and effective-offset
checks pass for all 48 sidecars.

| Layer | Pair | Balanced S / P / I | Dominant axis | Interaction RMS | Argmax position |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | `b_c` | 0.357 / 0.301 / 0.343 | spatial | 0.415 | 124 |
| 1 | `c_d` | 0.418 / 0.246 / 0.336 | spatial | 0.418 | 374 |
| 1 | `d_e` | 0.434 / 0.292 / 0.275 | spatial | 0.354 | 295 |
| 1 | `e_f` | 0.524 / 0.265 / 0.210 | spatial | 0.321 | 312 |
| 3 | `b_c` | 0.405 / 0.284 / 0.312 | spatial | 0.654 | 596 |
| 3 | `c_d` | 0.434 / 0.250 / 0.315 | spatial | 0.674 | 601 |
| 3 | `d_e` | 0.455 / 0.276 / 0.269 | spatial | 0.564 | 567 |
| 3 | `e_f` | 0.507 / 0.259 / 0.234 | spatial | 0.557 | 222 |
| 27 | `b_c` | 0.360 / 0.314 / 0.326 | spatial | 1.730 | 146 |
| 27 | `c_d` | 0.377 / 0.299 / 0.324 | spatial | 1.757 | 57 |
| 27 | `d_e` | 0.370 / 0.324 / 0.306 | spatial | 1.604 | 82 |
| 27 | `e_f` | 0.387 / 0.318 / 0.294 | spatial | 1.606 | 59 |

All 12 selected full tensors are spatial-dominant. The early scalar summary can
therefore emphasize an interaction difference while equal-scale energy over
the complete tensor is spatial-dominant.

Across the six source-pair combinations, interaction-vector cosine medians are
0.274 at layer 1, 0.224 at layer 3, and 0.084 at layer 27. All 18 cosines are
positive, but alignment is weak and decreases with depth. The result supports
a repeatable early scalar profile plus a weakly shared full-vector direction,
not one common Phi interaction vector.

Phi's processor-expanded image positions remain unresolved. The recorded image
marker ID is absent from the reconstructed token sequence, so the analysis now
reports only the complete `all_effective` region. It does not relabel all 917
positions as non-image or assign raw argmax positions an image role.

## Bounded Granite Vision Pilot

`mlx-community/granite-vision-3.2-2b-4bit` completed the source-cache smoke and
passed the fresh direct-readout integrity contracts on the fixed `c_d` pair
under MLX-VLM `0.4.4`. The adapter uses the tokenizer's Granite Vision chat
template. All 16 standard direct sidecars and all 64 prompt-control sidecars
pass hash, vocabulary-size, and log-normalization checks; the 16
prompt-baseline sidecars reproduce the direct run bitwise. The source prompt
returned `The image`, not the requested `ACK`, so source behavior compliance is
not counted as a pass.

The standard direct readouts separate by probe family:

| Probe | Max pair JS | Interaction L1 | Balanced S / P / I | Dominant axis |
| --- | ---: | ---: | --- | --- |
| family | 0.00905 | 0.0775 | 0.660 / 0.212 / 0.128 | spatial |
| frequency | 0.00682 | 0.1888 | 0.266 / 0.166 / 0.567 | interaction |

Forced-choice compliance is weak: direct candidate probability mass spans
0.019-0.065, and generated first tokens are prose tokens rather than declared
candidate labels. The prompt controls sharpen the visible/measured split:

| Probe family | Baseline | Paraphrase | Reversed order | Rotated labels |
| --- | --- | --- | --- | --- |
| family | spatial | spatial | spatial | spatial |
| frequency | interaction | spatial | spatial | spatial |

All 32 generated cell semantics are unresolved because no first token maps to a
declared candidate. Candidate mass across the prompt audit spans 0.019-0.169,
yet the complete distributions remain analyzable. Granite therefore contributes
one bounded diagnostic in which the visible forced-choice readout is unusable
while 7/8 full-vocabulary records are spatial-dominant and 1/8 is
interaction-dominant.

The rotated-family `A` token was absent from that probe's saved top-logprob
slice even though its full-vocabulary probability was present. The analyzer now
shares observed token-ID labels across sibling probes from the same factorial,
with a regression test, rather than declaring the complete candidate summary
missing because one local top-k slice omitted a label.

Granite exposes 40 cache layers. Its standard scalar interaction maximum is
late layer 35 `values` with negative interaction -313.536 and relative magnitude
0.062. The largest sampled position summary is layer 1 `keys`, raw position 709,
also negative. Those positions do not yet receive token roles: the reconstructed
processor layout has 1,474 tokens, while the captured cache has raw shape
`[1, 8, 3328, 64]`, effective offset 3,073, and captured shape
`[1, 8, 3073, 64]`. Full-vector Granite factorials are therefore deferred until
the model-specific processor-to-cache coordinate expansion is reconstructed.

## Updated Reading

What is visible and what is measured must now be kept separate:

- **Visible:** one generated candidate, or one four-cell semantic pattern.
- **Measured readout:** the complete prompt-conditioned first-token
  distribution and its balanced 2x2 contrast.
- **Measured source trace:** a separate fresh ACK cache summary or selected
  full tensor, with token roles used only when the processor-expanded layout is
  identified.

The visible answer is a coarse projection of a conditional response surface.
It can repeat while the measured distribution changes axis, and a stable scalar
cache locus can repeat while full-vector directions remain only weakly aligned.

The strongest manuscript-safe statement is:

> Across six local VLMs and four independent visual source pairs, 29 of 48
> fixed model-family-prompt records retained one generated four-cell semantic
> pattern over every pair, while only 3 retained one dominant equal-coefficient
> full-vocabulary factorial axis and only 2 retained both. In the 288 dependent
> pairwise comparisons, the semantic pattern repeated while the dominant axis
> changed 118 times. Within Phi-3.5 Vision, an early positive `keys` scalar
> profile repeated across all four pairs, but all 12 fixed full-tensor targets
> were spatial-dominant and their cross-pair interaction directions were only
> weakly aligned. In a separate one-pair Granite Vision pilot, none of 32
> prompt-generated first tokens mapped to a declared candidate even though 7/8
> full-vocabulary records were spatial-dominant and 1/8 interaction-dominant.
> The protocol uses separate fresh forwards and does not test persistence or
> causal cache mediation.

## Claim Boundaries

- The independent visual replication count is four source pairs, not 48 prompt
  records or 288 pairwise comparisons.
- Prompt variants are repeated diagnostics, not independent visual evidence or
  a fully crossed language factorial.
- Dominant-axis equality is weaker than equality of the complete share vector.
- Scalar-summary locus replication is weaker than full-vector direction
  replication.
- Phi image/non-image localization is unavailable until its expanded token
  sequence is reconstructed.
- Granite is a one-pair contract pilot, not a seventh balanced replication, and
  its raw cache positions remain role-unassigned until the 1,474-to-3,073
  coordinate expansion is reconstructed.
- Source ACK caches and direct probes are separate fresh forwards; no causal
  cache-to-readout path follows.
- No result establishes persistent state, a valid cache intervention, semantic
  steering, a universal layer, or a prompt-invariant visual decoder.

## Highest-Value Next Experiments

1. Reconstruct Granite's processor-to-cache coordinate expansion, then extend
   it to four source pairs only if token roles and full-vector contracts pass.
2. Cross wording, candidate order, and label-to-semantics mapping as separate
   factors with multiple verbalizers, neutral scoring, and open readouts.
3. Add independent generated trajectories plus matched natural, geometric,
   noise, and processor-frequency controls.
4. Reconstruct Phi's processor-expanded token layout and resolve fixed vectors
   by KV head and token position.
5. Build source-level permutation references before promoting inferential
   language.

## Primary Artifacts

- `runs/phi35_vision_expansion/prompt_controls/six_model_four_pair_1f_forced_choice_robustness_seed0/prompt_robustness_multi_pair_matrix.json`
- `runs/phi35_vision_expansion/six_model_four_pair_direct/cross_model_replication.json`
- `runs/phi35_vision_expansion/four_pair_source_cache_three_target/layer_001_keys_replication.json`
- `runs/phi35_vision_expansion/four_pair_source_cache_three_target/layer_003_keys_replication.json`
- `runs/phi35_vision_expansion/four_pair_source_cache_three_target/layer_027_keys_replication.json`
- `runs/granite_vision_expansion/c_d_1f_direct_seed0/full_vocab_factorial.json`
- `runs/granite_vision_expansion/prompt_controls/granite_vision_c_d_1f_forced_choice_robustness_seed0/prompt_robustness.json`
- `runs/granite_vision_expansion/c_d_1f_source_cache_smoke/mm_capture_mlx.json`
