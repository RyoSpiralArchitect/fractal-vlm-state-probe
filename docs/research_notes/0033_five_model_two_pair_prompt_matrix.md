# Research Note 0033: Five-Model Two-Pair Prompt Matrix

Date: 2026-07-14

## Status

Completed for five local VLMs, two independent source pairs, four visual
factorial cells, two probe families, and four semantically aligned prompt
variants.

> **Follow-up:** [Note 0034](0034_four_pair_prompt_replication_and_phi35_expansion.md)
> extends the same five-model audit to all four existing source pairs and adds
> an unbalanced Phi-3.5 Vision pilot. Its four-pair result is the current prompt
> replication boundary; the two-pair numbers below remain a reproducible
> intermediate slice.

This extension adds `b_c` prompt audits for Gemma 3, Qwen2.5-VL, SmolVLM2, and
InternVL3. Together with the LFM2-VL `b_c` audit and all five `e_f` audits from
Note 0032, the prompt surface now contains:

- 5 models x 2 source pairs,
- 40 four-cell run records,
- 640 complete first-step vocabulary sidecars, and
- 160 baseline sidecars that are bitwise exact copies of their standard direct
  runs.

All 640 sidecars pass recorded hash, shape, dtype, and finite-value checks. The
new 256 `b_c` sidecars and 64 baseline comparisons pass independently as well.

## Protocol Boundary

Every audit uses one selected image, temperature 0, stream seed 20260604,
probe seed 0, and separate fresh direct multimodal forwards. The eight probes
are baseline, paraphrase, reversed-order, and rotated-label variants for family
and frequency forced choices.

Candidate probabilities are aligned by declared meaning. Complete
first-vocabulary distributions are decomposed with equal-coefficient Hadamard
contrasts:

```text
spatial     = JJ + JM - MM - MJ
palette     = JJ + MJ - MM - JM
interaction = JJ + MM - JM - MJ
```

Squared L2 norms are normalized into spatial, palette, and interaction energy
shares. The direct readout comes from a separate fresh forward from the source
ACK cache. This matrix does not measure a causal cache-to-readout path.

## Paired Replication Criteria

The new matrix analyzer compares the two source pairs within each fixed
model, probe family, and prompt variant. It records three distinct criteria:

1. **Generated-pattern agreement:** the MM/JJ/MJ/JM semantic label pattern is
   identical across `b_c` and `e_f`.
2. **Balanced-axis agreement:** the largest complete-distribution energy share
   is the same spatial, palette, or interaction axis across pairs.
3. **Balanced-share L1:** the L1 distance between the two S/P/I share vectors.

These are descriptive repeated measurements. The 40 paired records are not 40
independent visual replications; the independent visual pair count is two.

## Global Result

| Generated pattern same | Balanced axis same | Both same | Balanced-share L1 median | Range |
| ---: | ---: | ---: | ---: | --- |
| 34/40 | 11/40 | 10/40 | 0.9521 | 0.0359-1.9243 |

The agreement contingency is:

| Generated pattern | Balanced axis | Records |
| --- | --- | ---: |
| same | same | 10 |
| same | different | 24 |
| different | same | 1 |
| different | different | 5 |

Categorical prompt response is therefore much more source-pair-stable than the
dominant full-distribution factorial axis in this matrix. In 24/40 records the
generated pattern repeats while the dominant axis changes. Only 10/40 repeat
both.

The median balanced-share L1 is 0.952 on a three-component probability simplex
whose maximum L1 distance is 2. This is not merely frequent argmax switching
near identical share vectors; many pair changes redistribute substantial
factorial energy.

## Model Profiles

| Model | Generated agreement | Axis agreement | Both | Family generated / axis | Frequency generated / axis | Share-L1 median |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| Gemma 3 | 5/8 | 5/8 | 4/8 | 3/4 / 2/4 | 2/4 / 3/4 | 0.751 |
| InternVL3 | 8/8 | 3/8 | 3/8 | 4/4 / 2/4 | 4/4 / 1/4 | 0.997 |
| LFM2-VL | 6/8 | 0/8 | 0/8 | 2/4 / 0/4 | 4/4 / 0/4 | 1.229 |
| Qwen2.5-VL | 7/8 | 3/8 | 3/8 | 3/4 / 1/4 | 4/4 / 2/4 | 0.798 |
| SmolVLM2 | 8/8 | 0/8 | 0/8 | 4/4 / 0/4 | 4/4 / 0/4 | 1.051 |

SmolVLM is the cleanest categorical/distributional split. All eight generated
patterns repeat across pairs, while none of the eight dominant axes does. All
eight `b_c` records are interaction-dominant; all eight `e_f` records are
palette-dominant.

InternVL likewise repeats every generated pattern, but only 3/8 dominant axes.
Its baseline frequency output is unclear in every cell for both pairs, while
the dominant axis changes from interaction on `b_c` to spatial on `e_f`; the
share-vector L1 is 1.621.

LFM2 retains the result from Note 0032: 6/8 generated patterns agree, no
dominant axes agree, and the median share-vector L1 is the largest of the five
models at 1.229.

Qwen supplies both a near-match and the largest mismatch. Its reversed-order
frequency readout is low-frequency in every cell for both pairs, palette is
dominant in both, and share L1 is only 0.0359. Its rotated-label frequency
readout is high-frequency in every cell for both pairs, but dominance changes
from spatial to palette and share L1 reaches 1.924.

Gemma is the only model where generated-pattern and dominant-axis agreement
have the same marginal count, 5/8 each. Even there, only 4/8 records agree on
both criteria.

## Source-Pair Profiles

The two source pairs produce different aggregate factorial regimes even though
they have the same number of prompt-induced generated-pattern changes.

| Source pair | Changed non-baseline patterns | Shared pattern across all models | Dominant S / P / I | Interaction above 1/3 |
| --- | ---: | ---: | --- | ---: |
| `b_c` | 20/30 | 0/8 | 10 / 9 / 21 | 24/40 |
| `e_f` | 20/30 | 0/8 | 15 / 14 / 11 | 14/40 |

Across both pairs, balanced dominance is spatial in 25/80 records, palette in
23/80, and interaction in 32/80. Interaction exceeds the exchangeable `1/3`
reference in 38/80.

The equal 20/30 generated-change count does not imply equal distributional
behavior. `b_c` is interaction-dominant in more than half of its records,
whereas `e_f` is more evenly split and has only 11/40 interaction-dominant
records. No prompt variant yields one four-cell generated pattern shared by all
five models on either source pair.

## Updated Reading

The result separates three layers of repeatability:

1. prompt-conditioned categorical output,
2. complete-distribution factorial geometry, and
3. source-cache tensor geometry measured in a separate forward.

The first can repeat while the second changes substantially. That does not
prove that the prompt overrides the image, nor that balanced readout axes are a
direct view of one internal representation. It shows that categorical
stability is a weak criterion for distributional visual replication.

The source-pair shift also argues against treating prompt robustness as one
architecture scalar. The same model and prompt formulation can preserve its
visible four-cell pattern while moving between spatial, palette, and
interaction-dominant probability contrasts.

The strongest manuscript-safe statement is:

> Across five local VLMs and two independent visual source pairs,
> prompt-conditioned four-cell generated patterns agreed in 34 of 40 paired
> model-family-variant comparisons, while the dominant equal-coefficient
> full-vocabulary factorial axis agreed in only 11. In 24 comparisons the
> categorical pattern repeated as the dominant spatial, palette, or interaction
> axis changed. Thus categorical prompt-response replication did not establish
> replication of complete-distribution visual factorial geometry.

## Claim Boundaries

- The independent visual replication count is two source pairs. Models,
  families, variants, cells, and sidecars do not multiply that count.
- Prompt controls are diagnostic variants, not a fully crossed orthogonal
  wording/order/label factorial.
- Dominant-axis changes summarize complete first-token distributions; they are
  not direct claims about one hidden-state coordinate or mechanism.
- Balanced-share L1 is descriptive and has no fitted null distribution here.
- Source caches and direct probes are separate fresh forwards, so no causal
  mediation claim follows.
- No result establishes prompt-invariant semantic decoding, persistent state,
  valid cache intervention, semantic steering, or subjective state.

## Highest-Value Next Experiments

1. Build a fully crossed prompt factorial that varies wording, candidate order,
   and label-to-semantics mapping independently with multiple verbalizers.
2. Extend prompt audits to `c_d` and `d_e` so source-pair variability can be
   estimated over all four existing independent pairs.
3. Add neutral candidate scoring, open-ended descriptions, and multi-token
   sequence scoring to reduce dependence on one-letter first-token choices.
4. Add independently generated trajectories plus matched natural, geometric,
   noise, and processor-frequency controls with within-design permutation
   references.
5. Resolve selected source-cache balanced vectors by KV head and image-token
   position while keeping cache and direct-readout surfaces separate.

## Artifacts

```text
runs/source_pair_expansion_50_v1/prompt_controls/
  {gemma3,internvl3,lfm2_vl,qwen,smol}_{b_c,e_f}_1f_forced_choice_robustness_seed0/
  five_model_b_c_1f_forced_choice_robustness_seed0/
  five_model_e_f_1f_forced_choice_robustness_seed0/
  five_model_two_pair_1f_forced_choice_robustness_seed0/
examples/research_notes/0033_five_model_two_pair_prompt_matrix/summary.json
```
