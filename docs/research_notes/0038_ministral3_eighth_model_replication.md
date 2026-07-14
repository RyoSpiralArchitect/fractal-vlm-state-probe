# Research Note 0038: Ministral 3 Eighth-Model Replication

Date: 2026-07-14

Status: completed for the balanced four-source-pair, one-frame Ministral 3
direct, prompt-robustness, processor, scalar-cache, and selected full-vector
protocols.

This note extends the independent one-frame core from seven to eight local VLM
architectures. It does not restore cache reuse. Every readout is a fresh
multimodal forward, and every retained source cache comes from a separate fresh
source-only ACK forward.

## What Changed

The eighth model is
`mlx-community/Ministral-3-3B-Instruct-2512-4bit`. It adds a Mistral 3 / Pixtral
vision stack with 26 language layers, eight KV heads, hidden size 3,072, and
head dimension 128. The same four independent fractal source pairs used in the
balanced core were run without changing the factorial, prompt, or artifact
contracts.

The resulting expansion contains:

- 16 standard direct cells and 64 complete-vocabulary sidecars,
- 16 prompt-audit cells and 256 complete-vocabulary sidecars,
- 16 fresh source-only ACK cells and 64 selected cache tensors,
- 16 selected full-vector factorial analyses,
- four processor-space factorial audits, one per source pair.

## Runtime Compatibility

The local runtime is MLX `0.31.1`, MLX-VLM `0.4.4`, and Transformers `4.57.6`.
The model's tokenizer metadata names the Transformers 5
`TokenizersBackend` class, which the local Transformers 4 runtime cannot load.
The probe now has a narrow Mistral 3 fallback that rebuilds the fast tokenizer
from `tokenizer.json`, preserves the repository chat template and stopping
criteria, and delegates model loading to MLX-VLM. Other load failures still
propagate unchanged.

Every fallback run records
`model_load_compatibility=mistral3_transformers4_tokenizer_backend`. No global
package upgrade or model-file mutation was required.

The contract smoke produced `ACK`, retained all 26 cache layers, and resolved
an identity processor-to-cache layout:

| Field | Observed value |
| --- | ---: |
| Effective source-cache positions | 241 |
| Pre-image positions | 38 |
| Image positions | 99 |
| Post-image positions | 96 |
| Image runs | 9 runs of 11 positions |
| Selected tensor shape | `[1, 8, 241, 128]` |

The raw cache allocation is 256 positions, but the validated effective length
is 241. All retained tensor sidecars use the effective sequence length.

## Protocol And Integrity

The four source pairs are `b_c`, `c_d`, `d_e`, and `e_f`; the visual cells are
`MM`, `JJ`, `MJ`, and `JM`. All runs use one frame, stream seed `20260604`,
probe seed `0`, and temperature `0`.

| Check | Result |
| --- | ---: |
| Direct before records, four-cell bitwise equality | 8/8 |
| Direct after records, four distinct sidecars | 8/8 |
| Standard direct sidecar hash / finite checks | 64/64 |
| Standard generated outputs in declared candidates | 32/32 |
| Prompt sidecar hash / finite checks | 256/256 |
| Prompt baseline sidecars equal standard direct sidecars | 64/64 |
| Prompt generated outputs in declared candidates | 128/128 |
| Source-only outputs equal `ACK` | 16/16 |
| Source-only cache layouts resolved as identity | 16/16 |
| Direct/source cache summaries equal | 16/16 cells |
| Selected cache tensor hash / finite checks | 64/64 |

Standard candidate probability mass lies in
`[0.9649438435, 0.9859774277]`; across prompt variants it lies in
`[0.8719290801, 0.9859908939]`. Candidate calibration is therefore usable for
this model, unlike Granite's unresolved generated outputs.

## Processor-Space Audit

Processor tensor mean is palette-dominant in 4/4 pairs. Tensor standard
deviation is palette-dominant in 3/4 and spatial-dominant in 1/4. Spectral
centroid in cycles per patch is spatial-dominant in `b_c`, `d_e`, and `e_f`,
and palette-dominant in `c_d`.

The processor spectral axis agrees with the direct full-vocabulary axis in
5/8 pair-by-probe records: 2/4 family probes and 3/4 frequency probes. A
palette-dominant processor mean therefore does not determine one downstream
readout axis.

## Direct Full-Vocabulary Result

The generated token order below is `MM/JJ/MJ/JM`. Axis shares use balanced,
equal-coefficient spatial/palette/interaction contrasts.

| Pair | Family tokens | Family dominant axis (S/P/I share) | Frequency tokens | Frequency dominant axis (S/P/I share) |
| --- | --- | --- | --- | --- |
| `b_c` | A/A/C/A | interaction (0.070/0.327/0.602) | L/H/L/H | spatial (0.650/0.036/0.314) |
| `c_d` | A/C/C/A | palette (0.070/0.848/0.082) | L/L/L/L | spatial (0.710/0.021/0.269) |
| `d_e` | A/A/C/A | palette (0.131/0.546/0.324) | L/L/L/L | spatial (0.646/0.304/0.050) |
| `e_f` | C/A/B/A | spatial (0.870/0.052/0.078) | L/L/H/L | spatial (0.773/0.032/0.195) |

Across the eight standard model/pair/probe records, dominance is spatial in
5, palette in 2, and interaction in 1. Family full-vocabulary interaction L1
has median `0.7487` and range `0.4549-1.3940`; frequency has median `0.3342`
and range `0.0980-0.5122`.

The visible family label changes in every source pair, while frequency is
visibly invariant in two pairs. The full distributions change in every record,
including those with invariant generated labels.

## Scalar Source-Cache Summary

All four scalar interaction maxima occur in `keys`, but exact layer, sign, and
normalized depth remain pair-dependent.

| Pair | Layer / tensor | Signed interaction | Relative absolute interaction |
| --- | --- | ---: | ---: |
| `b_c` | layer 16 `keys` | -14.5274 | 0.01784 |
| `c_d` | layer 16 `keys` | -9.8043 | 0.01203 |
| `d_e` | layer 1 `keys` | -11.0693 | 0.01321 |
| `e_f` | layer 0 `keys` | 22.4532 | 0.04245 |

Layer 16 is the mode in only 2/4 pairs, the sign is negative in 3/4, and
normalized depth spans `0.00-0.64`. This is a component-level regularity, not
an exact shared locus.

## Selected Full Vectors

Four targets were retained before inspecting their full-vector factorial
results:

- layer 16 `keys`, the modal scalar-argmax location,
- layer 0 `keys` and layer 1 `keys`, the two exception maxima,
- layer 25 `values`, a late depth control not selected by an observed effect.

The first three targets are still post-selection exploratory because they were
chosen from scalar results. Layer 25 is a depth control, not a preregistered
confirmatory target.

Across 16 pair-by-target analyses:

- balanced dominance is spatial/palette/interaction in 15/1/0,
- all 16 pre-image interaction effects are exactly zero,
- all 16 interaction argmaxes fall in image positions,
- median image interaction-energy fraction is `0.9704`, with range
  `0.8927-1.0000`,
- 14/16 image energy fractions exceed 0.9,
- all 16 interaction shares are at or below the exchangeable `1/3` reference.

The sole palette-dominant target is `d_e` layer 25 `values`. Nonzero
interaction and image localization again coexist with overwhelmingly
non-interaction-dominant full vectors.

## Cross-Pair Direction

Pairwise cosine medians compare six dependent source-pair combinations per
target. They are descriptive direction calibration, not independent samples.

| Target | Image interaction cosine median | Post-image interaction cosine median | Image energy median |
| --- | ---: | ---: | ---: |
| layer 0 `keys` | 0.0790 | undefined: post effect is zero | 1.0000 |
| layer 1 `keys` | 0.0876 | 0.4218 | 0.9992 |
| layer 16 `keys` | 0.0523 | 0.3993 | 0.9311 |
| layer 25 `values` | 0.0505 | 0.3332 | 0.9059 |

The eighth architecture therefore repeats the separation between high-energy,
weakly aligned image effects and lower-energy, more aligned post-image effects
at layers 1, 16, and 25. Layer 0 has no post-image interaction at all, so this
late directional structure appears only after the earliest captured layer. The
medians are lower than the strongest Qwen and Granite fractal results and do
not establish a universal direction.

## Prompt Robustness

For Ministral alone, only 2/8 fixed family/variant records retain one generated
four-cell pattern across all four source pairs. Only 1/8 retains one balanced
axis, and 0/8 retains both. Across 48 dependent pairwise records, generated
patterns agree in 21, axes in 19, and both in 8. The median balanced-share L1
difference is `0.8987` with range `0.1035-1.8516`.

Adding Ministral creates an eight-model four-pair prompt matrix:

| Measure | Eight-model result |
| --- | ---: |
| Model/source-pair audits | 32 |
| Prompt cell runs | 128 |
| Complete-vocabulary sidecars | 2,048 |
| Exact baseline sidecars | 512/512 |
| All-pair fixed records | 64 |
| Generated pattern agreement | 39/64 |
| Balanced-axis agreement | 4/64 |
| Both | 2/64 |
| Dependent pairwise records | 384 |
| Generated pattern agreement | 275/384 |
| Balanced-axis agreement | 151/384 |
| Both | 108/384 |

Full categorical compliance does not make Ministral's measured axis robust.
Across the eight-model matrix, visible-pattern stability remains much more
common than all-pair full-distribution axis stability.

## Updated Aggregate Surface

With Ministral included, the valid direct surface contains 62 factorial points,
248 cell runs, and 992 complete-vocabulary sidecars. The independent one-frame
core is balanced over eight models and four source pairs: 32 factorials, 128
cells, and 512 standard direct sidecars.

The selected full-vector surface now contains 192 fresh source-only ACK cells,
704 tensor sidecars, and 176 analyses. Balanced
spatial/palette/interaction dominance is 146/24/6. Of 164 partition-resolved
analyses:

- pre-image interaction is zero in 164/164,
- the interaction argmax is an image position in 160/164,
- image interaction-energy fraction exceeds 0.9 in 160/164.

Interaction share is at or below `1/3` in 165/176 analyses. Phi's 12 analyses
remain position-role-unassigned and are excluded from partition claims.

## Current Reading

Ministral strengthens four bounded observations:

1. Fresh multimodal cross-palette cells produce distinct complete
   distributions in an eighth architecture.
2. Processor mean, direct readout, scalar cache summary, and full-vector
   dominance are different measurement surfaces and do not reduce to one
   palette or geometry main effect.
3. Selected full-vector interaction energy remains concentrated in image
   positions even when interaction is almost never the dominant balanced axis.
4. Prompt-stable visible categories remain a weak proxy for the direction of
   complete-vocabulary factorial change.

The result does not establish persistence, causal cache mediation, one shared
layer, one cross-model vector basis, or a prompt-invariant semantic decoder.
The late post-image direction remains source-class-conditional evidence: it
repeats here over the original fractal panel, but Note 0037 already showed that
the same measurement collapses over matched and non-fractal controls.

## Manuscript-Safe Statement

> Across eight local VLM architectures and four independent one-frame source
> pairs, fresh multimodal cross-palette factorial cells produced non-identical
> complete first-step distributions. In selected source-cache tensors with
> resolved positions, interaction effects were usually concentrated in image
> tokens while balanced vector dominance was usually spatial. Prompt-stable
> generated patterns were substantially more common than prompt- and
> source-pair-stable full-distribution axes. Separate fresh forwards were used,
> so these observations do not test persistent state or causal cache mediation.

## Highest-Value Next Experiments

1. Add independently seeded source pairs within natural, geometry, noise, and
   fractal classes, then build source-label-preserving permutation references.
2. Factor neutral similarity, free description, wording, candidate order, and
   verbalizer mapping instead of relying only on forced domain labels.
3. Resolve Ministral's selected effects by KV head and image patch. Its identity
   cache layout makes it the cleanest next target for position-level analysis.
4. Resolve Phi's expanded image coordinates so all eight models contribute to
   the same partition claims.
5. Rebuild a valid multimodal suffix intervention path with exact prefix and
   cache-length invariants before making any causal test.

## Primary Artifacts

- `runs/ministral3_expansion/c_d_contract_smoke/`
- `runs/ministral3_expansion/processor/four_pair_1f/`
- `runs/ministral3_expansion/four_pair_1f_direct_seed0/`
- `runs/ministral3_expansion/eight_model_four_pair_direct/cross_model_replication.json`
- `runs/ministral3_expansion/four_pair_1f_source_cache_four_target_seed0/`
- `runs/ministral3_expansion/prompt_controls/four_pair_1f_forced_choice_robustness_seed0/`
- `runs/ministral3_expansion/prompt_controls/four_pair_1f_forced_choice_robustness_seed0/eight_model_four_pair_matrix/prompt_robustness_multi_pair_matrix.json`
- `examples/research_notes/0038_ministral3_eighth_model/summary.json`
