# Research Note 0036: Granite Four-Pair Completion and Cache Coordinates

Date: 2026-07-14

## Status

Completed for Granite Vision over the four independent one-frame source pairs
`b_c`, `c_d`, `d_e`, and `e_f`. The one-pair contract pilot in Note 0035 is now
superseded by:

- 4 standard direct factorials, 16 visual cells, and 64 complete-vocabulary
  sidecars,
- 4 prompt audits, 16 visual cells, and 256 complete-vocabulary sidecars,
- 64/64 prompt-baseline sidecars reproduced bitwise against the standard
  direct runs,
- 16 fresh source-only ACK cells, 48 selected tensor sidecars, and 12
  full-vector factorial analyses, and
- one validated processor-to-cache coordinate map for the single-image Granite
  source-cache protocol.

Joining Granite to the preceding six-model core produces a balanced seven-model
by four-source-pair one-frame surface. The direct core contains 28 factorials,
112 cells, and 448 standard sidecars. The prompt core contains 28 model/pair
audits, 112 cells, and 1,792 sidecars, with 448/448 baseline sidecars reproduced
bitwise. The wider valid direct surface, including nested lengths from earlier
models, contains 42 factorials, 168 cells, and 672 sidecars.

The selected source-cache tensor surface now contains 112 ACK cells, 512 tensor
sidecars, and 128 analyses across seven VLMs. The earlier Granite `c_d` smoke is
superseded by the fixed three-target four-pair batch and is not counted twice.

## Protocol Boundary

Every direct and prompt cell uses one selected image, temperature 0, stream seed
20260604, probe seed 0, and a fresh multimodal forward. Source-cache tensors
come from separate fresh ACK forwards and are not reused by direct probes.

The prompt audit retains baseline, paraphrase, reversed-order, and
rotated-label variants for family and frequency. The full-vocabulary spatial,
palette, and interaction contrasts use the same equal-coefficient Hadamard
formulas as Notes 0031-0035.

The Granite coordinate mapping is deliberately narrow. It applies only when:

1. the model type is `granite_vision`,
2. exactly one contiguous processor image-placeholder run is present,
3. all cache entries report one common effective sequence length, and
4. replacing that placeholder run closes the cache-length arithmetic exactly.

Unknown length mismatches and multi-run Granite layouts remain role-unassigned.
This is a validated single-image coordinate contract, not a generic multi-image
Granite rule.

## Processor-To-Cache Coordinate Resolution

The Granite source prompt contains 1,472 processor tokens:

```text
51 prefix + 1,317 image placeholders + 104 suffix = 1,472
```

The generated source response contributes two tokens, so the saved processor
history contains 1,474 tokens. Granite replaces the 1,317 placeholder tokens
with 2,916 vision features before the language-model cache is formed:

```text
51 prefix + 2,916 image positions + 106 suffix/response = 3,073
```

The resulting effective-cache map is:

| Region | Effective cache positions | Count |
| --- | --- | ---: |
| pre-image | 0-50 | 51 |
| image | 51-2966 | 2,916 |
| post-image | 2967-3072 | 106 |

The expansion delta is 1,599 positions. The raw cache allocation has shape
`[1, 8, 3328, 64]`, the effective offset is 3,073, and saved target tensors are
trimmed to `[1, 8, 3073, 64]`. The partition closes exactly in the runtime smoke
and in every selected factorial analysis.

## Granite Direct Four-Pair Result

The standard direct dominant axes are pair-dependent:

| Source pair | Family axis | Frequency axis | Scalar cache argmax |
| --- | --- | --- | --- |
| `b_c` | spatial | interaction | layer 33 `keys`, +149.800 |
| `c_d` | spatial | interaction | layer 35 `values`, -313.536 |
| `d_e` | palette | palette | layer 33 `keys`, +152.376 |
| `e_f` | interaction | palette | layer 0 `keys`, -199.328 |

All eight complete distributions are non-identical across their four visual
cells. The cache-summary mode is `keys` in 3/4 pairs, but exact layer agreement
is only 2/4, sign agreement is 2/4, and normalized depth spans 0-0.897. Granite
therefore adds another architecture whose direct readout and scalar cache locus
depend on source pair; it does not add a universal layer candidate.

## Seven-Model Prompt Matrix

The balanced matrix contains 56 fixed model-family-prompt records, each
observed over the same four source pairs.

| Criterion | All four pairs | Six dependent pairwise comparisons |
| --- | ---: | ---: |
| Generated pattern agrees | 37/56 | 254/336 |
| Dominant balanced axis agrees | 3/56 | 132/336 |
| Both agree | 2/56 | 100/336 |

The pairwise contingency is:

| Generated pattern | Balanced axis | Comparisons |
| --- | --- | ---: |
| same | same | 100 |
| same | different | 154 |
| different | same | 32 |
| different | different | 50 |

The median balanced-share L1 distance is 0.743, with range 0.015-1.983.
Across all 224 model-pair-family-prompt records, balanced dominance is spatial,
palette, and interaction in 87, 68, and 69 records; interaction exceeds the
exchangeable `1/3` reference in 85/224.

Granite requires a careful reading. None of its 128 prompt-generated first
tokens maps to a declared candidate, and candidate probability mass spans only
0.018-0.169. Consequently all eight Granite generated four-cell patterns are
the same unresolved pattern over all source pairs. That raises the seven-model
generated-pattern agreement count by 8, but it is missing categorical
compliance rather than positive semantic replication.

The measured full-vocabulary result moves in the opposite direction:

- Granite all-pair generated-pattern agreement: 8/8,
- Granite all-pair dominant-axis agreement: 0/8,
- Granite pairwise generated-pattern agreement: 48/48,
- Granite pairwise dominant-axis agreement: 12/48, and
- Granite pairwise balanced-share L1 median: 0.654.

Across Granite's 32 prompt records, dominant axes are spatial/palette/
interaction in 12/9/11. No fixed family/variant retains one dominant axis over
all four source pairs. The visible channel is unresolved everywhere while the
complete distribution remains structured and pair-dependent.

## Granite Full-Vector Factorials

Layer 1 `keys`, layer 35 `values`, and layer 39 `values` were fixed before the
four-pair comparison. Layer 1 was selected from the `c_d` sampled-position
maximum, layer 35 from the `c_d` scalar maximum, and layer 39 as a late control.
All 48 sidecars pass hash, dtype, shape, finite-value, and effective-offset
checks.

| Target | Median balanced S / P / I | Dominant pairs | Image energy median | Image argmax | Pre-image zero |
| --- | --- | ---: | ---: | ---: | ---: |
| layer 1 `keys` | 0.409 / 0.304 / 0.290 | spatial 4/4 | 0.999991 | 4/4 | 4/4 |
| layer 35 `values` | 0.374 / 0.321 / 0.302 | spatial 4/4 | 0.999624 | 4/4 | 4/4 |
| layer 39 `values` | 0.358 / 0.335 / 0.306 | spatial 4/4 | 0.999366 | 4/4 | 4/4 |

The scalar summary can emphasize an interaction maximum at layer 35 while the
equal-scale full tensor is spatial-dominant. This repeats the scalar-locus
versus vector-energy separation seen in Phi, now with token regions resolved.

The direction result separates the high-energy image region from the
low-energy fixed suffix:

| Target | Image interaction cosine median | Post-image interaction cosine median |
| --- | ---: | ---: |
| layer 1 `keys` | 0.124 | 0.099 |
| layer 35 `values` | 0.088 | 0.696 |
| layer 39 `values` | 0.086 | 0.653 |

Image tokens contain nearly all interaction energy, but their cross-pair
directions are only weakly aligned. At the late targets, the much smaller
post-image interaction is strongly aligned across source pairs. This supports
a candidate fixed-suffix transformation account; it does not establish that
the suffix mediates the direct readout or that one shared cross-model basis
exists.

Adding Granite changes the complete selected full-vector surface to:

- 128 analyses with spatial/palette/interaction dominance 108/14/6,
- 116 analyses with identified image partitions,
- 116/116 exact-zero pre-image interactions,
- 112/116 interaction argmaxes in image tokens,
- 114/116 image interaction-energy fractions above 0.9, and
- 117/128 interaction shares at or below `1/3`.

Phi's 12 analyses remain role-unassigned and are not included in the
partition-localization denominators.

## Updated Reading

The seventh balanced architecture strengthens two separations.

First, categorical output availability is not measurement validity. Granite
never emits a declared candidate first token, yet its complete distributions
form stable, non-identical factorial contrasts. Its apparent 8/8 categorical
agreement is therefore a replicated absence, while its measured axis changes
in 8/8 fixed records across source pairs.

Second, scalar maxima, full-tensor energy, and cross-pair direction are three
different objects. Granite's scalar maximum changes layer, component, and sign;
all selected full tensors are spatial-dominant; image-region interaction
directions are weakly shared; and late post-image directions are strongly
shared despite carrying little energy.

The manuscript-safe statement is:

> Across seven local VLMs and four independent visual source pairs, every fresh
> direct factorial produced non-identical complete first-step distributions.
> Across prompt controls, 37/56 fixed records retained one generated four-cell
> pattern, but only 3/56 retained one dominant full-distribution factorial axis
> and only 2/56 retained both. Granite contributed 8/8 generated-pattern
> agreements only because every generated candidate semantic was unresolved;
> none of its eight measured axes repeated over all four pairs. Its validated
> cache-coordinate map localized nearly all selected interaction energy to
> image positions, while late low-energy post-image directions aligned more
> strongly across pairs than high-energy image directions. Separate fresh
> forwards were used, so persistence and causal cache mediation were not
> tested.

## Claim Boundaries

- The independent visual replication count is four source pairs, not 56 prompt
  records or 336 pairwise comparisons.
- Granite's unresolved generated patterns are missing forced-choice compliance,
  not positive agreement on one visual category.
- The coordinate map is validated for one image-placeholder run only.
- Dominant-axis equality is weaker than equality of the complete share vector.
- Scalar-summary locus replication is weaker than full-vector direction
  replication.
- Image localization and suffix alignment do not establish causal mediation.
- Source ACK caches and direct probes are separate fresh forwards.
- No result establishes persistent state, semantic steering, a universal layer,
  or a prompt-invariant visual decoder.

## Highest-Value Next Experiments

1. Add independently generated fractal trajectories and matched natural,
   geometric, noise, and processor-frequency controls at one frame before
   extending nested lengths.
2. Cross wording, candidate order, and label-to-semantics mapping as separate
   factors with neutral scoring and open readouts.
3. Resolve Phi's processor-expanded image-token layout and compare its fixed
   vectors by KV head and token position.
4. Build source-level permutation references over a larger independent source
   set before promoting inferential language.
5. Rebuild cache intervention logic only after a multimodal suffix path passes
   exact prefix, length, tensor-shape, reciprocal, and sham invariants.

## Primary Artifacts

- `runs/granite_vision_expansion/seven_model_four_pair_direct/cross_model_replication.json`
- `runs/granite_vision_expansion/prompt_controls/seven_model_four_pair_1f_forced_choice_robustness_seed0/prompt_robustness_multi_pair_matrix.json`
- `runs/granite_vision_expansion/four_pair_1f_source_cache_three_target_seed0/replication/layer_001_keys.json`
- `runs/granite_vision_expansion/four_pair_1f_source_cache_three_target_seed0/replication/layer_035_values.json`
- `runs/granite_vision_expansion/four_pair_1f_source_cache_three_target_seed0/replication/layer_039_values.json`
- `runs/granite_vision_expansion/c_d_1f_cache_layout_smoke/mm_capture_mlx.json`
