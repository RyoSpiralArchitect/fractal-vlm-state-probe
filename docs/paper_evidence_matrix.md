# Paper Evidence Matrix

Last updated: 2026-07-14

This file is the compact bridge from run artifacts to a manuscript draft. It
separates valid observations, descriptive interpretation, withdrawn results,
and missing evidence. Counts refer to actual run units; pairwise distances from
the same four factorial cells are not independent samples.

## Current Evidence Units

| Evidence unit | Model / protocol | Replication unit | Main observation | Claim strength | Primary artifact |
| --- | --- | --- | --- | --- | --- |
| Cross-palette input audit | SmolVLM processor, 50-frame `MM/JJ/MJ/JM` | 2 source pairs | Palette donor and spatial luminance-rank field interact non-additively in raw and processor space | replicated input-level observation | [Note 0020](research_notes/0020_true_50_frame_cross_palette_replication.md) |
| Control-specificity panel | Qwen and Granite, fresh direct probes plus selected source-only tensors | 2 new fractal pairs, 3 dependent matched ablations, and 3 non-fractal panels; 64 direct cells, 256 sidecars, 64 source cells, 128 tensors | Raw marginals are preserved; all 32 after records are distributionally non-identical; interaction and image localization generalize, while late direction alignment collapses when six heterogeneous controls are pooled | historical two-model control observation; directional interpretation superseded by Notes 0039 and 0040 | [Note 0037](research_notes/0037_control_specificity_panel_and_conditional_cache_directions.md) |
| Seeded fixed-pairing direction panel | Qwen and Ministral, fixed source-only tensor targets | 3 fixed ordered generator pairings x 4 seeds; 96 source cells, 192 tensors, 48 analyses | Exact source-level permutations find stronger alignment within each seeded group than between groups at all four targets and two regions (`p=6/34,650`), but every group contains only one ordered generator pairing | valid pairing-conditioned seed observation; broad-class interpretation superseded by Note 0040 | [Note 0039](research_notes/0039_seeded_source_class_direction_permutation.md) |
| Generator-pairing direction hierarchy | Qwen and Ministral, fixed source-only tensor targets plus fresh direct probes | 2 broad classes x 4 ordered pairing families x 2 seeds; 128 source cells, 256 tensors, 64 vector analyses; 128 direct cells, 512 sidecars | All 8 exact matching tests support same-pairing seed repetition; under 70 pairing-family block assignments only Qwen image-region transfer is unadjusted `p < 0.05`, and no post-image test is | two-model hierarchical exact test separating seed repetition from pairing transfer | [Note 0040](research_notes/0040_generator_pairing_direction_hierarchy.md) |
| Cache-prefix audit | MLX-VLM `0.4.4`, Qwen and SmolVLM reuse paths | 2 audit runs, 7 available checks | No checked incremental or text-only branch reuse path retained a safe full prefix/cache-length relation | direct protocol-failure observation | [Note 0027](research_notes/0027_cache_prefix_audit_and_direct_full_vocab.md) |
| Qwen direct factorial trajectory | Qwen2.5-VL-3B 4bit, fresh ACK plus fresh direct probes | 6 fractal pairs at 1 frame; 2 of them extend to 2/4/8/16; 56 cells total | All 14 direct after-factorials are non-identical; fresh ACK scalar argmax is layer 33 `values` at all 14 points, while the added `g_h` point leaves the sign negative in 13/14 and 5/6 one-frame pairs | exact scalar-locus replication with a revised sign boundary | [Note 0037](research_notes/0037_control_specificity_panel_and_conditional_cache_directions.md) |
| SmolVLM direct factorial trajectory | SmolVLM2-2.2B, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 2 of them extend to 2/4; 32 cells total | All 8 direct after-factorials are non-identical; one-frame ACK argmax spans layers 1/21/22 and keys/values | replicated pair-dependence under the valid protocol | [Note 0028](research_notes/0028_source_pair_replication_and_prompt_robustness.md) |
| Gemma 3 direct factorial trajectory | Gemma-3-4B-it 4bit, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 2 of them extend to 2; 24 cells total | All 6 direct after-factorials are non-identical; all four one-frame maxima are early `values`, but exact layer and sign vary; frequency readout can change sharply | component-level regularity plus pair-dependent exact locus | [Note 0028](research_notes/0028_source_pair_replication_and_prompt_robustness.md) |
| InternVL3 direct factorial replication | InternVL3-2B 4bit, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 16 cells, 64 sidecars | All four direct after-factorials are non-identical; all ACK maxima are late layer 25-27 `values` with negative sign | component/sign/depth-band replication with pair-dependent exact layer | [Note 0029](research_notes/0029_cross_model_prompt_and_internvl_expansion.md) |
| LFM2-VL direct factorial replication | LFM2-VL-1.6B 4bit, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 16 cells, 64 sidecars | All after-cell distributions are distinct; visible family labels vary in 3/4 pairs while frequency labels stay fixed; balanced readout axes remain pair-dependent | fifth-architecture replication with deterministic artifact integrity | [Note 0031](research_notes/0031_balanced_contrasts_five_model_expansion.md) |
| Selected full-vector source-cache surface | Nine VLMs, with fixed-target control, seeded-pairing, and pairing-hierarchy extensions, fresh source-context forwards | 360 cell runs, 1,064 tensors, 266 analyses | Balanced axis dominance is spatial/palette/interaction in 227/32/7; in 254 partition-resolved analyses every pre-image effect is zero and 248 argmaxes are image tokens | targeted vector localization plus exact pairing-conditioned direction calibration | [Note 0041](research_notes/0041_fastvlm_ninth_model_replication.md) |
| Cross-model direct aggregate | Nine-model four-pair core, nested earlier lengths, and two-model control/pairing extensions, complete first-step vocabulary | 104 factorial points, 416 cells, 1,664 sidecars | Every direct after-factorial is non-identical; balanced readout dominance is model-, pair-, generator-, and probe-dependent | balanced nine-architecture core plus bounded two-model extensions | [Note 0041](research_notes/0041_fastvlm_ninth_model_replication.md) |
| Four-pair prompt robustness core | Eight VLMs on `b_c`, `c_d`, `d_e`, and `e_f`, fresh direct probes | 32 model/source-pair audit units, 128 cell runs, 2,048 sidecars | All 512 baseline sidecars repeat bitwise; generated patterns agree over all four pairs in 39/64 records, balanced-axis dominance in 4/64, and both in 2/64 | eight-model four-pair categorical versus distributional replication matrix | [Note 0038](research_notes/0038_ministral3_eighth_model_replication.md) |
| Phi-3.5 Vision full expansion | Phi-3.5 Vision 4bit on four source pairs, fresh direct probes and source-only ACK | 16 standard direct cells and 64 sidecars; 16 prompt cells and 256 sidecars; 16 ACK cells, 48 tensors, 12 full-vector analyses | Scalar ACK maxima are early positive `keys` in 4/4; all selected full tensors are spatial-dominant; cross-pair direction alignment is weak and image-token partition is unresolved | sixth-architecture scalar-locus versus vector-direction replication | [Note 0035](research_notes/0035_six_model_four_pair_completion_and_phi_full_vector.md) |
| Granite Vision full expansion | Granite Vision 3.2 2B 4bit on four source pairs, fresh direct probes and source-only ACK | 16 direct cells and 64 sidecars; 16 prompt cells and 256 sidecars; 16 ACK cells, 48 tensors, 12 full-vector analyses | All 128 generated candidate semantics are unresolved while no measured prompt axis repeats over all four pairs; all selected full tensors are spatial-dominant and image-localized under a validated single-image coordinate map | seventh-architecture visible/measured separation plus token-region full-vector replication | [Note 0036](research_notes/0036_granite_four_pair_completion_and_cache_coordinates.md) |
| Ministral 3 full expansion | Ministral-3-3B 4bit on four source pairs, fresh direct probes and source-only ACK | 16 direct cells and 64 sidecars; 16 prompt cells and 256 sidecars; 16 ACK cells, 64 tensors, 16 full-vector analyses | All generated candidates resolve; scalar maxima use `keys` in 4/4 but vary in layer/sign; selected vectors are spatial/palette/interaction-dominant in 15/1/0 and image-localized in 16/16 | eighth-architecture component-level scalar and token-region vector replication | [Note 0038](research_notes/0038_ministral3_eighth_model_replication.md) |
| FastVLM full expansion | FastVLM-0.5B bf16 on four source pairs, fresh direct probes and source-context forwards | 16 direct cells and 64 sidecars; 16 source cells, 48 tensors, 12 full-vector analyses | All eight after records are non-identical; all selected vectors are spatial-dominant and image-localized, with positive four-pair image and post-image cosines; source response is `The image`, not `ACK` | ninth-architecture fresh-forward replication with explicit suffix difference | [Note 0041](research_notes/0041_fastvlm_ninth_model_replication.md) |

## Withdrawn From The Evidence Set

| Historical result | Why withdrawn | What remains usable |
| --- | --- | --- |
| SmolVLM 25/50-frame persistent cache trajectories | Reconstructed history omitted processor-inserted image tokens and failed prefix/cache-length invariants | deterministic stimuli, manifests, raw image statistics, and processor-space statistics |
| Mid/after equality from text-only cache branches | The branch did not retain the full multimodal prefix; a Qwen legacy rerun produced byte-identical sidecars across image cells | fresh source-cache observations made in one valid multimodal forward |
| Layer 23 and layers 10-13 values-swap conclusions | Reciprocal and sham controls did not repair the invalid underlying multimodal suffix path | intervention code as a historical scaffold and audit target |
| Top-k or generated-letter equality claims | Partial readout cannot establish equality over the complete vocabulary | saved top-k records as descriptive slices only |

## Draft Claim Ladder

### Supported Now

1. Luminance-rank cross-palette transfer is not an additive palette control; it
   creates non-additive input and processor-space perturbations.
2. The audited MLX-VLM `0.4.4` incremental and text-only branch paths do not
   preserve the multimodal prefix needed for persistence or intervention
   interpretation.
3. Under fresh direct multimodal probes, every one of the 104 tested
   `MM/JJ/MJ/JM` factorials has non-identical full-vocabulary first-step
   distributions at saved numerical precision. The independent one-frame core
   is balanced over nine models and four source pairs; the wider surface adds
   bounded nested-length, control, seeded-pairing, and pairing-hierarchy
   extensions.
4. At one frame, Qwen's layer 33 `values` scalar interaction locus repeats over
   six independent fractal pairs, but its negative sign now repeats in 5/6.
   InternVL repeats a late negative `values` band over the original four, Gemma
   stays in early `values` but changes layer/sign, SmolVLM changes
   layer/component/sign, LFM2 exposes six hybrid-attention cache entries, and
   Granite changes layer/component/sign across pairs. Ministral uses `keys` in
   4/4 while changing exact layer and sign.
5. Across 266 selected full-vector cache analyses in nine VLMs, balanced
   spatial/palette/interaction dominance occurs in 227/32/7 analyses. In the
   254 analyses with identified token partitions, all shared pre-image prefixes
   have zero effect and 248 interaction maxima fall in image tokens.
6. A generated letter or top-k set can remain fixed while the complete
   distribution changes; visible-label equality is not distribution equality.
7. Across eight models and four prompt-audited source pairs, generated semantic
   patterns agree over all four pairs in 39/64 fixed model-family-variant
   records, while balanced-axis dominance agrees in 4/64 and both criteria in
   only 2/64. Every one of the 512 baseline sidecars reproduces exactly.
8. Over the six dependent source-pair comparisons per prompt record, generated
   patterns agree in 275/384 comparisons and balanced-axis dominance in
   151/384. In 167 comparisons, the categorical pattern repeats while the
   dominant full-distribution axis changes.
9. Phi repeats an early positive `keys` scalar-summary profile over all four
   pairs. All 12 fixed full-vector `keys` targets are spatial-dominant, while
   cross-pair interaction-vector cosine medians decrease from 0.274 at layer 1
   to 0.084 at layer 27. Its image-token partition is not yet identified.
10. Granite passes direct and prompt full-vocabulary integrity checks on all
    four source pairs. None of its 128 prompt-generated first tokens maps to a
    declared candidate; all eight generated patterns therefore repeat as
    unresolved, while the measured dominant axis repeats in 0/8 records.
11. Ministral passes direct and prompt full-vocabulary integrity checks on all
    four pairs, resolves every generated candidate, and still retains both one
    generated pattern and one measured axis in 0/8 fixed prompt records. All 16
    selected tensor interactions peak in image positions.
12. Granite's validated single-image coordinate map replaces 1,317 processor
    placeholders with 2,916 vision positions. All 12 original four-pair tensors
    are spatial-dominant and image-localized; the added `g_h` pair remains
    image-localized but is palette-dominant at all three fixed targets.
13. In the Qwen/Granite control panel, raw palette-donor marginals are preserved
    to `1.11e-16`, processor tensor means are palette-dominant in 16/16
    model-by-panel records, and processor spectral axes agree across models in
    only 2/8 panels.
14. All 32 Note 0037 full-vector analyses localize interaction maxima to image
    positions and have zero pre-image effects, but interaction dominates 0/32.
    Late post-image alignment collapses when its six heterogeneous matched or
    non-fractal controls are pooled; Notes 0039 and 0040 supersede the resulting
    fractal-specific interpretation.
15. In the earlier seeded panel, all four Qwen/Ministral targets have greater
    within-group than between-group interaction-vector cosine in image and
    post-image regions under 34,650 exact source-pair assignments
    (`p=0.000173`). Because each group uses one fixed ordered generator pairing,
    this supports pairing-conditioned seed repetition rather than broad-class
    transfer.
16. In the 16-pair generator hierarchy, all eight target-region exact matching
    tests support repetition across two seeds of the same pairing. Under 70
    exact pairing-family block assignments, only Qwen's image-region transfer
    is unadjusted `p < 0.05` (`2/70`), and no post-image transfer test is.
17. FastVLM reproduces non-identical complete direct distributions over four
    source pairs. All 12 architecture-fixed tensors have zero pre-image effects,
    image-localized interaction maxima, spatial-dominant energy, and positive
    four-pair image/post-image direction cosines. Its fixed source response is
    `The image`, not the requested `ACK`.

### Provisional

- The measured object is distribution-coupled visual perturbation under fresh
  multimodal inference. The strongest direction result is organized by ordered
  generator pairing, not restricted to fractals and not persistent latent-state
  steering.
- Nonzero interaction and image localization do not imply interaction-axis
  dominance. Equal-coefficient calibration places 252/266 selected
  source-cache interaction shares at or below the exchangeable `1/3` reference.
- Qwen layer 33 `values` is a reproducible scalar summary locus for six
  one-frame fractal source pairs and all 14 tested pair-by-length points. Its
  sign and balanced full-vector axis now vary, so a stable scalar locus is not
  one shared directional mechanism.
- The late negative `values` profiles in Qwen and InternVL are a candidate
  architecture grouping, not evidence of one shared mechanism.
- Post-image interaction is more strongly seed-aligned inside the same ordered
  pairing than the high-energy image interaction at the seeded Qwen/Ministral
  targets. A fixed suffix alone is insufficient; generator identity, processor
  geometry, low-level statistics, and learned structure remain candidate
  explanations, not established mechanisms.
- Cache-summary magnitude and direct readout interaction are neither equivalent
  nor monotonically coupled in the current trajectories.
- Probe family and candidate calibration matter in all eight balanced models;
  the magnitude, visible-label response, and balanced factorial axis are
  architecture- and source-pair-specific. In the balanced eight-model core,
  only 4/64 prompt records retain one dominant axis across all four pairs, and
  only 2/64 retain both one axis and one generated pattern.
- `b_c` is more distributionally distinct from the other three pairs in this
  four-pair matrix, but four source pairs do not establish a population-level
  cluster.
- Granite's 8/8 generated-pattern agreements are replicated unresolved outputs,
  not positive category agreement. Its measured axis changes across source
  pairs in every fixed prompt record.
- Granite's late post-image interaction directions align more strongly over the
  fractal source set than over its pooled heterogeneous controls. Granite has
  not yet received the within-class seeded permutation used for Qwen and
  Ministral. This is not evidence of cache-to-readout mediation.

### Not Supported Yet

- persistent multimodal state across incremental turns,
- a valid cache intervention or causal mediation effect,
- semantic or subjective-state steering,
- a model-general layer, tensor, normalized depth, or token position,
- a prompt-invariant categorical decoder of the visual/cache state,
- a generator-independent direction or semantic class generality beyond the
  observed ordered pairings,
- multiplicity-adjusted class-specific or pairwise direction effects,
- independence of nested replay lengths or within-factorial pairwise distances.

## Statistical Boundaries

- The balanced nine-model one-frame replication count remains four source
  pairs. Qwen and Granite now have six independent fractal pairs; the three
  `c_d` ablations are dependent matched controls, and the three non-fractal
  panels belong to different source classes.
- Pairwise distances within one four-cell design and pairwise cosines among one
  six-panel group are dependent descriptive comparisons, not new replicates.
- The Note 0039 seeded direction test keeps each complete four-cell source pair
  atomic and exactly enumerates 34,650 group-size-preserving assignments. Each
  group uses one fixed ordered generator pairing, so its p-value calibrates seed
  repetition inside that nested design rather than broad-class transfer.
- The Note 0040 hierarchy randomizes perfect seed matchings within fixed broad
  classes over 11,025 assignments. Its transfer test moves complete two-seed
  pairing families over 70 class-balanced assignments. The eight target-region
  tests share source pairs and models, and reported p-values are unadjusted.
- Replay lengths are nested contexts, not independent stimulus replicates.
- Source-context caches and direct probes come from separate fresh forwards
  with different prompts; their relationship is descriptive, not causal.
- The full-vector surface retains selected target tensors but does not make
  different layer or model coordinate bases comparable. Dominance counts are
  within-target summaries, not cross-model vector alignment.
- "Non-identical" means unequal at the saved numerical precision. It is not a
  null-hypothesis significance test.
- Probe seeds isolate stochastic generation variation but do not create new
  visual source-pair replications.
- Prompt variants are repeated measurements over one visual factorial, not new
  visual source-pair replications.
- The 64 all-pair prompt records share eight models, four source pairs, two
  families, and four prompt variants. They are not 64 independent visual
  replications.
- The 384 prompt pairwise comparisons are six dependent source-pair
  comparisons from each of those 64 records, not 384 independent samples.
- Phi's reconstructed layout does not identify image-token positions; its raw
  sequence positions must remain role-unassigned.
- Granite's 1,474-token saved processor history maps to 3,073 effective cache
  positions only under the validated single-image replacement contract.
  Multi-image and unknown mismatch layouts remain role-unassigned.
- FastVLM's 142-token processor history maps to 397 effective cache positions
  only for the validated single-run `llava_qwen2` replacement contract. Its
  source response differs from the requested and cross-model-typical suffix.

## Highest-Value Next Data

1. Add a third and fourth seed replicate per current pairing, then estimate
   pairing-conditioned uncertainty with held-out seeds.
2. Add more stochastic, fractal, and provenance-controlled natural-image
   pairing families; evaluate leave-one-pairing-family-out transfer.
3. Frequency-match pairing families, then resolve interaction direction by KV
   head and image patch rather than only tensor-wide regions.
4. Run the fixed FastVLM targets over the generator-pairing hierarchy and add
   verbalizer/order controls for its direct readout.
5. Balance wording, candidate order, and token-to-semantics mapping as separate
   factors, including neutral and non-forced readouts.
6. Reconstruct Phi's processor-expanded image-token layout, then resolve
   balanced full-vector axes by KV head and token position.
7. Rebuild intervention logic only on a multimodal suffix path that asserts
   exact prefix and cache sequence-length compatibility before mutation.

## Manuscript-Safe Wording

Preferred:

> In nine local VLMs, fresh multimodal cross-palette factorial cells produced
> non-identical complete first-step distributions across a balanced core of
> four independent source pairs. The wider valid surface contains 104
> factorials and shows that input interaction and image-token localization
> generalize across fractal, ablated, geometric, noise, and natural inputs. In
> a balanced 16-source-pair Qwen/Ministral hierarchy, interaction-vector
> direction repeated across independent seeds of the same ordered generator
> pairing, while exact pairing-family block tests found weak transfer across
> different geometry or stochastic pairings: one of eight global target-region
> tests had unadjusted `p < 0.05`, and no post-image test did. Across prompt
> controls, generated-pattern stability remained much more common than
> full-distribution axis stability. Separate fresh forwards were used, so
> semantic class generality, state persistence, and causal cache mediation were
> not tested.

Avoid:

> The fractal stream creates a persistent hidden state that is causally stored
> in a universal cache layer.
