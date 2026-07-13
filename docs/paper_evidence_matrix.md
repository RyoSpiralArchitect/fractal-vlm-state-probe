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
| Cache-prefix audit | MLX-VLM `0.4.4`, Qwen and SmolVLM reuse paths | 2 audit runs, 7 available checks | No checked incremental or text-only branch reuse path retained a safe full prefix/cache-length relation | direct protocol-failure observation | [Note 0027](research_notes/0027_cache_prefix_audit_and_direct_full_vocab.md) |
| Qwen direct factorial trajectory | Qwen2.5-VL-3B 4bit, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 2 of them extend to 2/4/8/16; 48 cells total | All 12 direct after-factorials are non-identical; fresh ACK scalar argmax is layer 33 `values` with negative sign at all 12 points and all 4 one-frame pairs | exact scalar-locus replication within this model and source set | [Note 0028](research_notes/0028_source_pair_replication_and_prompt_robustness.md) |
| SmolVLM direct factorial trajectory | SmolVLM2-2.2B, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 2 of them extend to 2/4; 32 cells total | All 8 direct after-factorials are non-identical; one-frame ACK argmax spans layers 1/21/22 and keys/values | replicated pair-dependence under the valid protocol | [Note 0028](research_notes/0028_source_pair_replication_and_prompt_robustness.md) |
| Gemma 3 direct factorial trajectory | Gemma-3-4B-it 4bit, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 2 of them extend to 2; 24 cells total | All 6 direct after-factorials are non-identical; all four one-frame maxima are early `values`, but exact layer and sign vary; frequency readout can change sharply | component-level regularity plus pair-dependent exact locus | [Note 0028](research_notes/0028_source_pair_replication_and_prompt_robustness.md) |
| InternVL3 direct factorial replication | InternVL3-2B 4bit, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 16 cells, 64 sidecars | All four direct after-factorials are non-identical; all ACK maxima are late layer 25-27 `values` with negative sign | component/sign/depth-band replication with pair-dependent exact layer | [Note 0029](research_notes/0029_cross_model_prompt_and_internvl_expansion.md) |
| LFM2-VL direct factorial replication | LFM2-VL-1.6B 4bit, fresh ACK plus fresh direct probes | 4 pairs at 1 frame; 16 cells, 64 sidecars | All after-cell distributions are distinct; visible family labels vary in 3/4 pairs while frequency labels stay fixed; balanced readout axes remain pair-dependent | fifth-architecture replication with deterministic artifact integrity | [Note 0031](research_notes/0031_balanced_contrasts_five_model_expansion.md) |
| Full-vector source-cache core | Five VLMs, 26 model-local layer/tensor targets, fresh source-only ACK | 4 source pairs x 26 target groups; 80 cell runs, 416 tensors, 104 analyses | Every pre-image effect is zero; 100/104 argmaxes are image tokens; balanced axis dominance is spatial/palette/interaction in 84/14/6 analyses | targeted vector localization plus calibrated descriptive direction replication | [Note 0031](research_notes/0031_balanced_contrasts_five_model_expansion.md) |
| Cross-model direct aggregate | Five-model four-pair core plus Phi-3.5 Vision on `c_d`, complete first-step vocabulary | 35 factorial points, 140 cells, 560 sidecars | Every direct after-factorial is non-identical; balanced readout dominance is model-, pair-, and probe-dependent; sixth-model coverage is unbalanced | replicated architecture heterogeneity plus a bounded sixth-architecture pilot | [Note 0034](research_notes/0034_four_pair_prompt_replication_and_phi35_expansion.md) |
| Four-pair prompt robustness core | Five VLMs on `b_c`, `c_d`, `d_e`, and `e_f`, fresh direct probes | 20 model/source-pair audit units, 80 cell runs, 1,280 sidecars | All 320 baseline sidecars repeat bitwise; generated patterns agree over all four pairs in 27/40 records, balanced-axis dominance in 2/40, and both in 2/40 | five-model four-pair categorical versus distributional replication matrix | [Note 0034](research_notes/0034_four_pair_prompt_replication_and_phi35_expansion.md) |
| Phi-3.5 Vision expansion pilot | Phi-3.5 Vision 4bit on `c_d`, fresh direct probes and source-only ACK | 4 standard direct cells and 16 sidecars; 4 prompt cells and 64 sidecars; 4 ACK cells, 8 tensors, 2 full-vector analyses | Prompt dominance is S/P/I in 2/6/0; both selected full tensors are spatial-dominant; image-token partition is unresolved | sixth-architecture feasibility and heterogeneity pilot, not balanced replication | [Note 0034](research_notes/0034_four_pair_prompt_replication_and_phi35_expansion.md) |

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
3. Under fresh direct multimodal probes, every one of the 35 tested
   `MM/JJ/MJ/JM` factorials has non-identical full-vocabulary first-step
   distributions at saved numerical precision. Thirty-four points form the
   balanced five-model core; Phi currently contributes one `c_d` point.
4. At one frame, Qwen's layer 33 `values` scalar interaction locus and negative
   sign repeat over all four independent source pairs. InternVL repeats a late
   negative `values` band, Gemma stays in early `values` but changes layer/sign,
   SmolVLM changes layer/component/sign, and LFM2 exposes six hybrid-attention
   cache entries with pair-dependent direct readouts.
5. Across 104 selected full-vector cache analyses in five VLMs, all shared
   pre-image prefixes have zero effect, 100 interaction maxima fall in image
   tokens, and balanced spatial/palette/interaction dominance occurs in
   84/14/6 analyses.
6. A generated letter or top-k set can remain fixed while the complete
   distribution changes; visible-label equality is not distribution equality.
7. Across five models and four prompt-audited source pairs, generated semantic
   patterns agree over all four pairs in 27/40 fixed model-family-variant
   records, while balanced-axis dominance and both criteria agree in only 2/40.
   Every one of the 320 baseline sidecars reproduces exactly.
8. Over the six dependent source-pair comparisons per prompt record, generated
   patterns agree in 185/240 comparisons and balanced-axis dominance in
   100/240. In 105 comparisons, the categorical pattern repeats while the
   dominant full-distribution axis changes.
9. The Phi `c_d` pilot produces non-identical direct readouts and a distinct
   prompt profile. Its two selected full-vector `keys` targets are
   spatial-dominant, but its image-token partition is not yet identified.

### Provisional

- The measured object is distribution-coupled visual perturbation under fresh
  multimodal inference, not yet persistent latent-state steering.
- Nonzero interaction and image localization do not imply interaction-axis
  dominance. Equal-coefficient calibration places 95/104 source-cache
  interaction shares at or below the exchangeable `1/3` reference.
- Qwen layer 33 `values` is a reproducible scalar summary locus for these four
  one-frame source pairs and all 12 tested pair-by-length points, but the four
  one-frame image-region interaction vectors are only weakly aligned. A stable
  scalar locus is not yet a shared directional mechanism.
- The late negative `values` profiles in Qwen and InternVL are a candidate
  architecture grouping, not evidence of one shared mechanism.
- The low-energy post-image interaction is more directionally aligned than the
  high-energy image interaction in the initial Qwen/InternVL targets. A
  fixed-suffix transformation is a candidate explanation, not an established
  mechanism.
- Cache-summary magnitude and direct readout interaction are neither equivalent
  nor monotonically coupled in the current trajectories.
- Probe family and candidate calibration matter in all six piloted models;
  the magnitude, visible-label response, and balanced factorial axis are
  architecture- and source-pair-specific. In the balanced five-model core,
  only 2/40 prompt records retain one dominant axis across all four pairs.
- `b_c` is more distributionally distinct from the other three pairs in this
  four-pair matrix, but four source pairs do not establish a population-level
  cluster.

### Not Supported Yet

- persistent multimodal state across incremental turns,
- a valid cache intervention or causal mediation effect,
- semantic or subjective-state steering,
- a model-general layer, tensor, normalized depth, or token position,
- a prompt-invariant categorical decoder of the visual/cache state,
- statistical significance from the current four source pairs,
- independence of nested replay lengths or within-factorial pairwise distances.

## Statistical Boundaries

- The independent one-frame visual replication count is four source pairs, not six
  pairwise distances within each four-cell design.
- Replay lengths are nested contexts, not independent stimulus replicates.
- ACK caches and direct probes come from separate fresh forwards with different
  prompts; their relationship is descriptive, not causal.
- The full-vector surface retains selected target tensors but does not make
  different layer or model coordinate bases comparable. Dominance counts are
  within-target summaries, not cross-model vector alignment.
- "Non-identical" means unequal at the saved numerical precision. It is not a
  null-hypothesis significance test.
- Probe seeds isolate stochastic generation variation but do not create new
  visual source-pair replications.
- Prompt variants are repeated measurements over one visual factorial, not new
  visual source-pair replications.
- The 40 all-pair prompt records share five models, four source pairs, two
  families, and four prompt variants. They are not 40 independent visual
  replications.
- The 240 prompt pairwise comparisons are six dependent source-pair
  comparisons from each of those 40 records, not 240 independent samples.
- Phi currently contributes only `c_d`; pooled six-model artifact counts must
  not be described as a balanced six-model four-pair design.

## Highest-Value Next Data

1. Extend Phi-3.5 Vision to `b_c`, `d_e`, and `e_f` to complete a balanced
   sixth-model prompt and direct matrix.
2. Balance wording, candidate order, and token-to-semantics mapping as separate
   factors, including neutral and non-forced readouts.
3. Add independently generated trajectories and matched natural, geometric,
   noise, and processor-frequency controls with source-level permutation
   references.
4. Reconstruct Phi's processor-expanded image-token layout, then resolve
   balanced full-vector axes by KV head and token position.
5. Rebuild intervention logic only on a multimodal suffix path that asserts
   exact prefix and cache sequence-length compatibility before mutation.

## Manuscript-Safe Wording

Preferred:

> In five local VLMs, fresh multimodal cross-palette factorial cells produced
> non-identical complete first-step distributions across four independent
> source pairs. Source-cache scalar loci formed distinct architecture profiles,
> while selected full-vector contrasts localized interaction energy mainly to
> image tokens. Equal-coefficient calibration showed that spatial structure,
> rather than interaction, dominates most captured source-cache targets. Across
> prompt controls, 27/40 fixed records retained one generated semantic pattern
> over all four pairs, but only 2/40 retained one dominant full-distribution
> factorial axis. A sixth-architecture `c_d` pilot produced another distinct
> profile but is not yet a balanced replication. The protocol does not test
> state persistence or causal cache mediation.

Avoid:

> The fractal stream creates a persistent hidden state that is causally stored
> in a universal cache layer.
