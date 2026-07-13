# Paper Evidence Matrix

Last updated: 2026-07-13

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
| Cross-model direct aggregate | Three VLMs, complete first-step vocabulary | 26 factorial points, 104 cells, 416 sidecars | Every direct after-factorial is non-identical; cache loci and readout interactions vary by architecture, source pair, probe, and frame count | replicated heterogeneity; Qwen-specific exact-locus result | [tracked summary](../examples/research_notes/0028_source_pair_replication_and_prompt_robustness/summary.json) |
| Prompt robustness audit | Gemma-3-4B, `e_f`, fresh direct probes | 1 source pair x 4 cells x 8 probes, 64 sidecars | Baseline repeats byte-for-byte, while paraphrase, candidate order, and label mapping change semantic candidate distributions and 2x2 interactions by orders of magnitude | direct single-model/pair prompt-sensitivity observation | [Note 0028](research_notes/0028_source_pair_replication_and_prompt_robustness.md) |

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
3. Under fresh direct multimodal probes, every one of the 26 tested
   `MM/JJ/MJ/JM` factorials has non-identical full-vocabulary first-step
   distributions at saved numerical precision.
4. At one frame, Qwen's layer 33 `values` scalar interaction locus and negative
   sign repeat over all four independent source pairs. Gemma stays in early
   `values` but changes layer/sign, while SmolVLM changes layer/component/sign.
5. A generated letter or top-k set can remain fixed while the complete
   distribution changes; visible-label equality is not distribution equality.
6. In the Gemma `e_f` audit, semantic forced-choice distributions and
   factorial interactions are not invariant to prompt and candidate
   formulation, even though the baseline is exactly reproducible.

### Provisional

- The measured object is distribution-coupled visual perturbation under fresh
  multimodal inference, not yet persistent latent-state steering.
- Qwen layer 33 `values` is a reproducible scalar summary locus for these four
  one-frame source pairs and all 12 tested pair-by-length points, not a
  universal mechanistic locus.
- Cache-summary magnitude and direct readout interaction are neither equivalent
  nor monotonically coupled in the current trajectories.
- Probe family and candidate calibration matter: Gemma 3 is nearly saturated
  for the standard family probe, strongly sensitive for the standard frequency
  probe, and sharply variant-dependent in the prompt audit.

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
- Cache summaries collapse vector direction and sample token positions. A
  scalar argmax is not a full hidden-state vector contrast.
- "Non-identical" means unequal at the saved numerical precision. It is not a
  null-hypothesis significance test.
- Probe seeds isolate stochastic generation variation but do not create new
  visual source-pair replications.
- Prompt variants are repeated measurements over one visual factorial, not new
  visual source-pair replications.

## Highest-Value Next Data

1. Replicate the prompt audit across Qwen, SmolVLM, and additional source pairs.
2. Balance wording, candidate order, and token-to-semantics mapping as separate
   factors, including neutral and non-forced readouts.
3. Save full-vector cache contrasts at the stable Qwen and Gemma candidate
   loci, with image-token region reporting.
4. Rebuild intervention logic only on a multimodal suffix path that asserts
   exact prefix and cache sequence-length compatibility before mutation.
5. Add a fourth architecture plus natural and geometric matched controls under
   the same direct protocol.

## Manuscript-Safe Wording

Preferred:

> In three local VLMs, fresh multimodal cross-palette factorial cells produced
> non-identical complete first-step distributions across four independent
> source pairs. Qwen reproduced one late source-cache summary locus, while a
> Gemma control showed that forced-choice semantic interactions can change
> sharply with prompt formulation. The protocol does not test state persistence
> or causal cache mediation.

Avoid:

> The fractal stream creates a persistent hidden state that is causally stored
> in a universal cache layer.
