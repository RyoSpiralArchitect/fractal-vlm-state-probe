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
| Qwen direct factorial trajectory | Qwen2.5-VL-3B 4bit, fresh ACK plus fresh direct probes | 2 pairs x 5 nested lengths, 40 cells | All 10 direct after-factorials have non-identical full-vocabulary distributions; fresh ACK scalar argmax is layer 33 `values` at all 10 points | replicated within this model and source-pair set | [Note 0027](research_notes/0027_cache_prefix_audit_and_direct_full_vocab.md) |
| SmolVLM direct factorial trajectory | SmolVLM2-2.2B, fresh ACK plus fresh direct probes | 2 pairs x 3 nested lengths, 24 cells | All 6 direct after-factorials are non-identical; fresh ACK scalar argmax moves across layers and tensors | replicated absence of the old fixed-locus pattern under the valid protocol | [Note 0027](research_notes/0027_cache_prefix_audit_and_direct_full_vocab.md) |
| Gemma 3 direct factorial trajectory | Gemma-3-4B-it 4bit, fresh ACK plus fresh direct probes | 2 pairs x 2 nested lengths, 16 cells | All 4 direct after-factorials are non-identical; ACK loci are stable within pair but early and pair-specific; frequency readout can change sharply | third-architecture replication with short trajectory | [Note 0027](research_notes/0027_cache_prefix_audit_and_direct_full_vocab.md) |
| Cross-model direct aggregate | Three VLMs, complete first-step vocabulary | 20 factorial points, 80 cells, 320 sidecars | Readout interactions and cache-summary loci vary by architecture, source pair, probe, and frame count | replicated heterogeneity; not a universal mechanism | [tracked summary](../examples/research_notes/0027_cache_prefix_audit_and_direct_full_vocab/summary.json) |

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
3. Under fresh direct multimodal probes, every one of the 20 tested
   `MM/JJ/MJ/JM` factorials has non-identical full-vocabulary first-step
   distributions at saved numerical precision.
4. Fresh ACK cache-summary interaction loci are architecture- and pair-dependent:
   stable and late in Qwen, moving in SmolVLM, and stable-within-pair but early
   in Gemma 3.
5. A generated letter or top-k set can remain fixed while the complete
   distribution changes; visible-label equality is not distribution equality.

### Provisional

- The measured object is distribution-coupled visual perturbation under fresh
  multimodal inference, not yet persistent latent-state steering.
- Qwen layer 33 `values` is a reproducible scalar summary locus for these two
  source pairs and five nested lengths, not a universal mechanistic locus.
- Cache-summary magnitude and direct readout interaction are neither equivalent
  nor monotonically coupled in the current trajectories.
- Probe family and candidate calibration matter: Gemma 3 is nearly saturated
  for the family probe but strongly sensitive for the frequency probe.

### Not Supported Yet

- persistent multimodal state across incremental turns,
- a valid cache intervention or causal mediation effect,
- semantic or subjective-state steering,
- a model-general layer, tensor, normalized depth, or token position,
- statistical significance from the current two source pairs,
- independence of nested replay lengths or within-factorial pairwise distances.

## Statistical Boundaries

- The independent visual replication count is two source pairs, not six
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

## Highest-Value Next Data

1. Add independent source pairs before extending the same nested trajectories.
2. Run prompt paraphrases, family/frequency candidate-order permutations, and
   neutral candidate controls.
3. Save full-vector cache contrasts at the stable Qwen and Gemma candidate
   loci, with image-token region reporting.
4. Rebuild intervention logic only on a multimodal suffix path that asserts
   exact prefix and cache sequence-length compatibility before mutation.
5. Add a fourth architecture plus natural and geometric matched controls under
   the same direct protocol.

## Manuscript-Safe Wording

Preferred:

> In three local VLMs, fresh multimodal cross-palette factorial cells produced
> non-identical complete first-step distributions. Fresh source-cache summary
> loci differed across architectures, and the current protocol does not test
> state persistence or causal cache mediation.

Avoid:

> The fractal stream creates a persistent hidden state that is causally stored
> in a universal cache layer.
