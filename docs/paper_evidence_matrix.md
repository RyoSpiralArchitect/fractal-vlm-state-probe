# Paper Evidence Matrix

Last updated: 2026-07-13

This file is the compact bridge from run artifacts to a manuscript draft. It
separates replicated observations, provisional interpretation, and missing
evidence. Counts below refer to actual run units; pairwise distances derived
from the same four cells are not treated as independent samples.

## Evidence Units

| Evidence unit | Model / protocol | Independent replication unit | Main observation | Current claim strength | Primary artifact |
| --- | --- | --- | --- | --- | --- |
| Cross-palette input audit | SmolVLM processor, 50 frames, `MM/JJ/MJ/JM` | 2 source pairs | Palette and spatial rank interact non-additively after image transformation and processor conversion | replicated input-level observation | [Note 0020](research_notes/0020_true_50_frame_cross_palette_replication.md) |
| Persistent cache-summary factorial | SmolVLM2-2.2B, 50-frame stream | 2 source pairs | Mid scalar interaction argmax repeats at layer 23 `values`; after repeats at layer 0 `keys` | replicated summary-stat locus | [Note 0020](research_notes/0020_true_50_frame_cross_palette_replication.md) |
| Forced-choice top-k20 readout | SmolVLM2-2.2B, 50-frame rerun | 2 source pairs | First-token sets are identical and maximum saved common-token interaction is about `0.004` logprob | replicated readout/trace dissociation within one model | [Note 0020](research_notes/0020_true_50_frame_cross_palette_replication.md) |
| Targeted values-swap | SmolVLM2-2.2B, 25-frame mid context | 2 source pairs x 3 probe seeds | Layer 23 swaps do not steer tokens toward donor; layer 12 gives the largest tested top-k perturbation, about `22%` of baseline separation | replicated intervention null/dissociation for this readout | [Note 0022](research_notes/0022_two_pair_values_swap_intervention.md) |
| Dense values-swap profile | SmolVLM2-2.2B, layers 8-23, seed 0 | 2 source pairs | Layer profiles are highly similar (`r=0.964`, `rho=0.982`), with the same layer 10 peak and top three layers 10/13/12 | replicated screening profile; controls pending | [Note 0024](research_notes/0024_dense_mid_layer_values_swap_profile.md) |
| Cross-model factorial pilot | Qwen2.5-VL-3B 4bit, 1 frame, all 36 layers | 2 source pairs | Surface/top-k10 readout is cell-invariant while interaction argmax repeats at layer 33 `values`, position 128 | replicated one-frame cross-architecture pilot | [Note 0023](research_notes/0023_qwen_cross_model_factorial_pilot.md) |

## Draft Claim Ladder

### Supported now

1. Luminance-rank cross-palette transformation is not an additive palette
   control; it creates non-additive processor-space perturbations.
2. Under SmolVLM's 50-frame protocol, saved cache-summary factorial interaction
   loci repeat across two independent source pairs while forced-choice readout
   remains quiet.
3. The repeated layer 23 summary locus is not sufficient for donor-directed
   generation under a whole-layer values-only creative-probe swap.
4. In two seed-0 source-pair screens, values-swap susceptibility has a shared
   mid-layer shape with the same layer 10 peak and top-three layer set.
5. A different architecture, Qwen2.5-VL, also shows one-frame cache/readout
   separation, with its own repeated late-layer locus.

### Provisional

- Persistent VLM source-cache state is sensitive to a nonlinear combination of
  RGB marginal distribution, spatial luminance rank, and processor-space
  frequency structure.
- Summary-stat salience and intervention leverage are distinct quantities.
- Qwen layer 33 `values`, position 128 is an architecture-specific mechanistic
  candidate, pending token-region mapping and multi-frame compatibility.

### Not supported yet

- semantic or subjective-state steering,
- equality of full-vocabulary output distributions,
- a model-general layer number or normalized depth locus,
- persistent multi-frame replication in Qwen2.5-VL,
- causal irrelevance of SmolVLM layer 23,
- inference from six pairwise distances as six independent observations.

## Statistical Boundaries

- The independent replication count for the current cross-palette spine is two
  source pairs, not the six pairwise distances within each four-cell design.
- Probe seeds isolate stochastic readout variation but do not create new visual
  stimuli or new source-pair replications.
- Mid and after are identical in a one-frame Qwen run by design and cannot be
  used as temporal persistence replicates.
- Cache summaries collapse vector direction; a signed scalar contrast is a
  contrast of the saved statistic, not a hidden-state vector contrast.
- Top-k overlap and RMSE are partial readouts. Full-vocabulary KL/JS or
  teacher-forced candidate scoring remains a required measurement upgrade.

## Highest-Value Next Data

1. Add full-vocabulary first-step sidecars or teacher-forced candidate scores
   for the existing SmolVLM and Qwen factorial runs.
2. Confirm SmolVLM layers `10`, `12`, and `13` over matched seeds with reciprocal
   and self-sham branches, then compare values, keys, key-value pairs, and
   position-local swaps.
3. Add early/middle/late single-frame Qwen factorials and map position 128 to
   text/image token regions.
4. Implement a clearly labeled cumulative visual-context replay lane for Qwen
   before making a persistent cross-model claim.
5. Add a third local model only after the scoring and token-region contracts are
   model-neutral enough to produce comparable artifacts.
