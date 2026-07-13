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
| Controlled mid-layer values-swap | SmolVLM2-2.2B, layers 10/12/13, 3 seeds | 2 source pairs x reciprocal directions | Directional ranks repeat within each pair; exact peak is pair-dependent within a layers 10-13 band; all tokens stay origin-identical and shams are zero | replicated controlled intervention profile | [Note 0025](research_notes/0025_controlled_mid_layer_values_swap_confirmation.md) |
| Cross-model factorial pilot | Qwen2.5-VL-3B 4bit, 1 frame, all 36 layers | 2 source pairs | Surface/top-k10 readout is cell-invariant while scalar interaction argmax repeats at layer 33 `values` | replicated one-frame cross-architecture pilot | [Note 0023](research_notes/0023_qwen_cross_model_factorial_pilot.md) |
| Qwen cumulative-replay trajectory | Qwen2.5-VL-3B 4bit, 1/2/4 ordered images, all 36 layers | 2 source pairs x 3 nested replay lengths | All six scalar argmaxes repeat at layer 33 `values`; readout stays cell-invariant; exact local position does not persist | replicated scalar locus under a non-incremental multi-image protocol | [Note 0026](research_notes/0026_qwen_cumulative_replay_trajectory.md) |

## Draft Claim Ladder

### Supported now

1. Luminance-rank cross-palette transformation is not an additive palette
   control; it creates non-additive processor-space perturbations.
2. Under SmolVLM's 50-frame protocol, saved cache-summary factorial interaction
   loci repeat across two independent source pairs while forced-choice readout
   remains quiet.
3. The repeated layer 23 summary locus is not sufficient for donor-directed
   generation under a whole-layer values-only creative-probe swap.
4. SmolVLM values-swap susceptibility forms a controlled layers 10-13 band;
   direction repeats within pair while the exact peak changes across pairs.
5. A different architecture, Qwen2.5-VL, shows cache/readout separation and a
   repeated scalar layer-33 `values` locus over 1/2/4-frame cumulative replay.

### Provisional

- Persistent VLM source-cache state is sensitive to a nonlinear combination of
  RGB marginal distribution, spatial luminance rank, and processor-space
  frequency structure.
- Summary-stat salience and intervention leverage are distinct quantities.
- Qwen scalar layer 33 `values` is an architecture-specific summary-stat locus.
  The exact local sequence position is pair- and replay-length-dependent.

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
- Qwen 1/2/4 replay lengths are nested contexts, not three independent stimulus
  replications; cumulative replay is not incremental temporal persistence.
- Sequence-position argmax is over saved sampled positions and can change when
  the sampling plan is enriched.
- Cache summaries collapse vector direction; a signed scalar contrast is a
  contrast of the saved statistic, not a hidden-state vector contrast.
- Top-k overlap and RMSE are partial readouts. Full-vocabulary KL/JS or
  teacher-forced candidate scoring remains a required measurement upgrade.

## Highest-Value Next Data

1. Add full-vocabulary first-step sidecars or teacher-forced candidate scores
   for the existing SmolVLM and Qwen factorial runs.
2. Compare SmolVLM values, keys, key-value pairs, layer windows, and
   position-local swaps within layers `10-13`.
3. Extend Qwen cumulative replay to 8/16 frames and compare image-run-local
   interaction profiles rather than one sampled argmax.
4. Repair or replace Qwen's incremental second-image cache path before making a
   persistent multi-turn claim.
5. Add a third local model now that replay, token-region, and trajectory
   contracts are model-neutral enough to produce comparable artifacts.
