# Research Note 0061: Controlled Overlap Phase 2 Decision

Date: 2026-10-29

Status: Phase 2 complete. Interpretation and next-stage scope only; no new
model forwards, statistical tests, baseline fitting or confirmation release.

## What Changed

[Note 0060](0060_controlled_overlap_cache_correspondence.md) completes the
registered FastVLM panel after 384 exact same-process calibration comparisons.
Its 2,048 experimental cells are complete and all twelve endpoints are
independently verified. The original input and cache specifications remain
unchanged. All r1-r4 sources and these maps remain development material.

Three observations now have to be explained together:

1. At zero exact source-coordinate coincidence, image-region cache margins
   are +0.032 to +0.040, and post-image L12/L23 margins are +0.261 / +0.218.
   Their five registered exploratory ranges are above zero. L1 post-image
   remains imprecise, with a negative white/blue-noise family case.
2. All six paired 25%-minus-zero ranges span zero. Four mean differences are
   negative and two positive; 28/48 family differences are negative. This is
   not a supported uniform increase, but neither is it an equivalence test.
3. Similar endpoint margins coexist with low cross-level direction cosines,
   near-unit image norm ratios and much higher same-map image margins.
   Correspondence, direction, amplitude and coordinate identity are distinct.

The raw/processor inputs had a different profile: near-zero M(0) point
estimates and about +0.034 for D. That comparison motivates explanation; it
does not identify causal gain or prove that a learned component is necessary.
The image-token intermediate means also dip below both endpoints. There is
no basis for imposing one monotonic response curve across representations.

## Bounded Reading

The strongest current formulation is:

> In the registered development panel, known-pairing correspondence is
> measurable in selected fresh source-cache representations even when
> reference/test maps have zero exact content-coordinate coincidence.
> The measured zero-to-25% overlap contrast is imprecise, and similar
> correspondence scores do not imply stable vector directions.

This is conditional on the sources, map algorithm, processor, model targets,
prompt and actual suffix. Each map still fixes 25% of its own content sites;
q=0 removes shared exact source indices, not every form of common spatial
information. Pixel colors can repeat, local neighborhoods can overlap in
receptive fields, and processing can mix positions. Between-level spectra
also differ. A scalar raw dot-product decomposition does not exhaust possible
input statistics.

Five exploratory range exclusions are not untouched confirmation. A range
spanning zero does not establish absence or a practical bound. The study does
not support semantic invariance, learned-versus-unlearned attribution,
persistent state, causal cache-to-output mediation or a universal cache locus.
The actual `The image` suffix is a protocol check, not a semantic classifier.

## Next Package

Continue the existing Note 0052 scope with Phase 3, not a larger overlap panel.
Keep all six cache views, all three overlap levels, same-map controls and all
family exceptions as the comparison surface. Do not discard L1 post-image
or the intermediate-level dip because their patterns are inconvenient.

The next bounded task is a **baseline specification and no-model calibration**:

- Register a compact set of non-learned processing paths, including local
  mixing/features followed by pooling. Fix scales, pooling grids, boundary
  handling, normalization, zero-vector rules and compute budget before scoring.
- Keep raw RGB and actual-processor starts separate. A permutation-invariant
  raw joint-RGB feature is an algebraic negative calibration: MM=JM and
  JJ=MJ imply a zero interaction. Treat its direction as unavailable, not a
  zero-cosine baseline. Do not assume this cancellation after resampling.
- Apply each feature path to all four cells before the factorial contrast.
  A nonlinear transform of an already formed interaction is a different
  object. Separate linear mixing from nonlinear local-feature/pooling steps.
- Reuse the fixed reference/test split and aggregation, with full-region
  normalization and explicit missingness. Retain all baseline profiles,
  including poor or cancelling ones; do not select a winner on confirmation.
- Qualify any encoder/projector extraction separately with an unchanged
  original-path replay before interpreting it as a representation-path
  observation. Such observations alone do not identify training effects.

Exact baseline definitions, scales, precision requirements and execution
commands are **not yet registered or implemented** by this note. This is a
scope decision, not an automatic launch. First establish which profiles simple
processing can reproduce; then consider the already planned prompt/suffix and
untouched source/map confirmation work. A sufficient baseline would explain
that tested profile, not prove the VLM implements that algorithm.

## Preprint Boundary

Phase 1 and Phase 2 are complete. The paper now has an input-to-cache
development comparison with fixed gates, an explicit negative/imprecise
overlap contrast and independently checked receipts. That is useful even if
the next baselines explain much of the profile.

The confirmation cohort is still **not generated and not sealed**. Its source
and map seeds, precision/budget justification, prompts, endpoints, baselines,
failure policy and multiplicity rules remain later preregistration work.
No current effect size authorizes more sampling until a desired conclusion
appears. Do not extend this completed run into Phase 3 or confirmation without
the separate specification and release required by the roadmap.
