# Research Note 0066: Non-Learned Baseline Decision

Date: 2026-11-03

Status: non-learned baseline panel complete. Scope decision only; no new model
forward, fitting, statistical test or confirmation release in this note.
Encoder/projector extraction remains unimplemented and unqualified.

## What The Baselines Resolve

[Note 0065](0065_nonlearned_processing_correspondence.md) independently verifies
all 2,048 feature cells and retains all 24 registered endpoints. Raw global
moments cancel in all 512 blocks, leaving their two endpoints unavailable.
The other eleven views have positive zero-overlap margins in every family,
and every available M(0) exploratory range lies above zero. All eleven overlap
difference ranges span zero, with 38/88 negative family-level differences.

The decisive simple case is **raw 8 x 8 mean pooling**, without local blur,
squaring or learned feature weights: M(0)=+0.236247 and D=-0.012610. Thus a
positive zero-coordinate-overlap correspondence score is not by itself a
diagnostic of a learned source-cache mechanism. The data-derived r1/r2
reference prototypes and known-family test remain part of the assay; this is
not a claim of unsupervised semantic recognition.

## Why Linear Pooling Can Change The Reading

For a fixed linear feature map P, the factorial contrast commutes with it:

```text
z_pool = P z_raw
cos(P a, P b) = (a^T P^T P b) / (||P a|| ||P b||)
```

The raw full-vector comparison uses the identity metric. Block averaging
instead gives a P^T P with cross-coordinate terms inside each pool and changes
the normalization. Zero exact source-index coincidence does not force those
cross-coordinate terms to cancel. This explains why the original raw dot-product
accounting cannot simply be carried into a pooled feature space.

This is a mathematical possibility plus its measured realization here, not
a measured decomposition of every pooled contribution or a causal model of
the VLM. Each map still fixes 25% of its own content, and the two maps use
the same underlying source. Common coarse statistics, nearby receptive-field
mixing and repeated colors can remain when exact indices do not coincide.

Raw global moments provide the complementary negative calibration: identical
raw multisets force zero interactions. Processor global moments do not have
that guarantee after resampling. They have small nonzero interaction norms
but a large normalized margin (+0.674669), independently verified far above
the registered tolerance-derived numerical scale. Amplitude, marginal
preservation and normalized correspondence are different measurements.

## What Is Not Resolved

The baselines do **not** reproduce the entire cache profile. All ten local
baseline means decrease across q=0/6.25/25%, whereas Note 0060's image-cache
means dip at the intermediate level. Baseline cross-level direction cosines
are about 0.89-0.92 for local features, unlike the 0.069-0.081 image-cache
medians. Same-map comparisons differ too. These scalar contrasts involve
different representations and dimensions, not aligned vector bases or
registered tests of representation differences.

There was no equivalence width, validated sufficiency criterion or
trained-versus-untrained design. We therefore cannot say that simple pooling
fully explains the cache, that learning contributes nothing, that a larger
margin means causal gain, or that the high-minus-zero effect is absent.
The reused development images and fixed-reference exploratory ranges do not
become untouched confirmation because their means are positive.

## Next Bounded Step

Keep all six baseline paths, both starts, same-map controls, family exceptions
and the complete cache comparison surface. Do not choose a baseline winner
or retune scales using these outcomes. A new representation-path stage should:

- Register exact encoder/projector observation sites, tensor layouts, pooling
  and normalization before new representation scores.
- Implement read-only extraction and first prove that enabling observation
  leaves the original qualified inference path, prepared inputs, suffix and
  selected cache tensors unchanged. A hook that changes execution is not a
  qualified observation site.
- Retain the non-learned baselines and raw/processor starts alongside those
  representations, with no cross-layer vector alignment assumption and no
  attribution to training from profile differences alone.
- Then proceed to separately registered prompt/suffix controls and final
  untouched source/map confirmation. Precision, budget, multiplicity and
  practical-equivalence decisions remain pre-confirmation work.

No encoder/projector, prompt-control or confirmation experiment is launched
by this decision. The approximate Note 0080 plan remains a work-package scope,
not a requirement to produce one positive result at every numbered note. The
current paper contribution is an auditable distinction between correspondence
under a chosen representation and claims about its learned or semantic origin.
