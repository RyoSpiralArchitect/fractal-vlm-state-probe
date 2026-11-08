# Research Note 0094: Own/Other Accounting Of Matched-Input Margins

2026-11-09. This is a new **post-hoc, saved-score-only analysis plan** after
the outcomes in [Note 0093](0093_stat_matched_source_results.md) were known.
It is not a new pre-outcome preregistration, independent replication or
scientific acquisition. The [plan](../../examples/research_notes/0094_margin_decomposition/plan.json)
freezes the original profiles, verification receipts, source specification,
all eighteen views/floors, sixteen test pairs and eight families by hash.

## Fixed Accounting

For each original test pair and shift, retain the original reference set:

```
own(s)   = cosine to the original own-family reference
other(s) = mean cosine to the other three same-class family references
margin(s) = own(s) - other(s)
delta(x) = mean(x(1), x(2), x(3)) - x(0)
D = delta(own) - delta(other)
```

Publish both signed pieces, `delta_own` and `other_contribution=-delta_other`,
not an unstable percentage of a near-zero total. Keep the source margin D
beside their sum, and require closure within 1e-12 at every pair, family and
whole-profile aggregation. A decline in own-family cosine and an increase in
other-family cosine have different descriptive meanings even when they yield
the same D. Neither is a mechanism or a causal attribution.

All eighteen original profiles and all rows remain present. Missing required
scores make the corresponding fixed-denominator aggregate unavailable; do
not remove a family or fill a direction with zero. There are no new reference
fits, transform selections, hypothesis tests, p-values, confidence intervals,
equivalence tests or ratio-attribution endpoints. Note 0093's source tables,
primary results and claim ceiling remain unchanged.

## Implementation And Replay Boundary

Commit the plan, implementation, independent arithmetic oracle and toy tests
before computing this decomposition. Record the implementation commit and
source hashes with each fresh run. The independent check uses exact rational
arithmetic over the saved binary64 score values; it does not rerun inference.
Two fresh verification roots must reconstruct the same outputs before public
results are accepted. Keep the first failure without overwriting an old root.

The plan lives beside the public analysis artifacts, not in the frozen source
configuration inventory. No preceding source configuration, implementation,
reference, capture or qualified profile is edited. The only required inputs
are the existing portable JSON/JSONL scores and metadata; full image/site/cache
arrays are not needed. New images, processor calls and model contexts are zero.

This analysis can locate the two algebraic contributors to a margin change.
It cannot determine whether all vectors share a common rotation, whether a
retransformed reference would restore correspondence, whether information is
lost, or whether training/semantics caused the result. Those comparisons
would require separate plans; none is launched here.

## Preparation Checkpoint

At preparation commit `d1c79e2`, the plan, score-only implementation, exact-rational oracle
and 22 analytic-toy/guard tests are prepared. All 45 focused decomposition,
preceding-publication and example checks pass. The preceding 135 frozen files
and Note 0093 source tables remain unchanged. No new real-score decomposition,
verification outputs, source images, processor calls or model contexts have
been produced. At that checkpoint execution/publication awaited the separate
operator scope decision. The operator subsequently approved the saved-score
decomposition, independent review and PR/main integration, with all new
acquisition forbidden. This approval is not an experiment or result itself.

Before real-score computation, the parent implementation ledger was added as
a fifth pinned input. The omission guard brings toy coverage to 23 tests;
46 focused checks pass. The six plan/code/test files were committed and
publicly pushed at `d72e6f4228f1e20c4c37d2453351f9658cfa3f70` before the
single real-score run. No preceding file was changed by this hardening.

## Verified Result

The score-only run completed once. Both fresh exact-rational checks returned
`VERIFIED_SCORE_DECOMPOSITION`; their complete output trees are byte-identical.
The [summary](../../examples/research_notes/0094_margin_decomposition/summary.json),
[all decompositions](../../examples/research_notes/0094_margin_decomposition/decomposition.jsonl),
[oracle checks](../../examples/research_notes/0094_margin_decomposition/oracle_checks.jsonl)
and [provenance/verification receipts](../../examples/research_notes/0094_margin_decomposition/verification_receipts.json)
retain every original profile and all rows.

| Site | Change in own cosine | Other contribution = -change in other cosine | Reconstructed D |
| --- | ---: | ---: | ---: |
| Encoder | -0.104124 | +0.008467 | -0.095657 |
| Projector | -0.102939 | +0.007884 | -0.095054 |

The exact own-cosine changes are -0.1041243644886229 and
-0.10293861516264867. The other-cosine changes are also negative,
-0.00846723860519043 and -0.0078843935409991, so their signed contributions
to the margin are positive. **In the overall average, decreased own-family
agreement is the larger negative term; decreased other-family agreement
partly offsets it.** The average is not an increased-other-agreement pattern.
These are arithmetic descriptions, not causal contributions or percentage
attributions.

The original Note 0093 D values are retained separately. Reconstructing them
through a different floating-point operation order can change final bits:
maximum difference from source D is **3.3306690738754696e-16** across all
pairs, families and profiles. Maximum component-closure residual is
**5.551115123125783e-17**, and maximum difference from the exact-rational
oracle is **2.220446049250313e-16**, all within the fixed 1e-12 check.

## Every Family, Both Signed Terms

| Family | Encoder own | Encoder other contribution | Encoder D | Projector own | Projector other contribution | Projector D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Voronoi / quasicrystal | -0.066906 | +0.001748 | -0.065158 | -0.073202 | -0.001441 | -0.074642 |
| Checkerboard / hex | -0.397330 | +0.042569 | -0.354761 | -0.398552 | +0.041871 | -0.356681 |
| Square / triangle | -0.240213 | -0.012328 | -0.252540 | -0.246633 | -0.008834 | -0.255467 |
| Voronoi / hex | -0.116615 | +0.026055 | -0.090559 | -0.114819 | +0.025461 | -0.089357 |
| White / blue noise | -0.045455 | -0.007512 | -0.052967 | -0.042974 | -0.009435 | -0.052408 |
| White noise / random dots | -0.001358 | +0.019956 | +0.018599 | +0.009684 | +0.020722 | +0.030406 |
| Blue noise / random dots | -0.016417 | +0.001920 | -0.014497 | -0.019812 | +0.001103 | -0.018708 |
| Sparse / dense dots | +0.051298 | -0.004672 | +0.046626 | +0.062797 | -0.006373 | +0.056425 |

The positive families are not one pattern. At encoder, white-noise/random-dots
has slightly decreased own agreement, but a larger decrease in other agreement
leaves a positive margin change. At projector both signed terms support its
positive D. Sparse/dense dots has increased own agreement at both sites, partly
offset by increased other agreement. Some negative-D families instead have
increased other agreement, adding a negative term. No exception is removed.

All 18 profiles, including all original learned floors, remain present.
The raw global-moment profile remains unavailable, with 0/16 available test pairs,
not zero.
The 288 pair rows, 144 family rows and 72 shift-summary rows include repeated
views/floors and are not additional independent samples. The underlying sample
remains the original sixteen test pairs in eight known families.

No source image, native array, processor or model is used for this follow-up;
all new acquisition counts are zero. This adds an algebraic reading of existing
cosines, not a new experiment or independent confirmation. Common rotation,
information loss, trained/untrained specificity, semantic effects and causal
mechanisms remain untested here. Note 0093's primary results and claim ceiling
are unchanged.
