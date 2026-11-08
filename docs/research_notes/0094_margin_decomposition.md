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
equivalence tests or ratio-attribution endpoints. Note 0093 remains unchanged.

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
