# Research Note 0097: Saved Array Crossing Execution Release

2026-11-09. The operator explicitly approved implementation, one execution,
observation, summary and integration, with a separate read-only reviewer.
This [execution release](../../examples/research_notes/0097_reference_crossing_execution/release.json)
adds authority without rewriting Note 0096's preparation-only registration.
No new image, processor or model invocation is authorized or required.

## Implementation Boundary

The production scorer reconstructs all 2,048 selected full arrays from their
original file/word receipts. Packed visual bfloat16 words are expanded and
checked against their saved expanded-word hashes. Four views, all 4 x 4
reference/test crossings and three amplitude floors remain fixed.

Floor-independent vectors and dot products are computed once, then all three
registered masks are applied; no floor is chosen from outcomes. Production
scores use NumPy dot products and means. The independent verifier separately
loads and hashes native files, independently expands bfloat16, reconstructs
interactions/references and uses 32,768-coordinate product/sum blocks followed
by math.fsum for cosine numerators and aggregation. Norm receipts deliberately
use the same NumPy L2 definition and cell addition order to preserve strict
absolute comparison of large unscaled metadata; this is not an independent
norm-library implementation. The verifier does not consume production vectors.

The original six learned s0 profiles are recomputed and compared before new
crossed scoring. New learned rows and the independent reconstructions must
also reproduce original scores, margins, missingness, D and available counts.
Registered score/closure checks use absolute 1e-12, relative tolerance zero.

The release pins the prepared inventory and source plan. All execution
plan/code/test files must match a committed ancestor-descended implementation
before scoring. The first run root and both verification roots are distinct
and exclusive-create; an unexpected failure is retained without retry.
Only two complete, byte-identical fresh native-verification trees qualify
the results. File replay is not independent model inference or confirmation.

## Preparation Status

The runner, independent scorer and analytic tests are implemented. Identity,
common orthogonal changes, collapsed family geometry, zero directions,
low-amplitude guard differences, reference cancellation, missingness,
native word corruption and original-row corruption have toy checks.
No real-array crossing result is reported at this preparation checkpoint.
The [manuscript](../paper/manuscript.md) retains only the preceding results
until the bounded run and both verifications qualify the new observations.

The pre-execution read-only review found two lifecycle gaps: a failed first
verification did not block the replay, and the runner did not enforce exact
two-tree equality. Both were fixed before any real-array scoring. Dedicated
fixtures now reject replay after failure, replay before the first verification,
and sub-tolerance score drift that would change the output bytes. The latter
must leave a retained failure without a successful replay summary.
Follow-up review also found a source/verification root symlink bypass. All
canonical roots and ancestors are now checked before reading; source-root
and first-verification-root symlink fixtures both fail before new output.
