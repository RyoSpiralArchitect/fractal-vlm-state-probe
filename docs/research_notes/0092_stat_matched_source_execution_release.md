# Research Note 0092: Bounded Matched-Source Execution Release

2026-11-08. The operator explicitly approved execution after the successful
Note 0091 zero-call preparation. This separate
[execution authorization](../../configs/stat_matched_source_acquisition_release_v1.json)
permits exactly one acquisition under the unchanged Note 0088 scientific
design and Note 0090 implementation. It changes no source, feature, map,
threshold, endpoint, model, runtime, suffix policy or stopping rule.

## Preserved Evidence

The original start-withheld Note 0091 registration is preserved byte-for-byte
at [its historical snapshot](../../configs/stat_matched_source_acquisition_registration_v1.json),
SHA-256 `d32f360b1bd4dfecd35cd29565cd931956f51976791b4d4e782fd51a9994692c`.
Its original public version remains in PR 12 history. Note 0091's link and
registration tests now use that snapshot rather than misreading the new
authorization as an earlier approval.

All 135 selected implementation/configuration/test/note files remain exactly
at the PR 11 merge commit `45ad7db162194ebe7cb596526ec1953605af6700`.
The code-publication receipt and complete file freeze remain unchanged.
No new runtime implementation is introduced by this release.

The prior preparation remains `PREPARED_NOT_RELEASED`, with zero source
generation, processor calls and model contexts. Its full frozen specification
and original receipt remain local and unchanged. That preparation is **not**
reused as the approved execution preflight.

## Fresh Public-Release Gate

Commit and publicly merge this authorization before use. Collect a separate
saved PR receipt for that public merge. The existing CLI must validate both
the committed authorization and its public ancestry, the 135-file code freeze
and the saved code-publication receipt before qualification.

Run a fresh approved zero-call preflight at
`runs/stat_matched_source_preflight_v1`, supplying the canonical authorization
and the separate public-release receipt. It must return
`PASS_FOR_APPROVED_RELEASE`, with zero processor/model calls, no weight load
and at least 28 GiB free. Capture must reproduce that exact frozen state.
Neither a failed gate nor the old preparation root can substitute for it.

## Fixed Acquisition And Verification

The maximum planned budget is **64 donor frames, 665 processor-only calls
and 920 model source contexts**. The 665 include 128 historical, 25 synthetic
and 512 candidate inputs; the 920 include 384 historical, 24 sham, 256
reference and 256 test contexts. Decoder calls are separately recorded.

All 256 reference cells, full site/cache artifacts and fixed family references
must pass independent file checks and seal at context 664 before any test
source generation. The first test model context is 665. Retain unavailable
directions, negative outcomes and every planned family; no replacement or
complete-case substitution. Keep the full storage reserve throughout.

Stop on the first unexpected failure. Preserve the failure, attempted/returned
counts, partial files and inventory. **No retry, resume, replacement, gate
relaxation or implementation correction is authorized by this release.**
A partial failed acquisition receives only the fixed file-only failure audit;
it never yields primary profiles.

Only a complete capture may undergo the two fresh independent file checks at
the registered verification roots. Require byte-identical verification trees
before the fixed eighteen descriptive profiles and two learned endpoints.
No tuning or favorable floor selection is allowed after observing results.

At registration there are no new observations. Report actual execution counts,
success or first failure separately. This is known-family development, not
learning identification, semantic steering, mediation, equivalence or an
untouched confirmation release. Confirmation remains not generated and not sealed.
