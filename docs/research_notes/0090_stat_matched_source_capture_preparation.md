# Research Note 0090: Capture And Independent Saved-Data Verification

2026-11-08. **Preparation implementation only; no acquisition release.**
This implements the capture orchestration and saved-data verification for
the unchanged [Note 0088 design](0088_stat_matched_source_registration.md),
using the [Note 0089 core](0089_stat_matched_source_core.md).
The [preparation profile](../../configs/stat_matched_source_capture_preparation_v1.json)
has `execution_authorized=false`. No registered source, native processor call,
model load or model experiment is acquired by this implementation work.

## Preserved Contract

All 118 inherited files, three design-registration files and five reviewed
core files stay unchanged: 126 selected preserved files. New implementation
files are fixed separately before any approved run. Neither construction
success nor software verification grants execution authority.

The budget remains 64 source frames, 665 processor-only calls (128 historical,
25 synthetic, 512 candidates) and 920 model source contexts (384 historical,
24 sham, 256 reference, 256 test). Source contexts are not decoder steps.
The native adapter records actual language-model call attempts/returns and
prepared-input counts separately from the generation-step receipts.

## Acquisition State Machine

The [capture engine](../../src/fractal_vlm_state_probe/stat_matched_source_capture.py)
accepts a backend but performs no work on import. The production CLI supplies
only the [native adapter](../../src/fractal_vlm_state_probe/stat_matched_source_native.py).
Python dependency injection is for tests/library use, not an alternative
authorization path for a native command.

1. Require fresh canonical roots, qualified existing maps, file/runtime
   fingerprints, source inventory and the fixed 28 GiB launch gate.
2. Replay all 128 historical processor inputs exactly. Repeat all 25 fixed
   synthetic cases, including all matching/oracle checks, exact shams and
   expected rejection of the nonmatching sentinel. No model load yet.
3. Generate only the 32 reference donor frames. Save all 256 raw/native
   candidate cells, all twelve actual/oracle feature views, signed residuals,
   RGB word fingerprints, changed-pixel/value/word counts and spectra.
4. Let the unchanged historical calibration perform its normal single model
   load; retain that exact return. Re-run its 384 OFF/ON/OFF contexts, replay
   their file verification and compare all historical site bytes exactly.
   Read-only decoder counters preserve return objects and restore methods;
   the historical parity gates also cover their presence.
5. Acquire all 24 zero/constant sham contexts. Enforce prepared input,
   natural suffix, cache and ON-site parity across stages/shifts. Capture
   all 256 reference cells with one projector observation each.
6. Independently reconstruct saved reference inputs and re-read every full
   site/cache array. Save and independently recompute all fixed shift-0
   family references, including unavailable cases and all three learned
   sensitivity floors. Seal their identities, hashes and counts at context
   664, with exactly 32 sources and 409 processor calls returned.
7. Only after this artifact-backed seal generate the remaining 32 test donor
   frames and 256 test inputs. The first test context is 665. No scores are
   computed during capture, no references are reselected and no pair is dropped.
8. Check all counts and final code/runtime/model fingerprints. A complete
   capture remains `COMPLETE_UNVERIFIED`, with primary results withheld.

Every processor/model dispatch records its attempt before the actual call
and its return before downstream validation. A malformed returned payload
is not misreported as a call that never returned. Extra backend invocations
are refused before execution. A hash-chained operation ledger fixes identities
and order, including the reference-seal boundary. This is an audit trail, not
a cryptographic attestation that model inference occurred.

Caught Python exceptions, interruptions and `SystemExit` retain the first
failure, operation counts, partial artifacts and immutable file inventory.
There is no retry/resume or replacement. An uncatchable process kill can
leave an unsealed root; it has no complete manifest and cannot be promoted.

## Native Path And Provenance

Use the qualified processor-only path unchanged; native pixels are copied
to C order without arithmetic changes. New model contexts use the same ACK
prompt and prepared non-pixel payload, with no size override or forced
historical suffix. Keep the actual natural IDs and token layout. Both full
bfloat16 sites are stored losslessly with original expanded-word receipts;
all three selected caches and original run/audit sidecars are retained.

Inventory all configuration JSON, their available manifest references and
all available `runs/**/sources` manifests. Record effective RNG initializers,
decoded RGB hashes, file hashes and missing paths. Recheck known collisions
before generation and decoded collisions after each donor return. Missing
historical material remains explicit; scanning the available inventory does
not establish that a source was never seen. No untouched claim is released.

The uncompressed payload bound is 14 GiB, inside the design's 16 GiB allowance.
It includes canonical arrays, original native sidecars, source/candidate PNGs,
reference vectors and working arrays. Four GiB overhead and eight GiB reserve
remain separate: launch stays 28 GiB. Calls also require 32 MiB headroom and
array writes check their own size. Other processes can still consume space;
a shortage stops this run without deleting existing data or assuming compression.

## Independent Complete Verification

The [file-only verifier](../../src/fractal_vlm_state_probe/stat_matched_source_verify.py)
never invokes a processor, loads weights or calls the model. For complete
captures it verifies the exact file inventory and operation schedule, source
identity/RNG recipes, source manifests/PNGs, explicit-site maps and scalar
payload-first quartets. Generator frame reconstruction shares the pinned
original generator; the rank/embedding oracle is separately implemented.

It replays actual feature definitions and the separate convolution/slice
oracle from saved raw/native words, checks every stored feature/error/residual,
recomputes spectra, exact joint RGB multisets and changed-word/pixel counts,
and verifies all twelve input views. All full representation and cache files
are re-read; bfloat16 expansion uses the independent half-word decoder.
Original native sidecars must match canonical cache words. Unreferenced or
missing NPY/NPZ files and extra cache-archive members are rejected rather
than ignored. Coordinate-map shape is checked in addition to dtype and words.
The saved native contract is bound to the registered configuration, reviewed
Git objects, separately approved release, historical ACK anchors and the
exact zero-call preflight. Verification uses the saved public-receipt text
and local Git history; it makes no network request.
Historical summaries are hash-checked against their pinned configurations
before use. A separate inventory traversal recounts metadata, files, decoded
images and explicit gaps; omitted entries cannot silently narrow the audit.
Only this acquisition's new tree is excluded from the historical recount.

The reference vectors and availability/norm diagnostics are recomputed through
the scalar oracle before accepting the seal. The historical calibration's
existing independent file publisher is replayed and compared byte-for-byte;
its private historical anchors must still be present. This is saved-data
verification, not independent native-processor regeneration or a proof of
causal/semantic specificity.

The reference-prefix check requires the completed historical calibration's
own final snapshot, but not the outer acquisition's future final snapshot.
The latter becomes mandatory only after acquisition finishes. This keeps
the context-664 seal executable without weakening the final gate.

Two fresh verification roots must produce byte-identical output trees before
analysis. The analysis entry point then emits all eighteen fixed profiles
(twelve baselines plus two learned sites at three floors), spectra and the
two primary learned endpoints. Every score/margin is checked against the
scalar scorer. Missingness, zero norms and cancelling references are retained;
no p-values, equivalence claim or favorable sensitivity selection is added.

A failed capture receives only `FAILED_CAPTURE_AUDITED`: its file inventory,
operation-chain prefix and counts are checked. Partial semantic verification
is explicitly not promoted to complete verification or primary results.
Corrupt complete captures produce `INVALID_CAPTURE` with a retained verifier
failure. Neither status is a calibration/panel success.

## Software Checks Versus Native Qualification

Tests use 80px synthetic software fixtures: 8 x 8 ten-pixel pools with four
one-pixel tiles, seed 17 and analytic data. Full planned call schedules are
simulated; no registered generator or native runtime is invoked. Successful
software capture/verification has status `VERIFIED_SOFTWARE_FIXTURE`, never
`VERIFIED`; the production CLI cannot promote it into native results.

Coverage includes interruption at processor/model boundaries, no test-source
leakage before the seal, extra-call refusal, exact decoder-counter restoration,
reserve/authorization guards, malformed saved arrays and adversarially
refreshed file manifests. The last cases alter source pixels, spectra,
reference direction or operation order while updating integrity metadata;
semantic reconstruction must still reject them. These are software tests,
not native-path calibration or scientific observations.

Read-only review found and corrected five preparation defects: a premature
outer final-snapshot requirement, unchecked extra cache-archive members,
unchecked map shape, unanchored historical-summary reads and incomplete
inventory-coverage checks. Focused software regressions retain these cases.

## Separate Start Gate

The [CLI](../../scripts/run_stat_matched_source.py) has `preflight`, `capture`,
`verify` and `analyze` modes. An unapproved preflight uses a separate preparation
root and returns `PREPARED_NOT_RELEASED`, not an executable preflight PASS.
No preparation/native preflight is run as part of this implementation work.

Native capture still requires explicit approval and a **new**, separately
committed canonical acquisition-release configuration. Its hashes must cover
all 126 preserved files and every new implementation file, matching Git objects.
It binds a saved public code-freeze receipt; a second saved merged-PR receipt
must show publication of the acquisition release itself. Receipt content is
checked against hashes and Git ancestry offline; authenticity of the collected
GitHub receipts remains an operator/publication check, not an inferred fact.
The approved preflight must match this exact release and remain zero-call.

No acquisition-release file is created here. Fresh native calibration and
the real cohort have not run; native readiness beyond software checks is
therefore **unmeasured**. Confirmation remains not generated and not sealed.
