# Research Note 0103: Qwen Capture And Independent Verifier Preparation

2026-11-10. Software implementation of the immutable [Note 0102
registration](0102_qwen_calibration_registration.md), not a native result.
The two-model FastVLM plus Qwen confirmation scope remains unchanged.

## Capture

The new `qwen_calibration_capture`, `qwen_calibration_native`,
`qwen_calibration_admission` and `qwen_calibration_verify` modules separate
acquisition, native access, admission and saved-data verification.
`scripts/run_qwen_observation_calibration.py` exposes four separate actions:
`preflight`, `capture`, `verify`, `verify-replay`.

The adapter preserves the existing prompt/chat template and native prepared
objects. Every stage gets a fresh cache and native grid/position computation.
It consumes the first yield from unmodified `generate_step(max_tokens=1,
temperature=0, prefill_step_size=512)` and closes without resuming. The
first yield already includes a native prefetch decoder call: saved K/V
therefore contain the prefix plus the first sampled token, not prefix only.
No second sampled token is fed, and the sampled ID is auxiliary only.

All 36 active K/V pairs, both full-vocabulary last-position raw logits,
first-yield scores, exact prepared inputs and source dtype receipts are saved.
Only ON retains complete premerger, auxiliary merger and final vision arrays.
No pooling, lossy storage, score interpretation or correspondence calculation
is added. Processor-only and native pixels must match word-for-word; grid
integer values must match without erasing their actual host/native dtypes.

An append-only operation journal records attempts and normal returns separately.
The first unexpected failure leaves a consumed root and a failure record;
there is no overwrite, resume, replacement or automatic retry. Live disk
reserve and payload/tree caps are checked during acquisition.
Standalone invalid output stops before starting another preparation, and native
nonfinite pixels stop before a decoder call. Returned context arrays are
persisted within the fixed byte bounds before parity/layout judgment so that
the first failing observation remains available for diagnosis. If persistence
itself exceeds a bound, the already retained partial file inventory and the
error remain; bounds are not relaxed to retain an oversized packet.

## Independent Verification

The verifier imports neither capture code nor native runtime/observer code.
It reloads files with pickle disabled and checks original raw copies, file
and decoded-word digests, canonical paths, complete field/file coverage,
source dtypes, shapes, finiteness, all three stages, decoder inputs and
cache offsets. A padded block transpose reconstructs window order separately
from capture's raster-window enumeration. OFF/ON/OFF parity is exact,
including signed zero, metadata, scores and all active cache entries.

Two fresh ordered verification roots must contain byte-identical complete
trees. A failed first verifier prevents replay. A verifier result never
releases the cohort or automatically qualifies native runtime parity.
Software fixtures remain explicitly `SOFTWARE_FIXTURE_ONLY` and cannot be
read as a native saved-data result through the public command.

## Admission Still Withheld

No native preflight, model loading, processor call, native calibration context,
new source image, coordinate map or confirmation score was performed here.
Synthetic stand-ins are not native observations.

The future offline-only preflight uses a fresh
`runs/qwen_observation_preflight_v1` root. It requires committed observation
software, unchanged registered sources/jobs, the exact local model revision,
weight-file hashes, installed package versions/source/binary hashes, and the
18 GiB calibration-only launch gate. It does not instantiate a processor or
load model weights. No download is permitted by that preflight.

Acquisition additionally requires a separately approved
`examples/qwen_native_execution_release_v1.json` (absent at this checkpoint):
`approved_for_native_execution: true`, `cohort_generation_authorized: false`,
`scope: qwen_observation_calibration_v1_only`, exact `plan_sha256`,
`preflight_sha256` and `software` hash map. Software cannot author its own
release. Admission precedes backend construction; offline environment flags
are enforced. Runtime/snapshot hashes and launch space are rechecked for
capture. Saved-data verification does not require the launch-space margin
or an installed model runtime after capture.

These paths and gates do not amend the 34-case/102-context/204-decoder-call
registration. Native execution requires a separate explicit decision.
Passing this calibration would still not qualify the new map matching rule,
all twelve Qwen input views, joint precision/storage, or confirmation.

## Software Validation And Review

The focused synthetic tests cover successful capture/two-tree replay,
consumed paths, signed-zero changes, cropped caches/logits, wrong dtypes,
nonfinite sites, token/metadata/restoration mismatches, window permutation,
changed raw sources, duplicate rows, symlinks, extra files, live disk/payload
stops and absent-release rejection before backend construction.

An additional toy test extracts the hash-pinned installed `generate_step`
function without importing MLX or loading a model. Only external arithmetic,
cache and model objects are replaced by small stand-ins; the function body
is unchanged. Through the real adapter it verifies two decoder calls before
first yield, active prefix-plus-one cache extraction, OFF/ON/OFF restoration
and pre-forward input rejection. This is control-flow qualification only.

The [software receipt](../../examples/research_notes/0103_qwen_capture_verifier_preparation/summary.json)
pins the six capture/verification/test files. The focused subset passes
82 tests, or 84 including the two new publication checks. The
[independent read-only review](../../examples/research_notes/0103_qwen_capture_verifier_preparation/review_receipt.json)
identified three P2 issues: pre-context/pre-forward validation, retention of
the returned failing packet, and independent standalone grid dtype validation.
All were fixed with regression tests; re-review found no remaining actionable
P1/P2 by inspection. The reviewer did not execute native work or rerun tests.

The earlier frozen implementation and research artifacts remain unchanged;
unrelated local note edits are excluded from this integration. Full-suite and
clean-archive verification precede merge. No CI workflow is configured, so
local tests are not described as CI-verified.
