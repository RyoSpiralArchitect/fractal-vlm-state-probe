# Research Note 0102: Bounded Qwen Observation Calibration Registration

2026-11-09. This [registration](../../examples/research_notes/0102_qwen_calibration_registration/plan.json)
fixes development-only inputs, observation parity gates and a bounded Qwen
calibration budget. The metadata preparer and independent window/layout
oracle are implemented. **Native capture/verification runners are not yet
implemented or released.** No model weights, processor or native inference
are used at this checkpoint; confirmation source/map generation stays held.

## Fixed Cases And Execution Order

Select the r5 source pair of every one of the eight known families, arrangement
s0, in source-spec order and cell order MM/MJ/JM/JJ: 32 existing development
cells. Append the existing zero-s0 and constant-s0 fixtures. The
[34-case inventory](../../examples/research_notes/0102_qwen_calibration_registration/jobs.json)
retains raw file/decoded-word receipts. All raw arrays are 320 x 320 x 3 uint8;
none is regenerated. No r9-r18 candidate confirmation source is used.

For each case, run OFF-before, ON, OFF-after in order, each with a fresh cache
and exactly one prepared image. Use the existing frozen source prompt and
native Qwen chat template, with no resize/normalization/metadata overrides.
Only ON installs the Note 0101 visual observer. Preserve native grid/RoPE
preparation without manual position-state patches or extra text-only calls.

## First Yield Includes Native Prefetch

Static inspection of the installed `mlx_vlm.generate` found that `_step` is
called for the prefix, then again for `y[None]` before the first yield when
max_tokens=1. Closing after that first yield does not undo the second call.
The inspected source SHA256 is retained in the plan; this is source evidence,
not an observed runtime counter.

The registered native path therefore uses max_tokens=1 and
prefill_step_size=512, consumes one native yield and closes without resuming.
There are **two decoder calls per context**, not one. The first sampled token
has already been consumed once by prefetch; the final captured cache is
**prefix plus one token**, not a suffix-free prefix cache. The sampled ID is
auxiliary and not a semantic/readout endpoint. No second sampled token is fed.

| Budget item | Fixed upper count |
| --- | ---: |
| Existing development/sham cases | 34 |
| Fresh model contexts, three stages per case | 102 |
| Decoder calls, prefix plus native prefetch | 204 |
| Standalone processor-only calls | 34 |
| Internal processor preparations in model contexts | 102 |
| Combined processor calls | 136 |

Counters must distinguish processor-only calls, model contexts, internal
processor preparations and decoder calls. Unexpected counts stop the first
case without retry. This calibrates the declared one-image native path;
it is not an incremental multi-image cache-reuse qualification.

## Acceptance And Saved Evidence

Within each OFF/ON/OFF triple require exact source dtypes and prepared input
words/metadata, all 36 layers' complete K/V words, both decoder last-position
full-vocabulary raw logits, first-yield native scores and sampled ID. There
is no allclose tolerance or practical-equivalence claim. Save lossless float32
expansions with native dtype receipts; float16, bfloat16 or float32 are the
predeclared discovery allowlist, not an assertion that one has been observed.
Raw logits/native score equality is tested without interpreting low-precision
score mass as an already normalized probability distribution.

The single-image grid has t=1, positive even h/w and at most 1,024 patches.
Prepared pixel layout is N x 1,176, with temporal patch size 2, RGB channels
and spatial patch size 14. Prefix length is at most 512; final cache length
must be prefix+1, at most 513. All same-sized cases must retain the same
grid, IDs/mask, prefix length and per-site dtypes. A cap violation is a
retained failure, not permission to resize, crop or relax the cap.

The ON stage saves complete pre-merger N x 1,280, auxiliary merger-return
(N/4) x 2,048 and final vision-return (N/4) x 2,048 arrays. Actual grid and
dtype are discovered by calibration and frozen before future cohort use.
The independent oracle enumerates valid merged-token positions window by
window in raster order and reverses that order; final words must exactly
equal the reordered auxiliary words. The code does not call the native
window-index method or infer ordering from correlation.

Original observer class functions must be restored before every stage. Zero
and constant shams pass the same parity gates; zero directions are not
assigned scores. This registration computes no family correspondence,
interaction direction, semantic effect or confirmation statistic.

## Storage Bound And Preparation Result

The [byte-bound report](../../examples/research_notes/0102_qwen_calibration_registration/budget.json)
uses uncompressed float32 arrays and explicit shape caps, not assumed archive
compression. It counts all 72 cache arrays per context, all prepared pixels,
three vocabulary vectors, position/ID/mask/grid data, three ON visual arrays,
34 standalone processor outputs, raw copies and per-context header/metadata
allowance. The calculated upper payload is **5,139,318,144 bytes**, below the
fixed 8 GiB payload allowance. Reserve 8 GiB and overhead 2 GiB produce an
**18 GiB calibration-only** launch threshold. Runtime must enforce these
caps and stop before breaching the reserve.

The [metadata preparation](../../examples/research_notes/0102_qwen_calibration_registration/summary.json)
completed once at `runs/qwen_calibration_preparation_v1`; all 34 existing raw
file hashes match their pinned capture receipts. No raw array is decoded in
this preparer and no processor is instantiated. The installed generation
source and observer software hashes match; model-weight integrity and
runtime behavior are not qualified. Free space at preparation exceeds the
18 GiB threshold, but this is not an approved runtime preflight.

The [receipt](../../examples/research_notes/0102_qwen_calibration_registration/preparation_receipt.json)
pins plan, preparation code and output hashes. Historical preparation snapshots
in Notes 0099-0101 remain unchanged. This does not set the two-model cohort
storage budget or treat FastVLM's 44 GiB as a joint threshold.

## Gates Before Native Calibration And Cohort

Before native calibration, implement and commit the complete native adapter,
counter/snapshot capture and independent saved-data verifier; qualify the
first-yield/prefetch boundary on stand-ins; freeze runtime/model file hashes;
perform offline preflight and separate read-only review; obtain a separate
explicit execution release. Retain the first unexpected failure and block
the remaining stages/verifications. Require two ordered fresh saved-data
checks with byte-identical complete trees, without additional inference.

Passing this calibration would qualify only the registered observation path
and base input layout. It does not qualify the new maps or twelve Qwen input
feature gates. The s0-only calibration also cannot estimate reference-change
variance: a separately bounded development panel would be needed before
joint precision is justified. Calibration stages are not independent samples.
Both-model input/map qualification, joint sample size and storage contracts,
and the before-test reference seals remain required before cohort generation.
No actual-content confirmation seal or native execution approval is claimed.

## Preparation Checks

All 107 focused calibration-contract, saved-metadata, observer, prior
observation, joint-scope, cohort-preparation, historical scope and example
checks pass; lint and local link checks pass. The window oracle is checked
against a separate padded-transpose construction on square and rectangular
grids. Tests retain rejection of cropped arrays, nonfinite values, wrong
order and even signed-zero word changes. All 135 inherited implementation
hashes remain unchanged. These are local software/metadata checks; native
calibration, independent review and public push had not occurred at that
initial preparation checkpoint.
The full local regression suite subsequently passed **1,014 tests**. This
does not substitute for the separately gated native execution or review.

## Read Only Review And Integration Scope

The operator subsequently approved a separate read-only review and PR/main
integration of Notes 0099-0102 and preparation software only. Native Qwen
calibration and confirmation source/map generation or model execution remain
excluded. The [review receipt](../../examples/research_notes/0102_qwen_calibration_registration/review_receipt.json)
retains the no-actionable-findings result. The reviewer checked 68 focused
tests, source/seed and case inventories, the byte bound, shared-sample and
multiplicity contracts, and the installed prefetch control flow. It performed
no native or processor run and no independent runtime/resource qualification.

Historical preparation receipts retain their original review/push fields;
they are snapshots, not mutable current-status ledgers. Public integration
does not complete native capture/verifier implementation or release any
experiment. Future runtime admission still requires its separate gates.
