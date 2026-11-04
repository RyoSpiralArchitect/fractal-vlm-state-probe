# Research Note 0068: Representation Observation Calibration

Date: 2026-11-03

Status: PASS for the registered historical OFF/ON/OFF calibration and
independent artifact reload. Encoder/projector correspondence is **not
measured**, and its experimental panel remains unreleased.

Subsequent work: [Note 0069](0069_representation_correspondence_registration.md)
separately released that panel; [Note 0070](0070_representation_correspondence.md)
reports the results. The evidence unit in this note remains calibration only.

## Registration And Provenance

[PR #5](https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/5)
received a light self-review with no merge blocker and merged at
`dff3101fb2c0abf54048547a50d9e4d7942101e0`. All 494 local tests passed for
that PR. No independent reviewer approval or CI verification was claimed.
The merge preserves the pre-outcome non-learned release commit `99fb45c`.

[Note 0067](0067_representation_observation_registration.md), the
[new configuration](../../configs/representation_observation_calibration_v1.json),
capture, verifier and fixtures were committed at
`7ad7d8219e4ff4ad3c760c3e508e746ea2c16ffd` before model replay. All 523
local tests passed. The existing 51 measurement/config files are unchanged;
seven additional frozen files, including the reused calibration publisher,
bring this observation inventory to 58. All 58 hashes remain unchanged after
capture and verification. The inventory includes 7,716 source-file and 3,712
manifest hashes, not that many independent observations.

The committed-design preflight passed with no loaded model weights and no
installed observer. It checked the historical evidence, qualified inputs,
runtime, snapshot, installed observation-site sources and committed code.
Frozen specification SHA-256:
`17eb1f1687a0dc4a00680742ea069ef219c02d65457a8f1b6e3b76a1ec08a11f`.

There is also a retained **pre-registration operational failure**. Git author
identity was initially unset, so a commit failed; an erroneously issued
preflight then rejected the uncommitted path. It records zero attempted and
completed source contexts. It is not a passing preflight or a failed model
replay. Its original local receipt remains at
`runs/representation_observation_preflight_v1/execution_failure.json`, SHA-256
`7de10f564cb959a7594ea5817f3d8a6f00602600734fda1c8cf565e235a25807`.
Note 0067 records the distinct committed-design preflight root before model
calls. No input, site, acceptance condition or tolerance changed; the released
calibration itself had no retry, resume or replacement.

## Complete Bracketed Calibration

All 128 qualified historical cells, across the existing eight families and
four source replicates, ran in fixed per-cell OFF-before / ON / OFF-after
order. Each execution used a fresh source context and the same model/runtime
instance, original processor path, prompt and actual generated suffix.

| Check | Actual Denominator | Result |
| --- | ---: | --- |
| Fresh historical source contexts | 384 | Complete |
| Actual prepared-input fingerprints, prompt, suffix and token layout | 384 | All exact against historical receipts |
| L1 keys / L12 keys / L23 values against historical tensors | 1,152 pairs | All bytes exact |
| ON against OFF-before and OFF-after | 768 tensor pairs | All bytes exact |
| ON target projector calls | 128 | Exactly one per observed source cell |
| Method restoration before subsequent execution | 384 contexts | Exact original method restored |
| Complete saved encoder/projector arrays | 256 | All fixed layouts/dtypes and finite |

The selected caches retain their full `[1,2,397,64]` layout, including all
pre-image, image and post-image positions. Maximum absolute cache difference
is **zero**, without tolerance relaxation. Response is `The image` in every
context; actual generated step IDs are `[785,2168,2168]`. No suffix was forced.

The two observed native bfloat16 sites match registration:

- Encoder: existing projector input after tower H/W reshape, `[1,256,3072]`.
- Projector: exact projector return before multimodal insertion, `[1,256,896]`.

The wrapper holds the existing lazy input/output objects and returns the
original result. It does not run another encoder/projector call. Cast and
serialization occur only after the original replay and cache persistence
finish. Saved full tensors are lossless float32 expansions of native
bfloat16, retaining all tokens/channels without pooling or normalization.
No factorial interaction, cosine margin, retrieval or representation
bootstrap was computed from them.

## Independent Reload And Public Receipts

The precommitted verifier independently reloads and checks hashes, layouts,
finiteness, stored input/suffix receipts and actual cache bytes. It verifies:

- 1,152 historical tensor pairs / **58,540,032 scalar pairs**.
- 768 direct ON/OFF tensor pairs / **39,026,688 scalar pairs**.
- All 256 representation arrays / **130,023,424 scalars**, including exact
  bfloat16 expansion-bit layout and array-byte hashes.

This adds **zero model forwards and zero processor calls**. Native projector
argument/return identity follows frozen code, identity-scoped observation and
fixtures, not a second independent encoder implementation. Full prepared
pixel arrays are not stored or regenerated by this verifier; actual generation
fingerprints were checked at capture and are independently compared as receipts.

All five public files are byte-identical to the verified local outputs. A
second fresh-directory verification reproduced all five byte-for-byte without
new model/processor calls. Full representation and cache sidecars stay local;
the public package has receipts, not a standalone replay of model computation.

- [Summary, registration and frozen provenance](../../examples/research_notes/0068_representation_observation_calibration/summary.json)
- [384 context receipts](../../examples/research_notes/0068_representation_observation_calibration/cells.jsonl)
- [1,152 historical cache comparisons](../../examples/research_notes/0068_representation_observation_calibration/historical_comparisons.jsonl)
- [768 ON/OFF comparisons](../../examples/research_notes/0068_representation_observation_calibration/on_off_comparisons.jsonl)
- [128 two-site representation receipts](../../examples/research_notes/0068_representation_observation_calibration/representations.jsonl)

Publication summary SHA-256:
`8baf41b326186f90cba3b7666b31b48dc9ffb1fd7f5954042bd1d410acc089d5`.

All **525 local repository tests pass**, including the complete bracketed
mock run, independent-publication fixture, frozen hashes and portable receipt
coverage. Scoped Ruff, whitespace checks and 367 local documentation links
also pass. These are local checks, not CI or an independent reviewer approval.

The original run is `runs/representation_observation_calibration_v1`;
the committed-design preflight is
`runs/representation_observation_preflight_committed_v1`. See Note 0067 for
commands. A clean checkout need not contain these source images, model
weights or full calibration sidecars. Receipt tests therefore do not claim
an independent clean-checkout model reproduction.

## What This Qualifies

The calibration qualifies these two observation sites **for the measured
historical cells, model, runtime and suffix**, with exact parity of the selected
cache tensors. It does not prove equality of every unobserved activation,
performance/peak-memory equivalence, behavior on every future input, training
attribution, semantic invariance or a causal explanation of correspondence.
Holding lazy arrays can change memory lifetime even when selected bytes match.

There were 384 calibration source-context forwards, not new experimental
conditions. As in previous repo runs, this counter counts source contexts,
not internal autoregressive generation steps. Experimental cache totals stay
**3,816 cells / 11,304 tensors / 2,826 target analyses**; direct-probe totals
stay **104 factorials / 416 cells / 1,664 full-vocabulary sidecars**.
The non-learned panel and its 24 endpoints remain unchanged, including both
unavailable raw-moment endpoints and negative family differences.

Next, separately register and implement the full encoder/projector panel:
freeze the exact full-tensor contrast and normalization, complete comparison
surface, endpoint family, fixed-reference sealing, budget and independent
analysis/verification before representation scores. Repeat this observation
calibration in that same process. Do not select pooling or layers from these
calibration arrays, fit an alignment between coordinate bases, or promote
profile differences to learning effects. Prompt/suffix controls remain later
work; the untouched confirmation cohort is still not generated or sealed.

A later full-tensor storage budget must cover about 7.75 GiB of uncompressed
cell features and 3.875 GiB of float64 interactions, before calibration,
receipts and reserve. Do not assume compression will make that budget safe;
check available storage before that separate release. No data was deleted or
moved to prepare this calibration.
