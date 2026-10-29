# Research Note 0058: Controlled Overlap Cache Calibration

Date: 2026-10-29

Status: historical calibration and independent verification PASS. The
controlled-overlap experimental cache panel remains unimplemented and unstarted.

## Main Result

All **128 historical source cells** reproduce their actual prepared inputs,
source suffix and token layout. All **384 selected cache tensors** reproduce
their historical bytes exactly: 128/128 each at **L1 keys, L12 keys and L23
values**. Independent NPZ reload verifies **19,513,344 scalar pairs**, with
maximum absolute difference zero. No tolerance was relaxed and no retry,
replacement cell or failed attempt occurred.

This validates the current measurement path against the historical anchor.
It does **not** measure controlled-overlap cache correspondence, establish a
new effect, or extend the confirmation cohort. We stop at the boundary requested
for this step: registration, no-model-call preflight, calibration, verification.

## Pre-Outcome Registration

[Note 0057](0057_controlled_overlap_cache_registration.md),
the [configuration](../../configs/controlled_overlap_cache_phase2_v1.json) and
calibration code were committed at
`526950da184960f9a912e2f8201539841616603d` before this run. The 32 inherited/new
measurement-code files still match their frozen hashes. The same registration
fixes the later six target-region views and twelve descriptive endpoints,
but does not implement or release the experimental panel.

All previously observed Phase 1 inputs are development material. The
registration explicitly knows their outcomes. Historical calibration is a
replay, not another independent sample of a pairing family or source image.

## What Was Checked

| Stage | Coverage | New source-context forwards | Outcome |
| --- | --- | ---: | --- |
| No-model-call preflight | 3,584 Phase 1 input cells / 896 gates; historical 128 cells / 384 tensors; full model fingerprint | 0 | PASS |
| Historical replay | All 32 blocks x four cells, three fixed targets | 128 | 384/384 exact tensor matches |
| Independent publication audit | 128 prepared-input equalities, 384 tensor pairs, 19,513,344 scalar pairs | 0 | All exact; maximum absolute difference 0 |
| Controlled-overlap cache panel | Not started | 0 | Not measured |
| Direct probes / confirmation cohort | Not started | 0 | Not measured |

The calibration replays the `historical_3q` panel stored in Note 0049,
covering all eight families and r1-r4. It does not run any of Phase 1's new
controlled-overlap maps through the model. Historical target tensors are
reloaded and hash-checked before comparing them to fresh captures.

Model is FastVLM-0.5B bf16 at snapshot
`81ffe929046666c43de53691147b1669ba0f3a4c`. Weight/tokenizer/config hashes,
processor implementation and model/generation module hashes match the
qualified path. MLX is 0.31.1 and mlx-vlm is 0.4.4; the Phase 1 processor
runtime also remains unchanged. Execution used the offline cache.

The response is **`The image` in 128/128 cells**, with generation step IDs
**[785, 2168, 2168]**, exactly as historically saved. This is not ACK compliance
and no response coercion was performed. The fresh-source protocol, prompt,
seed 20260604, zero temperature and max_tokens = 2 remain unchanged.

Each selected tensor retains native bf16 provenance and float32 stored shape
`[1,2,397,64]`. The validated effective partition remains pre-image 42 positions,
image 256 positions and post-image 99 positions. Every prepared-input field,
including `pixel_values`, token IDs, attention mask and `image_sizes`, matches
the historical receipt. `image_sizes` remains `[[320,320]]`; no metadata
override was used. This is exact replay, not a new metadata intervention.

## Independent Verification

The publisher checks the preflight/frozen-specification chain, every source
and run hash, capture coverage and all actual prepared-input records. It
reloads each historical/current NPZ, checks file hash, shape, dtype and finite
values, then compares unsigned-byte views of the arrays. Thus a claimed
`bitwise_equal` flag alone cannot pass; even signed-zero differences are
rejected. It independently checks reported zero differences and identity
cosines, source suffix IDs and the effective token partition.

The verifier uses the same file format and recorded provenance, not a second
model implementation. It makes no new model calls. This establishes exact
saved-tensor agreement under the qualified runtime, not architecture-wide
invariance or accuracy of a broader causal interpretation.

All **437 repository tests pass**, including no-model preflight checks,
duplicate/missing/mismatched calibration rejection, signed-zero detection,
publication failure guards and checked-in receipt coverage. Ruff and diff
whitespace checks pass. All 32 frozen measurement-code files and the config
still match their pre-outcome registration bytes; no calibration code was
changed after observing results.

## Release Decision

**Calibration is qualified; the experimental panel is not released.** There
is no automatic continuation from this pass. The next bounded task is to
implement and fixture-test the registered panel capture/analysis, preserving
all maps, families, source identities and the twelve endpoints. Commit that
implementation before its first experimental forward.

The future run must repeat the 128-cell calibration in its own process before
capturing the planned 2,048 experimental source cells / 6,144 tensors. Its
256 reference-source cells must be frozen before test-source captures. Counts
including that future calibration are 2,176 source cells / 6,528 tensors;
these are **planned, not executed here**. A changed runtime, suffix, layout or
one nonidentical calibration tensor stops the run with its partial evidence
retained. Today's pass does not waive this later gate.

The experimental selected-cache totals therefore stay at **1,768 source
cells, 5,160 tensors and 1,290 analyses**. The direct-probe surface stays at
104 factorial points / 416 cells / 1,664 vocabulary sidecars. This note adds
128 calibration cells / 384 replay tensors separately, not new experimental
factorial analyses. No new p-value, equivalence test, causal gain, persistence
claim or semantic-steering result is added. Confirmation is still not
generated and not sealed.

## Artifacts And Reproduction

- [Summary and frozen provenance](../../examples/research_notes/0058_controlled_overlap_cache_calibration/summary.json)
- [128 cell receipts](../../examples/research_notes/0058_controlled_overlap_cache_calibration/cells.jsonl)
- [384 tensor comparisons](../../examples/research_notes/0058_controlled_overlap_cache_calibration/comparisons.jsonl)

The compact publication retains hashes and local paths. Historical images,
model weights and NPZ payloads are not embedded; a reproduction needs the
qualified local artifacts, not this compact bundle alone. The publication
summary SHA-256 is
`9e6d93665ee1429b5cc76c09816b9ddf7fbfe287c3b9e9c09dc9a2d8ea3f36c6`.

Measurement commands and the no-retry policy are in Note 0057. Saved-artifact
verification uses a fresh output directory and no model loading:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/summarize_controlled_overlap_cache_calibration.py \
  --run-root runs/controlled_overlap_cache_calibration_v1 \
  --output-root examples/research_notes/0058_controlled_overlap_cache_calibration
```
