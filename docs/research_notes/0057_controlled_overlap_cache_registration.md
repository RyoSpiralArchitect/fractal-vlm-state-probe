# Research Note 0057: Controlled Overlap Cache Registration

Date: 2026-10-29

Status: registration/code committed at `526950d` before new calibration calls.
The contract below is unchanged by execution. Only historical calibration is
implemented for this step; controlled-overlap cache capture is not released.

## Scope And Order

[Phase 1](0056_controlled_overlap_input_correspondence.md) completed all 896
four-cell input gates with 28 maps. Its raw/processor zero-overlap means were
-0.000315703 / +0.003352657; both exploratory ranges included zero. Those
already-seen input outcomes inform this development design. They are not
untouched confirmation and must not be relabeled as such.

Freeze [the Phase 2 configuration](../../configs/controlled_overlap_cache_phase2_v1.json)
and calibration code in Git before any new model call. Run a no-model-call
preflight first, then replay the **128 historical Note 0049 anchor cells**.
Stop after calibration and independent saved-tensor verification. The runner
has only `preflight` and `calibrate` modes, no panel mode or automatic handoff.

The current step adds no cache measurement on any controlled-overlap image,
no direct probe, and no confirmation image. Its 128 fresh source-context calls
are calibration replays, separately counted from experimental evidence.

## Immutable Inputs

The complete Phase 1 publication is pinned by SHA-256
`2df1804d0820ea02c82e299f3dfb74276d2aca202598059489578cbc16483f1c`.
The Note 0049 cache publication is pinned by
`5ae36346e86d1af93aca7477e4fb551d5dff375862827455f4031d895aae1b8d`.
Verify all 3,584 input manifests/images, coordinate recipes, 896 unchanged
gates, hierarchy metadata, publication ledgers and inherited measurement-code
hashes. Reuse the four Phase 1 masks, two destinations, q = 0/1/4 sixteenths,
eight families, r1/r2 references and r3/r4 tests without outcome-based selection.

Model is `mlx-community/FastVLM-0.5B-bf16`, snapshot
`81ffe929046666c43de53691147b1669ba0f3a4c`. Preserve the complete qualified
weight/tokenizer/config fingerprint, actual model-loading-path processor,
runtime versions and generation/model module hashes. Offline flags are
mandatory. Reading weight hashes is not loading weights or a model forward.
Any unavailable or mismatched prerequisite stops execution; do not substitute
a processor, snapshot, native input path or support-only input gate.

## Calibration Gate

Replay all 32 historical source blocks x MM/JJ/MJ/JM, not selected examples.
Use fresh source-only multimodal contexts, seed 20260604, temperature zero,
max_tokens = 2, the existing prompt, and no input metadata override.
The expected response is the observed **`The image`**, not the requested ACK;
the saved generation step IDs are **[785, 2168, 2168]**, retained exactly.
Do not coerce or rewrite a response to meet this contract.

Capture **L1 keys, L12 keys, L23 values** in every cell: 384 comparisons.
All must match historical tensor bytes exactly, with native bf16 and stored
float32 shape `[1,2,397,64]`. Also require exact actual prepared inputs,
prompt, suffix IDs, token layout and model/runtime identity. Effective token
positions are pre-image `[0,42)`, image `[42,298)`, post-image `[298,397)`.
These are validated token regions, not reconstructed two-dimensional patches.

One mismatch, changed suffix, missing/duplicate capture, nonfinite tensor or
integrity error stops the run. Preserve the partial outputs and failure record.
No relaxed tolerance, replacement cell, retry or resume is allowed. The
minimum free disk at launch is 2 GiB. Only a complete 128-cell/384-tensor
identity result may be called PASS; preflight or partial equality is not PASS
for calibration. The independent publisher must re-read the stored arrays and
compare their bytes, not trust a saved `bitwise_equal` flag.

A pass is a point-in-time calibration, not a permanent release. When the
later panel is implemented and explicitly launched, repeat the 128-cell
calibration in the same process before experimental capture. Revalidate all
hashes and model identity. Count that future replay separately from this one.

## Registered Panel Budget

The following is fixed before controlled-overlap cache outcomes but is **not
executed by this step**. Implement and fixture-test its capture/analysis code,
then commit it before the first panel forward. A passing calibration alone
must never bypass that release boundary.

| Planned unit | Count |
| --- | ---: |
| All sources on four reference maps | 128 factorial blocks |
| Sixteen test sources on 24 test maps | 384 factorial blocks |
| Total retained factorial blocks | 512 |
| Four-cell source forwards | 2,048 |
| Three target tensors per cell | 6,144 |
| Factorial target analyses | 1,536 |
| Required pre-panel calibration | 128 cells / 384 tensors |
| Later panel run including its calibration | 2,176 cells / 6,528 tensors |

The unused r1/r2-on-test-map combinations are omitted by design, never by
scores; all of their inputs already passed Phase 1 gates. Capture and hash-lock
the 256 r1/r2 reference source cells before all r3/r4 test-source captures,
including reference-map diagonals. Keep all eight families and all map levels.
Minimum free disk before the future panel is 8 GiB. There is no optional
stopping or outcome-dependent enlargement. Reference and test sources remain
reused development material. Confirmation is not generated and not sealed.

## Twelve Primary Descriptive Endpoints

At each of three fixed targets and two regions (image, post-image), construct
the complete region interaction JJ-JM-MJ+MM in float64. Normalize each complete
region vector before cosine scoring; reference direction is
`unit(unit(r1)+unit(r2))`. No cross-layer coordinate alignment or masked
renormalization is introduced. Keep the same reference map across q levels.

For each test source, subtract the mean cosine with the other three within-class
families from own-family cosine. Average the four-mask x two-destination score
matrices, then the two test sources within family, then eight families equally.
The two primary endpoints per view are M(0) and paired D = M(1/4)-M(0), for
**12 endpoints** total. The 16 averaged test rows are not 128 independent images.

Use 10,000 crossed-block bootstrap draws, seed 2026102930, resampling four
mask blocks, eight family blocks, two test sources within each sampled family,
and two destinations within each sampled mask. Use the same family/source
draws across masks and the same full draws across all q levels and six views.
References are fixed, not resampled. Family sampling is over all eight observed
families, not stratified by broad class. Match Phase 1's draw order: masks,
families, sources, then destination draws for each sampled mask.

The twelve endpoint ranges use quantiles 0.025/12 and 1-0.025/12. This is
exploratory percentile uncertainty with a Bonferroni tail allocation, **not
guaranteed simultaneous coverage** or a validated hypothesis test. Report
widths and all family signs; do not turn exclusion of zero into confirmation.
No p-values, equivalence test, absence claim or guaranteed minimum detectable
effect is registered. A separate precision design is still required for the
future untouched cohort. The input-stage four ranges retain their own scope.

Same-map controls, intermediate q, retrieval, norms, cross-level direction
cosines and pre-image zero checks are secondary/descriptive. Save all 168
view/map score matrices, including four diagonals per view. Missing, zero or
cancelling vectors remain explicit; if any primary-required score is unavailable,
both endpoints for that view are unavailable, without dropping its family/map
or changing denominator. Protocol, integrity, nonfinite or suffix failures
stop capture and withhold the panel's primary result. No suffix coercion.

## Claim Boundary And Reproduction

Input gates match two within-block spectral summaries, not between-level
spectra. Controlling overlap changes spatial arrangements too. Larger cache
margins, margin ratios or differing significance labels do not establish
causal gain, learning-specific transformation, persistence, semantic steering
or a cache-to-readout mechanism. Simple-processing baselines and prompt/suffix
controls remain separate later work. This is still a fresh-forward protocol.

After committing the registration/code, use fresh output roots:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_controlled_overlap_cache_calibration.py preflight \
  --output-root runs/controlled_overlap_cache_preflight_v1

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_controlled_overlap_cache_calibration.py calibrate \
  --preflight-root runs/controlled_overlap_cache_preflight_v1 \
  --output-root runs/controlled_overlap_cache_calibration_v1
```

[Note 0058](0058_controlled_overlap_cache_calibration.md) now reports actual
calibration counts and verification separately: all 128 cells / 384 tensor
pairs pass. The experimental panel is still not released.
