# Research Note 0087: Guarded-Tile Matching On The Native Processor

2026-11-07. **Synthetic construction calibration PASS.** All twelve named
raw/processor feature views match on the 18 rearrangement pairs. Full-canvas
and within-pool joint RGB multisets match exactly, including native float32
words. This is construction feasibility on the fixed fixtures, not a VLM
representation result or untouched confirmation.

## Fixed Inputs And Execution

[Note 0085](0085_stat_matched_input_calibration_registration.md) fixed the
recipe, fixtures, acceptance criteria and budget. The separate
[Note 0086 release](0086_stat_matched_input_calibration_release.md) was
committed at `b3325507b704f535182f3ee3f01948646186d73b` before preflight
or any registered map/image. All 118 selected files remain byte-identical.

The file-only preflight passed without new candidate arrays, processor
instantiation or model weights. Four coordinate maps and all 25 raw inputs
were then constructed, independently reconstructed and sealed with zero
processor calls started. All 128 historical jobs passed exact native-word
and complete non-pixel parity before the first synthetic processor call.

Actual execution used **153 attempted / 153 returned processor calls**:
128 historical replays plus 24 fixture/shift canvases and one deliberately
nonmatching sentinel. No retry, replacement, dropped condition or changed
tolerance. Model forwards and verification processor calls are both **zero**.

| Registered check | Completed denominator | Result |
| --- | ---: | --- |
| Coordinate maps / raw inputs | 4 / 25 | Independent reconstruction and raw seal pass |
| Historical processor parity | 128 | Exact native hashes and non-pixel payloads |
| Nonidentity versus identity pairs | 18 | All retained |
| Raw joint RGB multiset comparisons | 1,170 | All exact |
| Native joint float32-word multiset comparisons | 1,170 | All exact |
| Named matched feature vectors | 216 | All within 1e-10 |
| Actual/oracle feature vector pairs | 300 | All within 1e-10 |
| Nonconstant changed-array checks, both starts | 24 | All changed |
| Zero/constant word-exact shams, both starts | 12 | All identical |
| Sentinel grid-mean checks | 2 | Expected nonmatch detected |

The 18 pairs are six fixture types x three nonzero shifts, with twelve
feature views each. Every multiset denominator is full canvas plus 64 pools.
The sentinel is not a nineteenth matching pair. These are dependent
calibration checks, not independent samples or statistical test counts.

## Numerical Matching And Negative Control

| Start | Feature | Max matching residual | Max actual/oracle error |
| --- | --- | ---: | ---: |
| raw_rgb | global_moments | 0 | 0 |
| raw_rgb | grid_mean | 8.882e-16 | 2.409e-14 |
| raw_rgb | small_mean | 1.776e-15 | 2.420e-14 |
| raw_rgb | small_energy | 9.992e-16 | 1.243e-14 |
| raw_rgb | large_mean | 3.886e-15 | 2.420e-14 |
| raw_rgb | large_energy | 2.220e-15 | 1.243e-14 |
| processor_pixel_values | global_moments | 1.110e-16 | 0 |
| processor_pixel_values | grid_mean | 0 | 0 |
| processor_pixel_values | small_mean | 6.217e-15 | 1.854e-14 |
| processor_pixel_values | small_energy | 2.387e-15 | 1.105e-13 |
| processor_pixel_values | large_mean | 8.660e-15 | 2.864e-14 |
| processor_pixel_values | large_energy | 5.052e-15 | 1.102e-13 |

The largest matching residual is **8.659739592076221e-15**; the largest
actual/oracle error is **1.1046719095020308e-13**. Both are below the
precommitted absolute coordinate tolerance 1e-10, relative tolerance zero.
No residual is silently rounded to zero in the stored arrays. Numerical
matching of these vectors is not a practical-equivalence test.

The mandatory sentinel replaces one content cell by white. Its maximum
grid_mean change is **0.4999926470588253 raw** and
**0.49625576567822804 native processor**, both above the fixed 1e-3
negative-control threshold. Its full joint multiset differs at both starts;
all twelve signed residuals remain stored. Expected rejection passed the
negative control rather than being removed as an inconvenient failure.

## What This Qualifies

On these fixed synthetic fixture orbits, changing within-pool tile placement
can preserve the listed six feature maps at both starts and the exact joint
RGB word distributions in every registered pool. The conditional guard/
resize-phase construction therefore has an observed native-path realization,
rather than merely a resampling assumption or a raw-space-only match.

Only 6,912 source coordinates move: 9% of content and 6.75% of the full
canvas. All other coordinates remain fixed, and internal 6 x 6 tile texture
is retained. Pixel values need not change wherever source coordinates move;
the constant/zero shams demonstrate that distinction explicitly.

This does not preserve every low-level statistic or destroy all structure.
Untested neighborhood, spectrum and positional information can still differ.
No model weights were loaded and no encoder/projector/cache/readout response
was measured. A learned/unlearned distinction, semantics, mediation and
generalization to real sources remain untested.

Small per-image feature residuals do not ensure identical normalized
factorial directions: subtraction and near-zero normalization can amplify
them. There are no family references, correspondence scores, p-values,
confidence intervals or equivalence conclusions in this calibration.

## Receipts And File Replay

- [Verified summary](../../examples/research_notes/0087_stat_matched_input_calibration/summary.json)
- [Four map receipts](../../examples/research_notes/0087_stat_matched_input_calibration/coordinate_maps.jsonl)
- [25 raw-input receipts](../../examples/research_notes/0087_stat_matched_input_calibration/raw_inputs.jsonl)
- [128 historical processor receipts](../../examples/research_notes/0087_stat_matched_input_calibration/historical_processor.jsonl)
- [25 synthetic processor receipts](../../examples/research_notes/0087_stat_matched_input_calibration/synthetic_processor.jsonl)
- [Every case audit and joint-word fingerprints](../../examples/research_notes/0087_stat_matched_input_calibration/cases.jsonl)
- [600 complete actual/oracle feature arrays](../../examples/research_notes/0087_stat_matched_input_calibration/features.npz)
  and [228 signed residual arrays](../../examples/research_notes/0087_stat_matched_input_calibration/residuals.npz)
- [Eight-output byte-identical replay receipt](../../examples/research_notes/0087_stat_matched_input_calibration/verification_replay.json)

Two fresh file-only verifications produce all **eight registered outputs
byte-for-byte**. They independently reconstruct raw images/maps and recheck
all saved native arrays, feature/oracle calculations, signed residuals,
metadata joins and complete case audits. They make no processor or model call.
Historical receipts are joined to pinned expectations, not independently
regenerated by verification.

Public archives support all 300 oracle comparisons, 216 matching-vector
comparisons and sentinel residual replay without weights, private runs or
a processor. Raw recipes/maps can also be regenerated under the registered
RNG/runtime. Native tensor bit-level multiset re-auditing requires the saved
native arrays, which remain local under `runs/stat_matched_input_calibration_v1`.
The public feature archives alone do not reconstruct those native tensors.

Summary SHA-256: `e2b856dc67d67ee7898a2d39240160b8a6c099b9b8f65d980825022d2cd38270`.
Capture: `a075986553571ad7f4dae6260e3d98525b890aca6f4495235959056a508536c7`.
Preflight: `d5e996d3909e978a877993c396013d0a881d58918d5b1ba0bfc421ad047941b9`.

All 697 local tests, scoped Ruff, whitespace and 503 local documentation
links pass. Public-only tests join all receipts, independently reconstruct
the raw recipes/maps and replay complete feature/oracle/residual archives.
The 118 frozen file hashes remain unchanged. These are local checks, not
GitHub CI, an independent processor regeneration or confirmation.

## Next Boundary

Calibration success does not automatically release a real source cohort or
learned representation panel. First separately fix source/reference/test
identities, target estimands, low-norm sensitivity policy and compute budget.
Retain all named baseline views, shams and negative controls rather than
selecting a matched winner. Untouched confirmation is still not generated
or sealed. No further experiment is launched by this result.
