# Research Note 0085: Guarded-Tile Input Matching Registration

2026-11-07. **Design registration only; not implemented or released.**
No new image, coordinate map, processor call, feature score or model forward
is acquired by this note. The [configuration](../../configs/stat_matched_input_calibration_v1.json)
registers a bounded construction/calibration candidate, not a confirmation
cohort or a learned-representation experiment.

## Question And Scope

[Note 0065](0065_nonlearned_processing_correspondence.md) found correspondence
after non-learned pooling. [Note 0066](0066_nonlearned_baseline_decision.md)
separates that observation from baseline sufficiency, learning and semantics.
[Note 0084](0084_first_token_prompt_panel.md) closes the registered
instruction/readout control, not those remaining explanations.

The next bounded question is whether an actual raw-image rearrangement can
leave **all six registered feature paths on both starts** matched while
changing within-pool tile placement. This is a feasibility/calibration
question. No cache, encoder/projector or output-distribution comparison is
released, and no real source seed is allocated.

The configuration pins the feature definitions/oracle, their qualified
publications and Note 0084, whose 108 frozen file hashes must stay unchanged.
Prior outcomes informed this development design; it is not an outcome-blind
confirmation. Existing measurement files are not amended.

## Candidate Construction

Use a full 320 x 320 RGB uint8 canvas. The content rows are [40,280);
its six rows x eight columns of 40 x 40 cells align with the full-canvas
8 x 8 pooling grid. Top/bottom rows and all non-tile pixels retain a
fixture-constant background. This is **not palette-mean padding** and does
not preserve any original source image or donor palette distribution.
Only each fixture's matched arrangements share their RGB multisets.

Inside each content cell place four 6 x 6 tiles at local top-left offsets
(7,7), (7,27), (27,7), (27,27), indexed 0..3 in row-major order.
For shifts s=0,1,2,3 set output tile k to identity tile (k+s) mod 4,
without rotating, reflecting or resampling its internal pixels.
All 48 content cells use the same shift. No seed search or rejection loop.
Store full uint32 output-to-source maps; reconstruct them independently
using explicit site assignments rather than a constructor copy.

Every nonidentity map moves exactly 6,912 sites and fixes 95,488:
**9% of content coordinates / 6.75% of full-canvas coordinates move**.
Thus 91% of content coordinates and all other canvas sites remain fixed.
RGB changes can be fewer when values coincide, as in the zero/constant
shams. This is not a zero-overlap map,
wholesale structure destruction, global geometry matching or semantic
invariance test. Within-tile texture and much of the image stay intact.

The 20-raw-pixel tile translation corresponds to 64 processor pixels at
the qualified 1024/320 ratio. Raw radius-4 box responses do not overlap
between tiles or leave the pool. For the processor, a *candidate* radius-2
raw resize support plus radius-14 processor box gives
2 + 14/(16/5) = 51/8 raw pixels, below the seven-pixel guard.
The combined support width 6 + 2(51/8) is below the 20-pixel displacement.

This conditional argument motivates the placement; **it does not qualify
the processor**. Its actual resize/crop/normalization behavior is unchanged
and must pass every empirical gate. No override is allowed to manufacture
matching. If the support/phase reasoning fails in the native path, retain
that failure instead of moving tiles or changing tolerances.

## Fixed Synthetic Inputs

Six fixture types in fixed order: zero, constant, ramp, checkerboard,
impulse, noise. The config gives all RGB formulas, backgrounds and coordinate
conventions. Noise uses PCG64 with SeedSequence([2026110701,r,c,k]) for each
identity tile, not a new draw after rearrangement. The independent constructor
oracle shares this specified RNG, not its assignment implementation.

Four arrangements per fixture give **24 matched canvases**. One additional
negative sentinel copies noise/shift-0 and overwrites raw rows [40,80),
columns [0,40) with RGB [255,255,255], for **25 synthetic canvases** total.
The sentinel must be rejected as a match; its expected rejection is a
passing negative control, not an acquisition failure.

Zero/constant arrangements must be word-identical despite different
coordinate maps. Every nonzero shift on ramp/checkerboard/impulse/noise
must change both the raw and processor arrays. These are calibration
witnesses, not families or independent scientific observations.

## Matching And Oracle Gates

Reuse the pinned raw uint8-to-float64/255 and native normalized processor
float32-to-float64 starts, full canvases, 8 x 8 grid, edge replication and
raw 3/9 versus processor 9/29 box widths. No denormalization or cropping.
Energy remains **square after local mean, before pooling**; it is not a
linear pixel-energy statistic. Disjoint-response reasoning is conditional
and cannot substitute for checking the nonlinear paths.

All 18 nonidentity-versus-identity fixture pairs must satisfy:

- Exact full-canvas and all 64 within-pool joint RGB multiset equality at
  both starts: 18 x 2 x 65 = **2,340** comparisons. For processor arrays,
  sort complete RGB tuples of little-endian float32 words, not marginal
  channels or rounded colors. Shape/dtype/finiteness are separate gates.
- All twelve feature vectors matched at maximum absolute coordinate error
  <= **1e-10**, relative tolerance zero: **216 vector comparisons**.
- All 25 x 12 = **300** actual/oracle vector pairs within the same fixed
  1e-10 tolerance, using the existing independent convolution/slice oracle.
- All **24** nonconstant array-change checks and **12** zero/constant
  word-exact shams pass, with the prescribed coordinate movement counts.

The sentinel must change grid_mean by >1e-3 at **both** starts and differ
in full joint RGB multiset. Retain all twelve sentinel residuals, not just
the rejecting view. These numerical thresholds are fixed before generating
any fixture; they are not statistical or practical-equivalence margins.

Save complete feature vectors, norms, signed coordinate residuals, maximum
absolute and L2 errors. Near-zero vectors are not thresholded into zero.
Per-image feature matching does **not** guarantee matching normalized
factorial directions: later subtraction/normalization can amplify residuals.
There are no family references, correspondence scores, p-values, CIs or
learned-model comparisons in this stage.

## Future Execution Contract

This file grants **no execution authority**. First implement and commit the
constructor, independent index oracle, fixtures, tests and file-only verifier.
A separate release receipt must precede the first new map/image or processor
call; a passing contract test here is not that release.

Then a file-only preflight must check pins, the old 108 hashes, implementation
commit, qualified runtime, release and capacity. Require **9 GiB** launch
free space: 8 GiB reserved plus 512 MiB payload and 512 MiB overhead.
Do not assume compression savings or remove existing runs/models for space.

After coordinate/raw-fixture audit and sealing, replay **all 128 historical
processor jobs** in Note 0063's recorded order, with exact native pixel
hashes and complete non-pixel payloads; compute no new real-image features.
Only after historical parity, prepare the 25 synthetic images in fixture/
shift order, sentinel last. Use the qualified processor-only call with
text=["<image>"], return_tensors="np", not an ACK/chat-template model path.
Synthetic non-pixel payloads must equal the fixed first historical job:
input_ids=[[-200]], attention_mask=[[1]], image_sizes=[[320,320]].
Native pixel shape is [1,3,1024,1024], with float32 dtype.

Budget: **153 processor calls maximum, zero model forwards**. Historical
replays save new hash/non-pixel receipts rather than duplicate pixel arrays.
Save all 25 raw/native processor arrays, maps, actual/oracle features and
every completed comparison. Two file-only verifications reconstruct maps,
raw fixtures and feature calculations; they make **zero processor calls**,
so do not claim independent processor regeneration.

Stop on the first unexpected failure, retaining attempted/completed/planned
denominators and partial outputs. No retry/resume, replacement fixture,
alternative guard/shift/seed, feature removal or tolerance relaxation.
Any amendment requires a new version before affected outcomes.

## Registration Validation

Five data-free contract tests check release boundaries, artifact pins and
all inherited 108 hashes, feature definitions, candidate geometry and
budget/negative-control arithmetic. All 676 local tests, scoped Ruff,
whitespace checks and 484 local documentation links pass. These are
software checks, not processor matching, GitHub CI or a calibration pass.
No constructor or acquisition runner is implemented by this registration.

## Next Boundary

A pass establishes feasibility only on these synthetic fixture orbits and
the named measurements. It neither preserves every low-level statistic nor
identifies learned structure, semantics or cache mediation. Unmeasured
statistics and positional sensitivity remain alternative explanations.

Passing does not release real source generation, a representation panel,
cache/readout measurement or confirmation. New source/map identities,
reference/test hierarchy, target estimands, precision/sensitivity policy
and compute budget still require separate registration. The untouched
confirmation cohort remains not generated and not sealed.
