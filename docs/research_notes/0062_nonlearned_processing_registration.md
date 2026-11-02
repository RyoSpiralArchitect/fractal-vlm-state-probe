# Research Note 0062: Non-Learned Processing Baseline Registration

Date: 2026-11-02

Status: feature definitions, future estimands and bounded calibration committed
at `5b44429` before new experimental baseline scores. This release executes
calibration only, not the real-image correspondence panel.

## Question And Scope

[Note 0061](0061_controlled_overlap_phase2_decision.md) asks which parts of the
input-to-cache profile follow non-learned local processing and pooling.
The known Phase 2 outcomes informed this development design. Nothing here is
an untouched confirmation prediction or a test that a VLM implements a baseline.

[The configuration](../../configs/nonlearned_processing_phase3_v1.json) fixes
six feature paths on two starts. No fitting, parameter search or winner
selection is allowed. Existing 32 sources, 28 maps, all eight families and the
r1/r2-reference versus r3/r4-test split remain unchanged. No new images, maps,
cache forwards, direct probes or encoder/projector extractions are released.
Do not change the 38 qualified Phase 2 measurement/config files.

## Feature Definitions

Raw starts use the full 320 x 320 uint8 canvas divided by 255 in float64.
Processor starts use the qualified native `[1,3,1024,1024]` float32 normalized
pixel tensor, rearranged to HWC and promoted to float64. Do not denormalize,
crop padding, restrict to support, clip or center either start. The processor
normalization therefore changes the meaning of squared local responses;
start-to-start differences are descriptive, not a pure resize comparison.

| Path | Operation on each factorial cell | Dimensions |
| --- | --- | ---: |
| global_moments | Per-channel mean and uncentered second moment over the full canvas | 6 |
| grid_mean | Nonoverlapping 8 x 8 equal-area means, RGB retained | 192 |
| small_mean | Small centered box mean, then the same grid pool | 192 |
| small_energy | Small centered box mean, channelwise square, then grid pool | 192 |
| large_mean | Large centered box mean, then the same grid pool | 192 |
| large_energy | Large centered box mean, channelwise square, then grid pool | 192 |

Small/large windows are 3/9 raw pixels and 9/29 processor pixels, chosen by
nearest-odd approximation to the 1024/320 resolution ratio. These are only
approximately equal canvas-relative widths, not calibrated VLM receptive
fields. Each channel is mixed independently with uniform weights; boundaries
replicate the full-canvas edge. No cross-channel features, learned weights,
feature whitening or per-dimension normalization are used. Flatten pooled
features in row/column/channel order. Both grids divide their images exactly.

Raw global moments use uint64 sums of RGB and RGB squared, divided by the
registered pixel count and 255 or 255 squared. These bounded image sums fit
uint64. This makes identical multisets produce identical floating features,
without order-dependent floating reductions. Processor moments are ordinary
float64 reductions; their cancellation is not assumed after resampling.

For each feature path **apply the transform to all four cells first**, then
compute `z = (JJ - MJ) - (JM - MM)` in float64. Paired subtraction is an
explicit numerical convention for this new feature family: it gives exact
zero when JJ=MJ and JM=MM. It is algebraically the same factorial contrast,
not a post-hoc change to any previous input/cache vector. Do not apply a
nonlinear transform to an already formed interaction as a substitute.

A zero raw-moment interaction is the expected negative calibration. Its
direction is unavailable, never a score of zero. Exact nonzero norms are not
thresholded; nonfinite values or norm under/overflow are integrity failures.
Near-zero processor interactions, if present later, must retain their norms
and remain subject to the full registered availability rules.

## Calibration Release

First run a no-processor/no-model preflight. Check both pinned publications,
their ledgers, all 896 four-cell input gates, all 3,584 image/manifest hashes,
the coordinate construction, runtime and committed code. Do not instantiate
a processor or inspect new baseline correspondence scores at this step.

Then execute this fixed calibration budget:

- 84 feature/oracle pairs: six synthetic patterns at 32 x 32 on both starts,
  plus full-size noise at 320 and 1024, each through all six paths.
- An independent implementation uses separable 1D convolution and explicit
  grid slices, compared with cumulative-sum mixing and reshape pooling.
  Fixed maximum absolute tolerance is 1e-10, not a score-based acceptance.
- Six linear contrast-commutation checks and four nonlinear-order witnesses.
  The latter use constant cells 0, 1, 1/4, 1/2 and must differ by more than 0.1
  between the two orders. Retain the actual differences.
- One joint-RGB-permutation raw global-moment negative control must cancel
  exactly and have unavailable direction.
- Replay all 128 historical processor cells. Native pixel bytes, dtype,
  shape and every non-pixel payload must match the historical receipts exactly.
  No baseline feature is computed on those source images, and no model weights
  or cache forward is loaded. These are processor replays, not new cache data.

Synthetic generator seed is 2026110201. Save all 85 synthetic arrays, oracle
errors, contrast checks, input hashes, processor comparison receipts and frozen
code/source metadata. Require 2 GiB free disk for calibration. Stop and retain
partial outputs on failure; no retry, resume, tolerance change or fixture
replacement. A passing calibration does **not** release real-image scoring.

## Future Panel Contract

The unimplemented panel budget is 512 blocks / 2,048 cells / 12 views / 6,144
interaction vectors / 336 score matrices, including same-map controls. Freeze
all 256 reference-source cell receipts before the first test source. Use the
same reference map across overlap levels and unit-normalized r1/r2 references.
The margin is own-family cosine minus the mean of the other three families
within its broad class. Average score matrices across four masks and two
destinations, then two seeds per family and eight equal families. Do not
average vectors before normalization or replace the distractor set.

Twelve views give 24 planned endpoints: M(0) and M(25%)-M(0). Retain global
raw moments as unavailable rather than reducing the endpoint family.
Any missing required q=0 or q=25% score withholds both endpoints for that
view without dropping a family/map. Missing intermediate or diagonal scores
do not remove otherwise complete primary endpoints. Cancelling summed unit
references use the inherited 1e-12 threshold, distinct from exact zero inputs.

The planned bootstrap has 10,000 draws, seed 2026110203, fixed references,
the same crossed-block resampling order as Phase 2, and draws paired across
all twelve views and overlap levels. Tails are 0.025/24 and 1-0.025/24 even
when some views are unavailable. Ranges remain exploratory without guaranteed
coverage. No new p-values, equivalence bounds or baseline-sufficiency tests
are registered. A range exclusion difference from cache is not a difference
test, causal gain or proof that learning is necessary.

Retain all family signs, middle-level profiles, same-map controls, retrieval,
norms and cross-level directions. No baseline may be chosen for confirmation
using confirmation results. The minimum future disk reserve is 8 GiB. A
separate committed/tested capture, analysis and publication implementation,
fresh processor calibration and explicit release are still required before
real-image scoring. Encoder/projector extraction needs its own qualified path.

## Commands And Boundary

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_nonlearned_processing_calibration.py preflight \
  --output-root runs/nonlearned_processing_preflight_v1

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_nonlearned_processing_calibration.py calibrate \
  --preflight-root runs/nonlearned_processing_preflight_v1 \
  --output-root runs/nonlearned_processing_calibration_v1
```

Each output root must be fresh. Note 0063 will report success or failure with
the actual denominators. Confirmation is still not generated and not sealed.
This is the first bounded Phase 3 step, not completion of the whole package.

Execution report: [Note 0063](0063_nonlearned_processing_calibration.md)
records passing preflight, all 84 synthetic feature/oracle comparisons and
all 128 exact historical processor comparisons. No real-image baseline
correspondence panel was released, and no registered measurement file changed.
