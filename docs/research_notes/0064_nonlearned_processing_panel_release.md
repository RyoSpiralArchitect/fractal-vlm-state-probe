# Research Note 0064: Non-Learned Processing Panel Release

Date: 2026-11-03

Status: capture, analysis and independent publication implementation to be
committed before new real-image baseline scores. One bounded development
panel is released; encoder/projector paths and confirmation remain closed.

## Merge And Unchanged Registration

[PR #4](https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/4)
merged Notes 0062-0063 at `10ae3de86f22b4f8574feb9c8e372cbd417f6a5d` after
light self-review and a fresh 484-test local run. No blocking finding was
identified, and no GitHub check runs were reported. This was not independent
approval or CI verification. The merge retains registration commit `5b44429`
and calibration publication commit `57e0423` as ancestors.

[Note 0062](0062_nonlearned_processing_registration.md) and its original
[configuration](../../configs/nonlearned_processing_phase3_v1.json) remain
byte-identical, including the historical `not_implemented_not_released` field.
The separate [release](../../configs/nonlearned_processing_panel_release_v1.json)
pins that configuration and the passing [Note 0063](0063_nonlearned_processing_calibration.md)
publication. No feature definitions, windows, input starts, source/map seeds,
weights, primary endpoints or bootstrap tails are changed.

This implements the already specified reused-data comparison. Existing r1-r4
images and all inspected maps remain development material. No learned model,
direct probe, cache forward or encoder/projector extraction is executed.

## One-Shot Capture

Run a new no-processor/no-model preflight, including all 896 input gates,
3,584 source images, original code/source hashes and the merged registration.
Then repeat the 84 synthetic feature/oracle comparisons and the exact raw
zero control. A fresh qualified processor instance must reproduce all 128
historical cells before experimental feature capture. Keep this same processor
instance for the entire panel; this does not carry an image-conditioned cache.

Capture **256 reference-source cells**, seal their feature-file receipts at
total ordinal 384, then capture **1,792 test-source cells** beginning at ordinal
385. Test cells include the reference-map diagonal controls. Order is inherited
from the controlled-overlap panel: mask, reference or q=0/1/4, destination,
sorted label, MM/JJ/MJ/JM, with all r1/r2 references first.

Every experimental image is reloaded from its qualified manifest; native
processor pixels, shape, dtype and all non-pixel payloads must match the
existing Phase 1 receipt exactly. Apply the six unchanged feature paths to
raw and processor starts separately. Save all twelve per-cell arrays as
float64 in a small NPZ. Total: **2,048 cells / 24,576 feature arrays**.
The run uses **2,176 processor calls**, including 128 calibration calls, and
zero model forwards. Require the original 8 GiB disk reserve throughout.

Do not save full processor pixel tensors, avoiding a large repeated payload.
Their hashes remain recorded and independent verification regenerates them.
There is no retry, resume, source replacement, suffix coercion, scale tuning
or favorable family/map filtering. Retain partial outputs and current jobs on
failure; no primary publication is allowed from an incomplete run.

## Analysis Details

Save all **6,144 interaction vectors**, **336 score matrices**, **36 overlap
aggregates** and **24 primary endpoints**, retaining missingness. Reuse the
previous complete-region scorer, within-class distractor set and balanced
family/source/mask/destination averaging. The twelve views keep the same
crossed-block bootstrap draws and unchanged 24-endpoint tail allocation.

Implementation details resolved before outcomes:

- Use the registered paired subtraction `(JJ-MJ)-(JM-MM)` on saved features.
  All 512 raw global-moment interactions must cancel exactly or signal an
  integrity failure. Their directions remain unavailable, not scored zero.
- Check all vector norms for nonfinite values and under/overflow. Exact
  nonzero vectors are not thresholded. Cancelling reference sums retain the
  separately registered 1e-12 threshold. A missing required q=0 or q=25%
  score withholds both endpoints for the view, with no family/map deletion.
- Secondary cross-level directions compare q=0 with q=6.25% and q=25% at the
  same source/mask/destination identity, retaining all 3,072 records and null
  statistics for undefined zero directions.
- Report vector norm divided by `4*sqrt(dimension)*1e-10` as a conservative
  numerical-sensitivity diagnostic. This is not a new acceptance threshold,
  confidence interval or permission to omit a small-norm view.

There are no new p-values, baseline-winner selection, equivalence or
baseline-sufficiency tests. Profile differences from earlier raw/processor
or cache scores are descriptive, not tests of processing gain or learning.

## Independent Verification Budget

The committed separate publisher checks capture order, the reference seal,
all feature/vector hashes and the source inventory. It reconstructs every
interaction exactly from saved cell features, independently recomputes all
cosines, margins, family signs, bootstrap ranges and paired directions, and
retains unavailable endpoints without imputing zeros.

Then instantiate a separate qualified processor and replay **128 historical
cells followed by all 2,048 experimental cells**: **2,176 verification
processor calls**, explicitly separate from capture. Each regenerated native
pixel hash and non-pixel payload must match its original receipt. Recompute
all **24,576 feature arrays** with the convolution/explicit-pooling oracle;
the fixed maximum absolute tolerance remains 1e-10. Score/bootstrap checks
use 1e-12 tolerance on the saved-feature vectors. These are distinct checks;
near-zero feature differences are not silently snapped to zero or rescored
with oracle values. Verification has zero model forwards and adds no new
independent sources.

The complete planned processor budget is **4,352 calls** across capture and
verification. Do not repeat the full verification merely to seek favorable
scores. Store all 6,144 small interaction arrays in a portable public NPZ so
the score surface can be replayed without original images or model weights.
Cell features and original image/processor data remain local. Full feature
reconstruction still requires the original input inventory and qualified
processor, not merely the portable score bundle.

## Launch Commands

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_nonlearned_processing_panel.py preflight \
  --output-root runs/nonlearned_processing_panel_preflight_v1

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_nonlearned_processing_panel.py run \
  --preflight-root runs/nonlearned_processing_panel_preflight_v1 \
  --output-root runs/nonlearned_processing_panel_v1

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/analyze_nonlearned_processing_panel.py \
  --run-root runs/nonlearned_processing_panel_v1

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/summarize_nonlearned_processing_panel.py \
  --run-root runs/nonlearned_processing_panel_v1 \
  --output-root examples/research_notes/0065_nonlearned_processing_panel
```

All roots must be fresh. Tests include the complete synthetic 512-block
analysis and independent numerical oracle, missingness, same-map controls,
reference ordering, altered metadata, hash failures and no-resume behavior.
Notes 0065-0066 will report actual results and the bounded decision, including
failure or unavailable views. No new confirmation cohort is generated/sealed.
