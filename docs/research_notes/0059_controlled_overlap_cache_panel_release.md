# Research Note 0059: Controlled Overlap Cache Panel Release

Date: 2026-10-29

Status: implementation and release contract to be committed before new panel
outcomes. This releases one bounded development study, not confirmation.

## Unchanged Design

[Note 0057](0057_controlled_overlap_cache_registration.md) fixes the Phase 2
design; [Note 0058](0058_controlled_overlap_cache_calibration.md) qualifies the
historical measurement path. The original config remains byte-identical,
including its historical `status_at_registration` field. The separate
[release record](../../configs/controlled_overlap_cache_release_v1.json) pins
that config and the prior calibration publication. No endpoints, seeds,
sample counts, overlap maps, spectral gates or response requirements change.

The approved run uses **one model instance in one process** for 128 historical
calibration cells, then 2,048 experimental cells. It does not reuse an
image-conditioned cache between source contexts. Every cell uses the existing
fresh source-only path. There is no retry, resume, replacement or automatic
extension into the next study.

## Capture Order And Gates

First repeat all 128 historical cells and require 384 exact tensor comparisons,
with identical actual prepared inputs, generated suffix IDs, token layout,
runtime and snapshot. Failure stops before the panel. The prior passing run
does not substitute for this same-process calibration.

Then capture all 256 reference-source cells: four reference maps x sixteen
r1/r2 sources x four factorial cells. Freeze their run/audit/tensor registry
before the first test source. Finally capture 1,792 r3/r4 cells, including
the four reference-map diagonals and all 24 test maps. Total: 512 factorial
blocks / 2,048 cells / 6,144 tensors; with calibration, 2,176 cells / 6,528
tensors. The unchanged lower disk bound is 8 GiB.

Order is mask 0-3, reference map then q = 0/1/4 sixteenths, destination 0/1,
sorted source label, cells MM/JJ/MJ/JM, subject to reference-source-first
ordering. Calibration source labels are sorted too. An append-only capture
journal records each successful ordinal. Reference seal occurs after total
ordinal 384; the first test ordinal is 385. Partial outputs and current job
are retained on failure, and primary results are withheld.

For panel inputs, only the audited frame hash and actual processor pixel hash
may differ from the historical source comparator. Token IDs, attention mask,
image sizes, prompt, suffix, effective regions, dtype and tensor shape remain
fixed. A changed response is retained as a failure, never coerced to `The image`.

## Analysis Implementation Details

Reuse the existing full-tensor factorial analyzer and input cosine scorer.
Save all 1,536 target factorial reports, 3,072 complete-region interaction
vectors, 168 score matrices and 18 overlap aggregates. The six views retain
the registered twelve primary endpoints and 10,000 paired bootstrap draws.
All source/mask/family weighting follows Note 0057; no new p-values are added.

Implementation details resolved before outcomes:

- Exact zero vectors are unavailable. A summed reference-unit-vector norm at
  or below 1e-12 is cancelling, matching the earlier scorer.
- q = 0 and q = 1/4 are primary-required. Any unavailable score there withholds
  both endpoints for the affected view. Missing intermediate-q or diagonal
  scores remain visible but do not remove otherwise complete primary results.
- Bootstrap draws remain paired across all six views, including unavailable
  views; the twelve-endpoint tail allocation is not reduced. Do not use
  NaN-skipping means or discard a family/map.
- Secondary cross-level direction/norm diagnostics compare q = 0 with 1/16
  and 1/4 at the same mask, destination identity and test source. Save all
  1,536 records, with explicit nulls for undefined zero-norm quantities.
- Pre-image zero effects are reported for all 1,536 factorial targets, not
  used to select favorable rows. A later integrity concern remains an audit
  issue rather than permission to silently filter.

These clarify implementation, not a new outcome-selected hypothesis family.
Primary range exclusion of zero is still exploratory, not validated coverage
or confirmation. No input/cache ratio is a causal processing-gain estimate.

## Verification And Launch

Fixture tests cover capture order, reference sealing, allowed pixel-only
changes, missingness, fixed denominators and a separately implemented bootstrap
oracle. The no-model preflight checks existing artifacts and the new committed
code before loading model weights. After capture, verify saved tensors,
factorials, full vectors, scores, margins, bootstrap and ledger counts using a
separate publication path. Do not change measurement code after outcomes.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_controlled_overlap_cache_panel.py preflight \
  --output-root runs/controlled_overlap_cache_panel_preflight_v1

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_controlled_overlap_cache_panel.py run \
  --preflight-root runs/controlled_overlap_cache_panel_preflight_v1 \
  --output-root runs/controlled_overlap_cache_panel_v1

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/analyze_controlled_overlap_cache_panel.py \
  --run-root runs/controlled_overlap_cache_panel_v1
```

Every output root must be fresh. Confirmation remains not generated and not
sealed. Notes 0060-0061 will report results and the bounded next-stage decision,
including failures or unavailable endpoints if they occur.
