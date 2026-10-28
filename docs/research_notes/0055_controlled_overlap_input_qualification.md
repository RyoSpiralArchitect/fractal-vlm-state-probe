# Research Note 0055: Controlled Overlap Input Qualification

Date: 2026-10-29

Status: measurement completed; all publication replay checks passed.

## Main Result

All **28 maps x 32 source blocks = 896 four-cell audits** pass the unchanged
input gates. The panel contains **3,584 transformed cells** and is eligible
for a separately registered cache study. No model weights were loaded and
no model/cache forward was run. Input eligibility is not cache evidence.

The construction in [Note 0053](0053_controlled_overlap_coordinate_construction.md)
and measurement contract in [Note 0054](0054_controlled_overlap_phase1_registration.md)
were committed at `1e19014a54a999f14b091f9ce0467b4aa034afdf` before generating
the full-size maps or inspecting new input scores. No seeds, retry limits,
thresholds, normalization, sample counts or measurement code changed afterward.

This reuses all eight development families and all four source replicates.
There is no untouched confirmation cohort, new source family or fresh source
seed. Repeated transformed inputs are not independent source observations.

## Coordinate Controls

Every map fixes exactly 19,200 of 76,800 content sites and moves exactly
57,600. Each of four mask replicates has one reference map and two test
destination maps at each of three overlap levels. The reference is common
across levels. Destination draws share the fixed set within a mask/level.

| Test level | Shared fixed sites | Shared moved sites | Total content coincidence | Test maps |
| --- | ---: | ---: | ---: | ---: |
| q = 0 | 0 | 0 | 0% | 8 |
| q = 1/16 | 4,800 | 0 | 6.25% | 8 |
| q = 1/4 | 19,200 | 0 | 25% | 8 |

Four additional reference maps supply same-map controls. The q = 1/4 test
maps are not those diagonal controls: their fixed sets match the reference,
but their moved destinations do not coincide with it. The unchanged 25,600
padding pixels are outside the content denominator.

All 28 coordinate maps pass independent seeded reconstruction and exact
count checks. The 24 test maps require 1-3 coordinate-only attempts each,
below the fixed cap of 128. All rejected counts are retained. Rejection
uses source indices only, never pixels, spectra or correspondence scores.
Maps are conditional random single-cycle derangements, not uniform arbitrary
permutations. See the full recipe/rejection ledger in the publication summary.

## Input Gates

The original 240 x 320 content sits in a 320 x 320 palette-mean canvas.
Each whole-canvas joint RGB multiset is preserved exactly under the maps;
relative to the original content distribution it remains 75% original palette
plus 25% registered fill mass. This is not original-marginal preservation.
Padding, inverse roundtrip and non-pixel processor metadata are checked on
every cell. In particular, `image_sizes` remains `[[320, 320]]`.

| Map role | Maps | Accepted four-cell blocks | Transformed cells |
| --- | ---: | ---: | ---: |
| Reference | 4 | 128/128 | 512 |
| q = 0 | 8 | 256/256 | 1,024 |
| q = 1/16 | 8 | 256/256 | 1,024 |
| q = 1/4 | 8 | 256/256 | 1,024 |
| Total | 28 | 896/896 | 3,584 |

The largest within-block processor centroid relative error is **4.661376%**,
below the unchanged 5% limit. The largest HF pairwise absolute difference is
**0.001364249**, below 0.02. No partial family or favorable map was selected.

Qualification uses the original model-loading-path processor, snapshot
`81ffe929046666c43de53691147b1669ba0f3a4c`. All **128 historical anchor
processor cells** reproduce their exact pixel hashes before new measurements.
The frozen record retains processor implementation/config hashes and runtime:
Python 3.13.13, NumPy 2.4.4, Pillow 10.4.0, torch 2.11.0, torchvision 0.26.0,
transformers 4.57.6 and mlx-vlm 0.4.4. Execution is offline and does not load
model weights. Actual processor outputs, rather than a substitute image path,
are used for the processor interaction vectors.

## Between-Level Spectra

Within-block acceptance does not imply equal spectra between q levels.
Compare each of the 32 blocks x four masks x two destination identities with
its q = 0 counterpart. These 256 paired comparisons per nonzero level are
dependent descriptive audits, not tests or an additional selection gate.

| Level minus zero | Centroid target relative change, min / median / max | HF target difference, min / median / max |
| --- | --- | --- |
| q = 1/16 | -0.808893% / -0.028932% / +0.645626% | -0.00010790 / -0.00000128 / +0.00012453 |
| q = 1/4 | -0.755807% / -0.009373% / +0.813043% | -0.00015541 / +0.00001081 / +0.00011936 |

The means used here are the four-cell gate targets, not pooled pixels or a
claim of full-spectrum equivalence. Mask coupling and conditional destination
draws also change spatial arrangement with q. The next cache stage must keep
these differences explicit, not call the manipulation frequency-only.

## Retained Data And Verification

Measurement uses **3,712 processor calls**: 128 historical checks plus 3,584
new cells. Save all cell audits and exactly 512 interactions per view, 1,024
total: every source on four reference maps, and r3/r4 sources on 24 test maps.
Unused r1/r2 test-map vectors are not retained; their cells are still audited.
All retained norms are nonzero and finite: raw norms span 86.577165-237.166634;
processor norms span 228.517254-622.537436.

Publication verification separately replays the 3,584 new processor cells,
checks exact bytes/metadata, and reconstructs all 1,024 interactions from
integer RGB or promoted processor pixels. The spectral-statistic extractor
is shared with measurement; its replay is reproducibility verification, not
an independent FFT implementation. Coordinate assignment, RGB/inverse checks,
score arithmetic and bootstrap resampling have separate verification paths.
All 3,584 verification replays match exactly, with maximum interaction
reconstruction error `4.45e-16` (rounded upward). No tolerance was relaxed.

Public receipts: [summary](../../examples/research_notes/0056_controlled_overlap_phase1/summary.json),
[all cells](../../examples/research_notes/0056_controlled_overlap_phase1/cells.jsonl),
[vectors](../../examples/research_notes/0056_controlled_overlap_phase1/vectors.jsonl),
and [512 between-level audits](../../examples/research_notes/0056_controlled_overlap_phase1/between_level_spectra.jsonl).
Large PNG/vector payloads remain in the ignored local run roots; the ledgers
retain their paths and SHA-256 hashes. Reproduction requires the qualified
historical source artifacts as well as the processor/runtime; the compact
public bundle alone is not a standalone image dataset.

The eligible-input decision is now complete. Correspondence and the Phase 1
decision are reported separately in [Note 0056](0056_controlled_overlap_input_correspondence.md).
