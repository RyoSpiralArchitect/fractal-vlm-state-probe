# Research Note 0054: Controlled Overlap Phase 1 Registration

Date: 2026-10-29

Status: specification to be committed before the first new input score.
This is a bounded development study, not the untouched confirmation stage.

## Frozen Inputs And Budget

[Configuration](../../configs/controlled_overlap_phase1_v1.json) fixes four mask
seeds 2026102901-2026102904, reference seed 2026102910 and destination seeds
2026102911-2026102912. PCG64 receives explicit SeedSequence keys; the reference
uses [reference seed, mask seed], each test uses [destination seed, mask seed,
q in sixteenths]. The ordered mask uses its mask seed alone. See
[Note 0053](0053_controlled_overlap_coordinate_construction.md) for construction.

Reuse the eight existing families, r1/r2 references and r3/r4 tests: all 32
blocks are development material. The 28 maps give 896 four-cell audits and
3,584 transformed input cells. Replay all 128 historical anchor processor
cells first. There are 3,712 measurement processor calls in total, excluding
separately counted verification replay. No weights or cache forward are used.

Every map is audited on all 32 blocks. Save both raw float64/255 and native
processor float32-promoted-to-float64 interaction vectors. Retain all 32
vectors for each reference map, and the sixteen test-source vectors for each
of the 24 test maps: 512 vectors per view, 1,024 total. The omitted vectors
are unused reference-source/test-map combinations, not failed observations.

Use the existing qualified processor, snapshot, runtime and exact historical
pixel hashes. Preserve expanded RGB multisets, palette-mean padding and
non-pixel metadata. The unchanged centroid 5% and HF 0.02 gates apply to all
four cells within each block. Cache eligibility requires all 28 x 32 blocks
to pass. Between-level spectral drift is separately reported, never declared
matched merely because the within-block gates pass.

The input diagnostic includes all planned maps and families even if spectral
eligibility fails. It then cannot qualify a cache panel. Integrity failures
(hash, coordinate, runtime, pixel roundtrip, non-finite values) stop the stage.
Retain partial outputs. No replacement maps, threshold relaxation or repeated
measurement launch after an outcome is inspected.

## Scores And Estimands

The full interaction is JJ-JM-MJ+MM. Reference direction is the normalized
sum of separately normalized r1 and r2 vectors. A test score is its cosine
with each of the eight family references; the margin subtracts the mean of
the other three references in its own broad class. Full-vector normalization
always precedes component accounting.

Average the eight paired map score matrices per q (four masks x two
destinations), then test seeds within family and all eight families equally.
Keep 16 averaged test rows, not 128 independent test images. For each of raw
RGB and actual processor pixels, report these two primary descriptive endpoints:

1. M(0), the mean margin at zero total content coincidence.
2. D = M(1/4) - M(0), computed with the same reference, source seeds and
   destination-replicate identities paired across levels.

The intermediate level, same-map controls, retrieval, all family signs,
reference/test vector norms and raw fixed/moved/noncoincident/padding terms
are diagnostics. Save all 56 score matrices (28 per view). Raw terms reconstruct
every full dot product and aggregate margin within 1e-12. All diagonal raw
scores should agree by common-permutation isometry, not learned invariance.

## Uncertainty And Claim Limits

This phase does not assign a randomization p-value to D: q labels are not
exchangeable under the construction, and a family-label correspondence null
does not test an overlap effect. No new p-values or equivalence tests are
performed. These decisions refine Note 0052 before new outcomes, rather than
silently treating its open statistical fields as defaults.

Report exploratory percentile ranges from 10,000 draws, seed 2026102920.
Resample four mask blocks, two destination draws within each sampled mask,
eight family blocks, and two test-source seeds within each sampled family.
Use the same family/source draws across masks and the same complete draw
across q levels and both representations. Reference directions remain fixed.
The four endpoint ranges use quantiles 0.00625 and 0.99375, a Bonferroni tail
allocation. This does not guarantee simultaneous coverage in this small,
crossed, reused-data bootstrap. Family resampling describes sensitivity to
the observed families, not random sampling of all possible generator families.
No post-selection success claim is based on range exclusion of zero.

The budget is fixed for development and disk/memory feasibility, not powered
to prove absence. There is no guaranteed minimum resolvable effect or chosen
equivalence width; report the observed widths and retain imprecision. A
separate pre-outcome precision design is still required for confirmation.
Do not increase sample size based on these results. Minimum free disk before
capture is 20 GiB; the retained subset of float64 vectors needs about 14 GiB
including raw vectors, plus images and receipts.

## Release And Reproduction

The runner requires code/config bytes to match a committed Git revision and
validates prior qualified hashes. Coordinate construction and measurement use
fresh output roots and refuse overwrites. Verification is read-only with
respect to measurements; any processor replay is separately counted.

```bash
PYTHONPATH=src python3 scripts/run_controlled_overlap.py maps \
  --output-root runs/controlled_overlap_maps_v1

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_controlled_overlap.py measure \
  --map-report runs/controlled_overlap_maps_v1/coordinate_report.json \
  --output-root runs/controlled_overlap_input_v1

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/analyze_controlled_overlap.py \
  --run-root runs/controlled_overlap_input_v1
```

No confirmation images, reference/test cohort manifest, seal or release receipt
is created by this phase. No cache study, causal intervention or baseline
selection is launched. Notes 0055 and 0056 will separately report input
eligibility and correspondence, including failures and negative families.
