# Research Note 0044: All-Cell Frequency Control Feasibility

Date: 2026-10-07

Status: bounded input sweep complete; neither arm qualifies the complete
eight-family cache design. No new model-cache forward was performed.

Following [Note 0043](0043_fastvlm_pairing_holdout_and_input_baseline.md), test
whether all four transformed cells can pass the registered processor-space
frequency gates while preserving each palette donor's joint RGB multiset.

The [stage-4 execution addendum](../pairing_validation_protocol.md#stage-4-input-execution-addendum)
freezes a bounded low-pass rank-field sweep and a separate common pixel
permutation comparator before either input scoring or cache forwards.
The latter destroys geometry and preserves raw full-vector cosine relations
by coordinate permutation; it does not identify frequency alone.

## Frozen Input Design

The same 32 source-pair units contain eight known ordered generator pairings,
with r1/r2 references and r3/r4 tests. Images are not a new independent cohort.
The qualified actual model-path FastVLM processor is unchanged at revision
`81ffe929046666c43de53691147b1669ba0f3a4c`, implementation SHA-256
`1fc2d894a8128c713980c69e544d434a5cbe82f5c6c938260bd9c689b3c37de2`.
No neural model weights were loaded for the input sweep or padding diagnostic.

The structured arm low-passes each original floating-point luminance rank
field, then assigns unchanged palette RGB pixels by stable rank. One spatial
field is shared across both palette cells. Eight cutoffs per donor give
64 candidate cutoff pairs per block, 2,048 candidate pairs across 32 blocks.
Only input metrics choose cutoffs; no cache or direction score is inspected.
The comparator applies one fixed bijection to all pixels of all four cells
and all source-pair units. It preserves raw coordinate-vector cosine relations
but destroys geometry and is not a frequency-only intervention.

For every candidate, all four processor-space luminance centroids must lie
within 5% of their common mean. The largest pairwise HF-ratio difference must
be at most 0.02, with joint RGB multisets preserved, identical processor shape
and finite nondegenerate spectra. Luminance is weighted RGB after undoing
recorded channel normalization, not the older arithmetic-channel-mean view.
The radial FFT convention and normalized HF cutoff 0.35 are unchanged.

The input design was committed as `da877cd`; the qualified local custom-code
loader was made noninteractive in `eb7c0d9`, before any input measurement.
Its saved source hash is checked before executing that processor code.

## Acceptance Result

| Condition | Accepted source-pair blocks | Reference blocks | Test blocks | Complete four-seed families |
| --- | ---: | ---: | ---: | ---: |
| Original cells | 0/32 | 0/16 | 0/16 | 0/8 |
| Rank-field low-pass | 4/32 | 1/16 | 3/16 | 0/8 |
| Shared pixel permutation | 1/32 | 1/16 | 0/16 | 0/8 |

Accepted structured blocks, with A/B cutoffs:

- White/blue noise r1: `0.02 / 0.02`.
- Voronoi/quasicrystal r3: `0.12 / 0.04`.
- White/blue noise r3: `0.02 / 0.02`.
- White/blue noise r4: `0.02 / 0.02`.

The only accepted permutation block is square/triangle r1. White/blue noise r2
still fails the structured gate, so r1/r3/r4 do not form a complete eligible
family. No failed block is discarded to manufacture an eight-family study.

All 256 selected transformed-cell RGB-multiset checks pass. All 128 serialized
inverse-permutation shams have identical decoded pixels to their originals.
Processor shapes are `[3, 1024, 1024]` throughout. Every low-pass candidate's
HF spread is below the operational 0.02 tolerance (maximum `0.002083`); across
the two selected arms the maximum is `0.002476`. The limiting gate is the
spectral centroid, not HF spread, shape or palette preservation.

Both complete-panel capture gates reject the actual frozen input receipts
before any model loading or cache capture. New cache forwards, tensors,
factorials and held-out tests are all zero. Earlier selected cache counts
remain 616 / 1,704 / 426. Input candidates and serialized shams are not new
cache observations or independent trials.

## Raw Match Is Not Processor Match

The permutation arm passes the analogous raw-RGB frequency diagnostic in
32/32 blocks, versus only 1/32 using the registered whole processor image.
The raw diagnostic does not override the actual processor acceptance gate.

A concrete checkerboard/hex r1 example:

| Cell | Raw permuted centroid | Whole processor centroid |
| --- | ---: | ---: |
| MM | 0.539253 | 0.006224 |
| JJ | 0.540502 | 0.052392 |
| MJ | 0.540758 | 0.052250 |
| JM | 0.541371 | 0.006232 |

All values use the normalized radial centroid convention; these are not
cycles-per-patch values. Raw whitening nearly equates the four spectral
summaries, but the processor output sorts strongly by palette donor.
Different image dimensions and processing make raw/processor centroid levels
descriptive, not a shared-coordinate gain measure.

## Post-Hoc Padding Audit

The unchanged processor expands the 320 x 240 input to a black-padded square
before resize. This motivates a separate input-only support diagnostic after
the registered sweep. It does not alter the acceptance gate or select a new
mask for a successful model-cache result.

Uniform-white and uniform-black probes identify rectangular nonzero support
at rows `[126, 898)` and columns `[0, 1024)`. This includes resize-transition
pixels; it is not a verified visual-token or pure interior-content mask.
The exact-black exterior occupies `24.609375%` of output pixels.
All 128 measured permutation-cell whole-processor statistics reproduce
exactly before taking the support-only diagnostic.

Using that fixed support rectangle, 29/32 blocks pass the analogous centroid/HF
diagnostic. Checkerboard/hex r1, r3 and r4 still fail. None is promoted to
registered acceptance: changing metric support is changing the measurement,
and no cropped image is supplied to the model.

The luminance variance decomposes exactly into within-content and
between-content-and-black terms. For support fraction s and content mean m:

`Var(Y_whole) = s Var(Y_content) + s(1-s) m^2`.

The between-region term accounts for 17.33%-97.70% of whole-image variance
across the 128 permutation cells (median 65.52%); maximum reconstruction
error is `3.13e-17`. In checkerboard/hex r1 it is about 96.17% for the A palette
and 50.65% for the B palette. Palette mean and content variability couple to
the processor's black exterior even when raw pixel placement is whitened.
This is a variance identity, not an additive decomposition of FFT centroid
or a causal attribution of model-cache behavior.

## Revised Reading And Next Gate

The original question remains open: does known-pairing cache correspondence
survive accepted frequency matching? This bounded grid did not produce the
complete accepted panel required to answer it. Failure of this grid is not
proof that every palette-preserving matching transform is impossible.

The next input-only design should register padding policy as an explicit
factor. A black-pad sham can retain the current processor path, while a
source-derived nonblack fill changes the input marginal and must be named and
audited as such. Freeze the altered marginal, dimensions, sham and all-four-cell
gates before comparing caches. Do not silently crop padding away, change the
centroid metric, enlarge tolerance, or call a whitening comparator a
structure-preserving intervention.

This follows the same research shift as Note 0043: the measured object includes
input transformation and the actual processor path, not isolated macro geometry
or semantic steering. No accepted matched cache correspondence, persistence,
adaptation or causal mediation is established here.

Follow-up: [Note 0045](0045_padding_policy_and_explicit_marginals.md) executes
that registered comparison on all 32 blocks. Common permutation plus
palette-mean padding qualifies 32/32 whole-processor input blocks, with changed
RGB marginals explicitly retained. Black pixel shams reproduce exactly but
change `image_sizes`; model-cache calibration remains unmeasured.

## Artifacts

- `configs/frequency_control_fastvlm_v1.json`
- `scripts/prepare_frequency_control_panel.py`
- `scripts/analyze_frequency_padding_audit.py`
- `scripts/summarize_frequency_control_feasibility.py`
- `runs/frequency_control_fastvlm_v1/input/input_acceptance.json`
- `runs/frequency_control_fastvlm_v1/input/input_audits/`
- `runs/frequency_control_fastvlm_v1/input/sham/`
- `runs/frequency_control_fastvlm_v1/padding_audit.json`
- `examples/research_notes/0044_all_cell_frequency_control/summary.json`

The publication retains all 2,048 cutoff gate evaluations, selected per-cell
statistics/hashes, accepted and failed blocks, and the separate padding audit.
