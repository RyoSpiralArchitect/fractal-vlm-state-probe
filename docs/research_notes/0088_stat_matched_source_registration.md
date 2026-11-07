# Research Note 0088: Fresh-Source Matched-Input Panel Registration

2026-11-08. **Design only; acquisition not released.** The
[configuration](../../configs/stat_matched_source_panel_v1.json) allocates
identities, estimands, missingness rules and a bounded budget. No new source,
candidate image, coordinate map, processor call, model context or score is
created by this registration. Cohort hashes do not yet exist. Implementation,
independent toy oracles, a separate release and explicit execution approval
are still required. Publish the registration/release before acquisition;
a local commit alone is not an independent public timestamp.

## Question And Claim Ceiling

[Note 0087](0087_stat_matched_input_calibration.md) qualified the synthetic
guarded-tile construction, not learned representations or a real cohort.
The next question is whether a fixed within-pool rearrangement changes
FastVLM encoder/projector factorial direction correspondence on new seeds,
while all twelve named raw/native feature vectors remain matched.

This is a **fresh-seed development panel within eight known ordered
generator families**, not unseen-family transfer or the untouched
confirmation cohort in Note 0052. Prior results informed the design.
Differences from these finite baseline views cannot identify learning,
semantic steering, multimodal specificity, baseline sufficiency or cache
mediation. Untested positional, texture and spectral information remains.
No untrained model, training intervention, new prompt or direct probe is
included. Preserve all 118 files frozen by Note 0087.

## Sources And Split

Keep the existing eight ordered families and generator parameters from
`generator_pairing_seed_holdout_v1.json`. For each family allocate r5/r6
as references and r7/r8 as tests: **32 pairs / 64 single-frame sources**.
The config explicitly lists all role-specific seeds as
`2026110801 + 1009*i`, i=0..63, in family/replicate/A-B order (last seed
2026174368). The roles are never swapped to
manufacture independent replication. Generator frames are 240 x 320 RGB.

New integers do not by themselves prove untouched sources. Before generation,
inventory existing configured source specifications and available source
receipts, recording inventory gaps. Stop without replacement on a repeated
effective generator specification, derived RNG seed or decoded source RGB
hash, within this cohort or against prior available sources. A missing historical inventory
precludes an untouched claim; it is not silently treated as no collision.
Do not search seeds, choose attractive images or replace degenerate pairs.

Pre-acquisition review caught component-stream reuse in an initial draft
allocation with consecutive integers: r6 triangle seed 2026110820 uses
palette seed +1, equal to r7 square seed 2026110821. No source was generated
under that draft. The registered spaced allocation instead audits all
**104** generator RNG initializations, including palette/background offsets
and frame-0 salted/modular seeds, against each other and historical available
sources. This removes exact stream reuse, not every statistical dependence
or the deliberately shared deterministic geometry within a generator.

Generate and seal all reference inputs and all 256 reference representation
cells **before generating test sources**. The primary family reference uses
only r5/r6 shift 0. Other reference shifts remain audit material; they cannot
be selected as better templates after seeing test results.

## Payload-First Factorial Construction

Cropping an already palette-matched full-frame quartet would break palette
equality on the selected pixels. Instead, sample first and rank-match second:

1. From each original A/B frame, take the four 6 x 6 tiles in each of the
   6 x 8 content cells, at the qualified offsets (7,7), (7,27), (27,7),
   (27,27). Source coordinates omit the forty-row canvas padding.
2. Pack 6,912 RGB pixels in C order `[cell_row,cell_col,tile,u,v,channel]`
   into a 72 x 96 x 3 array. Packing is an ordering convention, not a resize.
3. Form MM=A, JJ=B, MJ=A-rank/B-palette and JM=B-rank/A-palette using the
   existing `luminance_quantile_match_rgb`. Its stable luminance sort keeps
   complete donor RGB tuples, including deterministic tie handling.
4. Invert packing into the fixed guarded tiles on a 320 x 320 canvas with
   **RGB [128,128,128] everywhere outside the tiles**. Apply each of the four
   already qualified tile maps once. Never redo palette ranks after shifting.

The identical background plus payload matching preserves exact **full-canvas
raw joint RGB multisets** for MM/JM and MJ/JJ, at every shift. It does not
imply those pairs match within each pool, or after spatially dependent native
resizing. Separately, each cell's shifts must match its own identity in all
64 pools and full canvas at both raw/native starts, as in Note 0087.

The artificial canvas does not preserve the whole original source image or
its palette. Only 6,912 coordinates move under a nonidentity shift: 9% of
content / 6.75% of the full canvas. Internal tile texture is unchanged.
Record actual raw/native value-change counts, including zero changes, rather
than dropping fixed-point sources. These are four fixed cyclic arrangements,
not independent masks or whole-geometry destruction.

## Input Gates And Negative Controls

First repeat all 128 historical and 25 synthetic processor calls and every
Note 0087 gate, including zero/constant shams and the expected-rejection
white-cell sentinel. Then apply the unchanged processor path and feature
definitions to every candidate. Native shape is [1,3,1024,1024] float32;
non-pixel payloads, including `image_sizes=[[320,320]]`, must match the
qualified synthetic contract. No processor override to obtain matching.

For 32 pairs x 4 cells x 3 nonzero shifts, require:

| Candidate input audit | Fixed denominator |
| --- | ---: |
| All twelve feature-vector matches, absolute 1e-10 / relative 0 | 4,608 |
| Exact joint RGB word multisets, both starts x (64 pools + full canvas) | 49,920 |
| All 512 candidates x twelve actual/oracle vector comparisons | 6,144 |
| Raw full-canvas palette-donor multiset matches, two pairs x four shifts | 256 |

Store complete vectors and signed residuals. Tolerance is numerical, not
practical equivalence. Any unexpected mismatch stops the entire release;
there is no accepted subset, adjusted tolerance or replacement source.
Keep mandatory controls even when the substantive result is negative.

## Observation And Reference Seal

Use only the previously qualified FastVLM-0.5B-bf16 snapshot, ACK source
prompt, seed 20260604, temperature 0 and maximum two output tokens. Repeat
the unmodified historical OFF/ON/OFF calibration: 384 contexts, complete
input/suffix/selected-cache parity and both site-array historical checks.
Keep its single model load in the same process for the new panel.

Also acquire the eight zero/constant x four-shift inputs in OFF/ON/OFF,
24 contexts. Prepared inputs, natural suffixes and selected caches must be
exact OFF/ON/OFF and across shifts within each fixture. Both ON site arrays
must be exact across shifts; OFF stages do not observe site arrays.
The nonmatching sentinel remains an input control, not a promised nonzero
model response. Neither shams nor sentinel enters family scores.

Observe full native bfloat16 encoder [1,256,3072] and projector [1,256,896]
tensors at the existing sites. The observer must retain exact objects, make
one target call and restore the original method; no extra forward, pooling,
crop, whitening, token selection or cross-site alignment. Store lossless
upper bfloat16 words with exact float32 roundtrip. For each fresh candidate,
prepared pixel words must equal its processor-only seal. Retain non-pixel,
actual natural suffix, token layout and selected-cache audit receipts.
**Do not require new images to say `The image` or force historical IDs.**
Fresh candidate cache tensors are audit artifacts, not new cache endpoints.

After calibration/shams, capture all 256 reference cells (16 pairs x four
cells x four shifts). Seal arrays, norms and fixed shift-0 references at
context ordinal **664**. Only then generate/audit the 256 test inputs;
their first model context is **665**. No test score is computed during
capture. A degenerate reference remains unavailable, never replaced.

## Estimands And Low Norms

For each site or baseline view, flatten the complete tensor in C order and
compute `z=(JJ-MJ)-(JM-MM)` in float64. Family reference is
`unit(unit(z_r5_s0)+unit(z_r6_s0))`; no fitting or alignment.
Test each r7/r8 direction at every shift against all eight references.
Margin is own-family cosine minus mean cosine to the other three families
in the same broad class. Retain all 56 eight-by-eight score matrices:
14 views x four shifts, averaging the two test pairs only at the score level.

Two primary **descriptive** endpoints, one per learned site:
`D = mean_s1,s2,s3(M_s) - M_0`. Pair shifts within each test source, then
equally average two test pairs within each family and eight families.
Secondary outputs retain all margins, full interaction direction cosines,
norm ratios and `norm(z_s-z_0)/(norm(z_s)+norm(z_0))`, plus every baseline
profile and descriptive raw/native spectra. No baseline winner selection.
A zero D need not imply unchanged vectors; direction/amplitude diagnostics
are required. A nonzero D does not identify learning or semantics.

Do not normalize numerical residue into an apparent direction:

- Exact-zero interactions are unavailable, never imputed.
- Raw global moments must have zero interaction because the full raw
  palette marginals match across spatial donors. This expected unavailable
  baseline is not a failed input gate and does not veto learned-site results.
- For each baseline dimension d, the four per-cell oracle coordinate
  budgets give `E_z=4e-10*sqrt(d)`. Require `norm(z)>100*E_z` for direction
  reporting. The analogous across-shift bound is `4e-10*sqrt(d)` from four
  matched-cell differences. These are operational budgets conditional on
  the gates, not a proof of real-arithmetic accuracy or statistical CIs.
- For learned sites require `norm(z)>1e-6*sum_cell_norms` for the primary
  profile. This is a fixed relative-amplitude guard, **not** a calibrated
  error bound or smallest meaningful effect. Also publish complete results
  at fixed floors 1e-8, 1e-6 and 1e-4, without choosing the favorable one.
- Both reference replicates must be available. A summed unit-reference
  norm <=1e-12 is unavailable independently of those amplitude guards.

Retain unnormalized norms, cell scales, thresholds and reasons for every
unavailable row. A margin needs all four same-class references and its test
direction. If any required margin is unavailable, the fixed-denominator
aggregate endpoint is unavailable; never silently renormalize to complete
cases. Conditional summaries may show their explicit denominators but
cannot substitute for a primary endpoint. Retain cross-class scores as
diagnostics even though the primary margin is within class.

The 16 test pairs are nested in eight known families. Cells, shifts and
pairwise distances are dependent repeated measures, not sample-size
multipliers. No p-values, confidence intervals, significance threshold,
equivalence test or guaranteed population precision is registered.

## Budget, Release And Verification

| Future acquisition | Maximum / fixed plan |
| --- | ---: |
| Fresh original source frames | 64 |
| Candidate images, references / tests | 512 = 256 / 256 |
| Existing coordinate maps reused / new maps | 4 / 0 |
| Processor-only calls, historical + synthetic + candidate | 665 = 128 + 25 + 512 |
| Model contexts, historical + sham + reference + test | 920 = 384 + 24 + 256 + 256 |
| Experimental full site arrays / interaction vectors | 1,024 / 256 |
| New direct probes, training or prompt variants | 0 |

Normal multimodal preparation within 920 source contexts is additional to
the 665 processor-only calls, not included inside that number. Record actual
internal processor/decoder calls separately from source contexts. No retry,
resume, alternative snapshot or second model load. Stop on unexpected failure
and retain attempted/completed counts, partial arrays and the first failure.

Launch requires **28 GiB** free: 16 GiB uncompressed payload allowance,
4 GiB metadata/verification overhead and 8 GiB reserve. This conservatively
covers native candidate arrays, calibration captures, packed site arrays,
selected caches and at most 64 full float64 scratch vectors. No compression
savings are assumed. The implementation must enumerate its actual bound
within these allowances before release. Recheck reserve during acquisition;
shortage stops rather than deleting old runs, models or user files.

Before any new source, commit separate construction/scoring implementations,
independent oracles, toy tests and file-only preflight/verifier. Fix a release
receipt and public registration timestamp; this note alone authorizes no
execution. Preserve first failures and unavailable cells across versions.
After capture, require two fresh zero-model, zero-processor file verifications
with byte-identical registered outputs before publishing all prespecified
results. Cohort/source/reference seals and runtime/snapshot receipts must
accompany claims. The untouched confirmation cohort remains **not generated
and not sealed**; this panel cannot be relabeled as that confirmation.

## Local Design Validation

Eight metadata-only contract checks and all 705 local tests pass. The
inherited 118 file hashes remain exact; scoped Ruff, whitespace and all
493 local link targets in the six touched Markdown documents pass. A
separate read-only review identified the draft RNG-stream reuse above;
after correction and regression coverage, its follow-up found no remaining
concrete defect in this design. These checks do not generate the allocated
cohort, qualify a future implementation or authorize acquisition.
