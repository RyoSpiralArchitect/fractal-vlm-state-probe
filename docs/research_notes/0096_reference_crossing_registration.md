# Research Note 0096: Reference Crossings And Full Input Comparator Registration

2026-11-09. This is a **post-hoc registration before new endpoint computation**,
not a pre-outcome preregistration relative to Notes 0093-0094, confirmation or
an execution release. The [plan](../../examples/research_notes/0096_reference_crossing_registration/plan.json)
fixes the saved-data diagnostic proposed in Note 0095. Registration and
metadata-only preparation are in scope; real-array scores remain withheld.

## Fixed Comparison

Retain all r5/r6 reference pairs and r7/r8 test pairs: sixteen pairs in each
split, eight known families, four cells and four arrangements. At each of
encoder, projector, full raw RGB and full processor tensors, report all
reference-arrangement x test-arrangement crossings. Each of twelve profiles
(four views x three relative amplitude floors) has score shape 4 x 4 x 16 x 8.
Floors and crossings are dependent checks on the same data, not new samples.

```
z = (JJ - MJ) - (JM - MM)
r[f,a] = unit(unit(z[f,r5,a]) + unit(z[f,r6,a]))
m[p,a,b] = cosine(z[p,b], r[own,a])
           - mean cosine to the other three same-class r[f,a]
fixed_change = mean_b=1..3 m[p,0,b] - m[p,0,0]
matched_change = mean_b=1..3 m[p,b,b] - m[p,0,0]
reference_difference = mean_b=1..3 (m[p,b,b] - m[p,0,b])
```

The closure matched_change - fixed_change = reference_difference and
independent scorer comparisons use absolute tolerance 1e-12, relative
tolerance zero, fixed before new scores. The learned a=0 rows must reproduce
all original scores, margins, missingness and D at all three original floors.
No reference is fitted from tests; no map, layer, reference arrangement or
alignment is selected from scores. Own/other values accompany all margins.

## Numeric And Missingness Policy

Use stored coordinate order, C flattening and float64 arithmetic. Expand
saved encoder/projector uint16 bfloat16 words before conversion, not numeric
uint16 values. Raw uint8 converts without division by 255; native float32
converts without reprocessing. No normalization, whitening or coordinate
realignment is added before the factorial interaction.

All four views use norm(z) > floor * sum(four cell norms), with primary
floor 1e-6 and sensitivity floors 1e-8 and 1e-4. The learned rule is unchanged;
for the two new full-input views this is a newly registered scale-relative
amplitude policy, **not** the twelve feature views' error envelope, a calibrated
quantization uncertainty or equivalence threshold. This avoids assuming that
a feature-matching tolerance applies to millions of input coordinates.

Both reference replicates must be available; their summed unit directions
must have norm greater than 1e-12. Zero or small directions remain
unavailable, never zero-filled. Publish unnormalized interaction hashes,
norms, cell scales, thresholds, reference cancellation and reasons. Overflow,
underflow of a nonzero norm and nonfinite data are errors, not missing data.

Keep all pairs and families. A required missing margin makes its endpoint
and corresponding fixed-denominator aggregate unavailable, without filtering.
Average three nonidentity arrangements, then two test pairs within family,
then eight equally weighted families. Publish all families regardless of sign.
There are no p-values, confidence intervals or equivalence tests.

## Pinned Inputs And Preparation Command

The plan pins nine metadata/public files, including the capture manifest,
frozen specification, model/candidate ledgers and original published profiles.
The capture manifest transitively binds all array file hashes. The preparer
extracts a complete selected inventory of 2,048 receipts: 512 per view,
half reference and half test. It checks identities, split, shape, storage
dtype, safe paths, existence and agreement with the pinned capture manifest.
All 135 inherited implementation files must still match.

```
python3 -m fractal_vlm_state_probe.cli.prepare_reference_crossing \
  --output runs/reference_crossing_preparation_v1
```

The command has no run/score mode and reads no array payload. Its result
is **METADATA_PREPARED_NOT_SCORE_RELEASED**, not native-array reverification
or completed scientific analysis. Code hashes are recorded, but committed
implementation status is not qualified by this preparation command. A fresh
canonical output root is mandatory; failures are retained and not retried.

## Gates Before Real Scores

The scoring runner and independent native-content verifier are not implemented
or released by this registration. Before execution, commit the complete
plan/code/tests and separately obtain scope approval. Verify all selected
file and word hashes and stored layouts. The independent verifier must
reconstruct from original native arrays, not trust new score summaries.

Analytic toys must cover identity, common orthogonal change, changed pair
geometry, zero directions, low-amplitude boundaries, reference cancellation
and fixed-denominator missingness. Verify the original learned s0 row before
accepting new crossings. After one authorized run, require two fresh,
byte-identical independent saved-data verification trees; preserve the first
failure without replacement, tuning or promotion.

No new images, processor calls or model contexts are needed or authorized.
The twelve inherited feature views and original primary endpoints remain
unchanged. Matched-reference recovery could indicate condition-dependent
correspondence, not prove a common rotation; lack of recovery would not
prove information loss. Input/model differences alone do not identify
training, semantics or causal gain. Untouched confirmation remains open.

## Observed Metadata Preparation

The single metadata-only preparation completed at the canonical root with
status `METADATA_PREPARED_NOT_SCORE_RELEASED`. The
[receipt](../../examples/research_notes/0096_reference_crossing_registration/preparation_receipt.json)
records plan, implementation, preparation summary and selected-inventory
hashes. All 2,048 selected receipts are present, split equally between
reference and test; all 135 preceding implementation hashes match.
No array payload was read or newly hash-verified. No new scores, sources,
processor calls or model contexts were produced.

The 91 focused registration, metadata-corruption, decomposition, publication,
source/acquisition registration, example and scope tests pass; lint passes.
These tests include the registered endpoint arithmetic on analytic scores,
not validation of an unimplemented full-array scorer. Independent review,
commit qualification and score execution release remain open. The retained
preparation receipt does not claim a committed or publicly published freeze.
