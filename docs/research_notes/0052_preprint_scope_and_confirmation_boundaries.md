# Research Note 0052: Preprint Scope And Confirmation Boundaries

Date: 2026-10-29

Status: scope and analysis policy recorded; measurement registration incomplete.
No new images, maps, processor calls, model forwards or statistical tests.

## Question And Contribution

Where and under which controls does input-conditioned pairing correspondence
persist across VLM representations and differ from visible output?

Working title: *From Pixels to Source Cache: Auditing Input-Conditioned
Correspondence in Vision-Language Models*.

The intended contribution is an audited distinction between input-coordinate
correspondence, transformation-conditioned representation correspondence and
visible readout. It is not merely that images change internal tensors. Fractal
specificity, persistent state, semantic steering and universal cache loci are
not current claims. New control results may narrow the title and conclusions.

The evidence boundary is commit `f8036f0`, through [Note 0051](0051_raw_correspondence_and_shared_fixed_coordinates.md).
Notes 0027-0051 established valid fresh-forward measurement, pairing-conditioned
seed repetition, input correspondence before inference, and additive raw
accounting. They did not establish causal processing gain or an untouched
validation cohort for the next overlap study. See the
[evidence matrix](../paper_evidence_matrix.md) for measured results and withdrawals.

The [machine-readable scope](../../configs/preprint_scope_v1.json) is a design
record, not a runnable study config. It has no execution runner. Its null
fields are unresolved decisions, not defaults. Contract tests check this record;
they do not enforce access control or provide a future runtime launch gate.

## Competing Explanations

| Explanation | Discriminating observation to seek | Claim ceiling |
| --- | --- | --- |
| Source-coordinate overlap dependence | Within each representation, the paired mean-margin change when overlap is reduced | Dependence under the specified map construction, not a universal coordinate law |
| Simple processing and pooling sufficiency | Registered non-learned local-feature/pooling baselines reproduce the relevant correspondence profile on new inputs | Sufficiency of those tested baselines, not proof of the model's implementation |
| Residual correspondence beyond tested baselines | Correspondence survives audited controls and untouched confirmation without being matched by the registered baselines | Beyond those baselines, not automatically semantic, learned or causal |

These explanations may coexist across families and representations. They are
not an exhaustive model selection contest with a compulsory winner. An
imprecise or unavailable result remains an acceptable outcome.

## Scope Through Approximately Note 0080

| Notes | Work package | Decision point |
| --- | --- | --- |
| 0052-0056 | Scope, map construction, independent coordinate oracle, all-cell input audit, raw/processor scores | Is the intended overlap contrast actually constructed, and what survives before the model? |
| 0057-0061 | Exact FastVLM calibration and the eligible cache panel | Does overlap sensitivity differ between image and post-image regions? |
| 0062-0066 | Non-learned local-feature/pooling baselines and calibrated encoder/projector/cache observations | Which changes already follow simple processing, and where do profiles diverge? |
| 0067-0071 | Prompt, fixed-suffix and readout controls; final confirmation registration | Is the result conditional on the readout context, and which predictions will be tested untouched? |
| 0072-0075 | New source/map cohort, FastVLM and Qwen confirmation | Do the frozen predictions repeat outside development inputs? |
| 0076-0080 | Independent replay, negative-case audit, figures and manuscript | Can another reader reproduce the bounded conclusions? |

These ranges are work packages, not 29 mandatory positive experiments. Notes
0056 and 0061 are scope-review points. A failed prerequisite changes the next
question; it does not trigger more seeds until significance. FastVLM is the
development model; Qwen is the planned second confirmation architecture.
Ministral is optional and requires its own pre-outcome specification. The
existing nine-model surface is context, not a requirement to expand every arm
to every model. A manuscript must not depend on all optional extensions.

## Candidate Input Design

Keep 240 x 320 content in the existing 320 x 320 palette-mean-padded canvas.
Every map selects 3/4 of content sites for permutation. Target exactly 1/4
actual fixed sites; selection alone does not guarantee this because a selected
site can accidentally remain fixed. Construction and its oracle must distinguish
those quantities before any input scoring.

For content index u and source-coordinate maps p_a and p_b, define
`q = count(p_a(u) = p_b(u) = u) / content_pixels`.
Candidate q levels are `0`, `1/16`, `1/4`, corresponding to 0, 4,800 and
19,200 shared fixed sites out of 76,800. These are feasible set-intersection
counts; this note does not yet prove a full permutation construction with
all other requested constraints. Freeze the final levels and construction
before generating or scoring the new experimental panel.

Within a construction replicate, keep the reference map and reference images
the same across q levels. Vary destination realizations within each selected
mask, and repeat mask construction independently. There is no claim that
changing masks can leave the same absolute destination permutation unchanged.
The construction must state any coupling used across levels.

Count shared moved coincidences `p_a(u) = p_b(u) != u` separately. A zero-q
comparison is not a total-zero-coincidence comparison unless those counts are
also zero. Decide whether to eliminate or model shared moved coincidence
before scoring; the decision is currently unresolved. At q = 1/4, shared fixed
masks can still have different permutations. This is not the same-map control.
Keep same-map and raw inverse-permutation controls separate.

Apply each map commonly to all four cells of each source factorial, without
changing each cell's expanded RGB multiset, padding or metadata. Preserve all
coordinate counts and construction attempts. Any bounded rejection procedure
may inspect coordinate constraints only, never correspondence scores. Its
algorithm, retry bound and failure output must be registered first.

The existing all-cell processor thresholds remain unchanged; passing them
only establishes the specified within-condition acceptance. Audit between-q
spectral differences separately. Do not call the q contrast frequency-isolated
unless an additional registered control establishes that claim. Report every
failed block; never rescue it by dropping cells or favorable map selection.

## Estimands Before Outcomes

For every representation v and map p, retain the original factorial vector
`z = v(JJ) - v(JM) - v(MJ) + v(MM)`.
For each family, form `r = unit(unit(z_r1) + unit(z_r2))` from reference seeds.
The test margin is own-family cosine minus the mean cosine with other family
references in the same broad class. No cross-class distractor substitution.

For M_v(q), average map-score matrices using registered balanced weights first,
then test seeds within each family, then families equally. Keep the same
source/reference units paired across q. Do not average vectors and normalize
that average in place of the registered score average.

The two planned input endpoints per view are:

1. `M_v(0)`: correspondence at zero shared fixed overlap, explicitly retaining
   the shared moved-coincidence audit.
2. `D_v = M_v(q_high) - M_v(0)`: the paired high-minus-zero overlap contrast.

Raw RGB and actual-processor pixels give four planned input endpoints if the
candidate design is adopted. The intermediate level, retrieval, vector norms,
direction cosines and additive raw terms are secondary diagnostics. All family
signs remain visible. Full-vector normalization precedes raw partitioning;
masked re-normalization is a different estimand. Zero vectors, cancelling
references and incomplete paired panels remain unavailable with reasons;
do not silently change the aggregation denominator.

FastVLM retains the existing targets L1 keys, L12 keys and L23 values, each in
image and post-image regions. Its cache test family, Qwen targets, intervals
and final multiplicity policy must be committed before the respective new
scores. The old 576-assignment correspondence null is not automatically a
test of the paired overlap effect D_v. Specify and validate each null's
exchangeability and dependence units separately.

Inference must preserve family, source-seed, mask-seed and destination-seed
dependence. Four cells, ordered map pairs, layers and token positions are not
independent replicates. Claims are conditional on the sampled families and
map-construction rule unless a separate family-sampling design warrants more.
Non-significance does not establish absence: a practical equivalence margin
and adequate precision must be fixed before any equivalence conclusion.

## Baselines And Readout Boundaries

An exactly permutation-invariant raw feature g of the joint RGB multiset has
`g(MM) = g(JM)` and `g(JJ) = g(MJ)` here. Its factorial interaction cancels
algebraically. Use such a histogram/mean feature as a negative calibration;
its zero direction is unavailable, not a zero-cosine performance score.
This identity need not survive processor resampling.

The meaningful simple-processing comparison therefore includes local features
followed by pooling, with feature definitions, scales and fitting choices
fixed on development inputs only. Select no baseline on confirmation results.
Observed encoder/projector/cache profiles do not alone identify the effect of
training; that stronger claim needs a suitable trained/untrained comparison.

Prompt and suffix controls use fresh forwards, retaining actual token IDs.
A proposed fixed-suffix teacher-forced path requires separate implementation
and calibration; it is not assumed to exist already. Compare full readout
distributions as well as visible candidates. Separate source-cache and direct
probe forwards do not establish cache-to-output mediation. Intervention work
is optional and excluded unless the exact multimodal execution path passes
prefix, length, tensor, sham and token-layout checks.

## Confirmation Isolation And Open Registration

All existing r1-r4 images and all previously inspected maps are development
material for this new study, even when earlier notes called r3/r4 held out.
No confirmation cohort has been generated, sealed, inspected or released.
Null manifest and hash fields in the scope record explicitly preserve this.

Before confirmation, record fresh reference and test source seeds, fresh map
seeds, generator parameters, exclusions, file hashes and provenance. Verify
seed and content-hash overlap against the development inventory. Freeze all
model targets, prompts, baselines, sample counts, tests and stopping rules in
a commit before scoring. Track input-only eligibility inspection separately
from model outcomes; any allowed exclusions must be specified in advance and
all failures retained. A written policy is not an implemented access barrier.

Giving a new family its own reference seeds tests replication in that family,
not zero-shot transfer to an unseen family. A true held-out-family transfer
task needs its own target definition and is not required for the core paper.

The following remain mandatory unresolved registration items:

- Map-construction algorithm, shared moved-coincidence policy and retry limit.
- Mask/destination replicate counts, exact seeds and aggregate map weights.
- Independent confirmation cohort, reference/test sample sizes and hashes.
- Paired-effect null, interval method, resampling units and multiplicity family.
- Practical equivalence width, precision target and bounded compute budget.
- Model-specific calibration, fixed-suffix contract and release receipt.

Do not substitute current observed effects for a sample-size justification.
Use development-only precision simulations and a fixed budget; report the
minimum effect the resulting design can resolve. Confirmation scores may not
change the frozen design. Any amendment creates a new version before affected
outcomes and retains the previous specification and failure record.

## Gates And Immediate Next Step

| Gate | Required before proceeding | Current state |
| --- | --- | --- |
| Input measurement | Committed construction, independent oracle, seeds, counts and analysis specification | Pending |
| Cache measurement | All required input cells accepted, separate between-level audit and exact runtime calibration | Pending |
| Confirmation | Complete registration, untouched cohort inventory and release receipt | Pending |
| Causal intervention | Verified exact multimodal path and sham controls | Outside core scope; unqualified |

Stop the affected stage on failed hashes, invalid maps, unexpected metadata,
non-finite tensors, unresolved token layouts or a failed calibration. Preserve
partial outputs and reasons. No automatic retries with replacement seeds,
threshold relaxation or expansion to chase a favorable result. Negative or
imprecise findings can narrow the paper without preventing completion.

The next bounded step is Note 0053: implement and test coordinate-only map
construction and an independent oracle on synthetic index arrays, without
source-image scoring, processor calls or model forwards. This note records
the scope; it neither launches that work nor reports any new measurement.

## Record Validation

All 371 repository tests pass, including eight new scope-contract tests.
The candidate overlap counts are checked with exact rational arithmetic;
this checks set-intersection counts, not the unimplemented permutation
construction. All 133 local links and anchors in the updated documents
resolve. The 28 frozen measurement-code files from Note 0051 remain unchanged.
These checks validate the design record and preserve previous artifacts;
they do not qualify a new input panel or seal a confirmation cohort.
