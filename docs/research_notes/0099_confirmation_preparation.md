# Research Note 0099: Confirmation Cohort Preparation And Precision Sensitivity

2026-11-09. The operator asked to move to the untouched cohort. This
[preparation plan](../../examples/research_notes/0099_confirmation_preparation/plan.json)
allocates source metadata and uses existing Note 0098 outcomes for a
development-only precision sensitivity calculation. **No confirmation source,
coordinate map, processor output, model context or confirmation score has
been generated.** This is not a completed executable confirmation registration
or an acquisition release.

## Model Scope Decision

The proposed first stage is the already-qualified FastVLM visual path.
At that checkpoint, whether to confirm there first or retain the historical
FastVLM plus Qwen stage together was submitted to the operator. The hashed
preparation plan retains the initial pending state. The operator subsequently
chose to maintain both models; [Note 0100](0100_two_model_confirmation_scope.md)
records that resolution. The historical Note 0052 scope is not rewritten or
treated as fulfilled. The budgets and precision calculation below concern
FastVLM only and cannot justify Qwen sample size or calibration.

## Candidate Cohort And Question

The budget candidate retains all eight known ordered generator families,
with **two new reference pairs and eight new test pairs per family**:
16 reference pairs, 64 test pairs, 80 pairs / 160 source images in total.
Generator parameters are inherited unchanged except seeds, IDs and split.
Reference replicate IDs are r9/r10; tests are r11-r18. Source images would
be 240 x 320 x 3; candidate canvases remain 320 x 320 x 3.

The primary candidate endpoints are encoder and projector
reference_difference, mean over arrangements b=1/2/3 of
`m[p,b,b] - m[p,0,b]`. The question is whether matching the reference condition
changes mean correspondence in the preregistered direction on fresh sources
and a fresh map orbit. It is not a test of learning, semantic invariance,
information loss, all-family recovery or a residual causal processing gain.

Fixed-reference changes, matched-reference changes, full raw/native vectors,
all sixteen crossings, all families, twelve input gates and spectra remain
secondary records. A positive reference difference alone would not establish
absolute contraction toward zero for every family or a nonzero learned-site
residual. No family or sign pattern is an exclusion rule.

## A Fresh Map Requires Fresh Calibration

The development maps use the same four-tile cyclic order in every active
cell. Reusing those maps with new image seeds would not meet the planned
fresh-source-and-map boundary. The candidate instead assigns one of six
four-cycles independently by a deterministic SHA256 cell-key rule to each
of the 48 active cells, then uses the identity and the three powers of that
cycle field. Guarded support, tile sizes, gray canvas and moved-site count
6,912 remain unchanged.

This is **one common map orbit**, not three independent map samples. The
primary inference conditions on that orbit. All cells and sources share it.
New map bytes have not been constructed; there is no processor qualification
or observed matching gate for this rule. A new independent constructor,
coordinate oracle, sentinel and native synthetic processor calibration are
required before confirmation source generation. The old guard/phase argument
is a rationale, not empirical transfer of qualification.

The source and map recipe are newly specified, so this would confirm a
prediction on a fresh map construction rather than exactly repeat the old
global cyclic transformation. No map search, replacement or retry is allowed
after an unexpected gate failure or historical word-hash collision.

## Source Metadata And Novelty Boundary

The [allocated metadata](../../examples/research_notes/0099_confirmation_preparation/allocated_source_metadata.json)
uses the fixed SHA256 namespace rule in the plan. The
[seed audit](../../examples/research_notes/0099_confirmation_preparation/seed_audit.json)
compares 260 derived RNG initializers for the 160 proposed sources with the
available metadata of 128 historical sources. There are no initializer
collisions or within-candidate duplicates. This is not decoded-image novelty,
a complete inventory of every possible past source, or pretraining novelty.
Actual source-hash overlap must still be checked after authorized generation.

All reference sources and their complete four-cell/orbit observations must
be generated, audited and sealed before any test source is generated.
Allocating IDs now does not seal actual content or implement that access
barrier. The old runners/scorers with hard-coded sixteen-test assumptions
cannot simply be declared qualified for a sixty-four-test panel.

## Development Precision Check

The fixed budget candidate is eight test pairs per family, not a number
selected to make the observed positive effect significant. Existing
reference_difference values are centered within each family, leaving paired
two-site residual vectors from the two existing test pairs. Ten thousand
simulated samples for n=4/8/16 per family preserve those site pairings and
family strata. The report retains all n values and standard-deviation stress
multipliers 1/2/4; no automatic sample-size escalation is implemented.

The diagnostic half-width is normal critical value 2.2414 times
`sqrt(sum_family(sample_variance/n))/8`. It is a planning approximation,
**not a computed bootstrap confidence interval, power estimate or coverage
validation**. The policy target half-width 0.025 is not an equivalence margin.
An independent finite two-point variance formula checks every simulation.

| Test pairs per family | SD multiplier | Encoder median / p95 half-width | Projector median / p95 half-width |
| --- | ---: | ---: | ---: |
| 8 | 1 | 0.007467 / 0.007724 | 0.007840 / 0.008112 |
| 8 | 2 | 0.014935 / 0.015449 | 0.015679 / 0.016224 |
| 8 | 4 | 0.029870 / 0.030898 | 0.031359 / 0.032448 |

The fourfold-SD stress fails the 0.025 target for most simulations; it is
retained rather than used to expand n after looking for significance. The
[complete report](../../examples/research_notes/0099_confirmation_preparation/precision.json)
includes all eighteen n/site/stress rows and the centered residuals. Maximum
variance-oracle half-width discrepancy is below 1e-12. Two development pairs
per family cannot characterize tails, new map variance or reference variance.
No achieved confirmation precision is claimed.

Precision and bounded resources are distinct sample-size rationales; see
[Lakens, Sample Size Justification](https://doi.org/10.1525/collabra.33267).
The candidate future interval procedure uses shared whole-pair draws within
family strata, not independent draws of sites/cells/shifts; stratification
is also explicit in the [bootstrap documentation](https://stat.ethz.ch/R-manual/R-devel/library/boot/html/boot.html).
The final interval implementation and its independent checks remain open.

## Candidate Inference And Fixed Stopping

Proposed inference conditions on the eight named families, the new frozen
reference pairs and the one shared map orbit. Resample complete test pairs
within each family, keeping all views and arrangements together. The
candidate uses 9,999 shared bootstrap draws and two-sided 97.5% percentile
intervals for each of the two primary endpoints: nominal Bonferroni familywise
95%, not guaranteed finite-sample coverage. Positive-reference-effect support
would require both lower bounds above zero and all required pairs available.
No permutation exchangeability is assumed and no p-values are proposed.
Seed-population interpretation would additionally rely on the deterministic
hash seed allocation behaving as independent generator draws; it is not
literal random treatment assignment or a design-based coverage guarantee.

Any missing required primary margin makes its endpoint and interval
unavailable, without filtering, zero fill or replacement. All families,
secondary signs and three amplitude floors must be published. Freeze the
executable intervals, stopping, eligibility and acquisition implementations
before generating confirmation sources. Complete the fixed count once if
released; stop and retain the first unexpected failure. Neither significance,
interval width nor unfavorable family signs can trigger another batch.

## Resource And Release Gates

The candidate FastVLM budget is 1,280 experimental contexts plus 384
historical and 24 sham contexts: 1,688 model source contexts. Separate new-map
calibration allows at most 153 processor-only calls; acquisition allows 1,433,
for a combined upper budget 1,586. None is released by this preparation.

The conservative start gate is **44 GiB free**: 32 GiB payload allowance,
8 GiB reserve and 4 GiB overhead. At preparation, free space was 39.718 GiB,
about 4.282 GiB below that gate. This is an observed preparation snapshot,
not a fresh runtime preflight. No cleanup or deletion is performed here.

The [preparation summary](../../examples/research_notes/0099_confirmation_preparation/summary.json)
records `PREPARED_NOT_CALIBRATED_NOT_RELEASED`. Remaining gates are the model
scope decision, independent map implementation and native qualification,
actual-content collision inventory, executable cohort/interval software,
precision/scope review, committed complete registration, separate launch
approval and fresh disk/runtime checks. The cohort remains ungenerated and
unsealed; the narrow paper can continue drafting from existing evidence.

## Preparation Checks

All 41 focused preparation, portable-artifact, historical scope, example-link
and preceding crossing-publication tests pass, as does lint. Public metadata
and all eighteen precision rows reconstruct without confirmation images or
local native payloads. The [receipt](../../examples/research_notes/0099_confirmation_preparation/preparation_receipt.json)
retains source-plan, output and preparation-code hashes. At this initial
checkpoint independent review and public push had not occurred; complete
executable confirmation registration remains open. Subsequent preparation-only
review/integration is recorded in [Note 0102](0102_qwen_calibration_registration.md).
