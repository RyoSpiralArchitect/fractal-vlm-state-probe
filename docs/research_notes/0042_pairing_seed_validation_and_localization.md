# Research Note 0042: Held-Out Pairing Direction And Localization

Date: 2026-09-30

Status: completed for the four fixed Qwen/Ministral targets, with two frozen
reference seeds and two new test seeds in each of eight ordered pairings.

## The Refined Question

[Note 0040](0040_generator_pairing_direction_hierarchy.md) separated repeated
seeds inside a generator pairing from transfer across different pairings.
Its two seeds per pairing established a correspondence inside an observed
panel, not prediction of an unseen seed. The next experiment freezes that
panel as a reference and asks whether it predicts two additional seeds.

Four questions are deliberately separated:

| Question | Measurement | What it does not establish |
| --- | --- | --- |
| New-seed prediction | Frozen-reference own-pairing cosine margin | Transfer to an unseen pairing |
| Localization | Prediction and energy for every fixed KV head and token band | A verified two-dimensional patch map |
| Architecture extension | Same hierarchy at three fixed FastVLM targets | A shared cross-model coordinate basis |
| Frequency explanation | Accepted processor-matched blocks versus original blocks | Semantic specificity or cache-to-readout causation |

The first two use the present inputs and implementation. The latter two remain
separate future stages; their design is in the
[registered protocol](../pairing_validation_protocol.md).

## Frozen Split

There are eight ordered generator pairings, four geometry and four stochastic.
Replicates 1 and 2 are the unchanged Note 0040 references; replicates 3 and 4
are new test seeds. No test vector can update a reference direction.

- Reference panel: 16 source-pair factorials and 32 unique source images.
- Test panel: 16 source-pair factorials and 32 new unique source images.
- All 64 source first-frame hashes are distinct across the combined panels.
- Source dimensions and generator parameters remain those of Note 0040.
- Test-panel raw marginal spatial/interaction leakage is at most `2.22e-16`.
  This checks palette-preserved summaries, not frequency matching.
- Qwen retains layer 33 `values`.
- Ministral retains layers 1 and 16 `keys`, and layer 25 `values`.
- Every cell is a fresh single-image source-context forward; no cache is
  reused for a readout or suffix continuation.

For each pairing f, the frozen direction is:

`reference_f = unit(unit(interaction_f,r1) + unit(interaction_f,r2))`.

Normalizing the two references separately prevents the seed with a larger
interaction norm from setting the direction by amplitude alone.

Each test seed is scored against every reference. The primary margin is its
own-family cosine minus the mean cosine with the three other families in its
same broad class. Average the two seed margins within each family, then
average the eight family margins with equal weight. Negative margins and tied
retrievals remain visible; zero vectors and cancelling references are
explicitly unavailable rather than imputed.

## Exact Test And Multiplicity

The conditional reference permutes complete two-test-seed family blocks onto
the four reference families within each broad class: `4! x 4! = 576` exact
assignments. This tests reference/test correspondence while retaining the
dependence between the two test seeds of each family.

Four fixed targets times image/post-image regions yield eight primary tests.
Their one-sided exact p-values receive one joint Holm correction. Reporting
also includes all eight family margins, individual seed/reference cosines,
and descriptive retrieval over all eight references.

This is inference conditional on the observed pairings. Neither 16 held-out
seeds nor the repeated target/region views become independent samples of a
generator-family population.

## Localization Without Selecting A Winner

All KV heads are measured separately in both regions. Four contiguous
image-token order bands are measured with all heads retained. Qwen contributes
8 exploratory views and Ministral contributes 20 per target: 68 exploratory
views in total, alongside the 8 primary views.

For each view, report both:

1. its share of the parent region's interaction energy, and
2. its new-seed direction margin.

High energy and reliable direction prediction are different properties.
All head/band p-values are exploratory and unadjusted. A strongest head is
not promoted to a confirmatory result from these held-out scores. Token order
is not assumed to map onto two-dimensional image patches.

## Historical Reference Gate

The July reference runs saved runtime versions and tensor hashes, but not
model weight revisions. Before any new test forwards, each model repeats all
four cells of one complete reference factorial at all its fixed targets.
Every float32 tensor must match the historical tensor elementwise exactly.
A mismatch stops pooling until audited; it does not become a negative
new-seed result or trigger reference replacement.

These calibration cells are counted separately from the new experimental
cells. Current model snapshot revisions and hashes of weights, processor,
tokenizer and configuration files are recorded. Calibration verifies this
reference computation, not the missing provenance of every historical run.

## Results

All 16 calibration tensors match their historical float32 bytes exactly. The
eight calibration cells are separate from the new experimental denominator.
All 128 new source responses are `ACK`; actual suffix token IDs remain
`[4032, 151645]` in Qwen and `[13832, 2]` in Ministral, including the ending
token. The new panel contributes 128 source cells, 256 tensor sidecars and
64 factorial analyses, with no new direct full-vocabulary probes.

| Target / region | Own cosine | Other same-class cosine | Margin | Positive families | Retrieval | Exact p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen L33 values / image | 0.17708 | 0.03405 | 0.14303 | 8/8 | 15/16 | 1/576 | 0.013889 |
| Qwen L33 values / post-image | 0.72693 | 0.21995 | 0.50698 | 8/8 | 15/16 | 1/576 | 0.013889 |
| Ministral L1 keys / image | 0.23300 | 0.02565 | 0.20735 | 7/8 | 13/16 | 2/576 | 0.013889 |
| Ministral L1 keys / post-image | 0.57546 | 0.10550 | 0.46996 | 8/8 | 14/16 | 1/576 | 0.013889 |
| Ministral L16 keys / image | 0.15047 | 0.02624 | 0.12423 | 6/8 | 11/16 | 3/576 | 0.013889 |
| Ministral L16 keys / post-image | 0.63649 | 0.12605 | 0.51043 | 8/8 | 16/16 | 1/576 | 0.013889 |
| Ministral L25 values / image | 0.11696 | 0.01735 | 0.09961 | 8/8 | 12/16 | 1/576 | 0.013889 |
| Ministral L25 values / post-image | 0.62435 | 0.14688 | 0.47747 | 8/8 | 16/16 | 1/576 | 0.013889 |

All eight primary correspondence tests pass their joint Holm reference.
This strengthens the result from observed seed matching to prediction of
previously unmeasured seeds within the same eight generator pairings.

Prediction is not uniform across every image-region family. Three negative
family-target margins are retained:

- Ministral L1, blue noise / random dots: `-0.0000958`.
- Ministral L16, blue noise / random dots: `-0.026606`.
- Ministral L16, white noise / blue noise: `-0.002949`.

Post-image family margins are positive in 32/32 target-family summaries.
Image-region margins are positive in 29/32. These are repeated summaries over
the same eight families, not 32 independent replications.

## Localization Result

All 68 fixed exploratory views have a positive average held-out margin.
Their p-values remain unadjusted; positivity is not a selected-head claim.

| Target | Image-head margin range | Post-image-head margin range | Image-band margin range |
| --- | ---: | ---: | ---: |
| Qwen L33 values | 0.1425-0.1436 | 0.5062-0.5077 | 0.0971-0.2056 |
| Ministral L1 keys | 0.1979-0.2201 | 0.4135-0.5067 | 0.1752-0.2621 |
| Ministral L16 keys | 0.1175-0.1268 | 0.4893-0.5242 | 0.1011-0.1492 |
| Ministral L25 values | 0.0932-0.1131 | 0.4572-0.4977 | 0.0840-0.1121 |

Both Qwen KV heads retain positive post-image margins for all eight families.
The same holds for all 24 Ministral head-target views. Among individual test
seeds, no Ministral head contributes more than `0.2203` of its region's
interaction energy; Qwen's two-head maximum is `0.5394`. The measured
prediction is distributed across the fixed storage-head views, not explained
by one selected high-energy head. KV storage heads are not independent causal
attention-head contributions.

The 64 new factorials have exactly zero pre-image effects and image-position
interaction argmaxes in 64/64. Image interaction energy exceeds 0.9 in 63/64.
The exception is checkerboard/hex replicate 3 at Ministral L25 values:
`0.899444`, retained below the threshold without changing it.

Image-region balanced dominance is spatial/palette/interaction in `54/10/0`;
whole-effective dominance is `56/8/0`. Interaction share is at or below the
equal-coefficient `1/3` reference in 59/64 under either region convention.
Reliable interaction direction therefore does not imply interaction-axis
energy dominance.

Both models have 241 effective positions and 99 image positions. Qwen has
43 pre-image and 99 post-image positions. Ministral has 38 pre-image,
96 post-image and eight non-image gaps within the image span, at positions
`49, 61, 73, 85, 97, 109, 121, 133`. Consequently, image/post-image views do
not exhaust Ministral's non-image partition. Token bands are order bands,
not verified image quadrants.

## Revised Reading

The supported statement is now:

> Frozen references from two visual seeds predict interaction direction for
> two new seeds of the same observed generator pairing at four fixed
> Qwen/Ministral tensor targets. All eight image/post-image correspondence
> tests pass joint Holm correction. Prediction is stronger in post-image
> direction than in image direction and remains spread across the fixed
> head and token-band views, while image-family exceptions remain visible.

This is pairing-conditioned prediction of a visual interaction, not prediction
of an unseen pairing, semantic class direction or persistent state. The next
stages are the same frozen split in FastVLM, followed by processor-frequency
controls accepted before cache inspection. Frequency matching must cover all
four factorial cells, not only the two original images.

## Claim Boundaries

- There are eight observed pairings and two test seeds per pairing.
- References and four tensor targets are selected from prior work, not a
  blind sample of all layers or pairings.
- Generator parameters, frequency structure and learned visual structure
  remain entangled; the present panel is not frequency-controlled.
- Full-vector raw/processor interaction prototypes have not been scored with
  the same held-out split. The result does not show that pairing alignment
  first emerges inside the model rather than in the transformed input.
- Post-image includes the source prompt and actual generated suffix.
  Agreement there does not identify a pure visual representation.
- Source-only runs add no independent direct-probe factorials.
- No result establishes persistence, adaptation, semantic steering, causal
  cache mediation or a cross-model vector basis.

## Primary Artifacts

- `configs/generator_pairing_seed_holdout_v1.json`
- `docs/pairing_validation_protocol.md`
- `scripts/run_pairing_seed_validation.py`
- `scripts/summarize_pairing_seed_validation.py`
- `runs/pairing_seed_validation_v1/panel/`
- `runs/pairing_seed_validation_v1/qwen/`
- `runs/pairing_seed_validation_v1/ministral3/`
- `runs/pairing_seed_validation_v1/study_summary.json`
- `runs/pairing_seed_validation_v1/study_summary.md`
- `examples/research_notes/0042_pairing_seed_validation/summary.json`

Execution and checked-resume instructions are in the protocol. A fresh
checkout must reconstruct full reference tensors; tracked historical summary
JSONs alone are not sufficient to run this test.
