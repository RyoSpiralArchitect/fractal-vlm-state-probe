# Pairing Direction Validation Protocol

Registered: 2026-09-30, before the new seed forwards.

## Questions And Sequence

| Stage | Question | Unit and measurement | Completion criterion |
| --- | --- | --- | --- |
| 1: held-out seeds | Does a pairing direction predict new seeds? | Eight ordered pairing families, two frozen reference seeds and two new test seeds each; image and post-image interaction direction | Complete all four fixed Qwen/Ministral targets and report every family, including failures |
| 2: localization | Is prediction concentrated in particular heads or token bands? | Same split; each KV head and four equally sized image-token order bands | Report all fixed views; select no view using held-out scores |
| 3: FastVLM transfer | Does the hierarchy repeat in the ninth architecture? | Same 32 pairing units; fixed layers 1 keys, 12 keys, 23 values | Verify source response, image layout and shape first; complete the same reference/test split |
| 4: frequency controls | Can measured frequency structure explain pairing transfer? | Matched source-pair blocks after each model processor | Input acceptance gates must pass before model-cache claims |

## Frozen Stage 1 Specification

- Reference seeds: replicates 1 and 2 in Note 0040. No reference seed is
  selected or replaced using the new results.
- Test seeds: replicates 3 and 4 in
  `configs/generator_pairing_seed_holdout_v1.json`.
- Source geometry, dimensions, generator parameters, prompt, stream seed,
  generation length and runtime versions remain those of the reference runs.
- Qwen: layer 33 values. Ministral: layer 1 keys, layer 16 keys and layer 25
  values. These are the four existing targets, not an exhaustive layer search.
- Each cell is a fresh source-context forward, temperature zero, one image,
  stream seed 20260604, at most two generated tokens. Record the actual suffix.
- Record all source and hybrid hashes. Repeated images, incompatible runtime,
  different tensor shapes or token partitions invalidate pooling until audited.
- No new direct readouts are required for this stage. Source-cache prediction
  and the earlier full-vocabulary result remain separately counted surfaces.

For family f, form the reference direction from its two unit-length interaction
vectors, then normalize their sum:

`reference_f = normalize(unit(interaction_f,r1) + unit(interaction_f,r2))`.

For each test vector, retain its cosine with every frozen family reference.
The primary score is its own-family cosine minus the average cosine with the
three other references in the same broad class. Average the two test-seed
margins inside each family, then give each of the eight families equal weight.

The conditional exact test permutes complete two-test-seed family blocks onto
the four reference families within each broad class. There are
`4! x 4! = 576` assignments. This tests reference/test correspondence while
preserving broad class and within-family dependence. It does not treat the 16
held-out seeds or their pairwise distances as independent population samples.

The eight primary target-region p-values (four targets x image/post-image)
receive Holm correction together. Report raw p, corrected p, the eight family
margins, all individual seed scores, and retrieval accuracy over all eight
references. Retrieval is descriptive; tied best scores are unresolved.
Zero vectors or a cancelling reference make the corresponding view unavailable.
Negative family margins remain visible.

## Fixed Localization Views

Report each KV head separately in image and post-image regions, and four
contiguous image-token order bands with all heads retained. Energy fractions
describe where the interaction norm is concentrated; held-out margins describe
where direction predicts new seeds. These are different measurements.

Head and band results are exploratory repeated views of the same eight
families. Their p-values are reported as unadjusted, with no strongest-view
claim. Image-token order is not a verified two-dimensional patch map. A patch
claim needs model-specific evidence for feature ordering, crop merging,
padding and special-token handling.

KV storage heads are not independent attention-head contributions. Image and
post-image views need not exhaust non-image positions: non-image gaps between
image-token runs are separately listed, not silently merged into post-image.
Generated suffix identity is checked at the token-ID level as well as in text.

## Interpretation Rules

- A positive average margin with negative family margins supports an average
  correspondence, not uniform prediction across every observed pairing.
- Failure to reject the exact reference does not establish no effect;
  magnitude, family margins and the two individual test scores remain visible.
- Stronger post-image than image prediction does not identify a pure visual
  representation. Post-image includes shared prompt and generated suffix.
- A high-energy head need not be a high-prediction head. Do not interchange
  energy localization and directional stability.
- Passing these tests supports new-seed prediction within existing pairings.
  Prediction of an unseen pairing requires a separate held-out-family design.
- A frequency-matched result is eligible only after all four input cells pass
  the pre-cache acceptance gates. Matching two spectral summaries does not
  identify or eliminate every frequency confound.

## Frequency-Control Acceptance Gates

Choose candidate transforms on input and processor statistics alone. Freeze
accepted transforms before inspecting their cache scores. Use source-pair
blocks with original and matched conditions, not independent cell filtering.

- Match processor-space luminance spectral centroid within 5% of its common
  target and high-frequency energy ratio within 0.02 absolute difference.
- Audit these metrics for all four transformed factorial cells. A palette
  transfer can reintroduce frequency differences after source matching.
- Preserve each palette donor's RGB multiset, or explicitly register a changed
  marginal condition. Report raw and processor mean, standard deviation,
  entropy and colorfulness beside frequency summaries.
- Retain a transform-only sham and matched source-pair provenance.
- If a block fails any acceptance gate, report it as unmatched; do not count
  its cache result as evidence from a frequency-controlled design.
- With accepted blocks, compare held-out direction margins and family transfer
  before/after matching at the same frozen tensor targets. Choose additional
  pairings only through input criteria, then test them as held-out families.

The thresholds above are operational acceptance tolerances, not universal
equivalence bounds or a proof that all spectral information is matched.

## Outputs

The live stage-1 root is `runs/pairing_seed_validation_v1/`. It contains panel
manifests, per-cell source runs and tensor hashes, per-pair factorials, frozen
reference/test score matrices, exact block tests and exploratory localization.
Tracked research notes and compact example JSONs retain results and boundaries.

## Execution And Resume

The following commands use the existing Note 0040 local reference artifacts.
A fresh checkout must reconstruct those references with the existing panel,
source-cache capture and tensor-factorial tools first. The tracked example
summaries alone do not contain full reference tensors.

```bash
python3 scripts/prepare_generator_pairing_panel.py \
  --config configs/generator_pairing_seed_holdout_v1.json \
  --output-root runs/pairing_seed_validation_v1/panel

python3 scripts/run_pairing_seed_validation.py \
  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --reference-replication 33:values=runs/generator_pairing_transfer_v1/replication/qwen_l033_values.json \
  --output-root runs/pairing_seed_validation_v1/qwen

python3 scripts/run_pairing_seed_validation.py \
  --model mlx-community/Ministral-3-3B-Instruct-2512-4bit \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --reference-replication 1:keys=runs/generator_pairing_transfer_v1/replication/ministral3_l001_keys.json \
  --reference-replication 16:keys=runs/generator_pairing_transfer_v1/replication/ministral3_l016_keys.json \
  --reference-replication 25:values=runs/generator_pairing_transfer_v1/replication/ministral3_l025_values.json \
  --output-root runs/pairing_seed_validation_v1/ministral3

python3 scripts/summarize_pairing_seed_validation.py \
  --execution runs/pairing_seed_validation_v1/qwen/validation_summary.json \
  --execution runs/pairing_seed_validation_v1/ministral3/validation_summary.json \
  --output-json runs/pairing_seed_validation_v1/study_summary.json \
  --output-md runs/pairing_seed_validation_v1/study_summary.md
```

An interrupted run resumes existing cells only after verifying their model,
runtime, prompt, image hash, suffix and tensor hashes. The input specification
must remain identical. `--analysis-only` requires all source cells and reruns
the offline factorial, held-out and localization analyses.

Before new test forwards, each model repeats one complete reference factorial
at all fixed targets and requires elementwise equality with the historical
float32 tensors. These four calibration cells are separately counted. The
historical runs did not save model weight revision hashes, so this is a
measured reference reproducibility check; it does not recover missing weight
provenance.

The current snapshot revision and SHA-256 hashes of model weights, processor,
tokenizer and configuration files are recorded in `model_snapshot.json`.
Reusing an output root with a changed snapshot fails before new forwards.

## Stage 3 Execution Addendum

Registered: 2026-10-01, before any FastVLM pairing reference/test forward.

FastVLM has the four fractal-pair artifacts from Note 0041, but no measured
generator-pairing references. Capture the unchanged r1/r2 inputs first and
freeze those references before evaluating the unchanged r3/r4 inputs. Both
panel hashes and the reference/test roles are locked before reference capture.
The fixed targets remain L1 keys, L12 keys and L23 values; no target is chosen
from this panel. The three targets times image/post-image give six primary
tests with a separate, stage-specific Holm family of six. Do not retroactively
change Note 0042's family of eight. There are 24 fixed exploratory head/band
views, with unadjusted p-values.

Before reference capture, replay all four historical `b_c` cells and require
bytewise equality at all three targets. These 4 calibration cells / 12 tensors
are separate from 64 new reference cells / 192 tensors / 48 factorials.
The seed-validation runner then rechecks one new reference factorial (another
4 calibration cells / 12 tensors), before 64 new test cells / 192 tensors /
48 factorials. Only the r3/r4 cells enter the held-out test score.

Require the historical source prompt, runtime, actual response `The image`,
recorded generation-step token IDs, `[1, 2, 397, 64]` tensor shape and the
validated 42 pre-image / 256 image / 99 post-image position partition.
Historical generation-step traces include a repeated final step; compare the
stored trace as-is, without calling it three distinct generated tokens.
A changed response, unresolved layout, altered shape or failed calibration
stops pooling; do not relax the gate based on observed held-out performance.
The architecture comparison remains descriptive: Qwen/Ministral's `ACK`
suffix, head dimensions and coordinates differ from FastVLM.
Require the same model snapshot fingerprint for reference and test execution,
including on an analysis-only resume.

```bash
python3 scripts/capture_pairing_references.py \
  --model mlx-community/FastVLM-0.5B-bf16 \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --historical-factorial 1:keys=runs/fastvlm_expansion/analyses/b_c/layer_001_keys/cache_tensor_factorial.json \
  --historical-factorial 12:keys=runs/fastvlm_expansion/analyses/b_c/layer_012_keys/cache_tensor_factorial.json \
  --historical-factorial 23:values=runs/fastvlm_expansion/analyses/b_c/layer_023_values/cache_tensor_factorial.json \
  --output-root runs/fastvlm_pairing_seed_validation_v1/references_model_path

python3 scripts/run_pairing_seed_validation.py \
  --model mlx-community/FastVLM-0.5B-bf16 \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --reference-replication 1:keys=runs/fastvlm_pairing_seed_validation_v1/references_model_path/replication/layer_001_keys.json \
  --reference-replication 12:keys=runs/fastvlm_pairing_seed_validation_v1/references_model_path/replication/layer_012_keys.json \
  --reference-replication 23:values=runs/fastvlm_pairing_seed_validation_v1/references_model_path/replication/layer_023_values.json \
  --reference-model-snapshot runs/fastvlm_pairing_seed_validation_v1/references_model_path/model_snapshot.json \
  --output-root runs/fastvlm_pairing_seed_validation_v1/tests

python3 scripts/summarize_pairing_seed_validation.py \
  --execution runs/fastvlm_pairing_seed_validation_v1/tests/validation_summary.json \
  --expected-primary-tests 6 \
  --output-json runs/fastvlm_pairing_seed_validation_v1/study_summary.json \
  --output-md runs/fastvlm_pairing_seed_validation_v1/study_summary.md
```

Stage 4 remains input-only preparation until all four frequency-matched cells
pass their registered processor gates. Stage 3 completion is not evidence that
those frequency gates passed.

An auxiliary input-space diagnostic was fixed before FastVLM cache
measurement: apply the same r1/r2 -> r3/r4 split to full raw RGB interaction
vectors and processor-pixel interaction vectors. These diagnostics are
exploratory, with unadjusted p-values; they do not extend the six-test primary
family or select cache targets. Record the processor configuration and
implementation hashes and verify the actual model-loading path. This is not
accepted frequency matching or a shared-coordinate model comparison.

Audit correction, before any held-out forward: the initial native MLX
processor substitution failed all 12 historical tensor comparisons despite
matching shape, suffix and pre-image tensors. Keep that failed four-cell
attempt under `references/`; exclude it from reference/test pooling. Direct
image-processor calls with explicit tensor backends exposed an API error, but
the original full model processor uses a different call path and works without
substitution. Removing the substitution reproduced all 12 historical tensors
bytewise under `references_model_path/`. The exact gate was not relaxed, and
no held-out score selected this implementation. The six primary tests, targets,
input seeds and direction formulas are unchanged.

The native pixel baseline remains an ineligible alternative, not the actual
model-input baseline. Recompute the pixel diagnostic through the unchanged
full model processor with a pixel-only `<image>` text placeholder. Record and
match its image-processor implementation hash to actual source captures. This
correction precedes held-out cache forwards; all input diagnostics remain
exploratory. The extra failed calibration contributes four cells / 12 tensors
to the audit denominator, not the selected full-vector surface.

```bash
python3 scripts/analyze_pairing_input_holdout.py \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --fastvlm-model-processor-snapshot /path/to/the/frozen/FastVLM/snapshot \
  --output-json runs/fastvlm_pairing_seed_validation_v1/input_model_path_holdout.json
```

## Stage 4 Input Execution Addendum

Registered: 2026-10-07, before the new input sweep or any stage-4 cache forward.

Start with the same 32 FastVLM pairing units and qualified actual model
processor. This is not an additional independent image cohort. The processor
implementation, configuration and revision must match Note 0043's eligible
input baseline. No model weights are loaded for input preparation.
The locally saved custom processor is explicitly trusted only after checking
its source hash against the qualified implementation; there is no substitution.

`configs/frequency_control_fastvlm_v1.json` fixes two distinct arms:

- `rank_low_pass`: Fourier low-pass the original spatial donor's floating-point
  luminance field, then assign each unchanged palette's full RGB pixels in
  stable luminance order. The cutoff grid is 0.02, 0.04, 0.08, 0.12, 0.2, 0.35,
  0.6 and 1.0 (identity). Source A and B may have different cutoffs; the same
  field/order is shared by both palette cells of each spatial donor. There are
  64 candidate cutoff pairs per source-pair block. Do not independently filter
  the four cells or alter palette marginals.
- `shared_pixel_permutation`: one bijection of all 76,800 positions, generated
  by NumPy `default_rng(20261007)`, is shared by all four cells and all 32
  pairings. This deliberately destroys geometry; it is a whitening comparator,
  not a structure-preserving frequency intervention. A common permutation
  commutes with the raw interaction and preserves raw full-vector cosine
  correspondence mathematically. It must not be sold as isolating frequency.

Each arm preserves joint RGB multisets exactly, not just separate channel
histograms. An inverse-permutation roundtrip is serialized as a transform-only
sham for every original cell and checked for identical decoded pixels.

Processor-space frequency uses reconstructed RGB (undo recorded channel
normalization) and luminance weights 0.2126 / 0.7152 / 0.0722. It uses the
existing radial `rfft2` convention and HF cutoff at normalized radius 0.35.
This is not the arithmetic-channel-mean diagnostic used by the older generic
processor statistics. Retain raw and processor mean/std, 256-bin luminance
entropy, colorfulness, shapes and tensor mean/std alongside the gate metrics.
Finite nonzero luminance variance/spectral centroid are required; constant
images cannot qualify as a trivial match.

The operational common centroid target is the arithmetic mean of the four
candidate cell centroids, fixed by this formula before the sweep. All four
must be within 5% of that target. Require the largest pairwise HF difference
to be at most 0.02. Record the common HF mean and every individual cell value.
These are within-block targets, not a claim that matching retains original
frequency values or equates spectra across different families.

Prefer an accepted cutoff pair with the largest cutoff sum, then largest A
and B cutoff, to avoid extra filtering. If none qualifies, retain the candidate
with the smallest maximum normalized gate violation, breaking ties by the same
cutoff order, and mark the block unmatched. Retain all 64 gate evaluations;
no cache score or input correspondence score selects a candidate.

Freeze both reference and test transformed manifests and the hashed input
acceptance receipt before any cache measurement. A complete arm needs 32/32
accepted blocks to reuse the registered eight-family reference/test design.
Partial acceptance is an input-feasibility result, not a pooled held-out cache
result. Both pairing capture entry points reject an unmatched or changed
frequency receipt/image. Original and matched reference/test roles stay r1/r2
and r3/r4; targets stay L1 keys, L12 keys and L23 values. A complete eligible
arm has its own six-test Holm family; do not modify the earlier families.
If both arms qualify, jointly correct their 12 primary tests rather than
reporting two selected families. All head/band views remain exploratory.

Recheck the four-cell historical factorial bytewise before matched references
and one complete new reference before tests, with separate denominators.
Require the same source suffix and layout as Note 0043. Stop on any changed
source contract rather than widening it after looking at test scores.
Failure to qualify the structured arm does not become a successful structured
experiment merely because the geometry-destroying comparator qualifies.

```bash
python3 scripts/prepare_frequency_control_panel.py \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --config configs/frequency_control_fastvlm_v1.json \
  --fastvlm-model-processor-snapshot /path/to/the/frozen/FastVLM/snapshot \
  --qualified-processor-provenance runs/fastvlm_pairing_seed_validation_v1/input_model_path_holdout.json \
  --output-root runs/frequency_control_fastvlm_v1/input
```

## Stage 4B: Padding Policy Comparison

Registered: 2026-10-07, before measuring any of the new padding policies.

This bounded input-only experiment crosses three fixed content states with
three padding policies on all 32 Note 0044 source-pair blocks. Content states
are original, the already selected low-pass cells (including failed blocks),
and the already fixed common pixel permutation. The prior low-pass cutoffs
remain frozen under their original black-padding selection; there is no new
cutoff, fill-color or family search using these outcomes.

Center each 320 x 240 image on a 320 x 320 canvas, retaining every content pixel
at the same square-canvas location used by the actual processor. Fill the
25,600 additional pixels using one of three policies:

- Black: RGB `(0, 0, 0)`, an externally materialized pixel sham.
- Fixed gray: RGB `(128, 128, 128)`, shared across every cell and family.
- Palette mean: the corresponding original palette's mean RGB, rounded to
  uint8 with NumPy round-to-nearest, ties-to-even. The same palette gives the
  same fill in both spatial cells and all three content states of a block.

Every whole-canvas marginal is explicitly
`0.75 * original_palette + 0.25 * point_mass(fill)`.
Verify unchanged content, exact constant fill, joint RGB multisets against
the registered expanded palette, and analytical RGB mean/variance. Report
the exact total-variation distance from the original RGB distribution.
Do not call the expanded canvas's original RGB multiset preserved.

Keep the actual FastVLM processor, normalization, full-image luminance metric,
5% centroid tolerance and 0.02 HF tolerance. Preserve all nine condition
results per block. A condition is input-eligible for the complete known-pairing
design only if all 32 blocks pass and all eight families retain all four seeds.
The processor comparison uses whole output images, without support cropping.

Reproduce the prior rectangular-image processor statistics and require exact
pixel-tensor equality with the explicit black-pad sham for all 384 content
cells. Record non-pixel processor outputs too: external square images can
change `image_sizes` despite equal pixel tensors. Such a sham does not establish
model-cache equivalence. This stage loads no model weights and measures no new
cache vectors. Any later cache experiment needs its own black-pad calibration,
source-response/layout checks, frozen targets and test family.

The crossed comparison identifies padding-policy effects on the measured
input statistics for these fixed images. Padding color and the added RGB mass
change together, so this does not identify a padding effect independent of
marginals, a spectral-only intervention, or a semantic model effect.

```bash
python3 scripts/run_padding_policy_study.py \
  --source-receipt runs/frequency_control_fastvlm_v1/input/input_acceptance.json \
  --published-source examples/research_notes/0044_all_cell_frequency_control/summary.json \
  --config configs/padding_policy_fastvlm_v1.json \
  --fastvlm-model-processor-snapshot /path/to/the/frozen/FastVLM/snapshot \
  --output-root runs/padding_policy_fastvlm_v1
```

Execution report (2026-10-07): [Note 0045](research_notes/0045_padding_policy_and_explicit_marginals.md)
retains all nine conditions, explicit marginal changes, the complete accepted
permutation/mean input panel, and the unresolved model-cache calibration gate.

## Stage 4C: Padding Cache Calibration And Matched Panel

Registered: 2026-10-07, before the new cache forwards.

Use the unchanged qualified FastVLM snapshot and model path from Stage 3.
Freeze the Note 0045 live and published receipt hashes, all selected image
manifests, existing historical source runs/tensors, implementation hashes and
`configs/padding_cache_fastvlm_v1.json` before measuring caches. The targets
remain zero-based L1 keys, L12 keys and L23 values. Record actual prepared
inputs inside the real generation path, including pixel hashes and all
non-pixel fields. Instrumentation must itself reproduce historical tensors.

Calibration precedes the new panel:

1. Replay all 128 original cells in the existing 32-block panel and compare
   each target with its qualified historical tensor, byte for byte: 384 checks.
2. Run the explicit original-content black-square versions of the same 128
   cells. Require identical actual prepared inputs except `image_sizes`, whose
   values change from `[[320, 240]]` to `[[320, 320]]`. Compare the three full
   tensors and their fixed token regions against the fresh rectangular runs.
3. In the fixed `geometry_checker_hex_r1` four-cell anchor, cross size metadata
   only: rectangular pixels with square size metadata and black-square pixels
   with rectangular size metadata. These eight additional forwards must
   reproduce the corresponding naturally prepared inputs exactly. Record the
   full cache comparisons without substituting historical tensors.

The 264 calibration forwards and 792 captured tensors are counted separately
from the experimental panel. All runs retain the fixed source prompt, stream
seed 20260604, temperature zero, two-token generation budget, actual response
`The image`, generation trace `[785, 2168, 2168]`, tensor shape
`[1, 2, 397, 64]` and identified 42/256/99 pre/image/post positions. The repeated
last trace step is not a third generated token. A shape, source suffix,
prepared-pixel or exact calibration mismatch stops panel capture; preserve the
failure and do not widen numerical tolerances or force metadata in the panel.
Equality supports this measured single-image path only, not all architectures.

After calibration, capture two fixed square-input conditions for every block:
common permutation with black padding, and common permutation with
palette-mean padding. The mean condition is the 32/32 accepted input panel;
black is an explicitly unmatched comparator (1/32 input blocks accepted).
The same source pixels are permuted in both and both retain native square
metadata. Record the changed full-image RGB marginal from Note 0045; no claim
of a frequency-only intervention is permitted.

Capture r1/r2 references for both conditions first and freeze their tensor
hashes before r3/r4 test captures. This gives 256 panel source cells, 768 target
tensors and 192 four-cell factorial analyses, including 96 reference and 96
test analyses. No input, condition, target or view is selected using cache
outcomes. Repeat the existing interaction-vector definition, two-unit-vector
reference direction, equal-family score, 576 within-class complete-family
assignments, and fixed image/post-image primary views. Correct all twelve
primary tests together (two conditions x three targets x two regions) with
Holm. Unavailable views retain a p-value of one for correction and remain
unavailable in reports. Report all negative family margins and retrieval ties.

All 48 fixed head/band views are exploratory and unadjusted. Cross-condition
margin changes and vector cosines are descriptive paired measurements, not
new independent tests or causal frequency mediation. Test seeds are held out
from reference direction construction, but their images were used to select
the input-matching condition in Note 0045; this is an input-selected diagnostic
on an existing cohort, not fully untouched held-out validation. No new direct
probe, persistence, unseen-family transfer or semantic steering is measured.

Execution report (2026-10-07): [Note 0046](research_notes/0046_padding_cache_calibration_and_matched_panel.md)
records exact calibration, both complete panels, all twelve primary tests and
the negative multiplicity-adjusted result. The original registration above
is unchanged by those outcomes.

## Stage 4D: Graded Permutation Under Fixed Marginals

Registered: 2026-10-08, before the new input measurements.

Following Stage 4C, hold the palette-mean square padding and each cell's
expanded joint RGB multiset fixed. Reuse the exact 32 four-cell source blocks,
qualified processor, and historical common permutation from Stages 4A-4C.
This is an input-only audit, with no model weights, cache forwards, new cache
tests or additional independent image samples.

Use selected-site fractions 0, 1/8, 1/4, 1/2, 3/4, 7/8 and 1, fixed in
`configs/graded_permutation_fastvlm_v1.json`. Draw one panel-wide selection
order with NumPy default_rng seed 20261008. For each level, select the first
floor(fraction * 76800) positions. Restrict each cycle of the historical
permutation to its selected positions, linking each to the next selected
position in the original cycle and fixing every unselected position. Apply
the resulting common bijection to all four cells and all 32 blocks.

The zero endpoint is exactly identity; the full endpoint is exactly the
historical permutation. Selected-site sets are nested, but destinations of
already selected sites may change between levels. A one-selected-site cycle
remains fixed. This is not alpha blending, a graded local block shuffle, or
a guaranteed monotone spectral or semantic intervention. Record the actual
moved-index fraction, changed-RGB-site fraction, spatial displacement and
retained undirected nearest-neighbor grid edges separately.

Require exact expanded RGB multisets, inverse pixel roundtrips, unchanged
exterior fill, native square size metadata and all non-pixel processor fields
at every level. Reproduce both historical mean-padding endpoints' raw pixels,
processor pixel hashes, statistics and gates for every source cell (256
endpoint checks). A structural, provenance or endpoint mismatch stops the
study and is retained as a failure; numerical tolerances are not widened.
Freeze all receipt, map, code and processor hashes before measurement.

Audit all 896 input cells and all 224 four-cell gates using the unchanged
whole-processor 5% centroid and 0.02 HF-spread tolerances, including the frozen
moment check. Report every failed block and complete-family denominator.
Only 32/32 passing blocks with all eight complete four-seed families make a
level input-eligible. Report the full curve; do not adapt the grid, search
additional seeds or use cache outcomes to choose levels. These gates compare
the four cells within a level, not frequency equivalence between levels.

Expanded marginals are fixed across levels, not equal across different
palette donors and not equal to unpadded originals. All four input seeds are
audited, so a later cache test remains input-selected, not fully untouched
holdout. Input acceptance, spatial disruption and cache correspondence must
remain separate observations. Any subsequent cache dose-response needs its
own frozen design and multiplicity family after this input audit.

Execution report (2026-10-08): [Note 0047](research_notes/0047_graded_permutation_under_fixed_marginals.md)
records all 896 input cells, 256 exact endpoint replays and complete input
eligibility at 3/4, 7/8 and 1. It adds no cache forwards or primary cache tests.
The registration above is unchanged by those outcomes.

## Stage 4E: Graded Cache Correspondence, Transfer And Direction

Registered: 2026-10-08, before new cache measurements.

Use only the Stage 4D levels 3/4, 7/8 and 1, all 32 blocks and the unchanged
qualified FastVLM snapshot. Freeze `configs/graded_cache_fastvlm_v1.json`,
the live/published input receipts, historical Note 0046 endpoint run/tensor
receipts, runtime identity and code hashes. Do not include the unmatched 1/2
level or adapt levels, targets, views or tests after cache measurement. Note
0047's within-level acceptance does not establish cross-level spectral equality.

First replay all 128 full-permutation/mean-padding endpoint cells against
Note 0046, requiring all 384 selected tensor comparisons to be byte-identical.
Actual generation-path prepared inputs must also match exactly. These are
separate calibration captures, including previously observed r3/r4 inputs;
they do not enter reference construction or the experimental panel.

After calibration, make 384 fresh panel captures (three levels x 32 blocks x
four cells), 1,152 selected tensors and 288 factorial analyses. Freeze all
192 r1/r2 panel captures before any r3/r4 panel capture. The 128 fresh full
endpoint panel cells are repeated-condition observations, not new stimuli;
require all 384 endpoint target rechecks to match Note 0046 as well. The two
intermediate levels add 256 newly transformed cache cells and 192 factorials.
All repeats, calibration captures and new transformed cells remain separately
identifiable in the ledger and cumulative counts.

Keep zero-based L1 keys / L12 keys / L23 values, source stream seed 20260604,
temperature zero, two-token budget, source response `The image`, trace
`[785, 2168, 2168]`, shape `[1, 2, 397, 64]`, native bfloat16 capture and
42/256/99 pre/image/post positions. The repeated trace entry is not a third
generated token. Intermediate prepared inputs may differ from the endpoint
only in pixels. Provenance, source suffix, layout, metadata or exact endpoint
mismatches stop capture; do not widen tolerances or force a replacement suffix.

Decompose the measurements into three fixed questions:

1. Within each level, repeat the existing frozen-reference pairing test:
   independently unit-normalized r1/r2 references, r3/r4 tests, equal family
   weights and all 576 within-class complete-family assignments. Correct the
   18 primary tests (three levels x three targets x image/post-image) together
   with Holm. Unavailable views retain p=1 for correction and remain unavailable.
   Preserve every negative family margin and retrieval tie. All 72 fixed
   head/band views are exploratory and unadjusted.
2. For each intermediate level, replace only its r1/r2 reference vectors by
   the full endpoint's r1/r2 vectors, leaving its r3/r4 tests unchanged. Report
   all twelve image/post-image transfer views, their retrieval and the change
   from native-reference margins descriptively. Do not introduce another
   inferential test family or use transfer to rescue failed primary tests.
3. For each of the three level pairs, fixed target, token region and source
   block, report the interaction cosine, norm ratio and parallel/perpendicular
   decomposition. With the higher-fraction vector `a` as reference and the
   lower-fraction vector `b`, define `alpha = dot(a,b)/dot(a,a)`, parallel delta
   `(alpha-1)*a`, and perpendicular delta `b-alpha*a`. Record their RMS values,
   perpendicular share of total delta energy and the Pythagorean residual.
   Zero-reference or zero-delta denominators remain unavailable/null. These
   576 dependent records are split by reference/test phase, not independent
   replications or geometry-only causal tests.

Retain each factorial's spatial/palette/interaction effects, balanced energy
shares and localization. Pairwise changes in primary family margins are
descriptive; different significance labels do not test a between-level
difference. Attach each block/level's input centroid and HF drift from Note
0047 without treating dependent distances or three levels as independent
correlation evidence. No cache-to-readout mediation, persistent state,
unseen-family transfer or semantic mechanism is established by this design.
All images informed input selection, so reference/test separation is specific
to reference construction, not fully untouched validation.

Execution report (2026-10-08): [Note 0048](research_notes/0048_graded_cache_correspondence_and_direction.md)
records all 768 exact endpoint comparisons and the complete three-level panel.
Three of 18 jointly corrected primary tests pass 0.05, all post-image; all
twelve descriptive endpoint-reference transfer margins are lower than native
margins. The registration above is unchanged; the report retains all negative
family margins, endpoint repeats, dependent decompositions and spectral drift.

## Stage 4F: Fixed-Fraction Map-Seed Input Audit

Registered: 2026-10-16, before new map measurements.

Fix the selected-site fraction at 3/4, all 32 source blocks, the qualified
FastVLM processor, palette-mean padding and the expanded RGB marginals from
Note 0047. Retain its historical 3/4 map as an exact anchor. Draw exactly three
new panel-wide maps with seeds 20261016, 20261017 and 20261018. Each seed uses
`SeedSequence(seed).spawn(2)`: child zero generates the selection order and
child one the full permutation, using explicit PCG64 generators. Restrict full
permutation cycles to the first 57,600 selected positions. The same map applies
to every source block and all four cells. Both the selected set and destinations
change; this does not separate those two aspects of a map.

Freeze `configs/map_seed_input_fastvlm_v1.json`, code/runtime/source hashes,
map arrays and the registration commit before measuring. Audit all 512 cells
(four maps x 32 blocks x four cells), retaining failed gates. Require all 128
anchor cells to reproduce historical pixels and statistics exactly. Verify
inverse mapping, expanded joint RGB counts, padding, non-pixel processor fields
and serialized pixels. Use the unchanged centroid-relative 5% and HF-absolute
0.02 gates in every block. Record mapping overlap and spectral drift; passing
within-map gates does not establish between-map spectral equality.

Do not replace seeds, relax tolerances, drop failing blocks or select a subset
of maps. Proceed to a separately registered cache comparison only if all four
maps accept all 32 blocks. Otherwise publish the complete input result and stop
before model-cache capture. This stage loads the processor but not model weights.

The intended cache comparison, conditional on this gate, will freeze all r1/r2
references before r3/r4 captures. Aggregate same-map scores across the three
new maps and cross-map scores across their six ordered off-diagonal pairs;
move family blocks synchronously across maps in the same 576 assignments.
This retains map dependence rather than treating maps or pairwise cells as
new independent images. Historical-anchor comparisons and the full map matrix
will be descriptive. Exact capture counts, calibration and multiplicity will
be registered before any new cache forward, not inferred from these input gates.

Execution receipt (2026-10-16): all four maps accept 32/32 blocks. Independent
publication checks reproduce all 512 pixel/marginal controls, 128 four-cell
gates and 128 historical-anchor cell checks. No model-cache forward was used
for this eligibility decision.

## Stage 4G: Same-Map And Cross-Map Cache Correspondence

Registered: 2026-10-16, after Stage 4F input eligibility and before new cache
forwards. Freeze `configs/map_seed_cache_fastvlm_v1.json` and the published
input receipt. Do not modify the maps, fraction, gates, targets or test split.

First replay all 128 historical 3/4 cells from Note 0048, with 384 byte-exact
tensor comparisons at zero-based L1 keys / L12 keys / L23 values. Actual
prepared inputs, model snapshot, runtime, response `The image`, trace
`[785,2168,2168]`, native bfloat16 dtype, `[1,2,397,64]` tensor shape and
42/256/99 token partitions must match. Stop on any mismatch without widening
tolerances. These calibration captures do not become panel references.

Capture four maps x 32 blocks x four cells = 512 fresh panel cells, 1,536
tensors and 384 factorials. All 256 r1/r2 panel captures must be frozen before
any r3/r4 panel capture. Recheck the 128 historical-anchor panel cells against
another 384 exact tensor comparisons; they are repeated-condition observations,
not new stimuli. The three new maps contribute 384 newly transformed cells,
1,152 tensors and 288 factorials. All prepared non-pixel fields stay identical
to the anchor; only `pixel_values` may change. No direct probe is added.

For each target and image/post-image region, form a 4x4 matrix of reference-map
by test-map scores using the unchanged unit-normalized r1/r2 family references
and r3/r4 tests. Keep all 96 matrix views descriptive, including historical
anchor comparisons. Do not select a best map or add head/band tests.

The twelve primary tests are two regimes x three targets x two regions:

1. Same-map: equally average cosine scores across the three new-map diagonal
   views, before computing family margins and all-family retrieval.
2. Cross-map: equally average cosine scores across all six ordered off-diagonal
   new-map pairs. Exclude the already observed historical map from both regimes.

In each exact test, assign complete two-test-seed family blocks within broad
class, using the same assignment synchronously in every map pair. Enumerate
all 576 assignments, preserving dependence across maps. Compare equal-family
mean margins to this null and apply Holm jointly to all twelve tests.
Unavailable views retain p=1 for correction; never drop a required map pair.
Retrieval is over the sixteen averaged test-seed score rows, not 48 or 96
independent examples. Preserve negative family margins and ties.

Report same-minus-cross margins descriptively, including each family's gap.
Significance in one regime but not the other is not a test of their difference.
Retain all six unordered map-pair vector comparisons per block/target/region
(1,152 dependent directional decompositions), using the earlier-listed map
as reference, and split reference/test phases. Carry forward scalar factorial
effects, balanced dominance, localization and every block's spectral drift.
The same 32 images and input-selected 3/4 fraction are reused. Three new maps
do not establish a population-level map law, geometry-only cause, semantic
mechanism, state persistence or cache-to-readout mediation.

Execution report (2026-10-16): [Note 0049](research_notes/0049_fixed_fraction_map_seed_transfer.md)
records the completed fixed-fraction study. All 768 anchor tensor comparisons
reproduce exactly, and all twelve registered aggregate tests pass joint Holm.
The 96 descriptive matrix views, 1,152 dependent directional records and all
negative margins remain separate from the primary test family. Post-image
L12/L23 retain substantial cross-map margins; this is not a test of the
same-versus-cross gap or proof of vector/semantic invariance.

## Stage 4H: Fixed-Map Input-Vector Baseline

Registered: 2026-10-26, after Note 0049 cache results and before computing this
input correspondence matrix. This is a targeted explanatory follow-up, not
an untouched confirmation cohort. Freeze
`configs/map_seed_input_baseline_fastvlm_v1.json` and implementation in Git
before execution. Preserve all four existing maps, all 32 blocks, all four
cells, the selected-site fraction, palette-mean padding and r1/r2 versus r3/r4.

Use two full coordinate systems: 320x320 RGB canvases including padding,
converted to float64 and divided by 255; and actual model-loading-processor
`pixel_values` of shape [1,3,1024,1024], promoted from native float32 to
float64 without centering, whitening, projection, downsampling or feature
selection. Compute `jj - jm - mj + mm` in each space. Do not use cache
coordinates, generated labels or spectral summaries as substitute vectors.

Every one of 512 raw canvases must match its saved manifest and byte hash.
Every regenerated processor tensor must match both the Note 0049 input audit
and its actual cache-input pixel hash, with unchanged processor identity,
runtime and non-pixel input-audit payload. No model weights or cache forwards
are needed. Save 256 interaction vectors and hashes locally; freeze all 128
reference vectors before materializing test vectors. Stop on any mismatch
without changing the processor or widening tolerances.

Reuse the existing individually normalized r1/r2 reference directions and
r3/r4 scoring rule. Publish all 32 descriptive reference-map by test-map
views (4x4x2). Exclude the historical anchor from the four primary tests:
same-map and cross-map averages for each input space. Average cosine scores
over three diagonals or six ordered off-diagonals before computing margins
and retrieval. Move complete test-family blocks synchronously across maps
under the same 576 within-class assignments. Holm-correct these four input
tests jointly, separately from the already measured twelve cache tests.
Unavailable primary views stay in the correction with p=1; zero vectors,
cancelling references, ties and negative family margins remain explicit.
Sixteen averaged test-seed rows are not 48 or 96 independent examples.

As a deterministic calibration, inverse-permuted raw interaction arrays must
be exactly equal across maps for every block (96 comparisons to the anchor).
The four same-map raw cosine matrices must agree within 1e-12. This follows
from a common coordinate permutation being an isometry; it is not evidence
of learned invariance or independent map replication. Record all 384 paired
directional decompositions (six map pairs x 32 blocks x two spaces), ordered
as in Note 0049 and identified by reference/test phase. Norm, direction,
retrieval and same-minus-cross gaps remain distinct descriptive quantities.

Compare input and published cache margins and retrieval descriptively, with
all negative cases. Their coordinate systems and separate multiplicity
families differ: score differences, significance differences or margin ratios
cannot establish causal processing gain, semantic extraction or mediation.
The raw/processor baseline adds no new images, maps or cache observations.

Execution report (2026-10-26): [Note 0050](research_notes/0050_cross_map_correspondence_in_input_coordinates.md)
records complete reproduction of all 512 raw/processor cells, 256 input
interaction vectors and 96 exact raw inverse checks. All four input tests
pass their separate Holm family, while 8/32 primary family margins are
negative. Cross-map mean margins retain 6.004% / 8.652% of same-map margins
in raw/processor coordinates. Independent score, exact-test and direction
recomputation passes. This establishes input correspondence, not a causal
explanation of the larger post-image cache score ratios.

## Stage 4I: Raw Source-Coordinate Overlap Decomposition

Registered: 2026-10-28, after Note 0050 and before inspecting the decomposition.
Freeze `configs/source_coordinate_overlap_v1.json` and its implementation in
Git. This is a descriptive accounting of the existing raw score, not a new
statistical test, an untouched validation cohort or a pixel/model intervention.

Reuse all 128 raw interaction vectors, all four maps, eight families and the
unchanged r1/r2 reference versus r3/r4 split. Verify original file hashes and
reproduce every raw matrix/aggregate score from Note 0050 within 1e-12. Do not
call the processor, load model weights, generate images or capture new caches.

At each content output site u, a map stores the source-coordinate index p(u).
Coincidence means p_a(u) = p_b(u), not equal RGB values, identical source images
or necessarily an unmoved pixel. Partition the full canvas into four disjoint
terms, each including all three RGB channels:

1. Shared fixed content: p_a(u) = p_b(u) = u.
2. Shared moved content: p_a(u) = p_b(u), but p_a(u) != u.
3. Noncoincident content: p_a(u) != p_b(u).
4. Padding: rows outside the original [40,280) content span.

Keep padding separate: its 25% canvas area must not inflate the content-only
coincidence denominator. Report exact map-specific counts, including shared
moved sites. The source-coordinate identity refers to corresponding positions
in different source images; it does not imply identical observed pixel values.

Normalize reference inputs, their summed family reference, and test vectors
using the unchanged full-vector norms **before** partitioning. Sum elementwise
products within each component. Never re-normalize the masked vectors. The
four signed contributions must reconstruct the original full cosine, test-seed
margin, family margin and mean family margin within 1e-12. Empty components
contribute zero; zero full vectors or cancelling references remain unavailable.

Retain all 16 ordered reference-map/test-map combinations, 16 test-seed rows
and eight reference-family scores: 2,048 planned dot products with 8,192
primitive additive terms. Diagonal maps have zero noncoincident content and
serve as a control. Aggregate the three new-map diagonals and six ordered
new-map off-diagonals separately, with exactly the Note 0050 weights, excluding
the historical anchor from these aggregates. Preserve all per-family signs.

Report the two coincident content terms separately and their sum. Ratios of
component mean margins to the full mean margin are signed descriptive ratios,
not percentages constrained to [0,100], energy shares, significance evidence
or causal contributions. Keep ratios null when |denominator| <= 1e-12; do not
drop the corresponding scores or families. Retrieval is nonlinear and must
not be additively decomposed or reinterpreted as component accuracy.

Independently reload raw cells and verify the saved raw interactions, derive
the coordinate partition through a separate implementation, and recompute
the score contributions and their aggregation. No new p-values are added.
Processor resampling and cache coordinates are outside this decomposition;
the result cannot identify their mechanisms or predict the effect of a
future pixel-mask intervention with different inputs and normalization.

Execution report (2026-10-28): [Note 0051](research_notes/0051_raw_correspondence_and_shared_fixed_coordinates.md)
completes the accounting registered at `f208698`. All 2,048 dot products and
8,192 additive terms are available and independently verified from all 512
raw cells. Shared fixed coordinates contribute 102.703% of the cross-map
mean margin, noncoincident coordinates -2.703%, and shared moved/padding
terms zero. Three of eight families have positive noncoincident terms; two
family-margin signs reverse in opposite directions. The largest independent
component error is `2.45e-13` (rounded upward), below the unchanged `1e-12`
tolerance. No new input vectors, processor calls, model forwards or statistical
tests are added. Fixed-overlap mask/destination controls remain proposed.

## Stage 4J: Controlled Overlap Design Scope

Current status: Phase 1 is complete; see the registration and execution report
below. The following scope paragraph preserves the pre-registration state.

Scope recorded: 2026-10-29. Measurement registration remains incomplete.
[Note 0052](research_notes/0052_preprint_scope_and_confirmation_boundaries.md)
and [the scope record](../configs/preprint_scope_v1.json) define the next
question and planned endpoints, not a launchable experiment. The candidate
shared-fixed fractions are 0, 1/16 and 1/4 of content, with selected fraction
3/4 and target actual fixed fraction 1/4. Shared moved coincidence must be
audited separately; zero fixed overlap does not imply zero total coincidence.

Keep reference maps common across overlap levels within each construction
replicate. Register mask/destination construction, coupling, seeds, counts,
retry bounds and the analysis contract before new input scoring. Preserve
all four factorial cells and the existing all-cell processor gates, and audit
between-level spectra separately. The planned input endpoints are zero-overlap
mean margin and the paired high-minus-zero margin, each in raw and actual
processor space. The old correspondence-label permutation test does not by
itself test the overlap effect.

All existing images/maps are development material for this follow-up. No
confirmation cohort has been generated or sealed. Sample sizes, inference,
equivalence width, multiplicity policy, compute budget and confirmation release
remain open; do not invent defaults. The design record is not a runtime
enforcement mechanism. No new observation is added, and the next bounded work
is coordinate-only construction/oracle testing for Note 0053.

### Phase 1 Registration And Execution

The preceding scope text records the state before registration. It is refined,
not retroactively rewritten, by [Note 0053](research_notes/0053_controlled_overlap_coordinate_construction.md),
[Note 0054](research_notes/0054_controlled_overlap_phase1_registration.md) and
[`controlled_overlap_phase1_v1.json`](../configs/controlled_overlap_phase1_v1.json),
committed with the measurement/analysis code at `1e19014` before any full-size
map generation or new scores. Four mask seeds, two destination identities,
three q levels and one reference per mask give 28 maps. Exact single-cycle
construction fixes 25% of content; coordinate-only rejection removes shared
moved coincidence against the reference with a fixed 128-attempt cap.

The fixed budget is 32 existing source blocks x 28 maps x four cells, with
128 historical processor checks, 512 retained vectors per input view and no
model forwards. All maps enter input diagnostics even if spectral eligibility
fails. All 896 blocks must pass unchanged gates for cache eligibility; no
threshold relaxation or outcome-driven replacement is allowed.

The four primary descriptive endpoints are raw/processor zero-overlap margin
and paired high-minus-zero margin. No p-values or equivalence tests are
attempted. The registered 10,000-draw crossed-block bootstrap gives exploratory
ranges with Bonferroni tail allocation, fixed references and no guaranteed
coverage. This resolves the Phase 1 inference/budget fields before outcomes;
confirmation precision and its untouched cohort remain open, not defaulted.

Execution report (2026-10-29): [Note 0055](research_notes/0055_controlled_overlap_input_qualification.md)
records 896/896 accepted blocks and 128/128 exact historical processor checks.
[Note 0056](research_notes/0056_controlled_overlap_input_correspondence.md)
records zero-overlap raw/processor means -0.000315703 / +0.003352657 and paired
increases +0.034846370 / +0.033963924. Both zero-level exploratory ranges
include zero, while two of eight family increases are negative in each view.
Independent replay verifies all 3,584 cells, 1,024 interactions, 56 matrices
and the bootstrap. Measurement uses 3,712 processor calls; publication replay
uses a separate 3,584. No model/cache observation or new p-value is added.
Phase 1 ends here. Phase 2 requires its own calibration/endpoint/budget
registration; confirmation remains not generated and not sealed.

## Stage 4K: Controlled Overlap Cache Registration And Calibration

Registered before new calibration forwards at `526950d` on 2026-10-29:
[Note 0057](research_notes/0057_controlled_overlap_cache_registration.md) and
[`controlled_overlap_cache_phase2_v1.json`](../configs/controlled_overlap_cache_phase2_v1.json).
This is reused-data development after input outcomes, not untouched confirmation.
No-model preflight verifies all input gates, images, source/code hashes and
the qualified weight fingerprint before loading a model. Historical calibration
then requires all 128 Note 0049 anchor cells / 384 tensors to match exactly,
including actual prepared inputs, suffix IDs, dtype, shape and token partition.
No retry/resume, relaxed tolerance, cell replacement or automatic panel release.

At registration, the planned panel retains 512 factorial blocks: all 32 sources
on four reference maps and 16 test sources on 24 test maps. Its 2,048 experimental cells / 6,144
tensors are planned, not executed here. Commit and fixture-test capture and
analysis before panel outcomes. Freeze 256 reference-source cell receipts
before test-source capture, and repeat 128-cell calibration in the same process
before the experimental panel. The complete future run budget including that
calibration is 2,176 cells / 6,528 tensors, excluding today's calibration.

Three targets x image/post-image regions give six views and twelve primary
descriptive endpoints: zero-overlap margin and paired high-minus-zero margin.
Keep the same family references across q, complete-region normalization and
all families/maps. Use 10,000 paired crossed-block bootstrap draws, seed
2026102930, with fixed references and tails 0.025/12 and 1-0.025/12. The
exploratory ranges do not guarantee coverage and add no p-values, equivalence
or absence claims. Any unavailable primary score leaves both endpoints for
that view unavailable without filtering. Protocol/nonfinite/suffix failures
stop capture and withhold primary panel results. See Note 0057 for full rules.

Execution report: [Note 0058](research_notes/0058_controlled_overlap_cache_calibration.md)
records PASS for no-model preflight and historical calibration. All 384 tensor
pairs and all 128 prepared inputs/suffixes/layouts are exact; an independent
NPZ reload verifies 19,513,344 scalar pairs with maximum difference zero.
There were 128 calibration source-context forwards, zero experimental panel
cells and zero direct probes at that calibration-only stage. Confirmation
remains not generated and not sealed.

### Stage 4L: Controlled-Overlap Cache Panel Execution

[Note 0059](research_notes/0059_controlled_overlap_cache_panel_release.md) and
[`controlled_overlap_cache_release_v1.json`](../configs/controlled_overlap_cache_release_v1.json)
release the unchanged Stage 4K design with tested capture/analysis at `3be6645`,
before panel outcomes. The original Phase 2 config remains byte-identical.
Required q levels are zero and 1/4; a missing intermediate or diagonal score
does not remove complete primary endpoints. Exact-zero vectors and cancelling
references remain unavailable. There is no retry, resume or confirmation launch.

[Note 0060](research_notes/0060_controlled_overlap_cache_correspondence.md)
reports COMPLETE: 128 same-process calibration cells / 384 tensors exact,
then 256 frozen reference cells and 1,792 test cells. The reference seal is
after total ordinal 384; first test ordinal 385. All 2,048 experimental cells,
6,144 tensor sidecars, 1,536 target factorials, 3,072 region vectors and 168
score matrices pass independent verification. All twelve endpoints are
available, with no family/map filtering. Actual source suffix is `The image`
in every panel cell; no direct probe is added.

Five of six M(0) exploratory ranges lie above zero; all six high-minus-zero
ranges span zero. These do not establish confirmation, absence, equivalence
or causal processing gain. [Note 0061](research_notes/0061_controlled_overlap_phase2_decision.md)
retains the full result surface and calls for separately registered
non-learned processing baselines. That next package remains unlaunched.
