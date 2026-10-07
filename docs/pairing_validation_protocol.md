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
