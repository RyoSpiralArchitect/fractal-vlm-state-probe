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
  --output-root runs/fastvlm_pairing_seed_validation_v1/references

python3 scripts/run_pairing_seed_validation.py \
  --model mlx-community/FastVLM-0.5B-bf16 \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --reference-replication 1:keys=runs/fastvlm_pairing_seed_validation_v1/references/replication/layer_001_keys.json \
  --reference-replication 12:keys=runs/fastvlm_pairing_seed_validation_v1/references/replication/layer_012_keys.json \
  --reference-replication 23:values=runs/fastvlm_pairing_seed_validation_v1/references/replication/layer_023_values.json \
  --reference-model-snapshot runs/fastvlm_pairing_seed_validation_v1/references/model_snapshot.json \
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

An auxiliary input-space diagnostic is also fixed before FastVLM cache
measurement: apply the same r1/r2 -> r3/r4 split to full raw RGB interaction
vectors and native FastVLM processor-pixel interaction vectors. These two
diagnostic views are exploratory, with unadjusted p-values; they do not extend
the six-test confirmatory family or select cache targets. Verify that the
native processor implementation matches the one used in actual model loading.
Record the processor configuration and implementation hashes. This baseline
checks whether the correspondence is already present in transformed inputs;
it is not accepted frequency matching or a shared-coordinate model comparison.

The current Transformers class resolver initially failed to register the
MLX-VLM native image processor and silently selected the incompatible remote
implementation. Before source forwards, the adapter registers the native
class and preserves image settings, tokenizer, detokenizer and chat template.
The actual runtime compatibility tag and native implementation hash are saved.
This loader repair does not relax the historical exact-byte calibration gate.

```bash
python3 scripts/analyze_pairing_input_holdout.py \
  --reference-panel runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json \
  --test-panel runs/pairing_seed_validation_v1/panel/generator_pairing_panel_summary.json \
  --fastvlm-processor-snapshot /path/to/the/frozen/FastVLM/snapshot \
  --output-json runs/fastvlm_pairing_seed_validation_v1/input_holdout.json
```
