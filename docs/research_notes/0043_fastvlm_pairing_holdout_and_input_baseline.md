# Research Note 0043: FastVLM Pairing Holdout And Input Baseline

Date: 2026-10-01

Status: input baselines complete; FastVLM reference/test capture pending.

## Question And Frozen Design

[Note 0042](0042_pairing_seed_validation_and_localization.md) established
new-seed correspondence within eight observed ordered generator pairings at
four Qwen/Ministral targets. This stage applies the unchanged panel to FastVLM
and asks a separate input-space question: is correspondence already present
in raw RGB or native processor pixel vectors?

The unchanged r1/r2 inputs supply the references; unchanged r3/r4 inputs supply
the tests. There are four geometry and four stochastic ordered pairing
families. All 64 source first-frame hashes across the panels are distinct.
References are the normalized sum of two individually normalized interaction
vectors. Test seeds do not update references or select targets.

FastVLM targets remain L1 keys, L12 keys and L23 values from Note 0041. Their
image/post-image views comprise six primary tests with a separate joint Holm
family of six. Note 0042's family of eight remains unchanged. All 24 fixed
head/token-band views are exploratory, with unadjusted p-values.

The exact test retains complete two-test-seed family blocks and assigns them
to reference families inside broad class: `4! x 4! = 576` assignments. This
tests correspondence conditional on these eight pairings, not independent
generator-family population sampling or unseen-pairing transfer.

The [execution addendum](../pairing_validation_protocol.md#stage-3-execution-addendum)
was written before FastVLM pairing reference/test forwards. It separates:

| Capture | Cells | Tensor sidecars | Factorials | Use |
| --- | ---: | ---: | ---: | --- |
| Historical b_c calibration | 4 | 12 | 0 | Exact-byte gate before new references |
| New r1/r2 references | 64 | 192 | 48 | Frozen directions, not test outcomes |
| New-reference recheck | 4 | 12 | 0 | Exact-byte gate before held-out tests |
| New r3/r4 tests | 64 | 192 | 48 | Held-out scores |

These are the registered planned counts, not an executed denominator while
capture is pending. No direct full-vocabulary probe is added by this stage.

## Source And Coordinate Gates

The existing FastVLM response is `The image`, not the requested `ACK`.
Require the same prompt, runtime and actual response across source cells,
with the historical recorded generation-step ID sequence `[785, 2168, 2168]`.
Its final recorded step is repeated; this is not a claim of three distinct
generated tokens. The generation budget remains two tokens.

The historical target shape is `[1, 2, 397, 64]`: 42 pre-image, 256 image and
99 post-image positions. The verified expansion strategy is
`llava_qwen2_single_image_run_replacement`. A failed source, shape, layout or
bytewise calibration gate stops pooling instead of redefining the condition.
Reference and test captures must share the same model snapshot fingerprint.

FastVLM's suffix, KV dimensions and coordinate basis differ from the other
architectures. Cross-model score profiles are descriptive comparisons, not
one aligned cross-model vector or a shared-suffix intervention.

## Input-Space Baseline

Two full-coordinate interaction views use the same frozen split:

- Raw RGB, shape `[240, 320, 3]`, values divided by 255.
- Native FastVLM processor pixels, shape `[3, 1024, 1024]`.

For each source pair, retain `JJ - JM - MJ + MM`, with no norm/target selection
from test data. The baseline uses the installed MLX-VLM native
`FastVLMImageProcessor`, not the Hugging Face remote-code implementation.
Processor configuration and implementation hashes are retained; its eligibility
requires matching the actual model-loading processor.

| Input view | Own cosine | Other same-class cosine | Margin | Positive families | Retrieval | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw RGB | 0.20127 | 0.06523 | 0.13603 | 6/8 | 5/16 | 2/576 |
| Native processor pixels | 0.20357 | 0.06740 | 0.13618 | 7/8 | 4/16 | 3/576 |

Both baselines have positive average correspondence. Their p-values are
exploratory and unadjusted; they do not extend the six-test cache Holm family.
Raw negative family margins are blue-noise/random-dots `-0.008919` and
white-noise/random-dots `-0.000325`. Processor blue-noise/random-dots remains
negative at `-0.010996`. Positive class-averaged correspondence is not uniform
individual-family success, and retrieval is a different descriptive outcome.

The square/triangle pairing's margin is about `0.641` in both input views;
family contributions must remain visible instead of being replaced by the
pooled average. Inputs and cache coordinates have different dimensions, so
their margin difference is not a measured processing gain or causal mediation.

## Reading Before Cache Results

Pairing correspondence is already measurable in transformed input vectors.
The model experiment therefore cannot establish that such correspondence
first emerges inside the VLM. It can measure how it is expressed in selected
image and post-image cache coordinates under the fixed source prompt.

These baselines neither match frequency nor remove generator-parameter,
rank-map or spectral confounds. Stage 4 still requires all four transformed
cells to pass the registered processor-frequency gates before cache inspection.
No accepted frequency-matched block is claimed here.

## Artifacts

- `docs/pairing_validation_protocol.md`
- `scripts/capture_pairing_references.py`
- `scripts/run_pairing_seed_validation.py`
- `scripts/analyze_pairing_input_holdout.py`
- `runs/fastvlm_pairing_seed_validation_v1/input_holdout.json`
- `runs/fastvlm_pairing_seed_validation_v1/references/` (pending)
- `runs/fastvlm_pairing_seed_validation_v1/tests/` (pending)

Neither persistence, adaptation, semantic specificity, a universal layer nor
cache-to-readout mediation is tested by these fresh source-only forwards.
