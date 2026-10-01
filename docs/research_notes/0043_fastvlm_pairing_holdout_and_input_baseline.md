# Research Note 0043: FastVLM Pairing Holdout And Input Baseline

Date: 2026-10-01

Status: complete at all three fixed FastVLM targets; qualified input baseline
and failed native-calibration audit retained separately.

## Question And Frozen Design

[Note 0042](0042_pairing_seed_validation_and_localization.md) established
new-seed correspondence within eight observed ordered generator pairings at
four Qwen/Ministral targets. This stage applies the unchanged panel to FastVLM
and asks a separate input-space question: is correspondence already present
in raw RGB or actual model-processor pixel vectors?

The unchanged r1/r2 inputs supply the references; unchanged r3/r4 inputs supply
the tests. There are four geometry and four stochastic ordered pairing
families. All 64 source first-frame hashes across the panels are distinct.
References are the normalized sum of two individually normalized interaction
vectors. Test seeds do not update references or select targets.
The test images were already measured in Qwen/Ministral; they are newly
measured for FastVLM, not a new image cohort unseen across the research program.
Repeated architecture views do not create additional independent source pairs.

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

All four rows are complete. The failed native attempt adds a separate 4 cells
and 12 tensors: 12 calibration cells / 36 tensors in total for this stage,
of which 8 cells / 24 tensors are qualified exact repeats. Calibration is
excluded from the 128 new source cells / 384 tensors / 96 factorials.
No direct full-vocabulary probe is added by this stage.

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
The first native-processor substitution failed all 12 historical comparisons.
Its suffix, shape and pre-image tensors agreed, but image/post-image tensors
did not. That four-cell attempt is retained under `references/` and excluded
from pooling. Direct pixel-processor calls exposed an API failure, whereas
the original full model processor call path works without substitution.
After removing the substitution, all 12 historical tensors match bytewise
under `references_model_path/`. The exact gate was never relaxed, and no
held-out score chose a processor. The proposed native compatibility code is
not retained in the final adapter. These extra four cells / 12 tensors are
failed calibration attempts, not selected factorial analyses.
The qualified historical gate and new-reference recheck both pass 12/12.
The reference/test model snapshots are identical at revision
`81ffe929046666c43de53691147b1669ba0f3a4c`, with weight SHA-256
`cf7eafb090c64f4bef1a8ba959903a34dc58919170ad506e750e57643bff9bcd`.
Actual and historical runtime records retain MLX `0.31.1` / MLX-VLM `0.4.4`.
This measured reference check does not recover missing historical weight or
processor provenance for every older run.

FastVLM's suffix, KV dimensions and coordinate basis differ from the other
architectures. Cross-model score profiles are descriptive comparisons, not
one aligned cross-model vector or a shared-suffix intervention.

## Input-Space Baseline

Two full-coordinate interaction views use the same frozen split:

- Raw RGB, shape `[240, 320, 3]`, values divided by 255.
- Actual model-path processor pixels, shape `[3, 1024, 1024]`.

For each source pair, retain `JJ - JM - MJ + MM`, with no norm/target selection
from test data. The qualified baseline uses the unchanged full model processor
with a pixel-only `<image>` text placeholder. Its image-processor implementation
and hash match actual qualified source captures. The separately recorded native
baseline remains an ineligible alternative, not the actual model-input view.
Processor configuration and implementation hashes are retained.

| Input view | Own cosine | Other same-class cosine | Margin | Positive families | Retrieval | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw RGB | 0.20127 | 0.06523 | 0.13603 | 6/8 | 5/16 | 2/576 |
| Actual processor pixels | 0.20357 | 0.06740 | 0.13617 | 7/8 | 4/16 | 3/576 |

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

## Held-Out Cache Result

All 64 qualified reference responses and 64 test responses are `The image`,
with the same recorded generation-step IDs and resolved layout. No source
label, layer, head or token band is chosen from held-out performance.

| Target / region | Own cosine | Other same-class cosine | Margin | Positive families | Retrieval | Exact p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L1 keys / image | 0.38067 | 0.08156 | 0.29912 | 8/8 | 16/16 | 1/576 | 0.010417 |
| L1 keys / post-image | 0.58171 | 0.12135 | 0.46036 | 8/8 | 16/16 | 1/576 | 0.010417 |
| L12 keys / image | 0.36490 | 0.08041 | 0.28449 | 8/8 | 16/16 | 1/576 | 0.010417 |
| L12 keys / post-image | 0.88116 | 0.33706 | 0.54409 | 8/8 | 16/16 | 1/576 | 0.010417 |
| L23 values / image | 0.34737 | 0.06999 | 0.27738 | 8/8 | 16/16 | 1/576 | 0.010417 |
| L23 values / post-image | 0.85736 | 0.31117 | 0.54619 | 8/8 | 16/16 | 1/576 | 0.010417 |

All six primary correspondence tests pass their stage-specific joint Holm
family. Each view retrieves its own family in all 16 new-seed observations;
these are repeated views of the same test images, not 96 independent trials.
The result extends the known-pairing new-seed correspondence to a third
architecture, not to unmeasured pairing families or a shared vector basis.

All 24 exploratory head/band views have positive average margins. All two-head
views have positive family margins for all eight families, but band 2 has
four negative family-target margins:

- L1, sparse/dense dots: `-0.006343`.
- L1, white/blue noise: `-0.015957`.
- L12, sparse/dense dots: `-0.002470`.
- L12, white/blue noise: `-0.049421`.

| Target | Image-head margin range | Post-image-head margin range | Image-band margin range |
| --- | ---: | ---: | ---: |
| L1 keys | 0.2944-0.3032 | 0.4421-0.5056 | 0.2319-0.3468 |
| L12 keys | 0.2682-0.3245 | 0.5410-0.5466 | 0.2262-0.3368 |
| L23 values | 0.2620-0.3173 | 0.5394-0.5521 | 0.2221-0.3350 |

The maximum individual-test storage-head energy fraction is `0.7944`, `0.7637`
and `0.7717` at the three targets. Energy is not uniformly spread, although
prediction survives both measured heads. These are KV storage-head views,
not causal attention-head contributions; token bands are not a verified 2D
patch map.

## Factorial Integrity And Denominators

All 96 qualified reference/test factorials have exactly zero pre-image
effects, image-position interaction argmaxes and image energy above 0.9.
Interaction share is at or below `1/3` in all 96 under both image and
whole-effective conventions. Spatial/palette/interaction dominance is
`80/16/0` under either convention. The test-only subset is `41/7/0` over 48
factorials; the other 48 reference factorials are not held-out outcomes.

Direction correspondence is therefore strong without the interaction becoming
the dominant balanced energy axis. The selected full-vector aggregate is now
616 source cells, 1,704 tensors and 426 analyses. Of 414 role-resolved analyses,
414 have zero pre-image interactions, 408 image-position interaction maxima
and 408 image-energy fractions above 0.9. The 12 older Phi analyses remain
unassigned. The direct-probe aggregate is unchanged at 104 factorials /
416 cells / 1,664 full-vocabulary sidecars. Calibration repeats and failed
calibration attempts remain outside these selected-surface counts.

## Revised Reading

Pairing correspondence is already measurable in transformed input vectors.
The model experiment therefore cannot establish that such correspondence
first emerges inside the VLM. It measures how it is expressed in selected
image and post-image cache coordinates under the fixed source prompt.

The blue-noise/random-dots pairing is a useful concrete contrast: its input
margin is negative in both raw RGB and actual processor pixels, but positive
in every FastVLM primary view (image margins `0.1083-0.1236`, post-image
`0.3229-0.4163`). Literal pixel-coordinate correspondence does not describe
the entire cache profile. This does not exclude frequency, texture, palette
statistics or other input features, and is not a causal processing-gain test.

The native alternative's average input margin `0.136176` is almost identical
to the qualified processor margin `0.136174`, yet its calibration tensors are
not bitwise equivalent. Similar descriptive input scores do not establish
computational equivalence. The full processor call path and exact artifact
checks matter even when shape, suffix and pre-image tensors agree.

These baselines neither match frequency nor remove generator-parameter,
rank-map or spectral confounds. Stage 4 still requires all four transformed
cells to pass the registered processor-frequency gates before cache inspection.
No accepted frequency-matched block is claimed here.

## Artifacts

- `docs/pairing_validation_protocol.md`
- `scripts/capture_pairing_references.py`
- `scripts/run_pairing_seed_validation.py`
- `scripts/analyze_pairing_input_holdout.py`
- `runs/fastvlm_pairing_seed_validation_v1/input_model_path_holdout.json`
- `runs/fastvlm_pairing_seed_validation_v1/input_holdout.json` (native alternative)
- `runs/fastvlm_pairing_seed_validation_v1/references/` (failed native calibration)
- `runs/fastvlm_pairing_seed_validation_v1/references_model_path/`
- `runs/fastvlm_pairing_seed_validation_v1/tests/`
- `runs/fastvlm_pairing_seed_validation_v1/study_summary.json`
- `examples/research_notes/0043_fastvlm_pairing_holdout/summary.json`

Neither persistence, adaptation, semantic specificity, a universal layer nor
cache-to-readout mediation is tested by these fresh source-only forwards.
