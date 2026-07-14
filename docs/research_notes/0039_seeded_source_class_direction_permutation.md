# Research Note 0039: Seeded Source-Class Direction Permutation

Date: 2026-07-14

Status: completed for a balanced three-class, four-source-pair direction panel
in Qwen2.5-VL and Ministral 3.

> This note supersedes the directional interpretation in Notes 0037 and 0038.
> Their artifact counts and within-panel observations remain valid, but pooling
> six heterogeneous controls hid coherent directions inside the geometry and
> noise generator families.

## Question

Note 0037 found strong late post-image interaction-vector alignment over six
fractal pairs and near-zero alignment after six heterogeneous controls were
pooled. That established that a shared text suffix was insufficient, but it
did not distinguish two explanations:

1. the repeated direction was specific to fractals, or
2. each ordered generator family had its own repeatable direction, which was
   erased when unlike controls were averaged together.

The new panel tests the second possibility with source-pair-level replication
and an exact class-label permutation reference.

## Design

The permutation surface contains 12 atomic source-pair factorials, balanced as
four units in each class:

| Class | Source A / source B | Pair labels |
| --- | --- | --- |
| fractal | Mandelbrot / Julia | `b_c`, `c_d`, `d_e`, `e_f` |
| geometry | Voronoi / quasicrystal | `geometry_00` through `geometry_03` |
| noise | white noise / blue noise | `noise_00` through `noise_03` |

Within every pair, the same luminance-rank palette transfer produces the
`MM/JJ/MJ/JM` cells. Source A/source B orientation is held fixed. The four
geometry and four noise units use independent generator seed pairs. All 12
newly generated source-frame hashes are unique.

Two models contribute four fixed tensor targets:

| Model | Target | Selection boundary |
| --- | --- | --- |
| Qwen2.5-VL-3B | layer 33 `values` | previously repeated scalar locus |
| Ministral-3-3B | layer 1 `keys` | previous scalar exception target |
| Ministral-3-3B | layer 16 `keys` | previous modal scalar target |
| Ministral-3-3B | layer 25 `values` | effect-unselected late depth control |

The targets were fixed before the new seeded geometry/noise results were
inspected. Ministral layer 0 `keys` was not included because the prior fractal
panel had exactly zero post-image interaction there, making post-image cosine
undefined. Target choice is therefore still exploratory and conditioned on
earlier results, except for the layer 25 depth control.

All direct probes are fresh multimodal forwards. All tensors come from separate
fresh source-only ACK forwards. No image-conditioned cache is reused.

## Input, Processor, And Direct Integrity

Across the eight geometry/noise factorials, the maximum absolute spatial or
interaction leakage in raw luminance mean, luminance standard deviation,
luminance entropy, or colorfulness is `1.67e-16`.

The two-model processor surface contains 16 model-by-pair factorials:

| Processor metric | Spatial | Palette | Interaction |
| --- | ---: | ---: | ---: |
| tensor mean | 0 | 16 | 0 |
| tensor standard deviation | 0 | 16 | 0 |
| spectral centroid per patch | 7 | 0 | 9 |

Processor marginals remain palette-led, while frequency structure is split
between spatial and interaction axes. The processor spectral axis agrees with
the direct full-vocabulary axis in only 11/32 aligned after records: 3/16 for
Qwen and 8/16 for Ministral.

The balanced geometry/noise direct surface contains 16 factorials, 64 cell
runs, and 256 complete-vocabulary sidecars:

- 32/32 before records are bitwise identical across their four cells,
- 32/32 after records contain non-identical complete distributions,
- all 256 generated first tokens map to a declared candidate, and
- candidate probability mass spans `0.8475-0.9986`.

After-record balanced dominance is spatial/palette/interaction in 9/11/12.
The off-domain family labels remain measurement instruments, not correct class
names for geometry or noise images.

## Full-Vector Integrity

The balanced direction surface assembles 96 valid source-only ACK cells, 192
target tensors, and 48 pair-by-target analyses. Existing artifact-complete
fractal and seed-0 control cells were reused; the new execution added 56 ACK
cells, 120 tensors, and 30 analyses.

All four target groups pass tensor shape and token-position alignment. Across
the 48 analyses:

- pre-image effects are exactly zero in 48/48,
- interaction argmaxes fall in image positions in 46/48,
- image interaction-energy fractions exceed 0.9 in 46/48,
- balanced dominance is spatial/palette/interaction in 46/2/0, and
- interaction energy share is at or below `1/3` in 45/48.

The two out-of-image maxima are `noise_02` at Ministral layers 16 and 25. The
two image-energy exceptions are the previously measured Ministral layer 25
fractal `b_c` and `d_e` points. The three interaction-share exceptions are
Qwen `c_d`, Qwen `noise_03`, and Ministral layer 16 `noise_00`. These exceptions
do not affect the source-level direction test and are retained rather than
rounded into an all-cells claim.

## Exact Source-Level Permutation

One complete source-pair factorial is the permutation unit. Its four cells and
the orientation of its interaction vector stay together. Only the class labels
are reassigned while preserving class sizes 4/4/4.

There are exactly

`12! / (4! 4! 4!) = 34,650`

assignments. The one-sided global statistic is mean within-class cosine minus
mean between-class cosine. The smallest attainable p-value is
`6/34,650 = 0.00017316`, because six class-name permutations produce the same
unlabeled partition.

| Model / target | Region | Within mean | Between mean | Difference | Exact p |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen L33 `values` | image | 0.0788 | 0.0035 | 0.0753 | 0.000173 |
| Qwen L33 `values` | post-image | 0.7099 | 0.0917 | 0.6183 | 0.000173 |
| Ministral L1 `keys` | image | 0.0930 | 0.0068 | 0.0862 | 0.000173 |
| Ministral L1 `keys` | post-image | 0.3666 | -0.0144 | 0.3810 | 0.000173 |
| Ministral L16 `keys` | image | 0.0374 | -0.0017 | 0.0391 | 0.000173 |
| Ministral L16 `keys` | post-image | 0.3716 | 0.0075 | 0.3641 | 0.000173 |
| Ministral L25 `values` | image | 0.0368 | 0.0005 | 0.0362 | 0.000173 |
| Ministral L25 `values` | post-image | 0.3481 | 0.0293 | 0.3188 | 0.000173 |

All four targets therefore show source-generator-class organization in both
regions under this conditional exact reference. The post-image separation is
much larger than the image separation.

## Class-Specific Post-Image Coherence

Each class-specific test compares the four within-class cosines with cosines
across that class boundary. Choosing four units from 12 gives 495 exact
memberships per test.

| Model / target | Fractal mean / p | Geometry mean / p | Noise mean / p |
| --- | ---: | ---: | ---: |
| Qwen L33 `values` | 0.6776 / 0.0061 | 0.6610 / 0.0040 | 0.7912 / 0.0020 |
| Ministral L1 `keys` | 0.4272 / 0.0040 | 0.3660 / 0.0081 | 0.3067 / 0.0020 |
| Ministral L16 `keys` | 0.4076 / 0.0020 | 0.3897 / 0.0040 | 0.3174 / 0.0061 |
| Ministral L25 `values` | 0.3311 / 0.0040 | 0.4069 / 0.0020 | 0.3063 / 0.0061 |

Every post-image class has positive within-class coherence and an unadjusted
one-sided exact p-value at or below 0.0081. Image-region class coherence is
weaker: Ministral noise is not individually separated at layer 16 (`p=0.287`)
or layer 25 (`p=0.099`), despite significant global class organization.

Pairwise class tests do not support a unique fractal advantage. The only
post-image pairwise difference below 0.05 is Qwen geometry versus noise
(`p=0.0286`, unadjusted); noise is the more coherent class there. Region,
target, class-specific, and pairwise p-values have not been multiplicity
adjusted.

## Revised Reading

The old result was not evidence that only fractals possess a repeated late
direction. It was evidence that one should not average unlike controls and
call the result a control population.

The current bounded reading is:

> Under fixed source-A/source-B generator pairings, independently seeded
> Mandelbrot/Julia, Voronoi/quasicrystal, and white/blue-noise factorials each
> produce coherent class-conditional post-image interaction directions. The
> direction is organized by generator family and is not fractal-specific.

This is more specific than generic image localization and less semantic than a
class concept. The repeated vector may encode ordered generator identity,
source-role orientation, processor geometry, low-level statistics, learned
visual structure, or a mixture. The present design does not separate those
possibilities.

The visible/readout separation also remains. Complete distributions change in
all after records, but generated candidates and balanced readout axes do not
provide a stable decoder for the tensor direction.

## Updated Aggregate

Including the new executions, the valid direct surface contains 76 factorial
points, 304 cell runs, and 1,216 complete-vocabulary sidecars.

The selected full-vector surface contains 248 fresh source-only ACK cells, 824
tensor sidecars, and 206 analyses. Balanced
spatial/palette/interaction dominance is 175/25/6. Of 194 analyses with
resolved token partitions:

- pre-image interaction is zero in 194/194,
- the interaction argmax is an image position in 188/194, and
- image interaction-energy fraction exceeds 0.9 in 190/194.

Interaction share is at or below `1/3` in 193/206 analyses.

## Claim Boundaries

- A class is one fixed ordered generator pairing, not a sampled semantic
  population. Four seeds do not establish geometry or noise in general.
- Source A/source B orientation is fixed. Reversing roles or changing the
  generator pairing may reverse or replace the observed direction.
- The natural-image class is not replicated in this panel.
- The exact test conditions on these 12 observed source pairs. It does not
  license inference to unobserved visual populations.
- Targets share source pairs and models, so the eight global tests are not
  eight independent replications.
- Class-specific and pairwise p-values are unadjusted exploratory tests.
- Three targets were selected from earlier scalar results; only Ministral layer
  25 is an effect-unselected depth control.
- Family-prompt candidate semantics are off-domain for geometry and noise.
- ACK tensors and direct readouts come from separate fresh forwards. No causal
  cache-to-readout path is tested.
- No result establishes persistence, adaptation, subjective state, a universal
  layer, a cross-model vector basis, or a valid cache intervention.

## Highest-Value Next Experiments

1. Replace seed-only replication with multiple generator pairings inside each
   class, reverse source roles, and use leave-one-generator-pair-out tests.
2. Add several independently sourced natural-image pairings with explicit
   content and licensing strata.
3. Match or sweep processor-space spectral structure inside each generator
   family to separate generator identity from low-level frequency geometry.
4. Resolve Qwen and Ministral effects by KV head and image patch, then test
   whether class organization is concentrated or distributed.
5. Cross neutral similarity, free description, wording, order, and verbalizer
   mapping before linking tensor direction to a semantic readout.

## Manuscript-Safe Statement

> In Qwen2.5-VL and Ministral 3, a balanced panel of 12 source-pair
> cross-palette factorials showed greater within-generator-class than
> between-class interaction-vector cosine for fractal, geometry, and noise
> pairings. Exact source-pair-level class-label permutations gave `p=0.000173`
> for image and post-image regions at four fixed tensor targets. All three
> classes showed positive post-image coherence, revising the earlier
> fractal-specific interpretation to a generator-family-conditional result.
> Each class still used one fixed ordered generator pairing, and separate fresh
> forwards were used, so semantic class generality, persistence, and causal
> cache mediation were not tested.

## Primary Artifacts

- `runs/seeded_class_direction_v1/replication/`
- `runs/seeded_class_direction_v1/permutation/`
- `runs/seeded_class_direction_v1/processor/`
- `runs/seeded_class_direction_v1/direct/`
- `runs/seeded_class_direction_v1/source_cache/`
- `examples/research_notes/0039_seeded_source_class_direction/summary.json`
