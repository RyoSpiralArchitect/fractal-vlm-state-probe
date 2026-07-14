# Research Note 0040: Generator-Pairing Direction Hierarchy

Date: 2026-07-14

Status: completed for a two-class, eight-pairing-family, two-seed hierarchy in
Qwen2.5-VL and Ministral 3.

> This note supersedes the broad generator-class interpretation in
> [Note 0039](0039_seeded_source_class_direction_permutation.md). Note 0039's
> measurements remain valid, but its four seeds per class were nested inside
> one fixed generator pairing. The present design separates seed repetition
> from transfer across different generator pairings.

## Question

Note 0039 found strong direction alignment among independently seeded
Voronoi/quasicrystal and white/blue-noise factorials. That left two distinct
possibilities confounded:

1. a direction repeats when the same ordered generator pairing is rerun with
   new seeds, or
2. a direction transfers across different generator pairings inside a broad
   geometry or stochastic class.

The first is pairing-conditioned replication. The second is the stronger
broad-class claim. This panel tests both levels directly.

## Design

The panel contains 16 atomic `MM/JJ/MJ/JM` source-pair factorials:

| Broad class | Pairing family | Replicates |
| --- | --- | ---: |
| geometry | Voronoi / quasicrystal | 2 |
| geometry | checkerboard / hex tiling | 2 |
| geometry | square tiling / triangle tiling | 2 |
| geometry | Voronoi / hex tiling | 2 |
| stochastic | white noise / blue noise | 2 |
| stochastic | white noise / random dots | 2 |
| stochastic | blue noise / random dots | 2 |
| stochastic | sparse / dense random dots | 2 |

All 32 source first-frame hashes are unique. The two Voronoi/quasicrystal and
two white/blue-noise units exactly reuse source and hybrid image hashes from
Note 0039. The remaining 12 pair units are new.

The same four fixed tensor targets are retained without inspecting the new
directions:

| Model | Target |
| --- | --- |
| Qwen2.5-VL-3B | layer 33 `values` |
| Ministral-3-3B | layer 1 `keys` |
| Ministral-3-3B | layer 16 `keys` |
| Ministral-3-3B | layer 25 `values` |

Simultaneous source-role reversal is not an independent interaction test. It
maps `MM <-> JJ` and `MJ <-> JM`, so

`JJ - JM - MJ + MM`

is algebraically unchanged. The implementation now records this invariant and
tests it directly.

## Input And Processor Integrity

Across all 16 factorials, the maximum absolute spatial or interaction leakage
in raw luminance mean, luminance standard deviation, luminance entropy, or
colorfulness is `4.44e-16`.

Processor dominance over the same 16 pairs is:

| Model / metric | Spatial | Palette | Interaction |
| --- | ---: | ---: | ---: |
| Qwen tensor mean | 0 | 16 | 0 |
| Qwen tensor standard deviation | 2 | 14 | 0 |
| Qwen spectral centroid per patch | 6 | 0 | 10 |
| Ministral tensor mean | 0 | 16 | 0 |
| Ministral tensor standard deviation | 2 | 14 | 0 |
| Ministral spectral centroid per patch | 8 | 0 | 8 |

The two processors agree on the spectral dominant axis in 14/16 pairs. Palette
still controls processor marginals, while spatial rank and its interaction
with the palette map control much of the frequency perturbation.

## Direct Full-Vocabulary Surface

The 12 new pairings add 24 model/pair factorials, 96 cell runs, and 384 complete
first-step vocabulary sidecars. Combined with the four reused pairings, the
two-model panel contains 32 factorials.

- 64/64 before records are bitwise identical over their four cells.
- 64/64 after records contain non-identical complete distributions.
- Qwen candidate mass spans `0.9933-0.9986`.
- Ministral candidate mass spans `0.9559-0.9907`.

After-record balanced dominance is:

| Model / probe | Spatial | Palette | Interaction |
| --- | ---: | ---: | ---: |
| Qwen family | 10 | 5 | 1 |
| Qwen frequency | 6 | 3 | 7 |
| Ministral family | 9 | 3 | 4 |
| Ministral frequency | 10 | 5 | 1 |

Replicate seeds retain one dominant readout axis in 6/8 Qwen family records,
2/8 Qwen frequency records, 7/8 Ministral family records, and 5/8 Ministral
frequency records. Qwen and Ministral agree on the axis in only 4/16 family and
8/16 frequency factorials. Processor spectral and direct axes agree in 29/64
aligned model/pair/probe records. Visible readout organization is therefore not
an interchangeable proxy for tensor direction.

## Full-Vector Integrity

The complete 16-pair direction surface contains 128 model/source cells, 256
target tensors, and 64 factorial analyses. The new execution contributed 96
cells, 192 tensors, and 48 analyses.

- pre-image effects are exactly zero in 64/64 analyses,
- interaction argmaxes are image positions in 60/64,
- image interaction-energy fraction exceeds 0.9 in 63/64,
- image-region dominance is spatial/palette/interaction in 55/8/1, and
- image interaction share is at or below `1/3` in 63/64.

The four out-of-image maxima are retained: the reused white/blue-noise
replicate 2 at Ministral layers 16 and 25, checkerboard/hex replicate 2 at
layer 16, and square/triangle replicate 2 at layer 25.

## Hierarchical Exact Tests

Three cosine categories are separated for every target and region:

1. eight within-pairing seed-replicate cosines,
2. 48 cross-pairing cosines inside the same broad class, and
3. 64 between-broad-class cosines.

The seed test randomizes perfect pairings independently inside the two fixed
broad classes. There are `105 x 105 = 11,025` exact assignments. The transfer
test moves complete two-seed pairing families between broad classes while
preserving four families per class. There are `C(8,4) = 70` exact assignments.

| Target / region | Within pairing | Cross pairing, same broad | Between broad | Seed p | Transfer p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen L33 image | 0.140 | 0.0268 | 0.0053 | 1/11,025 | 2/70 |
| Qwen L33 post-image | 0.665 | 0.1600 | 0.0641 | 1/11,025 | 8/70 |
| Ministral L1 image | 0.146 | 0.0148 | 0.0116 | 1/11,025 | 30/70 |
| Ministral L1 post-image | 0.489 | 0.0941 | 0.0905 | 4/11,025 | 32/70 |
| Ministral L16 image | 0.0836 | 0.0231 | 0.0080 | 162/11,025 | 10/70 |
| Ministral L16 post-image | 0.567 | 0.0845 | 0.0514 | 1/11,025 | 18/70 |
| Ministral L25 image | 0.0662 | 0.0159 | 0.0041 | 49/11,025 | 4/70 |
| Ministral L25 post-image | 0.508 | 0.1152 | 0.0535 | 1/11,025 | 8/70 |

All eight seed-replication tests are positive under their exact reference. In
contrast, only Qwen's image-region broad-class transfer reaches an unadjusted
`p < 0.05`; no post-image transfer test does. The very strong class result in
Note 0039 was therefore driven primarily by repeated seeds nested inside one
ordered generator pairing.

## Geometry Versus Stochastic Transfer

Cross-pairing alignment is consistently larger on the geometry side:

| Target / region | Geometry cross / boundary | Geometry p | Stochastic cross / boundary | Stochastic p |
| --- | ---: | ---: | ---: | ---: |
| Qwen L33 image | 0.0319 / 0.0053 | 2/70 | 0.0216 / 0.0053 | 9/70 |
| Qwen L33 post-image | 0.2585 / 0.0641 | 5/70 | 0.0615 / 0.0641 | 31/70 |
| Ministral L1 image | 0.0254 / 0.0116 | 12/70 | 0.0042 / 0.0116 | 48/70 |
| Ministral L1 post-image | 0.1519 / 0.0905 | 18/70 | 0.0364 / 0.0905 | 46/70 |
| Ministral L16 image | 0.0438 / 0.0080 | 6/70 | 0.0023 / 0.0080 | 36/70 |
| Ministral L16 post-image | 0.1516 / 0.0514 | 11/70 | 0.0174 / 0.0514 | 42/70 |
| Ministral L25 image | 0.0271 / 0.0041 | 4/70 | 0.0047 / 0.0041 | 25/70 |
| Ministral L25 post-image | 0.1687 / 0.0535 | 6/70 | 0.0618 / 0.0535 | 31/70 |

The geometry tendency is suggestive but not multiplicity-adjusted. The
stochastic family does not show general transfer beyond its fixed pairing in
this panel.

## Revised Reading

The strongest supported statement is now:

> Full-vector interaction direction repeats strongly across independent seeds
> of the same ordered generator pairing. Transfer across different pairings in
> a broad geometry or stochastic class is much weaker, target-dependent, and
> concentrated on the geometry side; only Qwen's image-region global transfer
> passes the present unadjusted family-block exact reference.

This is pairing-conditioned visual perturbation, not evidence for a generic
geometry/noise concept direction. The high post-image seed cosine remains a
real and measurable result, but it is more specific than Note 0039 inferred.

## Claim Boundaries

- Pairing family has only two seed replicates.
- There are four observed pairing families per broad class and no natural or
  multi-pair fractal class in this hierarchy.
- Exact p-values are unadjusted across four targets, two regions, and
  class-specific follow-ups.
- Generator type, parameter contrast, frequency structure, and learned visual
  structure remain entangled across pairing families.
- Simultaneous A/B reversal cannot test interaction-direction orientation
  because the factorial interaction is invariant to that relabeling.
- Source-cache tensors and direct probes come from separate fresh forwards; no
  cache-to-readout causal path is tested.
- No result establishes persistence, adaptation, subjective state, semantic
  steering, or a cross-model vector basis.

## Highest-Value Next Experiments

1. Add a third and fourth replicate for each current pairing, then leave one
   seed out to estimate pairing-conditioned direction uncertainty.
2. Add more pairing families, especially within stochastic and fractal inputs,
   and use leave-one-pairing-family-out transfer.
3. Build frequency-matched pairing families so broad-class transfer is not
   carried by processor spectral structure.
4. Resolve the effect by KV head and image patch to determine whether seed
   replication is concentrated or distributed.
5. Add independently sourced natural-image pairing strata with explicit
   content and license provenance.

## Manuscript-Safe Statement

> In Qwen2.5-VL and Ministral 3, interaction-vector direction repeated across
> two independent seeds of each of eight ordered generator pairings. Exact
> within-class matching tests were positive at four fixed tensor targets in
> image and post-image regions. After pairing families rather than individual
> seeds were treated as the broad-class permutation unit, transfer across
> different geometry or stochastic pairings was weak: one of eight global
> region-target tests had unadjusted `p < 0.05`, and no post-image test did.
> The result therefore supports pairing-conditioned replication, not generic
> semantic class direction or persistent state.

## Primary Artifacts

- `configs/generator_pairing_transfer_v1.json`
- `runs/generator_pairing_transfer_v1/generator_pairing_panel_summary.json`
- `runs/generator_pairing_transfer_v1/processor/`
- `runs/generator_pairing_transfer_v1/model_runs/`
- `runs/generator_pairing_transfer_v1/analyses/`
- `runs/generator_pairing_transfer_v1/replication/`
- `runs/generator_pairing_transfer_v1/permutation/`
- `runs/generator_pairing_transfer_v1/hierarchy/`
- `examples/research_notes/0040_generator_pairing_direction_hierarchy/summary.json`
