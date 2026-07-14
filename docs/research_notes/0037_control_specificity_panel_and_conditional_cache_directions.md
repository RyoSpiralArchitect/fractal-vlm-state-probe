# Research Note 0037: Control Specificity and Conditional Cache Directions

> Superseding aggregate counts and the balanced eighth-model Ministral 3
> expansion are recorded in [Note 0038](0038_ministral3_eighth_model_replication.md).
> The directional interpretation is superseded by
> [Note 0039](0039_seeded_source_class_direction_permutation.md): balancing
> independent seeds within geometry and noise reveals coherent class-specific
> directions that were erased by pooling the heterogeneous controls here.

Date: 2026-07-14

## Status

Completed for an eight-panel one-frame control-specificity study in Qwen2.5-VL
and Granite Vision. The panel adds two independently generated fractal source
pairs and six controls:

| Panel | Source A / source B | Role |
| --- | --- | --- |
| `f_g` | new Mandelbrot F / Julia G | independent fractal pair |
| `g_h` | new Mandelbrot G / Julia H | independent fractal pair |
| `phase_c_d` | quantile-matched phase controls from `c_d` | matched ablation |
| `low_c_d` | low-pass quantile-matched controls from `c_d` | matched ablation |
| `high_c_d` | high-pass quantile-matched controls from `c_d` | matched ablation |
| `geometry_v_q` | Voronoi / quasicrystal | generated geometry control |
| `noise_w_b` | white / blue noise | generated noise control |
| `natural_c_f` | china / flower sample photographs | natural-image control |

Every panel uses the same luminance-rank palette-transfer construction and the
same `MM/JJ/MJ/JM` cell keys. Reports now describe those cells neutrally as
source A/source B. The old letters remain stable identifiers, not claims that
every panel contains Mandelbrot and Julia images.

The measured surface contains:

- 32 validated canonical cell manifests,
- 16 model-by-panel processor factorials,
- 16 fresh direct factorials, 64 cell runs, and 256 full-vocabulary sidecars,
- 64 separate fresh source-only ACK runs,
- 128 selected cache-tensor sidecars, and
- 32 full-vector factorial analyses.

The phase, low-pass, and high-pass panels share the underlying `c_d` source
pair. They are matched perturbations, not three independent source-level
replications. The geometry, noise, and natural panels also represent different
source classes and are not pooled as one exchangeable population.

## Protocol

All model runs use one image, temperature 0, stream seed 20260604, probe seed
0, cumulative replay, and fresh multimodal forwards. Direct probes use
`direct_multimodal_replay`; source-cache tensors come from separate
`source-cache-only` ACK forwards. No image-conditioned cache is reused.

Qwen captures layer 33 `values`. Granite captures layer 1 `keys`, layer 35
`values`, and layer 39 `values`, matching the fixed targets selected before
this panel. Spatial, palette, and interaction dominance always uses the
equal-coefficient balanced contrasts from Note 0031.

The natural controls are the scikit-learn `china.jpg` and `flower.jpg` sample
images, center-cropped to 4:3 and resized to 320 x 240. Their bundled source
metadata identifies both as CC BY 2.0, attributed to danielbuechele and
vultilion respectively. The generated run manifests retain the final hashes.

## Input And Processor Audit

The rank transfer preserves each palette donor's RGB multiset. Across all eight
raw factorials, the maximum absolute spatial or interaction leakage in
luminance mean, luminance standard deviation, luminance entropy, or
colorfulness is `1.11e-16`.

Processor-space marginal behavior remains palette-led:

| Processor metric | Qwen dominant panels | Granite dominant panels |
| --- | --- | --- |
| tensor mean | palette 8/8 | palette 8/8 |
| tensor standard deviation | palette 7/8, spatial 1/8 | palette 8/8 |
| spectral centroid per patch | spatial 5/8, palette 3/8 | spatial 2/8, palette 4/8, interaction 2/8 |

The processor spectral axis agrees between Qwen and Granite in only 2/8
panels. It agrees with the model's direct balanced readout axis in 7/16 Qwen
panel-by-probe records and 9/16 Granite records. Processor confounds are real
and measured, but they do not map one-to-one onto the prompt-conditioned
readout axis.

## Direct Full-Vocabulary Result

Each of the 16 model-by-panel factorials contributes family and frequency
records:

- 32/32 before records have bitwise-identical sidecars across their four cells.
- 32/32 after records have non-identical complete distributions.
- Qwen returns a declared candidate semantic in 64/64 generated first tokens;
  candidate probability mass spans 0.994-0.999.
- Granite returns a declared candidate semantic in 0/64 generated first tokens;
  its first tokens are `Based` or `The`, and candidate mass spans 0.019-0.087.

Balanced readout dominance is deliberately mixed:

| Model / prompt | Spatial | Palette | Interaction |
| --- | ---: | ---: | ---: |
| Qwen family | 3 | 3 | 2 |
| Qwen frequency | 2 | 2 | 4 |
| Granite family | 3 | 5 | 0 |
| Granite frequency | 3 | 3 | 2 |

Qwen and Granite agree on the dominant axis in 6/16 aligned panel-by-probe
records. Granite therefore repeats the earlier visible/measured separation on
non-fractal inputs: categorical compliance is absent, while the complete
distribution remains structured and cell-dependent.

The family prompt is off-domain for geometry, noise, and natural images. Its
distribution is still an auditable prompt-conditioned measurement, but its
Mandelbrot/Julia candidate semantics are not a source-identity test for those
controls. The frequency prompt is more domain-general but remains a fixed
forced-choice instrument.

## Scalar Cache Update

Qwen's layer 33 `values` scalar interaction argmax appears in 6/8 current
panels. Adding `f_g` and `g_h` to the original four fractal pairs extends the
exact layer/component locus to 6/6 independently generated fractal pairs, but
the sign is now negative in 5/6 rather than 6/6. The new `g_h` point is
positive. Across all eight current panels, the sign mode is negative in 5/8.

Granite remains pair-dependent. Its eight current maxima occupy eight distinct
layer/component locations; `keys` is the component mode in 6/8, and the sign
mode is positive in 5/8. The two new fractal points are layer 0 `keys` negative
for `f_g` and layer 29 `keys` positive for `g_h`.

The source-only reruns reproduce the direct-run target summaries exactly:
32/32 Qwen layer-33 records and 96/96 Granite fixed-target records match with
maximum absolute difference zero.

## Full-Vector Result

All 128 tensor sidecars pass the analysis path's hash, dtype, shape, finite
value, offset, and token-partition checks. Across the 32 target-by-panel
analyses:

- 32/32 pre-image contrasts are exactly zero,
- 32/32 interaction maxima fall in image positions,
- 32/32 image interaction-energy fractions exceed 0.989,
- balanced dominance is spatial/palette/interaction in 23/9/0, and
- 32/32 interaction energy shares are at or below the exchangeable `1/3`
  reference.

Nonzero interaction and image localization are therefore generic to this
rank/palette perturbation panel. Interaction is not the dominant full-vector
axis at any selected target.

The added fractal pairs also revise two previous regularities. Qwen layer 33
was spatial-dominant for the original four pairs but palette-dominant for both
`f_g` and `g_h`, producing 4 spatial and 2 palette points over six fractal
pairs. Granite's three fixed targets were spatial-dominant for the original
four pairs and `f_g`, but all three are palette-dominant for `g_h`, producing
15 spatial and 3 palette points over the six-pair target surface.

## Direction Specificity

The strongest separation is directional rather than energetic.

| Model / target | Six fractal image cosine | Six fractal post-image cosine | Six controls image cosine | Six controls post-image cosine |
| --- | ---: | ---: | ---: | ---: |
| Qwen L33 `values` | 0.059 | 0.698 | 0.004 | 0.056 |
| Granite L1 `keys` | 0.109 | 0.105 | 0.007 | -0.007 |
| Granite L35 `values` | 0.084 | 0.575 | 0.003 | 0.013 |
| Granite L39 `values` | 0.077 | 0.557 | 0.001 | 0.013 |

The six-fractal groups combine the original `b_c`, `c_d`, `d_e`, and `e_f`
analyses with the new `f_g` and `g_h` analyses. Their late post-image alignment
survives the expansion. It does not survive the heterogeneous control panel,
even though the source prompt and suffix are unchanged and nearly all
interaction energy remains in image positions.

This narrows the fixed-suffix interpretation. A shared suffix alone is not
sufficient to create the observed alignment. The late direction is at least
source-class conditional and may reflect shared fractal statistics, shared
processor geometry, shared semantic structure, or their combination. The
present panel does not identify which.

## Updated Aggregate

Including this panel, the valid direct surface contains 58 factorial points,
232 cell runs, and 928 full-vocabulary sidecars. The balanced seven-model by
four-original-pair core is unchanged; the 16 new factorials form a two-model
control-specificity extension rather than a newly balanced seven-model core.

The selected full-vector surface now contains 176 fresh source-only ACK cells,
640 tensor sidecars, and 160 analyses. Balanced spatial/palette/interaction
dominance is 131/23/6. Of 148 partition-resolved analyses, all 148 have an
exact-zero pre-image effect, 144 interaction maxima fall in image positions,
and 146 image-energy fractions exceed 0.9. Interaction share is at or below
`1/3` in 149/160 analyses.

## Updated Reading

The control panel separates three claims that had been easy to blend:

1. Rank-field and palette interaction is not fractal-specific. It appears in
   raw structure, processor frequency, direct distributions, and source-cache
   tensors across fractal, ablated, geometric, noise, and natural inputs.
2. Image-token localization is also not fractal-specific. It follows the
   aligned fresh-forward visual path in every selected control analysis.
3. Cross-pair direction is specific. Strong late post-image alignment repeats
   over six fractal pairs in both models but collapses over the six controls.

The manuscript-safe statement is:

> In a two-model eight-panel control-specificity study, luminance-rank palette
> transfer preserved palette-donor RGB marginals while producing model-specific
> processor-frequency interactions. Every fresh direct after-factorial had a
> non-identical complete first-step distribution, including Granite records
> whose generated first tokens never matched a declared candidate. All 32
> selected full-vector analyses localized interaction energy to image tokens,
> but no interaction axis dominated. Late post-image interaction directions
> remained aligned over six independently generated fractal pairs and collapsed
> over six matched or non-fractal controls. Separate fresh forwards were used,
> so persistence and source-cache-to-readout mediation were not tested.

## Claim Boundaries

- The two new fractal pairs extend source-level replication to six only for
  Qwen and Granite, not for all seven models.
- The three `c_d` ablations are dependent matched controls, not independent
  source pairs.
- The six control panels are heterogeneous and do not define one population.
- Fifteen pairwise cosines from six panels are dependent descriptive
  comparisons and have no permutation-calibrated null yet.
- Family-prompt semantics are off-domain for non-fractal controls.
- Dominant-axis equality is weaker than equality of the full share vector.
- Full-vector interaction has no sign and must not be conflated with the sign
  of a scalar L2-summary contrast.
- Image localization does not establish semantic representation or causal
  mediation.
- No result establishes persistent state, a valid cache intervention,
  subjective state, a universal layer, or a universal visual direction.

## Highest-Value Next Experiments

1. Add several independently seeded pairs within each natural, geometry, and
   noise class, then build source-level label-preserving permutation references
   for image and post-image direction cosines.
2. Replace the off-domain family prompt with neutral similarity, description,
   and open-vocabulary scoring while crossing wording, order, and label mapping
   as separate factors.
3. Resolve the full-vector effects by KV head and map image positions back to
   processor patches or tiles where the model exposes a reliable layout.
4. Run a fresh-forward contract pilot on a distinct small architecture. The
   local MLX-VLM runtime makes Mistral3 the primary diversity candidate and
   FastVLM the lower-cost fallback.
5. Revisit cache intervention only after a true multimodal suffix path passes
   exact prefix, length, reciprocal, and sham invariants.

## Primary Artifacts

- `runs/control_specificity_panel_v1/two_model_control_replication.json`
- `runs/control_specificity_panel_v1/qwen_direct_seed0/`
- `runs/control_specificity_panel_v1/granite_direct_seed0/`
- `runs/control_specificity_panel_v1/qwen_source_cache_seed0/replication/six_fractal_pairs.json`
- `runs/control_specificity_panel_v1/qwen_source_cache_seed0/replication/control_six.json`
- `runs/control_specificity_panel_v1/granite_source_cache_seed0/replication/six_fractal_layer_035_values.json`
- `runs/control_specificity_panel_v1/granite_source_cache_seed0/replication/control_six_layer_035_values.json`
