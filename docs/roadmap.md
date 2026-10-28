# Roadmap

## Phase 0: Reproducible Stimuli

- Deterministic Mandelbrot and Julia streams.
- Manifest validation with hashes and timestamps.
- External frame manifests for natural and geometric controls.
- Condition metadata for comparison axes.

Status: scaffolded.

## Phase 1: MLX-VLM Stream Probe

- Frame-by-frame chat streaming.
- Isolated before/mid/after probes.
- Sampled KV-cache summaries.
- Honest run metadata and non-claims.

Status: scaffolded.

## Phase 2: Hugging Face Internal Trace

- `model_id` and `local_path` loading.
- Hidden-state extraction via supported model outputs or hooks.
- Token/logit metrics for probe prompts.
- Matched output schema with MLX runs.

Status: first adapter scaffolded. Model-specific hooks and broader compatibility
passes are next.

## Phase 3: Remote Logprob Adapters

- Provider capability probes.
- OpenAI/Google/xAI adapter boundaries.
- Forced-choice and open-ended logprob probes.
- Behavior-only fallback labeling.

Status: planned.

## Phase 4: Comparison Runs

Run matched conditions inside the same model:

- blank visual null,
- Mandelbrot ordered,
- Mandelbrot shuffled,
- Julia ordered,
- non-fractal geometry,
- natural city/street,
- natural animal,
- deterministic noise.

Status: design-ready after Phase 2/3 adapters.

## Phase 4.5: Trace Separability

- Extract cache-summary feature matrices from completed runs.
- Run simple condition classifiers such as nearest centroid, kNN, or logistic
  regression.
- Report duplicate seeds and deterministic repeat caveats explicitly.
- Use classifiers to decide which visual controls deserve deeper raw-trace
  instrumentation.

Status: first nearest-centroid cache-summary classifier implemented.

## Phase 4.55: Paired Stochastic Probes

- Keep stream generation deterministic while sampling probes.
- Apply matched probe seed series across conditions.
- Compare `mid` and `after` distributions after subtracting `before` variance.
- Start with lexical distance, then add embeddings, semantic word categories,
  and cache-summary classifier reads on the same runs.

Status: first MLX batch runner and lexical summary implemented.

## Phase 4.6: Low-Level Visual Controls

- Static repeat, shuffled order, reversed order.
- Phase-scrambled Mandelbrot and Julia.
- RGB-quantile and luminance-rank matched phase-scrambled Mandelbrot and Julia.
- Low-pass and high-pass frequency ablations, including luminance-rank matched
  variants.
- Arbitrary manifest batch runner for transformed-control sweeps.
- Voronoi, tiling, checkerboard, quasicrystal, random dot, blue-noise, and
  white-noise generators.
- Histogram-matched visual controls.
- Cross-family palette controls that preserve one manifest's exact frame-level
  RGB pixel multiset while using another manifest's luminance-rank spatial
  ordering.
- Feature extraction for image statistics such as brightness, contrast, edge
  density, spectrum, entropy, periodicity, and symmetry.
- Processor-space image statistics computed after `pixel_values` conversion,
  including architecture-aware cycles-per-patch spectral summaries.
- Raw and processor-space 2x2 factorial image-stat contrasts aligned with the
  cache-summary factorial contrast.
- Replicated cross-palette factorial preparation over multiple source pairs.
- MLX first-step top-k and complete-vocabulary readout capture with saved-run
  contrast analysis.

Status: the generator, transformed-control, external-manifest, raw image-stat,
processor-stat, 2x2 factorial, and arbitrary manifest-batch paths are
implemented. Cross-palette transfer is replicated over two source pairs and
shows pair-dependent non-additive input/processor interactions. Complete
first-step vocabulary capture and KL/JS/TV/Hellinger plus probability-space
factorial analysis are now implemented. Earlier persistent-cache and branched
top-k interpretations were withdrawn by the Note 0027 prefix audit; the input
and processor-space findings remain valid.

## Phase 4.65: Targeted Cache Interventions

- Exact source/donor multimodal prefix and cache sequence-length checks.
- Single-layer values swaps with reciprocal branches.
- Matched source-with-source sham branches.
- Multi-layer and multi-seed sweep runner with saved top-k readouts.
- Token edit distance, top-k RMSE, and normalized donor/source pull analysis.
- Dense layer, key/value, layer-window, and sequence-position intervention
  scans.

Status: historical values-swap scaffolds and runs exist, but the Note 0027
audit found that their image-conditioned branch did not preserve the required
multimodal prefix/cache-length relation. All layer-23 and layers-10-13 causal
or susceptibility claims are withdrawn. This phase is blocked on a rebuilt
multimodal suffix path that fails closed unless exact prefix, cache length,
tensor shape, and token-region invariants pass. Reciprocal directions,
self-sham branches, full-vocabulary scoring, keys/values, layer windows, and
position-local swaps remain required after that path is valid.

## Phase 4.7: Cross-Model Replication

- Reuse the same manifests, factorial formulas, and readout contracts across
  distinct local VLM architectures.
- Record runtime versions and validate cache summaries for finite values.
- Separate stateless/single-frame, cumulative-replay, and incremental-cache
  protocols explicitly.
- Compare loci by normalized depth and token region rather than raw layer or
  position number alone.

Status: the balanced one-frame direct core now covers nine local architectures:
SmolVLM2, Qwen2.5-VL, Gemma 3, InternVL3, LFM2-VL, Phi-3.5 Vision, Granite
Vision, Ministral 3, and FastVLM over four source pairs. Two-model control and
generator-pairing extensions add independent fractal pairs, matched frequency
and natural controls, and 16 two-seed geometry/stochastic pairing units. The
wider valid direct surface contains 104 factorial points, 416 cell runs, and
1,664 complete-vocabulary sidecars, with non-identical after-cell distributions
in every factorial. The selected full-vector surface contains 1,290 analyses
from 1,768 source cells and 5,160 tensors, including 96 repeated analyses each
in Notes 0048 and 0049; these are not independent stimuli. Of these, 1,278
have resolved token partitions, including 1,278/1,278 zero pre-image effects
and 1,272/1,278 image-position interaction maxima.
FastVLM's original 12 fixed targets are
all spatial-dominant and image-localized under a validated `llava_qwen2`
single-image map. In the Qwen/Ministral hierarchy, all eight exact matching
tests support direction repetition across seeds of the same ordered pairing.
When complete pairing families are moved across broad classes over 70 exact
assignments, only Qwen's image-region test has unadjusted `p < 0.05`, and no
post-image test does. Phi's 12 analyses remain role-unassigned. The eight-model
prompt audit remains balanced over all four pairs: generated patterns agree in
39/64 fixed records, balanced-axis dominance in 4/64, and both in 2/64. The
frozen-reference follow-up now predicts two new seeds per pairing at all four
targets, with joint Holm `p=0.013889` in eight primary tests and all 68 fixed
head/token-band average margins positive. Three stochastic image-family
margins remain negative. FastVLM now adds 64 reference and 64 test cells at
three fixed targets: all six primary tests pass their separate Holm family
(`p=0.010417`), with 16/16 own-family retrieval per view. All 24 exploratory
average margins are positive, while four band-family negatives remain visible.
Raw/actual-processor baselines already have pairing correspondence, and the
native alternative is excluded after failed exact calibration. The original
model path reproduces all 24 qualified historical/reference-recheck tensors.
This repeats known test images in another model, not a new source cohort.
The first all-four-cell input sweep is now complete: only 4/32 low-pass and
1/32 common-permutation blocks qualify, with no complete reference/test family.
Raw permutation matches 32/32 blocks, but the actual processor includes a
palette-coupled black exterior. The post-hoc support-only 29/32 diagnostic
does not replace the original gate; no new cache observation is added.
The registered padding comparison is now complete: common permutation passes
1/32 blocks with black fill, 6/32 with fixed gray, and 32/32 with palette-mean
fill. Original content passes 0/32 with every policy. All expanded RGB
marginals are explicitly changed and audited. Black-square shams reproduce
384/384 processor pixel tensors but change `image_sizes`. Model-side calibration
now passes all 792 historical, black-sham and metadata-only comparisons at the
three fixed targets. Both complete permutation panels are measured: none of
the twelve jointly corrected primary tests passes 0.05, despite positive
average margins. Mean-padding retrieval is 3-7/16 versus the original 16/16.
Image margins are similar between padding policies while post-image vectors
change substantially. The graded input follow-up now holds expanded marginals
and mean padding fixed across seven nested selected-site permutations. All
896 cells and 256 historical endpoint checks reproduce; 3/4, 7/8 and 1 qualify
complete 32-block panels. Their separately registered cache comparison is now
complete: all 768 endpoint tensor checks match exactly, and 3/18 jointly
corrected primary tests pass 0.05, all post-image. Every intermediate average
margin is lower with full-endpoint references than native references, while
3/4 image norm ratios near 1.04 coexist with paired cosine medians of only
0.07-0.08. Direction, amplitude and within-condition prediction separate.
Within-level acceptance still does not establish cross-level spectral
equality. The fixed-3/4 map-seed follow-up is now complete: the historical
anchor and three new maps each pass 32/32 input blocks, and all 768 anchor
tensor checks reproduce. All twelve aggregate same-map/cross-map primary
tests pass their joint Holm family, with eight negative family margins retained.
Cross-map mean margins retain 14.1%-15.7% of same-map margins in image positions
and 94.1% / 91.5% at post-image L12/L23. These are descriptive ratios, not
between-regime tests or retained semantic information. Paired directions still
change despite near-unit median norm ratios. One of 384 factorials is narrowly
interaction-dominant; neither universal spatial dominance nor a general
interaction-dominant mechanism follows. The full raw/processor input baseline
is now complete on the same maps and split: all four aggregate tests pass
their separate Holm family, with cross/same margin ratios 6.004% / 8.652% and
cross-map retrieval 6/16 / 4/16. All 512 raw/processor input checks reproduce;
96 inverse raw comparisons are exact. Raw same-map agreement is an algebraic
isometry, not learned invariance. Input correspondence already exists, while
larger post-image cache score ratios do not identify causal gain. Raw
source-coordinate accounting is now complete with full norms unchanged:
shared fixed coordinates contribute 102.703% of the cross-map mean margin,
noncoincident coordinates -2.703%, and shared moved/padding terms zero.
All 8,192 additive terms pass independent recomputation. Positive noncoincident
contributions remain in 3/8 families, and two family margins reverse sign in
opposite directions. No new statistical test, processor call or model forward
is added. Independently varying fixed-coordinate overlap, selection masks and
destinations is the next proposed input control, not an executed intervention.
Token bands are not verified 2D patches.

## Phase 5: Research Report

- Analysis notebook or script.
- Tables for condition metadata and run capabilities.
- Probe output diffs.
- Logprob/trace plots.
- Strict separation between observed effects and interpretation.

Status: the evidence matrix and Notes 0027-0051 encode the audited claim
boundary, balanced nine-model direct replication, eight-model prompt
replication, selected full-vector localization, scalar-locus versus
vector-direction separation, and the pairing-conditioned direction hierarchy.
Exact seed-matching, frozen-reference new-seed prediction and fixed head/band
decomposition are complete at the four Qwen/Ministral targets and three fixed
FastVLM targets. Input baselines and the processor-identity audit are also
recorded. The frequency and padding studies now include exact square-input
calibration and a complete input-eligible cache panel, without reconfirming
the earlier strong pairing correspondence after permutation. Full-spectrum
equality and a frequency-only explanation remain unsupported. The graded
cache study now separates conditional pairing correspondence from
cross-condition direction and amplitude, retaining all negative family margins
and repeated endpoint counts. The fixed-fraction map study now supports
cross-map known-pairing correspondence without vector identity, retaining
negative cases and all repeated-anchor counts. Its raw/processor input
baseline now shows correspondence before any model forward. Additive raw
source-coordinate accounting now identifies a positive shared-fixed aggregate
with a negative noncoincident remainder, without erasing family exceptions.
The current priority is separately controlled fixed-site overlap and
mask/destination changes, with pre-registered construction and unchanged
all-cell processor gates before cache comparisons, then
held-out pairing-family transfer, neutral wording/order/verbalizer controls,
Phi coordinate resolution, head/patch-level
vector analysis, and a valid intervention path before causal prose is promoted.

### Preprint Work Packages

[Note 0052](research_notes/0052_preprint_scope_and_confirmation_boundaries.md)
records the approved scope, with unresolved launch requirements in
[`preprint_scope_v1.json`](../configs/preprint_scope_v1.json). No study is
launched by this record. Note 0051 remains the latest measured evidence.

| Notes | Package | Status |
| --- | --- | --- |
| 0052-0056 | Controlled fixed-overlap construction, oracle and input audit | Scope recorded; construction and measurement pending |
| 0057-0061 | Calibrated FastVLM cache comparison | Not started |
| 0062-0066 | Simple-processing baselines and representation path | Not started |
| 0067-0071 | Prompt/suffix/readout controls and final registration | Not started |
| 0072-0075 | Untouched source/map confirmation in FastVLM and Qwen | Cohort not generated or sealed |
| 0076-0080 | Independent replay, negative-case audit and manuscript | Not started |

Ministral is optional. Notes 0056 and 0061 are decision points, not reasons to
expand sample size until significance. The immediate next task is synthetic
coordinate-map construction and an independent oracle, without source-image
scoring. Counts, seeds, shared moved-coincidence policy, paired inference,
equivalence width and budget must be registered before affected measurements.
