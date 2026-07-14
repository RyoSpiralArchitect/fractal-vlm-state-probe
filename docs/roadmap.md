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

Status: the balanced one-frame direct core now covers SmolVLM2, Qwen2.5-VL,
Gemma 3, InternVL3, LFM2-VL, Phi-3.5 Vision, Granite Vision, and Ministral 3 over
four source pairs. Two-model control and seeded-class extensions add two
independent fractal pairs, matched frequency and natural controls, and four
seeded Voronoi/quasicrystal and white/blue-noise pairs. The wider valid direct
surface contains 76 factorial points, 304 cell runs, and 1,216
complete-vocabulary sidecars, with non-identical after-cell
distributions in every factorial. Ministral adds a four-pair `keys`-component
scalar profile whose exact layer and sign vary. Its 16 selected full-vector
targets are spatial/palette/interaction-dominant in 15/1/0 and all interaction
maxima are image-localized. The total selected full-vector surface contains 206
layer-by-pair analyses; 194 have resolved token partitions, including 194/194
zero pre-image effects and 188/194 image-position interaction maxima. A
balanced Qwen/Ministral panel now contains four source pairs in each of the
fractal, geometry, and noise classes. Within-class direction cosine exceeds
between-class cosine in image and post-image regions at four targets under all
eight exact source-pair-level tests (`p=6/34,650`). This revises the pooled
control result from fractal-specific to fixed-generator-family conditional.
Phi's 12 analyses remain role-unassigned. The eight-model prompt audit is
balanced over all four pairs. Generated patterns agree over every pair in 39/64
fixed records, balanced-axis dominance in 4/64, and both in 2/64. The next
independent-evidence step is multiple generator pairings per class, source-role
reversal, and leave-one-generator-pair-out transfer before extending nested
lengths.

## Phase 5: Research Report

- Analysis notebook or script.
- Tables for condition metadata and run capabilities.
- Probe output diffs.
- Logprob/trace plots.
- Strict separation between observed effects and interpretation.

Status: the evidence matrix and Notes 0027-0039 encode the audited claim
boundary, balanced eight-model direct and prompt replication, selected
full-vector localization, scalar-locus versus vector-direction separation, and
the fixed-generator-family direction result. The first exact source-level
permutation null is complete; multiplicity-aware hierarchical tests and figures
remain future work. The current priority is generator-pair diversity and role
reversal, a neutral wording/order/verbalizer factorial, Phi coordinate
resolution, a distinct ninth-architecture replication, Ministral
head/position-resolved vector analysis, and a valid intervention path before
causal prose is promoted.
