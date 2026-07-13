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
- MLX first generated-token top-k readout capture and saved-run readout
  contrast analysis.

Status: first generator suite, paired pattern batch runner, image-statistic
reporter, RGB quantile-matched phase-scramble control, luminance-rank matched
phase-scramble control, low/high-pass frequency controls, and arbitrary
manifest batch runner implemented. The first image-stat/cache-distance
correlation reporter is implemented. Independent stream variants, a
forced-choice probe preset, and first sequence-position cache-delta reporting
are implemented. Cross-family palette controls and processor-space image
statistics are implemented. First-step top-k logprob readout comparison for
saved HF runs is implemented. The first 12-frame cross-family palette-control
smoke is documented, including a summary-stat 2x2 cache factorial contrast
for spatial main effect, palette main effect, and interaction. Replicated
cross-palette batch preparation and raw/processor image-stat factorial
contrasts are implemented. The first two-pair 12-frame cross-palette
replication is documented: forced-choice surface labels remained fixed, cache
interaction argmax locations repeated, and processor image-stat interactions
were pair-dependent. The first true 50-frame two-pair replication is documented:
surface labels stayed fixed and scalar cache interaction argmax locations again
landed at mid layer 23 `values` and after layer 0 `keys`. MLX first-token
top-k readout capture and a saved-run readout analyzer are implemented. The
top-k20 rerun found identical first-token top-20 sets across `MM/JJ/MJ/JM` for
every phase/probe record while the cache locus persisted. Transform seed/cutoff
aggregation and full-vocabulary logprob or teacher-forced reads are still next.
The first cache-swap intervention scaffold is implemented for layer-targeted
MLX `PromptCacheState` `values` swaps before creative branch probes.

## Phase 4.65: Targeted Cache Interventions

- Strict source/donor token-history and tensor-shape compatibility checks.
- Single-layer values swaps with reciprocal branches.
- Matched source-with-source sham branches.
- Multi-layer and multi-seed sweep runner with saved top-k readouts.
- Token edit distance, top-k RMSE, and normalized donor/source pull analysis.
- Dense layer, key/value, layer-window, and sequence-position intervention
  scans.

Status: the first two-pair sweep is complete for layers `0`, `12`, `22`, and
`23` over three matched probe seeds. All generated token sequences remained
origin-identical. Layer 12 produced the largest tested top-k perturbation in
both pairs, about `22%` of baseline source/donor separation, but remained
origin-like. Layer 23 was smaller and strongly origin-like. Self-swap shams were
exactly zero. A seed-0 dense scan over layers `8-23` then produced highly similar
profiles across both source pairs (Pearson `0.964`, Spearman `0.982`), with the
same argmax at layer 10 and the same top three layers `10`, `13`, and `12`.
This separates the replicated layer 23 summary-stat locus from single-layer
generative sufficiency and identifies a mid-layer confirmation target. Strict
multi-seed reciprocal/sham confirmation, keys, key-value pairs, position-local
swaps, and full-vocabulary scoring are next.

## Phase 4.7: Cross-Model Replication

- Reuse the same manifests, factorial formulas, and readout contracts across
  distinct local VLM architectures.
- Record runtime versions and validate cache summaries for finite values.
- Separate stateless/single-frame, cumulative-replay, and incremental-cache
  protocols explicitly.
- Compare loci by normalized depth and token region rather than raw layer or
  position number alone.

Status: the first Qwen2.5-VL-3B 4bit pilot is complete for two one-frame source
pairs with all 36 cache layers. Labels and saved first-token top-k10 readouts
were cell-invariant, while the cache-summary interaction argmax repeated at
layer 33 `values`, position 128. MLX-VLM 0.4.4 currently fails when reusing
Qwen's prompt cache across a second image turn, so persistent multi-frame
cross-model replication remains open. Early/middle/late frame sampling,
token-region mapping, and an explicit cumulative-replay lane are next.

## Phase 5: Research Report

- Analysis notebook or script.
- Tables for condition metadata and run capabilities.
- Probe output diffs.
- Logprob/trace plots.
- Strict separation between observed effects and interpretation.

Status: evidence matrix started. Manuscript tables and figures remain future;
the current priority is full-vocabulary scoring and intervention/cross-model
coverage before prose claims are promoted.
