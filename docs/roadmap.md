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
aggregation, full-vocabulary logprob or teacher-forced reads, and cache-swap
intervention scaffolding are next.

## Phase 5: Research Report

- Analysis notebook or script.
- Tables for condition metadata and run capabilities.
- Probe output diffs.
- Logprob/trace plots.
- Strict separation between observed effects and interpretation.

Status: future.
