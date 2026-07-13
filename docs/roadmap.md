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

Status: the valid direct lane now covers SmolVLM2, Qwen2.5-VL, Gemma 3,
InternVL3, and LFM2-VL. The standard matrix contains 34 factorial points, 136
cell runs, and 544 complete-vocabulary sidecars; the one-frame surface is
balanced at four source pairs per model. Qwen repeats a late scalar locus,
InternVL a late band, Gemma an early band, SmolVLM broad pair dependence, and
LFM2 six hybrid-attention cache entries. Every direct after-factorial has
non-identical cell distributions. Selected full-vector captures now cover 26
model-local targets and 104 layer-by-pair analyses. The next replication should
add independently generated trajectories and matched
natural/geometric/frequency controls before extending nested lengths. The
prompt audit now covers all five models on `b_c` and `e_f`. Generated patterns
agree across source pairs in 34/40 paired model-family-variant records, while
balanced-axis dominance agrees in 11/40. The next prompt replication should
extend to `c_d` and `d_e` rather than add more nested replay lengths.

## Phase 5: Research Report

- Analysis notebook or script.
- Tables for condition metadata and run capabilities.
- Probe output diffs.
- Logprob/trace plots.
- Strict separation between observed effects and interpretation.

Status: the evidence matrix and Notes 0027-0033 encode the audited claim
boundary, five-model direct replication, selected full-vector localization,
balanced factorial calibration, and the five-model two-pair prompt matrix.
Initial manuscript tables exist; calibrated nulls and figures remain future
work. The current priority is a balanced prompt factorial, independent
trajectory and control replication, head/position-resolved vector analysis,
and a valid intervention path before causal prose is promoted.
