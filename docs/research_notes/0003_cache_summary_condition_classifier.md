# Cache Summary Condition Classifier

Date: 2026-06-04

## Setup

- Source batch: `runs/null_fractal_50_seed_batch`
- Records: 9 MLX runs
- Conditions: `blank_visual_null`, `mandelbrot_zoom_a`, `julia_zoom_b`
- Seeds: `20260604`, `20260605`, `20260606`
- Frames per run: 50
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Feature source: saved sampled cache-summary statistics, not raw KV tensors

The classifier extracts summary features from captured stream caches and from
the source cache immediately before the `mid` and `after` probes. The feature
set contains layer/tensor summary statistics such as mean, variance, absolute
mean, standard deviation, and L2 norm.

## Result

Nearest-centroid classification on cache-summary features separated the three
conditions:

| Evaluation | Accuracy |
| --- | ---: |
| Leave-one-out | `9/9` |
| Leave-one-seed-out | `9/9` |

Centroid distances in standardized feature space:

| Left | Right | Distance |
| --- | --- | ---: |
| `blank_visual_null` | `julia_zoom_b` | `36.213` |
| `blank_visual_null` | `mandelbrot_zoom_a` | `28.119` |
| `julia_zoom_b` | `mandelbrot_zoom_a` | `33.734` |

All three seeds produced exact duplicate feature hashes within each condition.
Therefore this is not evidence of stochastic generalization across independent
runs. It is evidence that the deterministic run path reproducibly writes
condition information into the measured cache-summary features.

## Interpretation Boundary

The surface probe text was identical across null, Mandelbrot, and Julia in the
50-frame batch. The classifier therefore does not learn output wording
differences. It learns only the saved trace summaries.

The current claim should stay narrow:

> Under a deterministic 50-frame MLX-VLM run, cache-summary features retain
> enough condition information to separate blank, Mandelbrot, and Julia streams,
> even when the sampled probe text is identical.

Do not yet claim that raw internal states have been classified, because this
tool reads summary statistics, not full hidden-state or KV tensors.

## Next Questions

1. Does the same separation survive stochastic probe runs where seeds actually
   change the generated sequence?
2. Does a forced-choice/logprob probe expose condition tilt without relying on
   open-ended text?
3. Do lower-level image-statistic controls such as phase scrambling,
   histogram matching, static repeat, shuffled/reversed order, Voronoi tilings,
   and noise reduce or preserve the separation?
4. Does the effect replicate on larger VLMs or HF adapters with hidden-state
   features?

## Local Artifacts

- `runs/null_fractal_50_seed_batch/cache_classifier.md`
- `runs/null_fractal_50_seed_batch/cache_classifier.json`
- `runs/null_fractal_50_seed_batch/analysis_summary.md`

## Tracked Examples

- [Cache classifier JSON](../../examples/research_notes/0003_cache_summary_classifier/cache_classifier.json)
- [Batch summary JSON](../../examples/research_notes/0003_cache_summary_classifier/batch_summary.json)
