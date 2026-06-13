# Control Stimuli

This repository treats visual controls as first-class stimuli, not as
afterthoughts. The goal is to separate broad stream effects from lower-level
image statistics, temporal order, and recognizable semantics before making
claims about fractal families.

## Generated Controls

Use `scripts/generate_control_frames.py` for deterministic generated controls:

```bash
python3 scripts/generate_control_frames.py \
  --kind voronoi \
  --output runs/controls/voronoi_seed_7 \
  --width 320 \
  --height 240 \
  --total-frames 50 \
  --fps 1 \
  --seed 7 \
  --overwrite
```

Supported generated kinds:

| Kind | Family | Why it is useful |
| --- | --- | --- |
| `blank` | control | Null image-token stream. |
| `white_noise` | control | High-entropy low-semantic baseline. |
| `blue_noise` | control | High-frequency low-semantic baseline. |
| `random_dots` | control | Sparse point field without object semantics. |
| `checkerboard` | geometric | Simple periodic structure. |
| `square_tiling` | geometric | Grid regularity with visible line geometry. |
| `triangle_tiling` | geometric | Non-square periodic geometry. |
| `hex_tiling` | geometric | Sixfold local symmetry. |
| `voronoi` | geometric | Irregular cells with clear boundaries. |
| `quasicrystal` | geometric | Aperiodic interference-like structure. |

These are not claims about what the model "perceives." They are deliberately
plain visual families that help ask whether cache-summary or output drift
tracks brightness, entropy, periodicity, symmetry, or boundary density.

## Source-Manifest Transforms

The same CLI can transform any existing manifest:

```bash
python3 scripts/generate_control_frames.py \
  --kind phase_scrambled \
  --source-manifest runs/null_fractal_50_seed_batch/stimuli/julia/manifest.json \
  --output runs/controls/julia_phase_scrambled_seed_7 \
  --seed 7 \
  --overwrite
```

Supported transform kinds:

| Kind | Preserves | Breaks or changes |
| --- | --- | --- |
| `phase_scrambled` | Approximate per-frame spectrum and color statistics | Spatial phase and recognizable geometry. |
| `phase_scrambled_quantile_matched` | Phase-scrambled frame with exact per-channel RGB quantile matching to the source frame | Spatial phase and recognizable geometry; luminance entropy should still be measured because RGB-channel matching is not identical to direct luminance matching. |
| `phase_scrambled_luminance_quantile_matched` | Source RGB pixel multiset and source luminance distribution, reassigned by scrambled luminance rank | Original spatial arrangement and recognizable geometry; spectrum is no longer the same promise as plain phase scrambling. |
| `low_pass` | Coarse low-frequency structure after FFT masking | High-frequency texture, fine edges, and sharp local detail. |
| `high_pass` | Local high-frequency detail after FFT masking | Low-frequency luminance layout and coarse shape. |
| `low_pass_luminance_quantile_matched` | Coarse low-frequency rank order plus source RGB pixel distribution | Fine detail; spectrum is filtered before pixel reassignment. |
| `high_pass_luminance_quantile_matched` | High-frequency rank order plus source RGB pixel distribution | Coarse layout; spectrum is filtered before pixel reassignment. |
| `static_repeat` | One exact source frame repeated at the same cadence | Temporal evolution. |
| `shuffled` | Source frame content and count | Temporal order. |
| `reversed` | Source frame content and count | Forward temporal direction. |

Transform manifests record the source manifest hash, source condition, and
source frame index for each emitted frame. That keeps the control auditable when
later probe differences are small.

For the stricter structure-control ladder, compare plain `phase_scrambled`
against both quantile-matched variants. The luminance-rank version is the
stronger entropy control:

```bash
python3 scripts/generate_control_frames.py \
  --kind phase_scrambled_luminance_quantile_matched \
  --source-manifest runs/pattern_probe_smoke/stimuli/julia/manifest.json \
  --output runs/phase_scramble_luminance_quantile_smoke/julia_phase_luminance_quantile_seed_7 \
  --seed 7 \
  --overwrite
```

This transform is useful when the question is whether trace separation survives
after pushing the scrambled control back onto the source frame's luminance and
color distribution. It is still not a final "structure-only" proof by itself:
run image statistics after generation and check whether edge density, spectrum,
and temporal deltas remain plausible for the comparison being made.

For low-vs-high frequency ablations, keep the source stream fixed and sweep the
FFT cutoff:

```bash
python3 scripts/generate_control_frames.py \
  --kind low_pass_luminance_quantile_matched \
  --source-manifest runs/pattern_probe_smoke/stimuli/mandelbrot/manifest.json \
  --output runs/frequency_ablation_smoke/mandelbrot_low_pass_luminance_quantile_cutoff_018 \
  --frequency-cutoff 0.18 \
  --overwrite

python3 scripts/generate_control_frames.py \
  --kind high_pass_luminance_quantile_matched \
  --source-manifest runs/pattern_probe_smoke/stimuli/mandelbrot/manifest.json \
  --output runs/frequency_ablation_smoke/mandelbrot_high_pass_luminance_quantile_cutoff_018 \
  --frequency-cutoff 0.18 \
  --overwrite
```

These controls are the next way to test whether a trace jump is better
explained by destroying coarse macro-structure, suppressing fine detail, or
changing lower-level statistics. The luminance-quantile variants are usually
the cleaner first read because they keep source entropy and colorfulness fixed.

## Suggested Next Ladder

For one source stream such as Julia:

1. `blank`
2. `white_noise`
3. `blue_noise`
4. `static_repeat`
5. `shuffled`
6. `reversed`
7. `phase_scrambled`
8. `low_pass_luminance_quantile_matched`
9. `high_pass_luminance_quantile_matched`
10. `checkerboard`
11. `voronoi`
12. `quasicrystal`
13. original ordered Julia

The clean read is not "which pattern is mystical." The useful read is whether a
small set of image-statistic and temporal controls can explain the trace
separation seen in the first deterministic batch.

## Pattern Batch Runner

Once the generated controls exist, use the pattern batch runner to place them in
the same paired stochastic-probe design as the early null/fractal runs:

```bash
python3 scripts/run_mlx_pattern_probe_batch.py \
  --output-root runs/pattern_probe_smoke \
  --conditions null_blank mandelbrot julia checkerboard voronoi quasicrystal white_noise blue_noise \
  --probe-seeds 0 1 2 \
  --frames 12 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-cache-policy isolated \
  --overwrite
```

This runner writes one manifest per condition, one run JSON per
condition/probe-seed pair, all pairwise comparisons, and a top-level
`paired_stochastic_analysis.md/json`. It keeps the open-ended probe RNG paired
across conditions, so text mobility and source-cache summary separation remain
separate readouts.

To test whether geometric-control distances survive independent generated
variants, repeat the same condition set with different `--stimulus-seed` values:

```bash
for stimulus_seed in 7 8 9; do
  python3 scripts/run_mlx_pattern_probe_batch.py \
    --output-root "runs/pattern_geometry_variants_seed_${stimulus_seed}" \
    --conditions null_blank mandelbrot julia checkerboard voronoi quasicrystal \
    --probe-seeds 0 \
    --stimulus-seed "${stimulus_seed}" \
    --frames 12 \
    --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --temperature 0 \
    --probe-temperature 0.7 \
    --probe-cache-policy isolated \
    --overwrite
done
```

Here, `julia` and `mandelbrot` remain fixed while the generated controls vary.
That makes it a first check of whether Julia's proximity to geometric controls
depends on one control seed.

## Manifest Batch Runner

For transformed controls, use the manifest batch runner instead of hand-wiring
one `run_mlx_stream_probe.py` command per manifest:

```bash
python3 scripts/run_mlx_manifest_probe_batch.py \
  --output-root runs/frequency_cutoff_sweep_probe_seed_0 \
  --manifest mandelbrot_low_018=runs/frequency_ablation_smoke/mandelbrot_low_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest mandelbrot_high_018=runs/frequency_ablation_smoke/mandelbrot_high_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest julia_low_018=runs/frequency_ablation_smoke/julia_low_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --manifest julia_high_018=runs/frequency_ablation_smoke/julia_high_pass_luminance_quantile_matched_cutoff_018/manifest.json \
  --probe-seeds 0 \
  --max-frames 12 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-cache-policy isolated
```

The runner writes one run JSON per `KEY=PATH`, all pairwise comparisons for
each probe seed, `manifest_batch_summary.md/json`, and
`paired_stochastic_analysis.md/json`. It reuses completed run JSONs unless
`--overwrite` is passed, so longer cutoff sweeps can be resumed.

## Image Statistics

Before interpreting trace-summary clusters, measure the visual manifests
directly:

```bash
python3 scripts/analyze_image_stats.py \
  --manifest runs/pattern_probe_smoke/stimuli/null_blank/manifest.json \
  --manifest runs/pattern_probe_smoke/stimuli/mandelbrot/manifest.json \
  --manifest runs/pattern_probe_smoke/stimuli/julia/manifest.json \
  --manifest runs/pattern_probe_smoke/stimuli/checkerboard/manifest.json \
  --manifest runs/pattern_probe_smoke/stimuli/voronoi/manifest.json \
  --manifest runs/pattern_probe_smoke/stimuli/quasicrystal/manifest.json \
  --output-md runs/image_stats/pattern_probe_smoke.md \
  --output-json runs/image_stats/pattern_probe_smoke.json
```

The report includes luminance mean/std, luminance entropy, edge density,
edge-strength mean, spectral centroid, high-frequency energy ratio,
colorfulness, and mean absolute luminance change between adjacent frames. These
metrics are controls for image-level variation; they are not model-state
measurements.
