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
| `static_repeat` | One exact source frame repeated at the same cadence | Temporal evolution. |
| `shuffled` | Source frame content and count | Temporal order. |
| `reversed` | Source frame content and count | Forward temporal direction. |

Transform manifests record the source manifest hash, source condition, and
source frame index for each emitted frame. That keeps the control auditable when
later probe differences are small.

## Suggested Next Ladder

For one source stream such as Julia:

1. `blank`
2. `white_noise`
3. `blue_noise`
4. `static_repeat`
5. `shuffled`
6. `reversed`
7. `phase_scrambled`
8. `checkerboard`
9. `voronoi`
10. `quasicrystal`
11. original ordered Julia

The clean read is not "which pattern is mystical." The useful read is whether a
small set of image-statistic and temporal controls can explain the trace
separation seen in the first deterministic batch.
