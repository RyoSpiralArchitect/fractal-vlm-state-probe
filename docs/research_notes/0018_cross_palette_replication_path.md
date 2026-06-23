# Cross-Palette Replication Path

Date: 2026-06-22

This note turns the cross-palette smoke into a replication workflow.

The immediate target is the `1 + 3` path:

1. Repeat the 2x2 cross-palette design across multiple Mandelbrot/Julia source
   pairs.
2. Carry raw image statistics and processor-space image statistics into the
   same factorial coordinate system as the cache-summary contrast.

The goal is to test whether the asymmetric interaction observed in
[0017 Cross-Palette Control Smoke](0017_cross_palette_control_smoke.md) is
source-pair specific or a repeatable property of the stimulus construction.

## Factorial Cells

For each source pair:

| Cell | Meaning |
| --- | --- |
| `MM` | Mandelbrot spatial rank x Mandelbrot palette |
| `JJ` | Julia spatial rank x Julia palette |
| `MJ` | Mandelbrot spatial rank x Julia palette |
| `JM` | Julia spatial rank x Mandelbrot palette |

The same formulas are now used for cache summaries, raw image statistics, and
processor-space image statistics:

- `spatial_main_effect = ((JM - MM) + (JJ - MJ)) / 2`
- `palette_main_effect = ((MJ - MM) + (JJ - JM)) / 2`
- `interaction_effect = JJ - JM - MJ + MM`

## New Preparation Runner

The runner prepares multiple source pairs without launching MLX inference:

```bash
python3 scripts/prepare_cross_palette_factorial_batch.py \
  --output-root runs/cross_palette_replications \
  --pair c_d=runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json,runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --pair b_c=runs/source_variant_smoke/stimuli/mandelbrot_b/manifest.json,runs/source_variant_smoke/stimuli/julia_c/manifest.json \
  --max-frames 50 \
  --processor-model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --patch-size 14 \
  --overwrite
```

For each pair, it writes:

- `MJ` and `JM` cross-palette manifests,
- raw image statistics,
- raw 2x2 factorial image contrast,
- processor-space image statistics when `--processor-model` is provided,
- processor-space 2x2 factorial image contrast,
- suggested `run_mlx_manifest_probe_batch.py` and
  `analyze_factorial_cache_contrast.py` commands.

The current source-variant manifests contain 12 frames. `--max-frames 50` is
therefore a ceiling, not a guarantee. A true 50-frame replication should start
from fresh 50-frame source configs.

## Standalone Image-Contrast Analyzer

Any saved raw or processor image-stat batch can be analyzed directly:

```bash
python3 scripts/analyze_factorial_image_contrast.py \
  --stats-json runs/cross_palette_controls/cross_palette_processor_image_stats.json \
  --mm mandelbrot_zoom_c \
  --jj julia_zoom_d \
  --mj mandelbrot_c_spatial_julia_d_palette \
  --jm julia_d_spatial_mandelbrot_c_palette \
  --output-json runs/cross_palette_controls/processor_factorial_image_contrast.json \
  --output-md runs/cross_palette_controls/processor_factorial_image_contrast.md
```

This makes the evidence layers comparable:

- raw image interaction,
- processor-space image interaction,
- cache-summary interaction.

## Reading Target

The next useful question is not "does palette or structure win?" The more
precise question is:

> Do cache-summary interaction peaks line up more consistently with raw image
> interactions, processor-space image interactions, or neither?

If the answer is "processor-space more than raw," then cutoff and palette
controls should be reported in architecture-aware units such as cycles per
patch. If the answer is "neither," the apparatus is seeing a trace behavior not
reduced by these scalar image summaries, which motivates deeper hidden-state
or cache-swap instrumentation.

## Next Run

1. Prepare at least two source pairs with the runner above.
2. Run the suggested MLX manifest-batch command for each pair.
3. Run `analyze_factorial_cache_contrast.py` for each completed pair.
4. Add a small aggregation report over:
   - top cache interaction location,
   - top raw image interaction metric,
   - top processor-space interaction metric,
   - whether the direction and rank ordering repeat across pairs.

This should come before cache-swap intervention. Cache-swap will be more useful
after we know which pair and layer/phase/probe combination carries the most
repeatable interaction signal.
