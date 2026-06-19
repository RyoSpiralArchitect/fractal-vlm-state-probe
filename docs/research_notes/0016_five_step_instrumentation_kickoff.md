# Five-Step Instrumentation Kickoff

Date: 2026-06-19

This note turns the post-cutoff-sweep discussion into a concrete instrumentation
ladder. The goal is not to claim that a visual stream has a single stable
meaningful effect. The goal is to isolate which parts of the visual input and
which parts of the readout path carry the measured trace differences.

## Five Steps

1. **Processor-space image statistics.** Recompute image statistics after the
   model processor has resized, normalized, tiled, or otherwise converted frames
   into `pixel_values`.
2. **Cross-family palette matching.** Build controls where one source supplies
   spatial luminance rank and another source supplies the exact frame-aligned
   RGB pixel multiset.
3. **Logit readout metrics.** Compare first-token logits, top-k overlap,
   probability divergence, and teacher-forced logprob deltas for fixed probes.
4. **Cache-swap intervention.** Treat source-cache state as an intervention
   target: stream condition A, swap in condition B cache at probe time, then
   measure readout movement.
5. **Generalization axes.** Allocate compute to independent source instances,
   order controls, processor/model families, and natural/geometric controls
   before increasing isolated probe seeds that cannot change source-cache state.

## Implemented In This Pass

### Processor-space statistics

New module:

```bash
python3 scripts/analyze_processor_image_stats.py \
  --manifest runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json \
  --manifest runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --patch-size 14 \
  --max-frames 50 \
  --output-json runs/processor_stats/source_variant_processor_stats.json \
  --output-md runs/processor_stats/source_variant_processor_stats.md
```

The report includes normalized spectral centroid plus architecture-aware
`cycles_per_patch` summaries:

- `spectral_centroid_cycles_per_patch`
- `energy_ratio_above_half_cycle_per_patch`
- `energy_ratio_above_one_cycle_per_patch`

These metrics answer a narrower question than raw image FFTs: what frequency
content survived the actual processor path that the model saw?

### Cross-family palette controls

New control mode:

```bash
python3 scripts/generate_control_frames.py \
  --kind cross_palette_luminance_matched \
  --source-manifest runs/source_variant_smoke/stimuli/julia_d/manifest.json \
  --palette-manifest runs/source_variant_smoke/stimuli/mandelbrot_c/manifest.json \
  --max-frames 50 \
  --condition-id julia_d_spatial_mandelbrot_c_palette \
  --output runs/cross_palette_controls/julia_d_spatial_mandelbrot_c_palette \
  --overwrite
```

The transform preserves the palette manifest's exact RGB pixel multiset within
each aligned frame while using the source manifest's luminance-rank spatial
ordering. In plain terms: it can make "Julia structure with Mandelbrot palette"
and the reverse. This is a stronger control against the objection that the
cache-distance split is just brightness, contrast, or color histogram.

### First-step readout comparison

`compare-runs` now reads the first generation step's saved top-k logprobs when
the run contains them, as HF runs do through `output_scores=True`.

The comparison reports:

- top-k Jaccard distance,
- whether the top-1 token id matched,
- max and mean absolute logprob deltas over shared top-k token ids.

This is not a full-vocabulary KL/JS measurement. It is a compact readout probe
for the immediate question: are two conditions choosing the same visible token
from measurably different local candidate distributions?

## What This Does Not Yet Prove

- Cross-family palette matching only fixes marginal RGB pixel multisets at the
  frame level. It does not equalize all local texture statistics.
- Processor-space statistics audit what reaches `pixel_values`; they still do
  not tell us which features the model uses.
- Full-vocabulary KL/JS, teacher-forced logprob deltas, and cache-swap
  measurements still need to be wired into the MLX/HF runners before they can
  carry stronger evidence.

## Next Batch Shape

The next compact batch should use explicit condition IDs and four streams:

1. `mandelbrot_c`
2. `julia_d`
3. `mandelbrot_c_spatial_julia_d_palette`
4. `julia_d_spatial_mandelbrot_c_palette`

Then run:

- raw image stats,
- processor image stats,
- the existing manifest probe batch,
- image/cache correlation,
- forced-choice probes through the HF path when top-logprob readout is needed.

The strongest near-term read would be:

> If source-cache distances continue to follow spatial source identity after
> frame-level RGB multisets are swapped, the current evidence moves from a
> low-level palette confound toward an input-organization hypothesis.

That wording is still deliberately cautious: it says the control strengthens an
interpretation, not that it closes the causal story.
