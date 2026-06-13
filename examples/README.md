# Examples

Tracked example artifacts for the research notes.

The live `runs/` directory is intentionally ignored because it can contain
large model outputs, frame copies, and machine-specific paths. This directory
keeps lightweight JSON snapshots that make the note trail easier to inspect
from a fresh checkout.

These examples are not benchmark fixtures. They are curated pilot artifacts for
reading the current experiment logic and claim boundaries.

## Research Note Snapshots

| Note | Example JSON | What To Read |
| --- | --- | --- |
| [0001 Mandelbrot Smoke](../docs/research_notes/0001_mandelbrot_smoke_observation.md) | [mandelbrot_vs_julia.json](research_notes/0001_mandelbrot_julia_smoke/mandelbrot_vs_julia.json) | Early paired comparison shape after the first Mandelbrot/Julia smoke path existed. |
| [0002 Null vs Stream](../docs/research_notes/0002_null_vs_stream_smoke.md) | [probe_only_vs_visual.json](research_notes/0002_null_vs_stream/probe_only_vs_visual.json), [text_only_vs_visual.json](research_notes/0002_null_vs_stream/text_only_vs_visual.json), [blank_visual_vs_visual.json](research_notes/0002_null_vs_stream/blank_visual_vs_visual.json) | Delivery-mode controls before fractal-family interpretation. |
| [0003 Cache Classifier](../docs/research_notes/0003_cache_summary_condition_classifier.md) | [cache_classifier.json](research_notes/0003_cache_summary_classifier/cache_classifier.json), [batch_summary.json](research_notes/0003_cache_summary_classifier/batch_summary.json) | Cache-summary separability while surface text is fixed. |
| [0004 Unseeded Probe Temperature](../docs/research_notes/0004_unseeded_probe_temperature_smoke.md) | [null_vs_mandelbrot.json](research_notes/0004_unseeded_probe_temperature/null_vs_mandelbrot.json), [null_vs_julia.json](research_notes/0004_unseeded_probe_temperature/null_vs_julia.json), [mandelbrot_vs_julia.json](research_notes/0004_unseeded_probe_temperature/mandelbrot_vs_julia.json) | Unpaired sampled probe outputs; useful mobility, not a paired condition effect. |
| [0006 Paired Stochastic Probe](../docs/research_notes/0006_paired_stochastic_probe_smoke.md) | [paired_stochastic_analysis.json](research_notes/0006_paired_stochastic_probe/paired_stochastic_analysis.json), [paired_stochastic_batch_summary.json](research_notes/0006_paired_stochastic_probe/paired_stochastic_batch_summary.json) | Matched probe-seed surface text and source-cache summary separation. |
| [0007 Paired Stochastic 50-Seed](../docs/research_notes/0007_paired_stochastic_probe_50_seed.md) | [paired_stochastic_analysis.json](research_notes/0007_paired_stochastic_probe_50_seed/paired_stochastic_analysis.json), [paired_stochastic_batch_summary.json](research_notes/0007_paired_stochastic_probe_50_seed/paired_stochastic_batch_summary.json) | Fifty matched probe seeds: surface text remains seed-locked while source-cache summaries separate conditions. |
| [0008 Pattern Probe Smoke](../docs/research_notes/0008_pattern_probe_smoke.md) | [pattern_batch_summary.json](research_notes/0008_pattern_probe_smoke/pattern_batch_summary.json), [paired_stochastic_analysis.json](research_notes/0008_pattern_probe_smoke/paired_stochastic_analysis.json) | First non-fractal pattern-control batch; Julia is closer to generated geometry than to null or Mandelbrot in this pilot trace summary. |
| [0009 Stimulus Seed 8 Variant](../docs/research_notes/0009_stimulus_seed_8_variant_smoke.md) | [pattern_batch_summary.json](research_notes/0009_stimulus_seed_8_variant/pattern_batch_summary.json), [paired_stochastic_analysis.json](research_notes/0009_stimulus_seed_8_variant/paired_stochastic_analysis.json) | One generated-control seed variant; Julia remains geometry-leaning, but its nearest generated neighbor changes. |
| [0010 Phase Scramble Image Stats](../docs/research_notes/0010_phase_scramble_image_stats.md) | [pattern_phase_scramble_stats.json](research_notes/0010_phase_scramble_image_stats/pattern_phase_scramble_stats.json), [julia_vs_julia_phase_scrambled.json](research_notes/0010_phase_scramble_image_stats/julia_vs_julia_phase_scrambled.json), [mandelbrot_phase_vs_julia_phase.json](research_notes/0010_phase_scramble_image_stats/mandelbrot_phase_vs_julia_phase.json) | Phase scrambling moves both fractals toward a shared high-entropy/noise-like trace neighborhood in this pilot. |
| [0011 Quantile-Matched Phase Scramble](../docs/research_notes/0011_quantile_matched_phase_scramble.md) | [phase_scramble_quantile_stats.json](research_notes/0011_quantile_matched_phase_scramble/phase_scramble_quantile_stats.json), [quantile_comparison_summary.json](research_notes/0011_quantile_matched_phase_scramble/quantile_comparison_summary.json) | Luminance-rank quantile matching fixes the entropy confound introduced by plain phase scrambling, then changes the first source-cache summary read in an asymmetric way. |
| [0012 Frequency Ablation Smoke](../docs/research_notes/0012_frequency_ablation_smoke.md) | [frequency_ablation_stats.json](research_notes/0012_frequency_ablation_smoke/frequency_ablation_stats.json), [frequency_ablation_comparison_summary.json](research_notes/0012_frequency_ablation_smoke/frequency_ablation_comparison_summary.json) | Entropy-fixed high-pass controls collapse Mandelbrot/Julia closer together, while low-pass controls remain separated in this single-cutoff pilot. |
| [0013 Frequency Cutoff Sweep](../docs/research_notes/0013_frequency_cutoff_sweep.md) | [frequency_cutoff_sweep_stats.json](research_notes/0013_frequency_cutoff_sweep/frequency_cutoff_sweep_stats.json), [cutoff_sweep_core_comparison_summary.json](research_notes/0013_frequency_cutoff_sweep/cutoff_sweep_core_comparison_summary.json), [manifest_batch_summary.json](research_notes/0013_frequency_cutoff_sweep/manifest_batch_summary.json) | Cutoff sweep confirms a low-vs-high frequency split, strongest around middle cutoffs, using the arbitrary manifest batch runner. |

## Reading Order

1. Read the note first.
2. Open the linked example JSON for the machine-readable evidence shape.
3. Treat paths inside JSON as provenance from the original local `runs/`
   directory, not as guaranteed files in a clean checkout.
