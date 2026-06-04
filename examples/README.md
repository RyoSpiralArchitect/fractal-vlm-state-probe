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

## Reading Order

1. Read the note first.
2. Open the linked example JSON for the machine-readable evidence shape.
3. Treat paths inside JSON as provenance from the original local `runs/`
   directory, not as guaranteed files in a clean checkout.
