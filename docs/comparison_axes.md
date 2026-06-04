# Comparison Axes

The central design choice is to compare visual conditions inside the same model.
That keeps model scale, tokenizer, prompting style, and provider behavior fixed
while the visual stream changes.

## Stimulus Ladder

1. **Probe-only vs stream**
   Tests whether any frame-indexed stream context changes later probes beyond a
   clean baseline.

2. **Text-only stream vs visual stream**
   Tests whether the shift is explainable by transcript length, timecodes, and
   synchronization text without image tokens.

3. **Blank visual stream vs visual stream**
   Tests whether generic image-token/cache growth differs from structured
   visual content.

4. **Fractal ordered vs fractal shuffled**
   Tests temporal order while preserving frame content.

5. **Fractal ordered vs static repeat**
   Tests evolving context against repeated exposure to the same visual object.

6. **Fractal A vs Fractal B**
   Tests whether an effect is tied to a particular equation, zoom path, or color
   field rather than "fractal" as a broad label.

7. **Fractal vs non-fractal geometry**
   Uses low-semantic visual structure such as Lissajous curves, spirals,
   kaleidoscopes, Voronoi fields, tilings, quasicrystals, or cellular automata.

8. **Fractal vs low-level controls**
   Uses phase scrambling, white noise, blue noise, random dots, static repeat,
   shuffled order, and reversed order to test whether lower-level image
   statistics or temporal structure can explain the measured trace separation.

9. **Fractal vs natural texture or landscape**
   Introduces real-world image statistics without necessarily adding strong
   object narratives.

10. **Fractal vs semantic-rich video**
   Uses cats, streets, rooms, faces-with-care, vehicles, or other recognizable
   scenes. This is useful but interpretively noisy because semantic content can
   dominate the probe.

## Condition Metadata

Each stimulus condition should declare:

```json
{
  "condition_id": "mandelbrot_zoom_a",
  "condition_family": "fractal",
  "temporal_policy": "ordered",
  "semantic_load": "low",
  "deterministic": true,
  "source_kind": "generated",
  "comparison_role": "primary_fractal"
}
```

The same metadata shape is used for external frame sequences. Natural videos can
be extracted into frames by any tool, then imported through
`scripts/build_external_frame_manifest.py`.

## Matching Rules

Prefer matched:

- frame count,
- resolution,
- frame rate or sampling interval,
- probe prompts,
- cache policy,
- generation length,
- temperature,
- model checkpoint,
- adapter version.

When a natural-video condition cannot be deterministic, preserve provenance:
source, extraction settings, frame hashes, and any cropping or resizing choices.

## Interpretation Order

Start with the least semantic comparisons first. A fractal-vs-cat result may be
interesting, but it does not isolate fractal structure. A fractal-vs-geometric
or ordered-vs-shuffled result is usually more interpretable.
