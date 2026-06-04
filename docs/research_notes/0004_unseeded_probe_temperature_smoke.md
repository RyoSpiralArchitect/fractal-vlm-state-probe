# Unseeded Probe-Temperature Smoke

Date: 2026-06-04

## Setup

- Source manifests: `runs/null_fractal_50_seed_batch/stimuli/*/manifest.json`
- Conditions: `blank_visual_null`, `mandelbrot_zoom_a`, `julia_zoom_b`
- Frames per run: 12
- Model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Stream generation: greedy, `temperature=0`, `max_tokens=2`
- Probe generation: sampled, `probe_temperature=0.7`, `probe_max_tokens=80`
- Seed: omitted

This smoke separates stream-turn temperature from probe temperature. The stream
context is still greedy, but the before/mid/after probes are allowed to sample.

## First Reading

The sampled probe outputs moved visibly across conditions. However, the clean
`before` probes also differed across conditions because the runs were unseeded
and sampled independently. Therefore this smoke should not be interpreted as a
paired condition effect.

The useful observation is simpler:

> Once probe sampling is enabled and the seed is omitted, the surface text
> becomes highly mobile again.

This contrasts with the deterministic 50-frame batch, where all three conditions
produced identical probe text while cache-summary traces remained separable.

## Interpretation Boundary

Unseeded sampled text is useful for phenomenology and for discovering prompt
families that are sensitive to stream context. It is not, by itself, a clean
condition comparison. A stronger stochastic design needs repeated sampled probes
per condition, plus either matched RNG seeds for comparable probe draws or a
distributional analysis over many samples.

## Local Artifacts

- `runs/unseeded_output_smoke/null_blank_probe_temp_0_7.json`
- `runs/unseeded_output_smoke/mandelbrot_probe_temp_0_7.json`
- `runs/unseeded_output_smoke/julia_probe_temp_0_7.json`
- `runs/unseeded_output_smoke/null_vs_mandelbrot.md`
- `runs/unseeded_output_smoke/null_vs_julia.md`
- `runs/unseeded_output_smoke/mandelbrot_vs_julia.md`
