# Paired Stochastic Probe Design

Date: 2026-06-05

## Motivation

The deterministic 50-frame batch showed one useful asymmetry: surface probe text
can remain fixed while cache-summary traces separate conditions. The unseeded
probe-temperature smoke showed the opposite pressure: sampled probe text becomes
mobile again, but unpaired sampling noise already moves the clean `before`
probe.

The next design keeps the mobility while restoring pairing:

> Keep the visual stream deterministic, sample probes from matched RNG seeds
> across conditions, and compare mid/after probe distributions after subtracting
> before-probe variance.

## Batch Shape

- Stream: greedy, `temperature=0`
- Probe: sampled, usually `probe_temperature=0.7`
- Stream seed: fixed across all runs
- Probe seeds: repeated series, applied to all conditions
- Probe phase seeds: `probe_seed`, `probe_seed+1`, `probe_seed+2` for
  `before`, `mid`, and `after`
- Conditions: start with `null_blank`, `mandelbrot`, and `julia`
- Frames: `12` for smoke, `50` for a stronger pass

This isolates the first stochastic question:

> Given the same probe sampling seed, does the visual stream condition shift the
> distribution of later non-visual probe text beyond the clean before-probe
> variance?

## Implemented Entry Point

```bash
python3 scripts/run_mlx_paired_stochastic_probe_batch.py \
  --output-root runs/paired_stochastic_probe_smoke \
  --probe-seeds 0 1 2 \
  --frames 12 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-max-tokens 80 \
  --probe-cache-policy isolated \
  --overwrite
```

For the fuller design, use a longer seed series:

```bash
python3 scripts/run_mlx_paired_stochastic_probe_batch.py \
  --output-root runs/paired_stochastic_probe_50_seed_batch \
  --probe-seeds $(seq 0 49) \
  --frames 50 \
  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --temperature 0 \
  --probe-temperature 0.7 \
  --probe-max-tokens 80 \
  --probe-cache-policy isolated \
  --cache-summary-every 10 \
  --cache-summary-max-layers 4 \
  --overwrite
```

The runner writes:

- `paired_stochastic_batch_summary.md`
- `paired_stochastic_batch_summary.json`
- `paired_stochastic_analysis.md`
- `paired_stochastic_analysis.json`
- per-seed run JSONs and pairwise comparison reports

## Current Analysis Scope

The first analyzer reports lexical set distances:

- condition-level `mid`/`after` drift from `before`,
- pairwise condition distances per phase,
- pairwise `mid`/`after` distances after subtracting `before` distance.
- pairwise source-cache summary deltas for `mid` and `after`.

This is deliberately modest. It is a surface-text analysis, not an embedding or
semantic-vector analysis. The next layer should add embedding distances,
lexical category counts such as color/space/body words, and cache-summary
separability on the same paired stochastic batch.
