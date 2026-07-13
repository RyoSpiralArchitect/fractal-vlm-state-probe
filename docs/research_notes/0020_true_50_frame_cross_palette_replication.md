# True 50-Frame Cross-Palette Replication

Date: 2026-06-24

> **Protocol audit update (2026-07-13):** [Note 0027](0027_cache_prefix_audit_and_direct_full_vocab.md)
> found that MLX-VLM `0.4.4` did not retain the full multimodal token prefix
> across incremental turns or text-only probe branches. The input/processor
> analyses in this note remain valid, but persistent-cache and branched-readout
> interpretations below are withdrawn pending a valid rerun.

This note records the first true 50-frame cross-palette replication, the
first-token top-k readout instrumentation, and the first readout-enabled rerun.

The goal was to revisit the 12-frame cross-palette smoke from
[0019](0019_cross_palette_replication_readout.md) with source manifests that
actually contain 50 frames, not just `--max-frames 50` applied to shorter
manifests.

## Run Scope

Local run root:

```text
runs/cross_palette_replication_50_v1/
```

The run used two source-pair replications:

| Pair | MM frames | JJ frames | MJ frames | JM frames |
| --- | ---: | ---: | ---: | ---: |
| `c_d_50f` | 50 | 50 | 50 | 50 |
| `b_c_50f` | 50 | 50 | 50 | 50 |

The local summary is:

```text
runs/cross_palette_replication_50_v1/true_50_frame_cross_palette_replication_v1_summary.md
```

## Surface Readout

The forced-choice surface readout remained quiet.

For both source pairs, the paired stochastic summary reported pairwise lexical
condition distance `0.000` at before, mid, and after. This repeats the earlier
pattern: visible forced-choice labels are not the main measured object.

## Cache-Summary Replication

The scalar cache-interaction argmax locations repeated under true 50-frame
inputs.

| Pair | Mid scalar argmax | Value | After scalar argmax | Value |
| --- | --- | ---: | --- | ---: |
| `c_d_50f` | layer 23 `values` | 6.376 | layer 0 `keys` | -4.582 |
| `b_c_50f` | layer 23 `values` | 4.436 | layer 0 `keys` | 8.391 |

The sign should not be over-read because the saved summary collapses vector
direction. The replicated part is the locus:

- mid phase: late-layer value-cache norm,
- after phase: first-layer key-cache norm.

Effect magnitudes were smaller than the 12-frame smoke, but the locations
persisted across both source pairs.

## Processor-Space Image Statistics

Processor-space statistics again helped as an input audit, but did not provide
a single scalar mechanism.

| Pair | Top processor interaction | Value | Spatial frequency main effect |
| --- | --- | ---: | ---: |
| `c_d_50f` | `spectral_centroid_cycles_per_patch` | -0.241 | 0.265 |
| `b_c_50f` | `mean_abs_tensor_delta_from_previous` | -0.030 | 0.191 |

The frequency organization remains visible as a spatial main effect in both
pairs. The interaction field remains pair-dependent.

## Readout Instrumentation

The MLX run path now stores top-k logprobs for generated tokens when
`mlx_vlm.stream_generate` exposes them:

```text
generation.steps[].top_logprobs
generation.steps[].token_logprob
```

The manifest runner accepts:

```bash
--generation-readout-top-k 10
```

The saved-run analyzer compares first generated-token readouts over the same
2x2 cells:

```bash
python3 scripts/analyze_probe_readout_contrast.py \
  --mm runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/mm_mlx.json \
  --jj runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/jj_mlx.json \
  --mj runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/mj_mlx.json \
  --jm runs/cross_palette_replication_50_v1/c_d_50f/manifest_probe_seed_0/probe_seed_0/jm_mlx.json \
  --output-json runs/cross_palette_replication_50_v1/c_d_50f/probe_readout_contrast.json \
  --output-md runs/cross_palette_replication_50_v1/c_d_50f/probe_readout_contrast.md
```

Older run JSONs, including the first true 50-frame run above, do not contain
`generation.steps`. The run was therefore repeated with
`--generation-readout-top-k 20` under:

```text
runs/cross_palette_replication_50_v1_readout_topk20/
```

The compact tracked summary is:

```text
examples/research_notes/0020_true_50_frame_cross_palette_replication/readout_topk20_summary.json
```

## Readout Rerun

The readout-enabled rerun reproduced the cache-summary locus:

| Pair | Mid scalar argmax | Value | After scalar argmax | Value |
| --- | --- | ---: | --- | ---: |
| `c_d_50f` | layer 23 `values` | 6.376 | layer 0 `keys` | -4.582 |
| `b_c_50f` | layer 23 `values` | 4.436 | layer 0 `keys` | 8.391 |

But the first-token top-k readout was almost perfectly quiet:

| Pair | First-token top-20 sets | Max common-token interaction |
| --- | --- | ---: |
| `c_d_50f` | identical across all `MM/JJ/MJ/JM` phase/probe records | 0.004 |
| `b_c_50f` | identical across all `MM/JJ/MJ/JM` phase/probe records | 0.004 |

For every phase/probe record, mean top-k Jaccard was `1.000` and the count of
common top-k tokens was `20/20`. The generated first tokens also remained
condition-identical within each phase/probe record:

- before family probe: `C`,
- before frequency probe: `L`,
- mid family probe: `A`,
- mid frequency probe: `H`,
- after family probe: `A`,
- after frequency probe: `L`.

This does not prove the full vocabulary distribution is identical. It does show
that the current forced-choice first-token top-20 readout is largely insensitive
to the repeated cache-summary interaction locus.

## Current Reading

The strongest current claim is:

> In two true 50-frame cross-palette source-pair replications, forced-choice
> surface labels and first-token top-20 readouts stayed effectively fixed while
> the scalar cache-summary interaction argmax repeated at mid layer 23 `values`
> and after layer 0 `keys`.

This strengthens the state-geometry replication spine and weakens a simple
"surface readout drift" explanation. It still does not show that a single
processor-space image statistic is the mechanism, nor does it rule out
full-vocabulary or teacher-forced readout differences below the top-20 slice.

## Next Step

The next readout step is no longer top-k overlap. It is either:

1. full-vocabulary first-token divergence or teacher-forced scoring, or
2. a targeted cache-swap intervention at the repeated loci.

The top-k result makes cache-swap more interesting, because the measured cache
interaction appears real in the saved trace summaries while the current
forced-choice readout head remains nearly unchanged.
