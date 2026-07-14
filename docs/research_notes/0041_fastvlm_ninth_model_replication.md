# Research Note 0041: FastVLM Ninth-Model Replication

Date: 2026-07-14

Status: completed for the balanced four-source-pair, one-frame FastVLM core.

## Question

The balanced direct core previously covered eight local VLM architectures.
This experiment asks whether a smaller ninth model can satisfy the same
artifact contracts and whether its source-cache factorial directions repeat
over the established `b_c`, `c_d`, `d_e`, and `e_f` fractal source pairs.

## Model And Fixed Targets

The model is `mlx-community/FastVLM-0.5B-bf16`, loaded with MLX `0.31.1` and
MLX-VLM `0.4.4`. Its language cache has 24 layers, two KV heads, and head
dimension 64.

Three targets were fixed by architecture depth before inspecting any
factorial effect:

| Target | Selection rule |
| --- | --- |
| layer 1 `keys` | early depth |
| layer 12 `keys` | middle depth |
| layer 23 `values` | final depth and opposite K/V component |

The run adds:

- 16 direct cells and 64 complete-vocabulary sidecars,
- 16 separate source-only cells and 48 tensor sidecars,
- 4 full-vocabulary factorial analyses,
- 12 full-vector factorial analyses, and
- 3 four-pair tensor-replication analyses.

No image-conditioned cache is reused.

## Cache Coordinate Contract

FastVLM uses the MLX-VLM `llava_qwen2` path. Its processor history contains one
`image_token_index=-200` placeholder at position 42. The effective cache has
397 positions rather than 142 processor tokens because the single placeholder
is replaced by 256 vision positions:

`142 - 1 + 256 = 397`

The validated map is:

| Region | Effective cache positions | Count |
| --- | --- | ---: |
| pre-image | 0-41 | 42 |
| image | 42-297 | 256 |
| post-image | 298-396 | 99 |

All 16 source cells resolve this same
`llava_qwen2_single_image_run_replacement` strategy. Raw cache allocations are
`[1, 2, 512, 64]`; saved target tensors are offset-trimmed to
`[1, 2, 397, 64]`.

Unknown model mismatches still fail closed. The new mapping applies only to the
known `llava_qwen2` model type with exactly one image-token run.

## Source-Response Boundary

FastVLM does not follow the source prompt's `Return ACK only` instruction. All
16 source-only responses are exactly `The image`. The source response is
therefore fixed across cells but must not be described as an ACK.

This does not invalidate within-model factorial comparison: every source cell
uses the same prompt protocol, generated suffix, cache layout, and tensor
shape. It does mean that comparisons with models whose suffix is `ACK` are
architecture-level descriptive comparisons, not a shared textual-suffix
intervention.

## Direct Full-Vocabulary Result

All eight before records are bitwise identical over their four cells, and all
eight after records contain non-identical complete distributions. All 64
generated probe responses are declared candidate tokens: `C` in 53 and `B` in
11. After-record candidate probability mass spans `0.6593-0.9711`.

| Pair | Family tokens / axis | Frequency tokens / axis |
| --- | --- | --- |
| `b_c` | C/B/B/C / palette | C/C/C/C / interaction |
| `c_d` | B/B/C/B / interaction | C/C/C/C / interaction |
| `d_e` | B/B/C/B / spatial | C/C/C/C / spatial |
| `e_f` | B/B/C/B / interaction | C/C/C/C / palette |

Balanced dominance over the eight after records is spatial/palette/interaction
in 2/2/4. The fixed frequency label pattern therefore hides four different
full-distribution surfaces across the four source pairs.

## Full-Vector Integrity

All 12 fixed target analyses pass sidecar hash, tensor shape, and cache-position
alignment checks:

- pre-image effects are exactly zero in 12/12,
- interaction argmaxes are image positions in 12/12,
- image interaction-energy fraction exceeds 0.9 in 12/12,
- image-region balanced dominance is spatial in 12/12, and
- image interaction share is at or below `1/3` in 12/12.

The scalar interaction is therefore large and image-localized without becoming
the dominant equal-coefficient full-vector axis.

## Four-Pair Direction Replication

Every one of the six pairwise cosines is positive at every selected target in
both image and post-image regions:

| Target | Image cosine median (range) | Post-image cosine median (range) |
| --- | --- | --- |
| L1 `keys` | 0.188 (0.119-0.201) | 0.424 (0.249-0.484) |
| L12 `keys` | 0.198 (0.123-0.221) | 0.546 (0.262-0.694) |
| L23 `values` | 0.133 (0.091-0.156) | 0.593 (0.469-0.641) |

The post-image direction becomes more aligned with depth over these three fixed
targets, while the image direction remains modestly positive. This is a
descriptive four-pair profile, not a selected-layer maximum or a test over all
24 layers.

## Revised Aggregate

FastVLM expands the independent one-frame direct core to nine models x four
source pairs. Together with the generator-pairing extension in Note 0040, the
valid direct surface now contains 104 factorials, 416 cell runs, and 1,664
full-vocabulary sidecars.

The selected full-vector surface now contains 360 source-only cell runs, 1,064
tensor sidecars, and 266 analyses. Balanced spatial/palette/interaction
dominance is 227/32/7. Among 254 analyses with resolved token partitions:

- pre-image interaction is zero in 254/254,
- interaction argmax is an image position in 248/254, and
- image interaction-energy fraction exceeds 0.9 in 249/254.

Image interaction share is at or below `1/3` in 252/266 selected analyses.

## Claim Boundaries

- FastVLM source-only responses are `The image`, not `ACK`.
- The three tensor targets are fixed depth controls, not an exhaustive
  layerwise search.
- Four fractal source pairs do not establish transfer to non-fractal inputs.
- Different models do not share a directly comparable vector basis, token
  layout, KV-head count, or textual suffix.
- Direct readouts and source tensors come from separate fresh forwards; no
  cache-to-readout causal path is tested.
- No result establishes persistence, adaptation, subjective state, a universal
  layer, or semantic steering.

## Highest-Value Next Experiments

1. Run the same fixed FastVLM targets over the generator-pairing hierarchy.
2. Add prompt-order and verbalizer controls before treating its frequent `C`
   response as a stable semantic readout.
3. Sweep all 24 layers descriptively, then confirm any selected locus on held-
   out source pairs.
4. Resolve direction by the two KV heads and by the 16 x 16 image-position
   grid implied by 256 vision features.

## Manuscript-Safe Statement

> A ninth local architecture, FastVLM-0.5B, reproduced non-identical complete
> first-step distributions across four fresh `MM/JJ/MJ/JM` factorials. Three
> architecture-fixed source-cache targets had zero pre-image effects,
> image-localized interaction maxima, spatial-dominant full-vector energy, and
> positive four-pair direction cosine in image and post-image regions. Its
> source response was consistently `The image` rather than the requested
> `ACK`, so the result is a within-model fresh-forward replication, not a
> shared-suffix or persistent-state claim.

## Primary Artifacts

- `runs/fastvlm_expansion/contract_smoke/`
- `runs/fastvlm_expansion/four_pair_1f_direct_seed0/`
- `runs/fastvlm_expansion/four_pair_1f_source_cache_three_target_seed0/`
- `runs/fastvlm_expansion/analyses/`
- `runs/fastvlm_expansion/replication/`
- `examples/research_notes/0041_fastvlm_ninth_model/summary.json`
