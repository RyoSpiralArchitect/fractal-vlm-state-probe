# Research Note 0029: Cross-Model Prompt Audit and InternVL Expansion

Date: 2026-07-13

## Status

Completed as two extensions of the fresh-forward protocol established in Note
0027:

1. the `e_f` prompt-robustness audit now covers Gemma 3, Qwen2.5-VL, and
   SmolVLM2, and
2. the four independent one-frame source pairs now include a fourth
   architecture, InternVL3-2B.

The valid standard matrix now contains 30 `MM/JJ/MJ/JM` factorial points, 120
cell runs, and 480 complete first-step vocabulary sidecars across four VLMs.
The independent one-frame surface is four source pairs x four models. The
separate prompt audit contains 12 cell runs and 192 sidecars across three
models.

## Protocol Boundary

Every standard cell uses:

- one selected image,
- seed 0 and temperature 0,
- one fresh ordered multimodal ACK forward for the source-cache summary,
- separate fresh direct multimodal forwards for family and frequency probes,
- all available cache-summary layers, and
- the complete first-step vocabulary distribution.

No image-conditioned cache is reused between the ACK forward and either direct
probe. The cache and readout measurements remain related condition-level
observations from separate forwards, not a causal chain.

## Three-Model Prompt Audit

Qwen and SmolVLM were run over the same `e_f` four-cell audit previously run on
Gemma 3. Each model received baseline, paraphrase, reversed candidate order,
and rotated token-to-semantics variants for both family and frequency probes.
Candidate probabilities were aligned by declared meaning before comparison.

The audit is deterministic. For each model, all 16 baseline before/after
sidecar hashes match the corresponding prior standard `e_f` run. Across all
three models the baseline replication is therefore 48/48 exact SHA-256
matches.

| Model | Probe | Changed generated variants | Max semantic TV | Baseline semantic interaction L1 | Interaction L1 range |
| --- | --- | ---: | ---: | ---: | --- |
| Gemma 3 | family | 1/3 | 0.9998 | 0.000247 | 0.00000209-1.682 |
| Gemma 3 | frequency | 3/3 | 0.9999 | 2.172 | 0.001318-3.084 |
| Qwen2.5-VL | family | 1/3 | 0.5005 | 0.3978 | 0.2464-0.4642 |
| Qwen2.5-VL | frequency | 1/3 | 0.5988 | 0.06029 | 0.03300-0.1987 |
| SmolVLM2 | family | 2/3 | 0.1821 | 0.1122 | 0.06129-0.1283 |
| SmolVLM2 | frequency | 3/3 | 0.3167 | 0.09867 | 0.04927-0.1351 |

The family baseline generates Mandelbrot for all four cells in all three
models, yet its semantic interaction differs by more than three orders of
magnitude between Gemma and Qwen. Visible agreement therefore does not imply
either probability agreement or equal factorial structure.

Prompt sensitivity is also architecture-dependent rather than one shared
failure mode:

- Gemma can move almost the entire semantic probability mass when candidate
  order changes.
- Qwen keeps several generated patterns fixed while its semantic distributions
  still move by total variation above 0.5 across variants.
- SmolVLM has smaller probability movement, but more generated-label changes,
  especially for the frequency probe.

No frequency variant has one generated semantic pattern shared by all three
models. The readout surface is jointly calibrated by image, prompt wording,
candidate order, token mapping, and architecture.

## Adding InternVL3-2B

The fourth architecture is `mlx-community/InternVL3-2B-4bit`, local snapshot
`e474b39487e563dbfa12c12e6a7e7d743cf340d4`.

The installed MLX-VLM `0.4.4` and Transformers combination did not load this
model's custom processor correctly. The automatic path rejected the custom
InternVL image processor and then passed list-valued multimodal content to a
string-oriented chat template. A narrow compatibility adapter now:

1. reconstructs MLX-VLM's InternVL processor with its custom dynamic image
   expansion,
2. normalizes multimodal messages to InternVL's `<image>` string form, and
3. records `internvl_chat_custom_image_expansion` in run metadata.

This is a local compatibility boundary, not an upstream processor fix. Its
behavior was tested both at the adapter level and by complete image forwards.

## InternVL Integrity Audit

The one-cell smoke and all 16 factorial cells completed successfully. Across
the full four-pair batch:

- 16/16 run JSONs are present,
- 64/64 distinct sidecar paths are present and pass their recorded hashes,
- every sidecar contains 151,674 logprobs with no NaN or positive infinity,
- every source cache reports all 28/28 language layers,
- every selected image expands to one contiguous run of 3,328 image tokens,
- every ACK cache contains 3,471 tokens, and
- all four direct after-factorials contain non-identical cell distributions.

The large image-token count is model-processor behavior. It should not be
compared as if it were the same replay length or tokenization as Qwen, SmolVLM,
or Gemma.

## InternVL Cache Result

| Pair | Scalar argmax | Interaction | Relative abs interaction | Normalized depth |
| --- | --- | ---: | ---: | ---: |
| `b_c` | layer 25 `values` | -88.960 | 0.04425 | 0.9259 |
| `c_d` | layer 26 `values` | -156.665 | 0.06760 | 0.9630 |
| `d_e` | layer 27 `values` | -146.184 | 0.06227 | 1.0000 |
| `e_f` | layer 27 `values` | -76.290 | 0.03233 | 1.0000 |

InternVL does not reproduce one exact layer across all pairs. It does reproduce
a narrower component/sign/depth result: 4/4 maxima are `values`, 4/4 have a
negative interaction, and all lie in normalized depth 0.926-1.0. The exact
location mode is layer 27 `values` in 2/4 pairs.

This resembles Qwen's late negative `values` result at a coarse normalized
level, but it is weaker than Qwen's exact 4/4 layer-33 replication. It also
contrasts with Gemma's early `values` band and SmolVLM's broad layer/component
movement.

## InternVL Readout Result

| Pair | Family generated (MM/JJ/MJ/JM) | Family max JS | Family interaction L1 | Frequency generated | Frequency max JS | Frequency interaction L1 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `b_c` | M/M/M/M | 0.05059 | 0.6269 | U/U/U/U | 0.005610 | 0.2806 |
| `c_d` | M/M/U/M | 0.15018 | 0.6246 | U/U/U/U | 0.02438 | 0.4167 |
| `d_e` | M/M/M/M | 0.01716 | 0.3962 | U/U/U/U | 0.01641 | 0.1122 |
| `e_f` | M/M/M/M | 0.10055 | 0.3338 | U/U/U/U | 0.002362 | 0.06535 |

Here M means Mandelbrot and U means unclear. The family interaction L1 median
is 0.5104 (range 0.3338-0.6269); the frequency median is 0.1964 (range
0.06535-0.4167). Only the `c_d` MJ cell changes the visible family label, while
all complete distributions remain cell-dependent.

## Four-Model Reading

The one-frame cache surface now separates into four architecture profiles:

| Model | Exact location mode | Exact share | Component share | Sign share | Normalized depth range |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen2.5-VL | layer 33 `values` | 4/4 | 4/4 `values` | 4/4 negative | 0.943-0.943 |
| InternVL3 | layer 27 `values` | 2/4 | 4/4 `values` | 4/4 negative | 0.926-1.000 |
| Gemma 3 | layer 3 `values` | 2/4 | 4/4 `values` | 3/4 positive | 0.000-0.091 |
| SmolVLM2 | layer 1 `keys` | 2/4 | 3/4 `keys` | 3/4 positive | 0.043-0.957 |

The new result strengthens a model-conditional reading. There is no universal
layer, but there may be architecture-specific depth/component regimes worth
testing: two models are late-negative-`values`, one is early-`values`, and one
is broadly pair-dependent. Four models are enough to expose that candidate
structure, not enough to attribute it to a specific architectural cause.

The readout surface is at least as heterogeneous. Standard family labels agree
in 15/16 one-frame factorials, but full-vocabulary interaction magnitudes differ
substantially. Standard frequency labels split by model: Qwen and SmolVLM favor
low frequency, Gemma favors high/unclear patterns, and InternVL favors unclear.
The three-model prompt audit then shows that these categorical patterns are not
stable decoding targets in the first place.

## Updated Claim

The strongest manuscript-safe statement is now:

> Across four independent fractal source pairs and four local VLMs, fresh
> multimodal factorial cells produced non-identical complete first-step
> distributions. Source-cache scalar interaction loci formed distinct
> architecture profiles, including an exact late Qwen locus and a late
> InternVL band. In three models, semantically aligned prompt controls changed
> generated labels and candidate distributions in model-specific ways despite
> exact deterministic baseline replication.

## Claim Boundaries

- Four source pairs are independent visual replications; four cells and their
  six pairwise distances are not six independent samples.
- Prompt variants are repeated measurements over one source pair, not new
  visual replications.
- Cache effects are scalar summary-stat contrasts, not full-vector directions.
- Normalized depth does not make different architectures or token layouts
  mechanistically equivalent.
- The InternVL processor adapter is validated for this local model/runtime, not
  every InternVL checkpoint or MLX-VLM release.
- No result establishes persistent state, valid cache intervention, causal
  mediation, a universal layer, semantic steering, or subjective state.

## Local Artifacts

```text
runs/source_pair_expansion_50_v1/prompt_controls/cross_model_e_f_1f_forced_choice_robustness_seed0/
runs/internvl3_four_pair_1f_direct_seed0/
runs/internvl3_full_vocab_factorials/
runs/source_pair_expansion_50_v1/cross_model_four_arch_four_pair/
```

## Next Steps

1. Separate wording, candidate order, and token mapping into a balanced prompt
   factorial, then repeat it on a second source pair.
2. Add neutral scoring prompts and non-forced descriptions so the paper does
   not depend on one three-way categorical interface.
3. Save full-vector contrasts in Qwen's layer 33 and InternVL's layers 25-27,
   partitioned into image-token and non-image-token regions.
4. Add natural and synthetic geometric controls matched on processor-space
   spectral statistics.
5. Add another architecture from a different processor/token-layout family to
   test whether the late-negative-`values` grouping survives.

## Tracked Summary

```text
examples/research_notes/0029_cross_model_prompt_and_internvl_expansion/summary.json
```
