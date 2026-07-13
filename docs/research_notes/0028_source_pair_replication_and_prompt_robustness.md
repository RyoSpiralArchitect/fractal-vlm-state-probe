# Research Note 0028: Source-Pair Replication and Prompt Robustness

Date: 2026-07-13

## Status

Completed as two linked extensions of the valid fresh-forward protocol from
Note 0027:

1. two new deterministic 50-frame source pairs, tested at one frame in all
   three local VLMs, and
2. a four-variant forced-choice prompt audit on the strongest Gemma 3
   frequency factorial.

The standard evidence matrix now contains 26 factorial points, 104 cell runs,
and 416 complete first-step vocabulary sidecars. The independent one-frame
surface contains four source pairs x three models. A separate prompt audit adds
64 sidecars and shows that the direct readout interaction is not invariant to
prompt formulation.

## Why Add Pairs Before More Frames

The previous matrix had long nested trajectories but only two independent
source pairs. Increasing replay length again would add context points without
adding independent visual replication. This extension instead adds:

- `mandelbrot_zoom_d_50f` with `julia_zoom_e_50f` (`d_e`),
- `mandelbrot_zoom_e_50f` with `julia_zoom_f_50f` (`e_f`).

Each source manifest has 50 unique frame hashes. The new sources use distinct
fractal parameters, zoom paths, and seeds. Their cross-palette input
interactions also differ materially:

| Pair | Edge-density interaction | HF-energy interaction | Spectral-centroid interaction |
| --- | ---: | ---: | ---: |
| `d_e` | 0.0641 (relative 0.486) | 0.2147 (relative 1.338) | 0.1341 (relative 0.990) |
| `e_f` | 0.0620 (relative 0.858) | -0.0453 (relative 0.249) | -0.0174 (relative 0.116) |

The model replication therefore does not merely repeat two near-identical raw
frequency interactions.

## Fresh Direct Run Matrix

The new runs use one selected image, seed 0, temperature 0, all cache-summary
layers, a fresh ordered ACK forward, and separate fresh direct multimodal
probes. No image-conditioned cache is reused.

| Model | New pairs | New cells | New full-vocab sidecars | Vocabulary |
| --- | ---: | ---: | ---: | ---: |
| SmolVLM2-2.2B | 2 | 8 | 32 | 49,280 |
| Qwen2.5-VL-3B 4bit | 2 | 8 | 32 | 151,936 |
| Gemma-3-4B-it 4bit | 2 | 8 | 32 | 262,208 |

All six new direct after-factorials have non-identical cell distributions. The
full standard matrix is now 26 factorials, 104 cells, and 416 sidecars.

## Four-Pair Cache Replication

At one frame, cache-summary scalar argmax stability separates cleanly by
architecture:

| Model | Exact argmax mode | Exact share | K/V mode | K/V share | Sign share | Normalized depth range |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Qwen2.5-VL-3B | layer 33 `values` | 4/4 | `values` | 4/4 | 4/4 negative | 0.943-0.943 |
| SmolVLM2-2.2B | layer 1 `keys` | 2/4 | `keys` | 3/4 | 3/4 positive | 0.043-0.957 |
| Gemma-3-4B | layer 3 `values` | 2/4 | `values` | 4/4 | 3/4 positive | 0.000-0.091 |

The exact pair-level locations are:

- Qwen: layer 33 `values` for `b_c`, `c_d`, `d_e`, and `e_f`.
- SmolVLM: layer 1 `keys`, layer 21 `keys`, layer 22 `values`, layer 1
  `keys`.
- Gemma 3: layer 3 `values`, layer 1 `values`, layer 0 `values`, layer 3
  `values`.

Qwen is now the only architecture with an exact scalar locus and interaction
sign replicated over four independent source pairs. Gemma has a weaker but
real component-level regularity: all four maxima are early `values`, without a
single stable layer. SmolVLM remains pair-dependent in layer, component, and
sign.

This is evidence about scalar cache-summary factorial interactions. It is not
yet evidence that full cache vectors share one direction or mechanism.

## Standard Readout Across Four Pairs

The generated family token is `A/A/A/A` for every one-frame pair in every
model, but the complete distributions are not remotely equivalent:

| Model | Family interaction L1 median (range) | Frequency interaction L1 median (range) | Frequency generated patterns |
| --- | --- | --- | --- |
| Qwen2.5-VL-3B | 0.498 (0.205-0.716) | 0.088 (0.024-0.141) | `L/L/L/L` in 4/4 pairs |
| SmolVLM2-2.2B | 0.106 (0.034-0.249) | 0.093 (0.052-0.171) | `L/L/L/L` in 4/4 pairs |
| Gemma-3-4B | 0.000759 (0.000180-0.002605) | 0.809 (0.078-2.172) | `C/H/H/H` in 2 pairs; `H/H/H/H` in 2 |

The model families therefore calibrate the same probe very differently. The
fixed family token conceals substantial Qwen and SmolVLM distribution shifts,
while Gemma's family probe is saturated and its frequency probe carries the
larger interaction.

## Prompt Robustness Design

The `e_f` Gemma factorial was rerun with eight probes:

- baseline family and frequency choices,
- semantic-preserving paraphrases,
- reversed candidate presentation order,
- rotated token-to-semantics mappings.

Every probe record stores `probe_family`, `prompt_variant`, candidate labels,
candidate order, and the explicit label-to-semantics map. Analysis aligns
probabilities by meaning before comparing variants. The baseline subset is a
strict deterministic replication: all 16 before/after sidecar SHA-256 values
match the prior `e_f` Gemma run.

## Prompt Audit Result

Family readout:

| Variant | Generated semantics (MM/JJ/MJ/JM) | Full-vocab interaction L1 | Max semantic TV from baseline |
| --- | --- | ---: | ---: |
| baseline | M / M / M / M | 0.000247 | 0 |
| paraphrase | M / M / M / M | 0.00000209 | 0.000096 |
| reversed order | M / J / J / J | 1.682 | 0.999795 |
| rotated labels | M / M / M / M | 0.001700 | 0.000711 |

Frequency readout:

| Variant | Generated semantics (MM/JJ/MJ/JM) | Full-vocab interaction L1 | Max semantic TV from baseline |
| --- | --- | ---: | ---: |
| baseline | unclear / high / high / high | 2.172 | 0 |
| paraphrase | unclear / unclear / high / high | 3.084 | 0.698 |
| reversed order | low / high / low / high | 0.106 | 0.999 |
| rotated labels | high / high / high / high | 0.001319 | 0.970 |

The candidate semantic distributions move by nearly maximal total variation
under some prompt pairs. This is not explained by stochastic sampling: the run
is temperature 0, the baseline reproduces exactly, and the analysis uses the
complete first-token vocabulary.

The result also is not simply "the prompt overwrites the image." Several
variants preserve strong cell differences, but those differences rotate,
collapse, or expand with candidate context. The measured object is better
described as a prompt-conditioned readout surface where visual evidence and
linguistic/candidate priors interact.

## Updated Reading

The project now has three distinct measured surfaces:

1. **Input surface.** Luminance-rank palette transfer creates nonlinear raw and
   processor-space interactions.
2. **Fresh source-cache surface.** Scalar cache-summary interactions are
   architecture-dependent; Qwen's late layer 33 `values` locus replicates over
   four source pairs, Gemma stays early in `values`, and SmolVLM moves.
3. **Direct readout surface.** Complete first-token distributions change across
   cells, but their semantic factorial structure can be strongly prompt-
   dependent.

These surfaces should not be collapsed into one latent-state story. The cache
summary and direct probe come from separate fresh forwards, and the prompt
audit shows that a forced-choice distribution is not a prompt-invariant decoder
of the source cache.

The strongest manuscript-safe statement is:

> Across four independent fractal source pairs, Qwen reproduced one late
> source-cache summary locus while SmolVLM and Gemma showed different forms of
> pair dependence. All standard direct factorials had non-identical complete
> first-token distributions, but a Gemma control showed that their semantic
> structure can change sharply with prompt and candidate formulation.

## Claim Boundaries

- The four source pairs are independent visual replications; replay lengths
  within one pair are not.
- Cache effects are scalar summary-stat contrasts, not full-vector contrasts.
- Prompt robustness has been tested in one model and one source pair only.
- The current variants are diagnostic controls, not a complete balanced
  factorial over wording, order, and label mapping.
- No result establishes persistent multimodal state, valid cache intervention,
  causal mediation, semantic steering, or subjective state.
- Non-identical saved distributions are descriptive numerical observations,
  not null-hypothesis significance tests.

## Local Artifacts

```text
runs/source_pair_expansion_50_v1/factorials/
runs/source_pair_expansion_50_v1/full_vocab_factorials/
runs/source_pair_expansion_50_v1/cross_model_four_pair/cross_model_replication.{json,md}
runs/source_pair_expansion_50_v1/prompt_controls/gemma3_e_f_1f_forced_choice_robustness_seed0/prompt_robustness.{json,md}
```

## Next Steps

1. Replicate the prompt audit across Qwen, SmolVLM, and additional source pairs.
2. Balance wording, candidate order, and token-to-semantics mapping as separate
   factors, including neutral and non-forced readouts.
3. Save full-vector cache contrasts at Qwen layer 33 `values` and Gemma's early
   `values` band, with image-token-region reporting.
4. Add natural and geometric controls matched on processor-space statistics,
   then add a fourth architecture.
5. Revisit intervention only after a multimodal suffix path passes exact prefix
   and cache-length assertions.

## Tracked Summary

```text
examples/research_notes/0028_source_pair_replication_and_prompt_robustness/summary.json
```
