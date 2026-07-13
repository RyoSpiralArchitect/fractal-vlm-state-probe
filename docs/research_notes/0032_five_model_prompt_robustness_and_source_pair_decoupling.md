# Research Note 0032: Five-Model Prompt Robustness and Source-Pair Decoupling

Date: 2026-07-14

> **Follow-up:** [Note 0033](0033_five_model_two_pair_prompt_matrix.md)
> extends the second-pair audit to all five models. Its 5 x 2 matrix supersedes
> this note's LFM2-only source-pair replication limit while preserving the
> results reported here.

## Status

Completed as two extensions of the calibrated fresh-forward readout protocol:

1. the `e_f` forced-choice prompt audit now covers all five local VLMs, and
2. LFM2-VL repeats the same eight probes on a second independent source pair,
   `b_c`.

The prompt-audit surface now contains 24 four-cell runs and 384 complete
first-step vocabulary sidecars. It is separate from the 544-sidecar standard
direct matrix: the audit repeats each visual cell under four prompt variants
for both family and frequency probes.

Across the six model-by-source-pair audits, all 96 baseline sidecars are
bitwise exact copies of their corresponding standard direct runs. Prompt
sensitivity is therefore not a sampling or baseline-reproduction artifact.

## Protocol And Calibration

Each audit uses temperature 0, seed 0, one selected image, and separate fresh
direct multimodal forwards. The eight probe formulations are:

- baseline family and frequency choices,
- semantic-preserving paraphrases,
- reversed candidate presentation order, and
- rotated token-to-semantics mappings.

Candidate probabilities are aligned by declared meaning before variants are
compared. The analysis now also carries the equal-coefficient full-vocabulary
Hadamard contrasts introduced in Note 0031:

```text
spatial     = JJ + JM - MM - MJ
palette     = JJ + MJ - MM - JM
interaction = JJ + MM - JM - MJ
```

Their squared L2 norms are normalized into spatial, palette, and interaction
energy shares. This lets prompt variants and source pairs be compared without
the factor-of-two scale mismatch in the original main-effect estimands.

The aggregate CLI now requires the comparison meaning to be explicit when the
units are source pairs:

```bash
python3 scripts/analyze_prompt_robustness_aggregate.py \
  --comparison-axis source_pair \
  --analysis b_c=.../lfm2_vl_b_c.../prompt_robustness.json \
  --analysis e_f=.../lfm2_vl_e_f.../prompt_robustness.json \
  --output-json .../prompt_robustness_aggregate.json \
  --output-md .../prompt_robustness_aggregate.md
```

## Integrity Audit

The new runs add LFM2-VL on `b_c` and `e_f`, plus InternVL3 on `e_f`:

| Model / pairs | Cell runs | Sidecars | Hash/shape/dtype/finite | Baseline exact |
| --- | ---: | ---: | ---: | ---: |
| LFM2-VL / `b_c`, `e_f` | 8 | 128 | 128/128 | 32/32 |
| InternVL3 / `e_f` | 4 | 64 | 64/64 | 16/16 |

InternVL3 again used its audited custom image-expansion adapter and expanded
the selected image to 3,328 image tokens. All four cells completed in separate
processes. LFM2-VL also used separate cell processes after the earlier
multi-cell Metal lifetime timeout. No completed artifact failed validation.

Together with the earlier Gemma 3, Qwen, and SmolVLM audits, the complete
prompt surface has 384/384 valid sidecars and 96/96 exact baseline sidecars.

## Five-Model `e_f` Result

Every model changes at least one generated semantic pattern under a prompt
variant, but the response is architecture-specific. `Changed` counts compare
the three non-baseline variants with the baseline pattern. Dominance counts
pool the eight family/frequency variants within each model.

| Model | Changed family / frequency | Max semantic TV family / frequency | Balanced dominant S / P / I |
| --- | --- | --- | --- |
| Gemma 3 | 1/3 / 3/3 | 0.9998 / 0.9999 | 2 / 1 / 5 |
| InternVL3 | 0/3 / 3/3 | 0.3837 / 0.6652 | 5 / 1 / 2 |
| LFM2-VL | 3/3 / 3/3 | 0.4404 / 0.4434 | 6 / 2 / 0 |
| Qwen2.5-VL | 1/3 / 1/3 | 0.5005 / 0.5988 | 2 / 2 / 4 |
| SmolVLM2 | 2/3 / 3/3 | 0.1821 / 0.3167 | 0 / 8 / 0 |

Across all five models:

- 20/30 non-baseline model-family variants change the generated four-cell
  semantic pattern;
- none of the eight prompt variants has one four-cell generated pattern shared
  by all five models;
- balanced axis dominance over the 40 model-family-variant records is spatial
  15, palette 14, and interaction 11; and
- interaction exceeds the exchangeable `1/3` reference in 14/40 records.

The balanced result exposes architecture profiles that a generated-label table
alone conceals. Qwen is interaction-dominant for all four family variants.
SmolVLM is palette-dominant for all eight variants. LFM2 is spatial-dominant
for all family variants, while InternVL is spatial-dominant for all family
variants but splits its frequency variants across all three axes.

InternVL provides a particularly clean label/distribution separation. Its
family output is Mandelbrot in every cell under all four variants, yet the
maximum semantic total variation is 0.384 and the complete-distribution
interaction share ranges from 0.034 to 0.335. Its frequency output changes
globally from unclear to high, low, and high under the three controls, with
semantic total variation reaching 0.665.

## LFM2-VL Two-Pair Result

The second source pair reveals a stronger decoupling. The table gives complete
full-vocabulary balanced shares; generated patterns are ordered MM/JJ/MJ/JM.

| Probe / variant | Generated `b_c` | Generated `e_f` | `b_c` S/P/I | `e_f` S/P/I |
| --- | --- | --- | --- | --- |
| family baseline | J/J/J/M | M/M/U/M | 0.237/0.592/0.172 | 0.809/0.159/0.032 |
| family paraphrase | J/J/M/J | M/J/M/J | 0.039/0.153/0.808 | 0.722/0.202/0.077 |
| family reversed | J/J/J/J | J/J/J/J | 0.100/0.433/0.467 | 0.936/0.009/0.055 |
| family rotated | M/M/M/M | M/M/M/M | 0.228/0.713/0.059 | 0.560/0.424/0.016 |
| frequency baseline | U/U/U/U | U/U/U/U | 0.635/0.047/0.319 | 0.099/0.703/0.198 |
| frequency paraphrase | H/H/H/H | H/H/H/H | 0.873/0.097/0.030 | 0.168/0.762/0.070 |
| frequency reversed | L/L/L/L | L/L/L/L | 0.330/0.383/0.287 | 0.741/0.237/0.022 |
| frequency rotated | H/H/H/H | H/H/H/H | 0.284/0.705/0.011 | 0.697/0.288/0.015 |

Generated semantic patterns agree across pairs in 6/8 variants. Balanced-axis
dominance agrees in 0/8.

The frequency probe is the sharpest example: its generated pattern agrees in
4/4 variants (`U`, `H`, `L`, `H` in every cell), while its dominant factorial
axis flips in 4/4. Baseline and paraphrase are spatial-dominant on `b_c` but
palette-dominant on `e_f`; reversed and rotated are palette-dominant on `b_c`
but spatial-dominant on `e_f`.

The family probe tells the complementary story. The reversed and rotated
generated patterns replicate across pairs, but `b_c` is interaction- or
palette-dominant while every `e_f` family variant is spatial-dominant.

Thus categorical prompt response can replicate while the visual factorial
geometry of the complete readout distribution does not. Conversely, a changed
generated label does not by itself identify which balanced visual axis moved.

## Updated Reading

The forced-choice prompt is best treated as a measurement operator, not a
transparent semantic decoder. It jointly selects:

- candidate semantics and their token realizations,
- linguistic order and local token context,
- the source-pair-specific visual perturbation, and
- architecture-specific calibration of the complete vocabulary.

The most useful new distinction is between two kinds of replication:

1. **categorical response replication**, where the generated semantic pattern
   repeats, and
2. **distributional factorial replication**, where the balanced S/P/I geometry
   repeats.

LFM2 frequency prompts show the first without the second. Across models, even
the first does not hold universally: no audited variant has one generated
pattern shared by all five architectures.

The strongest manuscript-safe statement is:

> Across five local VLMs on one shared visual factorial, semantically aligned
> prompt variants changed generated patterns and redistributed balanced
> full-vocabulary spatial, palette, and interaction axes in
> architecture-specific ways. In a two-source-pair LFM2 replication,
> generated patterns agreed for six of eight variants while the dominant
> balanced axis agreed for none, demonstrating that categorical readout
> repeatability does not establish distributional factorial repeatability.

## Claim Boundaries

- In this note's snapshot, the five-model comparison uses `e_f` and only LFM2
  has a second-pair audit. Note 0033 subsequently closes that coverage gap.
- Paraphrase, candidate order, and token mapping are diagnostic variants, not a
  fully crossed orthogonal prompt factorial.
- The measured distribution is the first generated-token distribution, not a
  complete multi-token answer distribution.
- Source ACK caches and direct probes remain separate fresh forwards. This
  experiment does not identify a causal map from cache tensors to readout.
- The four visual cells and repeated prompt variants are not independent
  statistical samples.
- The result does not establish a prompt-invariant semantic representation,
  persistent state, causal cache mediation, or subjective-state steering.

## Highest-Value Next Experiments

1. Cross wording, candidate order, and label-to-semantics mapping as separate
   balanced prompt factors, using multiple verbalizers per semantic class.
2. Add neutral candidate scoring and open-ended/non-forced readouts, including
   multi-token sequence scores rather than only first-token letters.
3. Repeat the second source-pair audit in Qwen, Gemma, SmolVLM, and InternVL,
   then add independently generated trajectories.
4. Add matched natural, geometric, noise, and processor-frequency controls with
   within-design permutation nulls.
5. Resolve selected balanced cache vectors by KV head and image-token position,
   while keeping cache and readout surfaces descriptively separate.

## Artifacts

```text
runs/source_pair_expansion_50_v1/prompt_controls/
  five_model_e_f_1f_forced_choice_robustness_seed0/
  lfm2_vl_b_c_1f_forced_choice_robustness_seed0/
  lfm2_vl_e_f_1f_forced_choice_robustness_seed0/
  lfm2_vl_two_pair_1f_forced_choice_robustness_seed0/
  internvl3_e_f_1f_forced_choice_robustness_seed0/
examples/research_notes/0032_five_model_prompt_robustness/summary.json
```
