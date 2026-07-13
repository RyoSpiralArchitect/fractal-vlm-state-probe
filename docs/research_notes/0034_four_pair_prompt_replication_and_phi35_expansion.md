# Research Note 0034: Four-Pair Prompt Replication and Phi-3.5 Vision Expansion

Date: 2026-07-14

## Status

Completed for the five-model core prompt matrix over all four existing source
pairs. A sixth architecture, Phi-3.5 Vision, is complete as an intentionally
unbalanced `c_d` pilot with fresh direct readouts, prompt controls, source-cache
summaries, and two selected full-tensor contrasts.

The core prompt surface now contains:

- 5 models x 4 independent source pairs,
- 20 model/source-pair audit units and 80 visual-cell run records,
- 1,280 complete first-step vocabulary sidecars, and
- 320 baseline sidecars that reproduce their standard direct runs bitwise.

All 1,280 sidecars pass recorded hash, raw shape, raw dtype, loader hash, and
finite-value checks. The newly added `c_d` and `d_e` extension contributes 40
cell runs and 640 sidecars, including 160/160 exact baseline comparisons.

Including the Phi-3.5 Vision `c_d` pilot, prompt coverage reaches 21 audit
units, 84 cell runs, 1,344 sidecars, and 336/336 exact baseline comparisons.
That total is useful for artifact accounting, but it is not a balanced
six-model x four-pair replication.

## Protocol Boundary

Every prompt audit uses one selected image, temperature 0, stream seed
20260604, probe seed 0, and separate fresh direct multimodal forwards. The
eight probes are baseline, paraphrase, reversed-order, and rotated-label
variants for family and frequency forced choices.

Candidate probabilities are aligned by declared meaning. Complete first-step
vocabulary distributions are decomposed with equal-coefficient Hadamard
contrasts:

```text
spatial     = JJ + JM - MM - MJ
palette     = JJ + MJ - MM - JM
interaction = JJ + MM - JM - MJ
```

Squared L2 norms are normalized into spatial, palette, and interaction energy
shares. Source-cache measurements come from separate fresh ACK forwards and do
not form a causal path to the direct probes.

## Four-Pair Replication Criteria

For each fixed model, probe family, and prompt variant, the new analyzer keeps
two levels of comparison separate:

1. **All-pair agreement:** one generated four-cell semantic pattern or one
   dominant balanced axis must repeat over `b_c`, `c_d`, `d_e`, and `e_f`.
2. **Pairwise agreement:** the same criteria are evaluated over the six
   source-pair combinations, with balanced-share L1 also reported.

The all-pair table contains 40 repeated-measure records. The 240 pairwise
comparisons are six dependent comparisons derived from each record; they are
not 240 independent visual replications.

## Core Five-Model Result

### All four source pairs

| Generated pattern same | Balanced axis same | Both same |
| ---: | ---: | ---: |
| 27/40 | 2/40 | 2/40 |

Twenty-seven model-family-variant records retain one generated semantic
pattern across all four source pairs. Only two retain one dominant
full-vocabulary factorial axis, and those are the same two records that retain
both criteria.

The two axis-stable records are:

- Gemma 3 frequency with rotated labels: high-frequency in every cell and
  spatial-dominant on every pair.
- Qwen frequency with the paraphrase: low-frequency in every cell and
  spatial-dominant on every pair.

Even these are coarse argmax agreements, not equality of the share vectors.
Gemma's pairwise share-L1 reaches 1.252 and Qwen's reaches 0.750 on a simplex
with maximum L1 distance 2.

### All six pairwise comparisons

| Generated pattern same | Balanced axis same | Both same | Share-L1 median | Range |
| ---: | ---: | ---: | ---: | --- |
| 185/240 | 100/240 | 80/240 | 0.8138 | 0.0155-1.9832 |

The agreement contingency is:

| Generated pattern | Balanced axis | Comparisons |
| --- | --- | ---: |
| same | same | 80 |
| same | different | 105 |
| different | same | 20 |
| different | different | 35 |

The largest cell is therefore the split case: in 105/240 dependent pairwise
comparisons, the visible four-cell pattern repeats while the dominant spatial,
palette, or interaction axis changes.

## Model Profiles

| Model | All-pair generated / axis / both | Pairwise generated / axis / both | Pairwise share-L1 median |
| --- | --- | --- | ---: |
| Gemma 3 | 4/8 / 1/8 / 1/8 | 28/48 / 23/48 / 18/48 | 0.713 |
| InternVL3 | 7/8 / 0/8 / 0/8 | 45/48 / 20/48 / 18/48 | 0.821 |
| LFM2-VL | 5/8 / 0/8 / 0/8 | 35/48 / 16/48 / 11/48 | 0.854 |
| Qwen2.5-VL | 5/8 / 1/8 / 1/8 | 37/48 / 17/48 / 14/48 | 0.772 |
| SmolVLM2 | 6/8 / 0/8 / 0/8 | 40/48 / 24/48 / 19/48 | 0.744 |

InternVL and SmolVLM provide the clearest categorical/distributional split.
InternVL repeats seven of eight generated patterns over all four pairs but no
dominant axis; SmolVLM repeats six of eight generated patterns and likewise no
axis. No model has more than one all-pair axis-stable prompt record.

Family probes retain generated patterns in 12/20 all-pair records and retain
no dominant axes. Frequency probes retain generated patterns in 15/20 and
account for both axis-stable records.

## Source-Pair Structure

| Pair comparison | Generated agreement | Axis agreement | Both | Share-L1 median |
| --- | ---: | ---: | ---: | ---: |
| `b_c` vs `c_d` | 27/40 | 9/40 | 6/40 | 1.132 |
| `b_c` vs `d_e` | 35/40 | 11/40 | 10/40 | 0.861 |
| `b_c` vs `e_f` | 34/40 | 11/40 | 10/40 | 0.952 |
| `c_d` vs `d_e` | 29/40 | 23/40 | 17/40 | 0.593 |
| `c_d` vs `e_f` | 28/40 | 24/40 | 18/40 | 0.606 |
| `d_e` vs `e_f` | 32/40 | 22/40 | 19/40 | 0.610 |

`b_c` is the most distributionally distinct pair in this matrix. Every
comparison involving `b_c` has only 9-11/40 axis agreement and a larger median
share-L1 than comparisons among `c_d`, `d_e`, and `e_f`. This structure is
descriptive; with four source pairs it does not define a population-level
cluster.

| Source pair | Changed non-baseline patterns | Shared pattern across models | Dominant S / P / I | Interaction above 1/3 |
| --- | ---: | ---: | --- | ---: |
| `b_c` | 20/30 | 0/8 | 10 / 9 / 21 | 24/40 |
| `c_d` | 21/30 | 0/8 | 16 / 11 / 13 | 16/40 |
| `d_e` | 16/30 | 0/8 | 16 / 13 / 11 | 16/40 |
| `e_f` | 20/30 | 0/8 | 15 / 14 / 11 | 14/40 |

Across 160 model-pair-family-variant records, dominant axes are spatial,
palette, and interaction in 57, 47, and 56 records. No prompt variant produces
one generated four-cell pattern shared by all five models on any source pair.

## Phi-3.5 Vision Pilot

The sixth architecture uses
`mlx-community/Phi-3.5-vision-instruct-4bit` under MLX `0.31.1` and MLX-VLM
`0.4.4`. The `c_d` pilot adds four standard direct cells and 16 standard
sidecars, then four prompt-audit cells and 64 prompt sidecars. All 64 prompt
sidecars pass integrity checks and all 16 baseline sidecars reproduce exactly.

The standard direct after-factorials are non-identical:

| Probe | Generated semantics (MM / JJ / MJ / JM) | Max pair JS | Balanced S / P / I | Dominant axis |
| --- | --- | ---: | --- | --- |
| Family | unclear / Mandelbrot / Mandelbrot / Mandelbrot | 0.0432 | 0.349 / 0.455 / 0.196 | palette |
| Frequency | unclear / unclear / unclear / unclear | 0.0194 | 0.155 / 0.724 / 0.121 | palette |

Phi's family generated pattern changes under all three non-baseline variants,
while its dominant axis remains palette in all four. Frequency changes in two
of three non-baseline variants and alternates between palette and spatial
dominance. Across its eight prompt records, dominance is spatial/palette/
interaction in 2/6/0. Interaction never exceeds the exchangeable `1/3`
reference.

Adding Phi to the six-model `c_d` aggregate does not create a shared generated
pattern: cross-model agreement remains absent for all eight prompt variants.
The number of distinct six-model patterns is 3, 5, 4, and 3 for the family
variants and 3, 5, 3, and 4 for the frequency variants.

## Phi Source-Cache Boundary

The fresh source ACK exposes 32 cache layers with a reported effective token
count of 917. Raw cache entries have shape `[1, 32, 1024, 96]`; captured tensor
sidecars are offset-trimmed to `[1, 32, 917, 96]`.

The scalar summary has its largest interaction at layer 3 `keys`; the sampled
sequence-position summary has its largest interaction at layer 27 `keys`,
position 512. Full tensors captured at those two targets give:

| Target | Full-vector balanced S / P / I | Dominant axis | Interaction argmax position |
| --- | --- | --- | ---: |
| layer 3 `keys` | 0.434 / 0.250 / 0.315 | spatial | 601 |
| layer 27 `keys` | 0.377 / 0.299 / 0.324 | spatial | 57 |

Thus a large interaction in a saved scalar field does not imply
interaction-dominant full-vector energy. The full-tensor argmaxes also do not
match the sampled scalar position.

Phi's recorded image marker ID is absent from the reconstructed token layout,
so the current artifact reports zero identified image tokens. Positions 512,
601, and 57 must not be called image-token locations. Image/non-image energy
partitioning remains unavailable until Phi's processor-expanded token layout
is reconstructed compatibly.

## Updated Reading

The strongest result is now a replicated separation between categorical
readout stability and complete-distribution factorial stability. Extending from
two to four independent source pairs weakens, rather than rescues, the idea
that one visible response pattern identifies one stable spatial/palette/
interaction regime.

The measured object is therefore a conditional response surface:

> controlled visual transformation x source pair x architecture x probe
> formulation.

Some categorical outputs are highly repeatable within that surface, but the
underlying first-token probability geometry remains conditional. The Phi pilot
extends the architecture range and preserves that heterogeneity; it does not
yet supply a sixth balanced four-pair replication.

The strongest manuscript-safe statement is:

> Across five local VLMs and four independent visual source pairs, 27 of 40
> fixed model-family-prompt records retained the same four-cell generated
> semantic pattern across every pair, while only 2 retained the same dominant
> equal-coefficient full-vocabulary factorial axis. Across the six dependent
> pairwise comparisons per record, categorical patterns agreed in 185/240
> comparisons and dominant axes in 100/240; in 105 comparisons the categorical
> pattern repeated while the dominant spatial, palette, or interaction axis
> changed. A sixth-architecture `c_d` pilot produced another distinct prompt
> profile, but is not yet balanced across source pairs.

## Claim Boundaries

- The independent core visual replication count is four source pairs, not 40
  all-pair records or 240 pairwise comparisons.
- The Phi pilot adds an architecture, not a fifth source pair or a complete
  six-model matrix.
- Prompt variants are diagnostic repeated measurements, not an orthogonally
  crossed wording/order/verbalizer design.
- Dominant-axis equality is weaker than equality of the complete share vector.
- Source ACK caches and direct probes are separate fresh forwards, so no causal
  cache-to-readout mediation follows.
- Phi image-token localization is unresolved and no raw position is assigned
  an image role.
- No result establishes prompt-invariant semantic decoding, persistent state,
  valid cache intervention, semantic steering, or subjective state.

## Highest-Value Next Experiments

1. Extend Phi-3.5 Vision to `b_c`, `d_e`, and `e_f` to complete a balanced
   sixth-model four-pair matrix.
2. Cross wording, candidate order, and label-to-semantics mapping as separate
   factors with multiple verbalizers, neutral scoring, and open readouts.
3. Add independently generated source trajectories plus matched natural,
   geometric, noise, and processor-frequency controls.
4. Reconstruct Phi's processor-expanded image-token layout, then resolve
   selected balanced vectors by KV head and token position.
5. Use source-pair-level permutation or hierarchical repeated-measure analyses
   only after the independent trajectory count is expanded.

## Artifacts

```text
runs/source_pair_expansion_50_v1/prompt_controls/
  {gemma3,internvl3,lfm2_vl,qwen,smol}_{b_c,c_d,d_e,e_f}_1f_forced_choice_robustness_seed0/
  five_model_four_pair_1f_forced_choice_robustness_seed0/
runs/phi35_vision_expansion/
  c_d_1f_direct_seed0/
  c_d_1f_source_cache_smoke/
  c_d_1f_source_cache_tensors_seed0/
  prompt_controls/phi35_vision_c_d_1f_forced_choice_robustness_seed0/
  prompt_controls/six_model_c_d_1f_forced_choice_robustness_seed0/
examples/research_notes/0034_four_pair_prompt_and_phi35_expansion/summary.json
```
