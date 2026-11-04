# Research Note 0078: Same Suffix IDs, Different Contextual Cache

2026-11-05. **Verified development control.** The same 32 FastVLM images
were read under the historical ACK and YES instructions, then both prompts
received exactly [785,2168] (`The image`) one ID at a time. At L1 keys,
L12 keys and L23 values, **both supplied positions differed in all 32 pairs**.
L0 keys and values at those positions were word-exact in **all 32 pairs**.
Identical continuation IDs do not imply identical later-layer cache.

## What Was Controlled

[Note 0077](0077_common_suffix_prompt_registration.md) fixed the design at
`3d1546fd3908b8a29a75c64cad7f446c97b778a7`, before any new forward.
The same model/processor first repeated all 96 historical rows with native
plain / native traced / forced own (288 contexts). All 13,824 boundary-level
native/forced tensor comparisons and the corresponding Note 0076 full-vector
hashes matched. Only then did the 64 new common-suffix contexts run.
There was one attempt, no replacement, no new image or prompt wording.

The only differing prepared token was instruction position 132, ACK to YES.
Its effective cache position was 387. Frame, full prepared-pixel fingerprint,
attention mask, image sizes, all other prompt IDs, and both complete visual
representations were identical within each pair. The actual decoder saw
one [1,395,896] multimodal prefill, then [785], then [2168], with offsets
395/396/397 in all layers. Each context started with an empty cache.

All 64 common-suffix prefills matched their same-process own-suffix prefills
(3,072 all-layer tensor comparisons). All 32 ACK contexts and the two YES
contexts that already had those native IDs matched their own-suffix replay
at every checkpoint (4,896 comparisons). These shams were gates; the other
30 YES contexts were **retained**, not rejected for changing native tokens.
For those 30, the common suffix is explicitly off the native greedy trajectory.

## Primary Fixed-Suffix Results

RMS of YES minus ACK over both supplied positions [395,397), all KV heads
and channels. Each row summarizes all 32 paired development images. These
are raw units within each target, not a cross-layer importance ranking.

| Registered target | Median RMS | Per-image minimum | Per-image maximum | Both positions changed |
| --- | ---: | ---: | ---: | ---: |
| L1 keys | 0.012353 | 0.009308 | 0.021192 | 32/32 |
| L12 keys | 0.070050 | 0.066681 | 0.073687 | 32/32 |
| L23 values | 0.167282 | 0.154345 | 0.173098 | 32/32 |

The singleton RMS medians for the first/second supplied ID were
0.008195/0.014965 at L1 keys, 0.086279/0.048149 at L12 keys, and
0.197743/0.130697 at L23 values. Thus even the first-versus-second pattern
is target-dependent. These are overlapping descriptive views, not six new
independent endpoints. Complete family/cell values, norms, relative L2,
cosine, exact changed scalars and positions are retained in the region ledger.

The earlier naturally matched two images in Note 0072 are included. All
six of their previously available suffix/target records reproduce exactly
(RMS, maximum difference, scalar and position counts). This extends the
token-aligned comparison from that selected-by-native-response subset to
the full registered 32 images; it does not estimate the other 30 images'
natural suffix effects from forced trajectories.

## Equality And Localization Across All Layers

Every one of the 48 targets first differed at **387** on every image.
All positions [0,387) were word-identical, including image and earlier
post-image text: 76,087,296 scalar pairs. No equality constraint was imposed
on this between-prompt comparison. At L0, only the edited position changed;
the seven unchanged prompt/template positions and both fixed-suffix positions
were identical. At each of layers 1-23, suffix K/V differed for every image.
All 48 targets are retained, not selected by this result.

This L0 exception is consistent with the pinned Qwen2 implementation:
cached keys/values are projected from the current block input before that
block's attention output. At the first block, the same suffix ID and position
have the same embedding-derived input; later blocks receive contextualized
states. This implementation reading is not an intervention on a head or
layer, nor evidence that a particular cached component caused a response.

The new observation is a bounded **prompt-conditioned difference at
token-matched positions**, not visual-state steering, semantic invariance,
consciousness, multi-turn persistence, or learned family correspondence.
Forcing `The image` also says nothing about which visible answer the model
would choose: the phrase was supplied, not selected. No distributional
readout, equivalence test, p-value or new factorial correspondence score.
The 32 related cells come from eight reused r1 families, not independent
family/source replicates.

## Receipts And Reproducibility

- [Summary and all 384 target/region aggregates](../../examples/research_notes/0078_common_suffix_prompt_control/summary.json)
- [64 panel conditions and native/supplied IDs](../../examples/research_notes/0078_common_suffix_prompt_control/panel_cells.jsonl)
- [12,288 complete paired region records](../../examples/research_notes/0078_common_suffix_prompt_control/regions.jsonl)
- [192 actual decoder-boundary receipts](../../examples/research_notes/0078_common_suffix_prompt_control/decoder_boundaries.jsonl)
- [288 fresh calibration arms](../../examples/research_notes/0078_common_suffix_prompt_control/calibration_arms.jsonl)
  and [288 all-layer boundary checks](../../examples/research_notes/0078_common_suffix_prompt_control/calibration_boundaries.jsonl)
- [Independent word/metric audit](../../examples/research_notes/0078_common_suffix_prompt_control/independent_word_check.json),
  generated by a [NumPy-only reader](../../scripts/audit_common_suffix_words.py)

Both fresh verification directories produced all five registered publication
files byte-for-byte. Independent token-position byte scans and scalar
reductions matched the numeric metrics. The separate reader also rechecked
the actual NPY words, full boundary/per-tensor digests, real supplied IDs,
all shams and every paired region's positions/RMS. Neither verification
nor this audit ran a model or processor. Full primary arrays remain local
under `runs/common_suffix_prompt_control_v1` (about 2.4 GiB); tracked
receipts permit aggregate replay but not full-tensor reconstruction alone.

Summary SHA-256: `e6b7740cf7c4b15d775756f296989b48a104a62cf7640ad07a403baa16ef727c`.
Capture: `9a51d6d5b0df66e7ee3e5693ff75b00e29841858f8093f7afa02f6487c92815c`.
Preflight: `42907a101a121d07f0b5ee8c9bdff9731e06ce688da9732bab7801fcfb84e74b`.

All 615 local tests pass, including public-only receipt/ID joins, every
region aggregate and the six historical matched-suffix comparisons. All
90 registered files remain unchanged; scoped Ruff, whitespace and 432
local links in affected documentation pass. Both independent word-audit
outputs also match byte-for-byte. These are local checks, not CI.

The earlier v1 adapter failure stays unqualified and outside these
denominators. Core cache-factorial and direct-probe totals are unchanged.
Next, separately qualify/register a fresh first-step full-vocabulary
readout if the question is how the prompt changes the answer distribution
**before any supplied suffix**. This cache control does not provide that
distribution or cache-to-output mediation. Untouched confirmation is still
neither generated nor sealed; Note numbers are not a release criterion.
