# Research Note 0084: Paired First-Token Prompt Distributions

2026-11-06. **Verified development readout panel.** The mean paired
ACK/YES first-token total variation is **0.619766**, across all 32 images,
after fresh same-process calibration and two byte-identical verifications.

## Fixed Study And Provenance

[Note 0081](0081_first_token_prompt_panel_registration.md) fixes the design.
After the retained [zero-forward v1 storage stop](0082_first_token_prompt_preflight_stop.md),
[Note 0083](0083_first_token_prompt_panel_v2_registration.md) registers a
separate v2 release at `175b30229f6938722d47947ad2a1c002f4d3b9dc`.
Scientific conditions, allocation, endpoints and stopping rules are unchanged.
All 108 selected registration/source files remain frozen; v1 is not retried
or counted as experimental evidence.

One model/processor instance completed 96 rows x three calibration paths
(288 contexts), then 64 fresh suffix-free experimental prefills.
Each experimental condition starts from an empty cache. The ACK/YES pair
uses the same image and differs only in the registered instruction token.
The full effective prefix is 395 positions; the experimental path supplies
no generated or forced suffix. Both complete visual sites and prepared
input receipts remain tied to their historical qualified sources.

The 32 images are eight ordered source-pair families x four donor cells
(historical codes mm/jj/mj/jm, not fractal class labels),
reused development material. Closing ACK calibration repetitions are not
extra experimental observations. Native first-ID patterns were already
known; this is the prospectively registered paired score analysis, not an
unseen-label test or an untouched confirmation cohort.

## Fresh Calibration Before Measurement

The release seal records all 96 accepted rows / 288 completed contexts and
zero experimental contexts started at that point. Only then did the panel
begin. Each calibration row reproduces the accepted Note 0080 comparison.

| Check | Complete comparisons | Gate |
| --- | ---: | --- |
| Native stream-only vs decoder-observed first-step scores | 96 x 151,936 coordinates | Word-exact |
| Native first-step scores vs independent prefix scores | 96 x 151,936 coordinates | Word-exact |
| Native prefill raw logits vs independent prefix logits | 96 x 151,936 coordinates | Word-exact |
| Native vs independent prefix K/V, all 24 layers | 4,608 tensors | Word-exact |
| Experimental prefix vs matched same-process calibration prefix | 3,072 tensors | Word-exact |
| Experimental full score/logit vectors vs matched calibration | 64 conditions | Word-exact |

All 352 contexts completed in one acquisition, without retry, replacement
or reduced denominator. Calibration equality qualifies the measurement
path; it is not an additional independent scientific sample.

## Probability Definition And Results

Use the native saved bf16 scores, losslessly expanded to float32, then
`p = exp(float64(s) - logsumexp64(s))` over all 151,936 coordinates.
Native exp-score mass is not assumed to be one. This definition does not
undo bf16 rounding or substitute a float64 model/logit-softmax run.

Primary: equal-image mean total variation across all 32 ACK/YES pairs.
JS is divergence in nats, not its square root. Keep Hellinger, delta L2,
maximum absolute probability change, complete argmax sets and the fixed
top-20 coordinate diagnostics. All coordinate ordering and zero/tie rules
are unchanged from registration. No vocabulary filter or probability floor.

The registered primary value is **0.6197663143985033**. Every one of
32 paired distributions differs; these observed distances are not an
accuracy, a significance test or a proportion of changed natural answers.

| Metric | Mean | Median | Per-image min | Per-image max |
| --- | ---: | ---: | ---: | ---: |
| TV (primary mean) | 0.619766 | 0.620616 | 0.590613 | 0.644582 |
| JS divergence, nats | 0.306283 | 0.306335 | 0.286321 | 0.319545 |
| Hellinger | 0.611573 | 0.611777 | 0.591816 | 0.626023 |
| Probability delta L2 | 0.388632 | 0.384544 | 0.348641 | 0.431028 |
| Maximum absolute probability delta | 0.280641 | 0.270416 | 0.233362 | 0.331839 |

Min/max columns are observed per-image extrema, not uncertainty intervals.
The complete 32 values and all top-20 coordinate diagnostics are retained;
no coordinate is decoded or selected as a semantic primary endpoint.

Argmax sets overlap on **2/32** pairs. ACK has a singleton maximum on all
32 images; YES has 30 singleton maxima and two two-ID ties. Both overlap
cases are shown below, not a selected subset. Shared IDs are not identical
sets; their TV values remain 0.590613 and 0.596080.

| Source-pair ID / cell | ACK maximum IDs | YES maximum IDs | TV |
| --- | --- | --- | ---: |
| stochastic_blue_dots_r1 / mj | [785] | [785,9454] | 0.590613 |
| stochastic_sparse_dense_r1 / jj | [785] | [785,9454] | 0.596080 |

No paired native score vectors are word-identical. The tie sets are retained
rather than replaced by a single preferred ID.

### All Registered Strata

Each source-pair stratum contains its four donor cells. No extra test or
uncertainty calculation is attached to these dependent descriptive summaries.

| Source-pair ID | Images | Mean TV | Per-image min | Per-image max |
| --- | ---: | ---: | ---: | ---: |
| `geometry_checker_hex_r1` | 4 | 0.615007 | 0.594680 | 0.638397 |
| `geometry_square_triangle_r1` | 4 | 0.627108 | 0.599599 | 0.644582 |
| `geometry_voronoi_hex_r1` | 4 | 0.622478 | 0.606804 | 0.632181 |
| `geometry_vq_r1` | 4 | 0.633581 | 0.625443 | 0.639655 |
| `stochastic_blue_dots_r1` | 4 | 0.608664 | 0.590613 | 0.616930 |
| `stochastic_sparse_dense_r1` | 4 | 0.609806 | 0.596080 | 0.619888 |
| `stochastic_white_blue_r1` | 4 | 0.621480 | 0.610425 | 0.633267 |
| `stochastic_white_dots_r1` | 4 | 0.620005 | 0.613952 | 0.628951 |

| Cell | Images | Mean TV |
| --- | ---: | ---: |
| jj | 8 | 0.618195 |
| jm | 8 | 0.620466 |
| mj | 8 | 0.619576 |
| mm | 8 | 0.620829 |

Similar cell means do not establish cell equivalence, image independence or
absence of image-by-instruction interaction. The full mean/median/min/max
of all five metrics in every stratum remain in the summary.

## Interpretation Boundary

This comparison measures instruction-conditioned next-token distributions
on fixed development images. It does not measure a full natural-answer
sequence distribution, visual semantic classification, fractal steering,
learning, persistence, cache intervention or cache-to-answer mediation.
The source prompt is not a new forced-choice direct factorial probe;
historical core/direct-probe totals do not increase.

The eight source-pair and four cell strata are descriptive, not independent
replication samples. No p-value, confidence interval, equivalence test or
new confirmation input is introduced. A shared argmax is not equality of
the full distribution; a changed distribution is not semantic specificity.

## Receipts And Verification

- [Verified summary](../../examples/research_notes/0084_first_token_prompt_panel/summary.json)
- [All 32 paired results](../../examples/research_notes/0084_first_token_prompt_panel/pairs.jsonl)
- [64 experimental receipts](../../examples/research_notes/0084_first_token_prompt_panel/panel_cells.jsonl)
- [288 fresh calibration arms](../../examples/research_notes/0084_first_token_prompt_panel/calibration_arms.jsonl)
  and [96 accepted comparisons](../../examples/research_notes/0084_first_token_prompt_panel/calibration_comparisons.jsonl)
- [Complete lossless native-score archive](../../examples/research_notes/0084_first_token_prompt_panel/native_scores.npz)
- [Two-verification byte replay receipt](../../examples/research_notes/0084_first_token_prompt_panel/verification_replay.json)

Two fresh zero-model verifications produced all **six registered outputs
byte-for-byte**, including the NPZ. Each verification rechecks the original
complete calibration/capture arrays, then checks paired metrics against the
separately formulated float64/`math.fsum` oracle at the original tolerances.
These are file-only checks through the frozen implementation, not a second
model implementation or additional experimental samples.

The public NPZ contains `native_score_words` [64,151936], little-endian uint16,
and the 64 original anchor ordinals. All **9,723,904 score coordinates**
round-trip losslessly to the saved float32 expansion of native bf16. The
payload is native score words, not pre-rounded probabilities or top-k data.
Public-only tests reconstruct every pair, argmax/tie/top-20 record and
registered aggregate without model weights or source images, and join the
fresh calibration to Note 0080 and each experimental condition to its
same-process calibration. Complete local logits, visual arrays and cache
tensors remain under `runs/first_token_prompt_panel_v2`; the portable score
archive does not reconstruct those other arrays.

Summary SHA-256: `b565632eb64dc9887dc7cdd7ff5c6c5a233a2d12459b8a51e3fe377a3b397285`.
Score archive: `aa2ccca10d0e45f115c254bc0a01ae2fb7995fb7a2d518c2f76ab05bd3d99196`.
Capture: `1744e0ecda396cc41010b7e95467ac2bdac066938491137f703cf7eefbf74f2b`.
Preflight: `496814d9ccba56445fb6ff0c4f50f4a1a6cc3f35b5cd24e41323889a3f4cb40b`.

All **671 local tests** pass. The two public-only publication tests also
pass from a clean Git archive of `5101348b08f39cc566d9befcabcdc8e116dd5152`,
without local runs, model weights or source images. The 108 frozen file
hashes, scoped Ruff, whitespace checks and 475 local documentation links
pass. These are local checks, not GitHub CI or untouched confirmation.

## Next Boundary

Low-level/statistical structure, learned structural processing and ordinary
instruction-conditioned contextualization remain compatible explanations.
This panel closes its registered descriptive readout question; it does not
choose among all three. New input controls, learned/unlearned comparisons
and image-by-instruction controls require separate registration.
Untouched confirmation remains neither generated nor sealed.
