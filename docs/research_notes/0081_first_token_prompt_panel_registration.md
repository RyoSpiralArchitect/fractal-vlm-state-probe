# Research Note 0081: Paired First-Token Prompt Readout

2026-11-06. Register the first **paired distributional analysis** of these
reused development conditions, not untouched confirmation. Base commit
`fd0aab331109fb3c9d5066fc325175a8575f4afd`. Freeze new code/config/
fixtures/verifier and this note before any new forward or inter-prompt
distance inspection. Keep all 97 previously frozen files unchanged.

The native first-ID pattern was already visible in earlier notes. Note
0080 acquired calibration score vectors but did not inspect their paired
ACK/YES distances. This prospective endpoint specification cannot turn
those same development images into an unobserved or independent cohort.

## Fixed Inputs And Execution

Use [Note 0080's qualified source-prompt path](0080_first_step_readout_calibration.md).
The single prompt edit is ACK to YES at prepared ID 132; each source has
140 prepared IDs, 256 image positions and a 395-position prefix. There is
**no supplied suffix** and no sampler in experimental prefill-only contexts.
Both full visual representations and all prepared input fields are fixed
within each image, except the registered instruction ID. No probe question,
candidate order, image, model, dtype, seed, image size or tokenizer change.

Offline preflight verifies the original source/representation/cache chain,
all 97 frozen code files, installed/runtime/snapshot provenance, every prior
readout score/logit/prefix array and the published Note 0080 checks.
Commit the new implementation first. Do not load weights for preflight.

On **one** model/processor instance:

1. Reacquire all 96 historical rows (including closing ACKs) using the
   unchanged native stream-only / native decoder-observed / independent
   prefix-only routines. Complete 288 contexts before experiment release.
   Require all Note 0079 gates and byte-exact score/logit/prefix identities,
   and the same complete per-row comparison receipts as Note 0080.
2. Seal those 96 calibration rows, with zero experimental contexts started.
   Then acquire **32 images x ACK/YES = 64** prefill-only contexts in original
   family/cell order, ACK then YES. Closing ACK is a calibration control, not
   an extra experimental replicate. One fresh empty cache per context.
3. Every experimental condition must match its same-process calibration
   condition's complete native score/logit arrays, visual arrays, all-layer
   prefix words, actual decoder IDs/offsets and suffix-free path.
   A score or cache mismatch is a failed gate, not an effect estimate.
4. Finish all capture before computing any paired effects. Preserve the
   first failure and attempted/completed counts; no replacement, alternate
   acquisition root, retry, resume, tolerance change or sample expansion.

Budget: **352 contexts**, 288 calibration + 64 experimental. The 32 paired
images are eight reused r1 source pairs x four related factorial cells;
they are not independent families/seeds. Reacquiring a condition whose
scores must match calibration does not create independent confirmation.

## Frozen Probability, Endpoints And Aggregation

Use all **151,936** native output coordinates, including padded/rare IDs.
For saved native scores `s`, define
`log p = float64(s) - logsumexp64(float64(s))`, then `p = exp(log p)`.
Retain exact negative infinity as zero support. No probability floor,
tail truncation, verbalizer conditioning, raw-mass comparison or switch to
raw-logit softmax. Raw-logit normalization remains the earlier numerical
diagnostic, not this panel's endpoint.

**One primary descriptive endpoint:** equal-image mean of
`TV(p_ACK,p_YES) = 0.5 sum_i abs(p_YES_i - p_ACK_i)` over all 32 pairs.
The balanced design also equals an equally weighted mean of eight
four-cell source-pair means (up to arithmetic ordering). Publish all 32
values plus mean, median, minimum, maximum, eight source-pair strata and
four cell strata. No bootstrap, CI, p-values, significance, equivalence
threshold or inference to sampled families as a population.

Secondary, kept for every pair:

- Jensen-Shannon **divergence in nats**, not its square root:
  half KL from each probability vector to their equal mixture.
- Hellinger distance, full probability-delta L2, maximum absolute coordinate
  delta and its lowest-ID argmax/tie count. If delta is exactly zero,
  its representative argmax ID is null, with no missing image or imputation.
- Complete native argmax ID sets for each prompt and whether they overlap.
  Ties remain sets; no unique-choice inference or generated-answer claim.
- Top 20 absolute probability-delta coordinates, descending magnitude then
  ascending token ID. Record both probabilities and YES-minus-ACK values.
  These are display diagnostics **after full-vector metrics**, not selected
  coordinates for the endpoint, semantic labels or additional tests.

Use the existing distribution-distance implementation and an independent
probability/scalar-sum oracle (rtol 1e-12, atol 1e-14). Validate TV in
[0,1] and JS in [0,ln 2]; do not clamp a large computational discrepancy
into a result. Keep equality and unchanged-argmax cases. Every image must
be present; an incomplete or duplicate matrix withholds the endpoint.

## Portable Data And Storage

Publish all 64 native score vectors **losslessly** as a [64,151936]
little-endian uint16 native-bf16-word matrix, plus their source anchor
ordinals. Expanding each word back to float32 must match the original
score bytes, including signed zero and negative infinity. An independent
half-word decoder checks roundtrip. No low-rank projection or pooling.
This portable archive permits complete endpoint/summary replay without
model weights or research images; full raw logits/KV remain local.

| Allocation without compression savings | Bytes |
| --- | ---: |
| 288 calibration contexts, original complete arrays | 2,396,504,064 |
| 64 experimental prefixes and complete arrays | 518,455,296 |
| Metadata, portable scores and scratch allowance | 536,870,912 |
| Reserved free disk | 8,589,934,592 |
| **Minimum free at launch** | **12,041,764,864** |

Recheck reserve before each context; never delete/move earlier runs or
user data to pass a gate. Two fresh zero-model verifications must reproduce
all receipt files and the portable archive byte-for-byte before publication.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_first_token_prompt_panel.py preflight --output-root runs/first_token_prompt_preflight_v1
python3 scripts/run_first_token_prompt_panel.py capture --output-root runs/first_token_prompt_panel_v1
python3 scripts/run_first_token_prompt_panel.py verify --output-root runs/first_token_prompt_verified_v1
python3 scripts/run_first_token_prompt_panel.py verify --output-root runs/first_token_prompt_verified_replay_v1
```

Scope ceiling: this is a lexical instruction control of first-token output
on reused development images. It is not semantic visual classification,
fractal-specific steering, learning attribution, a full answer-sequence
distribution, cache intervention/mediation or multi-turn persistence.
No untouched source/map cohort is generated or sealed.
