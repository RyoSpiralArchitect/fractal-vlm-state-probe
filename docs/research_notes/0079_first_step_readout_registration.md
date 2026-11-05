# Research Note 0079: Suffix-Free Full-Vocabulary Readout Qualification

2026-11-05. Calibration and definition registration only. Commit this
document, config, capture/verifier, fixtures and existing sidecar code
before new forwards. Base `eca7a9d92092b11ba29a0270d5bf17eec66f14db`.
Do not edit the previous 90 frozen files or the retained v1 failure.

## Question And Three Routes

[Note 0078](0078_common_suffix_prompt_control.md) compared cache after
supplied IDs. It did not measure which first token the model would choose.
Here qualify a **fresh prefill-only** readout against the native stream's
first score vector, before using that path in a prompt-effect panel.
Keep all 151,936 model-output coordinates, including padded/rare IDs;
do not replace the vector with top-k or selected verbalizers.

The same 96 historical rows recur: 32 images, 64 unique image/prompt
conditions and deliberately repeated closing ACKs. Each row has three
fresh-cache contexts on one model/processor instance:

1. **Native stream only.** Original source-only generation, no decoder
   hook. Retain the first actual chunk's score object lazily, unchanged;
   collect all original chunks including the final repeated notification.
   Save the full vector only after source/cache artifacts are written.
2. **Native decoder observed.** Same native generator and sampler. Reuse
   the qualified decoder-cache observer at 395/396/397; additionally retain
   only the original prefill's last-logit view and first stream score object.
   No eval/cast or extra forward to obtain logits inside that wrapper.
3. **Prefix only.** A separate fresh multimodal forward using the same
   original chat template/system/source prompt, manifest image, processor
   fields and seed. Obtain [1,395,896] embeddings, call the real decoder
   **once** with the original 140 prepared IDs, then compute the identical
   native logprob expression. No sampler, next-token decoder call, source
   cache reuse, supplied suffix, stopping-token consumption or probe-question
   substitution. Record the actual empty-to-395 cache offsets and input IDs.

The source-only option continues to forbid its existing probe-sidecar flag;
do not relax that guard. Reuse its sidecar writer explicitly in this separate
calibration. A source first-score extraction is not a cache-branch readout.

## Mandatory Checks And Bounded Budget

Offline preflight checks original inputs/cache/calibration roots, the
Note 0078 publication and ledgers, model snapshot, installed sources,
previous frozen code and the new committed code. No weights loaded or
model/processor forwards. Minimum launch free space: 8 GiB reserve + 3 GiB
new-data allowance. Do not delete or move earlier runs.

Budget: **192 natural-generation + 96 independent prefix contexts = 288**.
For every row require:

- Identical actual prepared inputs, prompt, frame, complete visual bytes
  and runtime against the same Note 0072 row. Native responses and actual
  suffix IDs must remain historical, including end-of-turn/comma cases.
- Word-exact terminal cache for all 48 tensors between the two natural
  routes, and identical existing token/top-logprob step records.
- All observed natural checkpoints match Note 0076's qualified full
  all-layer hashes, not just the three historical target tensors.
- Independent prefix cache equals the natural prefill across all layers,
  keys, values, heads, positions and channels; its effective length is 395,
  not the final natural-generation length 397. It has zero suffix IDs and
  zero suffix decoder calls, with one actual projector call.
- Whole native first-score vector from stream-only equals decoder-observed
  scores; independent prefix scores equal both. Original last raw logits
  match between observed natural prefill and independent prefix. **No
  tolerance on bytes**, shapes, native dtype (bf16) or coordinate count.
- The native chosen first ID belongs to the full-vector maximum set;
  retain ties. Existing top-k values equal their coordinates in the complete
  vector. Do not claim argmax label equality proves distribution equality.

Persist native bf16 values as exact float32 expansions in NPZ using the
existing score writer; persist raw logits separately. Retain all complete
native KV word arrays/checkpoints and both visual-site arrays locally.
Record first-score hashes, native dtype, dimensions, complete coordinate
counts, IDs, cache role/length checks and source receipts for all rows.
Reload and compare independently, then require a second fresh verification
to reproduce the public receipt files byte-for-byte. Public digests/metadata
alone cannot reconstruct scores or cache arrays.

## Native Scores And The Probability Definition

The installed runtime applies `logits - logsumexp(logits)` in its native
dtype. Do **not** assume exponentiating bf16 scores sums to exactly one.
Keep their unmodified probability mass/logsumexp and nonfinite census.
Allow native negative infinity as exact zero support, never replace it with
a finite floor; reject NaN/+infinity or a vector with no finite support.
Raw logits must be finite. No vocabulary filtering or tail truncation.

Register the future probability definition now: for the saved native scores
`s`, compute `p = exp(s - logsumexp64(s))` in float64 over the **whole**
model output vocabulary. This is explicit renormalization of native scores,
not a claim to recover higher-precision logits. Check probability sum to
1e-12. As a numerical diagnostic only, also compute the TV distance from
the float64 softmax of saved raw logits. Do not silently switch the primary
definition according to which looks more normalized. Scalar-sum oracles
check numerical summaries at rtol 1e-12, atol 1e-14, separately from the
zero-tolerance model-route identity.

**No inter-prompt TV/JS/effect analysis is released by this calibration.**
Acquiring ACK/YES calibration vectors does not authorize effect inspection,
sample expansion, selected tokens, p-values, equivalence claims or untouched
confirmation. Stop and preserve the first failed gate, partial artifacts
and attempted/completed counts. One canonical acquisition; no retry, resume,
replacement, tolerance change or same-run patch.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_first_step_readout_calibration.py preflight --output-root runs/first_step_readout_preflight_v1
python3 scripts/run_first_step_readout_calibration.py capture --output-root runs/first_step_readout_calibration_v1
python3 scripts/run_first_step_readout_calibration.py verify --output-root runs/first_step_readout_verified_v1
python3 scripts/run_first_step_readout_calibration.py verify --output-root runs/first_step_readout_verified_replay_v1
```

A passing gate qualifies this measurement path, not its scientific effect.
A later separately registered 32-image/64-condition prompt panel must fix
effect summaries, dependence/aggregation, stopping and its fresh calibration
before release. Note 0080 is not a deadline for declaring confirmation done.
