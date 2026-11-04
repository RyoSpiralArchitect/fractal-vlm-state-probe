# Research Note 0073: Native-Suffix Replay Parity Registration

2026-11-04. Development calibration only, registered after the merge of
[PR 7](https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/7)
at `97df3fb`. Commit this note, the [fixed config](../../configs/native_suffix_replay_calibration_v1.json),
both measurement modules, launcher and fixtures before any new model call.
Do not change the 73 files registered through Note 0072.

**Question.** If we rebuild the multimodal prefix from the same image and
prompt and explicitly feed each originally generated ID, do we recover the
native *effective* cache, byte for byte, at prefill and after each token?
Historical [Note 0072](0072_prompt_only_source_calibration.md) saved three
full selected tensors and both visual sites, not all-layer KV. Use historical
identity for those observations and acquire a fresh all-layer native reference.
Never present a new 48-tensor reference as a historical 48-tensor capture.

## Fixed Design

Use every Note 0072 context, in its original 1-96 order. There are 32 unique
images, 64 distinct prompt/image conditions, and **96 historical rows**
because the closing ACK is repeated. Do not average or silently deduplicate.
The sealed own-suffix schedule, including the repeats, is 66 x [785,2168],
20 x [9454,151645] (end-of-turn), 10 x [9454,11] (comma). Do not use visible
text, re-tokenize the answer, drop punctuation, or use the final stream
notification as another ID. No counterfactual token is supplied to a real model.

Run three fresh, empty-cache contexts on one model/processor per row:

1. **Native plain.** Original Note 0071 natural source generation and visual
   observer, with no language-call/selection hook. Read *all* final KV only
   after the stream is exhausted.
2. **Native traced.** Same unmodified native sampler and stream. Observe the
   original decoder call, actual inputs/embedding shape, all layer offsets
   before/after it, and every effective KV word at each boundary. Check final
   equality with native plain. This bounds the effect of synchronously reading
   the cache; it is not an equality claim about performance.
3. **Forced own.** Reuse the real prompt/template, processor and outer
   source stream, never a preceding arm's cache or image features. Replace
   only the token-step generator. Obtain fresh multimodal embeddings and call
   the actual language model for prefill, then explicitly call it with
   `[[id_1]]` and `[[id_2]]` (uint32). Neither ID is chosen from logits.
   Source log probabilities are returned to the existing streaming interface;
   no direct full-vocabulary probe or score comparison. Consume each ID before
   yielding to the streamer, including end-of-turn: the qualified native
   schedule advances cache before it handles that stop. Independently observe
   what entered the decoder. Restore both hooks even when a call raises.

Expect 140 prepared IDs; a single image marker at 42 expands to 256 image
positions. Decoder inputs: whole prefix with embeddings [1,395,896], then
two singletons without embeddings; effective offsets **0 -> 395 -> 396 ->
397** in every layer. Compare all 24 layers, keys and values, complete shape
[1,2,offset,64], native bf16 words including signed zero. Never compare
allocated slack, only the effective sequence. At every arm also require
historical frame/prompt, prepared inputs, both full visual sites, actual
response/suffix/layout and original three final target bytes, including B
and closing-A rows. No cache-to-output claim.

## Receipts And Stopping

Offline preflight: verify the original and catalog Note 0072 publications,
all three ledgers, 96 historical source rows/288 selected tensors, previous
registration, model snapshot, installed runtime/cache implementations and
committed new code. No weights loaded or source forwards. Reserve 8 GiB;
require another 3 GiB at launch. Preserve previous runs. One canonical
preflight and acquisition directory, no overwrite/retry/resume/replacement.

Budget: **288** fresh source contexts (96 per arm); 192 explicitly supplied
token calls. 96 x 48 plain/traced terminal controls, 96 x 3 x 48 native/forced
boundary comparisons, 288 x 3 historical selected-cache checks. These are
dependent implementation checks, not independent scientific observations.
Zero tolerance on words, IDs, offsets and shapes. Stop on the first failed
gate; keep primary data, attempted/completed counts and the discrepancy.
Do not tune timing, dtypes, tolerances or exclusions after a failure.

Store native-plain terminal arrays. For each traced arm read the *full*
live array at each checkpoint, verify its prefix against the previous
checkpoint and persist the prefill plus the two appended positions in
lossless uint16 NPYs. On a rewritten prefix retain the full offending
checkpoint. Record full-word and per-tensor digests at every boundary.
Reload and reconstruct all checkpoints; compare each native/forced pair and
the native terminal's corresponding prefix. Publish complete arm/boundary
receipts only after successful verification and a byte-identical second
verification; both reloads use zero model calls.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_native_suffix_replay.py preflight --output-root runs/native_suffix_replay_preflight_v1
python3 scripts/run_native_suffix_replay.py capture --output-root runs/native_suffix_replay_calibration_v1
python3 scripts/run_native_suffix_replay.py verify --output-root runs/native_suffix_replay_verified_v1
python3 scripts/run_native_suffix_replay.py verify --output-root runs/native_suffix_replay_verified_replay_v1
```

Passing would qualify **own-suffix** replay on these inputs and this runtime
only. No cross-prompt common-suffix experiment, token substitution, cache
swap, direct readout, or untouched confirmation is authorized by this gate.
