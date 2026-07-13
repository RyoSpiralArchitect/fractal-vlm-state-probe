# Research Note 0027: Cache-Prefix Audit and Direct Full-Vocabulary Replay

Date: 2026-07-13

## Status

Completed as a protocol audit plus a fresh, direct multimodal replication over
three local VLMs. The audit withdraws the old incremental-cache and branched
readout/intervention claims from the supported evidence set. It replaces them
with 80 valid factorial cell runs whose image-conditioned first-step
distributions are saved over the complete vocabulary.

## Why the Protocol Changed

MLX-VLM `0.4.4` reuses `PromptCacheState` by finding the common token prefix
between the cached token history and a newly reconstructed prompt. It then
trims cache tensors using that token-prefix length. Our earlier history records
stored the text of prior visual turns, but not their processor-inserted image
tokens. Reconstructing a text-only probe or the next visual turn therefore did
not preserve the original multimodal prefix. Multimodal embedding expansion
also made cache sequence length differ from token-history length.

The new audit records both conditions explicitly:

| Run | Location | Source tokens | Common prefix | Cache length | Safe |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen, one-image replay | after branch probe | 241 | 42 | 256 | no |
| SmolVLM, two-frame incremental | second visual turn | 186 | 41 | 256 | no |
| SmolVLM, two-frame incremental | after branch probe | 251 | 106 | 297 | no |

Across the two audit runs, all seven available reuse checks failed. The Qwen
legacy branch then produced byte-identical full-vocabulary sidecars across all
four image cells. Running the same probes as fresh multimodal forwards made the
four distributions non-identical immediately. The old equality was therefore
a measurement-path artifact, not evidence of output-distribution equality.

## Evidence Consequence

The following earlier claims are now **withdrawn pending valid rerun**:

- persistent 25/50-frame SmolVLM cache-state trajectories,
- mid/after readout equality obtained through text-only cache branches,
- layer-23 or layers-10-13 intervention leverage from branched cache swaps,
- donor/source pull conclusions from those intervention branches.

This does not invalidate deterministic stimuli, frame manifests, image and
processor statistics, or fresh single-turn multimodal observations. The Qwen
cumulative-replay ACK cache was created in one fresh multi-image forward, so
its source-cache factorial remains usable. Its old branched readout does not.

## Replacement Protocol

Each factorial cell now has two explicitly separate measurements:

1. A fresh ordered multi-image ACK forward records the source cache summary.
2. Each forced-choice probe runs in another fresh forward containing the same
   ordered images and the probe question directly.

No image-conditioned cache is reused between turns. The after probe therefore
measures immediate multimodal readout, not persistence after the ACK turn.
First-step logprobs are promoted to float32 and stored as compressed `.npz`
sidecars with SHA-256, vocabulary size, dtype, and normalization diagnostics.
The analysis renormalizes in log space and reports pairwise KL/JS, total
variation, Hellinger distance, probability-space factorial contrasts, and
conditional forced-choice probabilities.

## Run Matrix

| Model | Revision | Frames | Pairs | Cells | Vocabulary |
| --- | --- | --- | ---: | ---: | ---: |
| SmolVLM2-2.2B | `482adb5` | 1/2/4 | 2 | 24 | 49,280 |
| Qwen2.5-VL-3B 4bit | `46d4cf0` | 1/2/4/8/16 | 2 | 40 | 151,936 |
| Gemma-3-4B-it 4bit | `9372490` | 1/2 | 2 | 16 | 262,208 |

Total: 20 factorial points, 80 cell runs, and 320 full-vocabulary first-step
sidecars. All before distributions are cell-identical within their matched
factorial. Every direct after factorial contains non-identical cell
distributions.

## Qwen Trajectory

The fresh ACK source-cache scalar argmax remains layer 33 `values` at every
pair-by-length point: 10 of 10 factorials, normalized depth `0.943`. The sampled
local position still moves across image-token runs.

The corrected direct readout is not quiet:

- `c_d` family interaction L1 is `0.716, 0.732, 1.014, 0.964, 0.416` over
  1/2/4/8/16 frames; max pairwise JS peaks at `0.0996` at 8 frames.
- `b_c` family interaction L1 is `0.598, 0.411, 0.519, 0.449, 0.242`; it
  weakens at 16 frames rather than following cache magnitude.
- `c_d` frequency interaction L1 peaks at `0.255` at 8 frames.
- `b_c` frequency interaction grows from `0.024` to `0.146` across 1 to 16
  frames.

The `c_d` MJ cell moves strongly toward `C` around 4-8 frames while the other
family cells remain predominantly `A`. Cache interaction magnitude and readout
interaction are therefore neither equivalent nor monotonically coupled.

## SmolVLM Trajectory

Under the valid fresh protocol, the scalar cache argmax is not a stable late
layer locus:

- `c_d`: layer 21 `keys`, layer 1 `keys`, layer 21 `keys` at 1/2/4 frames,
- `b_c`: layer 1 `keys`, layer 1 `keys`, layer 16 `keys`.

Normalized depths range from `0.043` to `0.913`. Every full-vocabulary after
factorial is non-identical even when generated letters match. Family
interaction L1 ranges from `0.020` to `0.249`; frequency ranges from `0.033`
to `0.171`. This does not reproduce the old persistent layer-23 `values` story.

## Gemma3 Third-Model Lane

Gemma3 adds a distinct architecture and a much larger vocabulary. Its scalar
cache argmax is pair-specific but stable from one to two images:

- `c_d`: layer 1 `values`, normalized depth `0.030`,
- `b_c`: layer 3 `values`, normalized depth `0.091`.

The sampled local peak is layer 0 `values` at the start of an image-token run
in all four pair-by-length points. Family choice is nearly saturated on `A`, so
its full-vocabulary interaction is small (`0.00024-0.00261` L1). Frequency
choice is highly sensitive: max pairwise JS reaches `0.288` for `b_c` at one
frame and `0.534` for `c_d` at two frames. In each of those cases the MM cell
generates `C` while the other cells generate `H`.

## Current Reading

The strongest supported result is no longer a cache/readout dissociation under
persistent streaming. It is this:

> Under valid fresh multimodal forwards, cross-palette factorial cells produce
> non-identical full-vocabulary readout distributions in three VLMs, while the
> location and stability of cache-summary interactions are architecture- and
> pair-dependent.

Qwen shows a stable late scalar locus, Gemma3 shows stable pair-specific early
loci, and SmolVLM moves across depth. The full-vocabulary effects can be large
while the generated letter remains fixed, but visible letter changes also
occur. A universal layer, universal token position, persistent state, or causal
cache mechanism is not supported.

## Statistical Boundaries

- There are two source-pair replications, not six independent pairwise
  distances per factorial.
- Replay lengths are nested contexts, not independent stimulus replicates.
- Cache summaries collapse vector direction and use sampled token positions.
- The ACK cache and direct probe are separate fresh forwards with different
  prompts; their relationship is descriptive, not a causal mediation test.
- Full-vocabulary equality is now directly testable and does not hold at the
  saved numerical precision for any direct after factorial in this run set.

## Local Artifacts

```text
runs/cache_prefix_audit/prefix_audit_analysis.{json,md}
runs/full_vocab_factorials/cross_model_direct_family_trajectory.{json,md}
runs/full_vocab_factorials/cross_model_direct_frequency_trajectory.{json,md}
runs/full_vocab_factorials/qwen_*_direct_all_layers_seed0/
runs/full_vocab_factorials/smol_*_direct_all_layers_seed0/
runs/full_vocab_factorials/gemma3_*_direct_all_layers_seed0/
```

## Next Steps

1. Add new visual source pairs before increasing nested frame lengths further.
2. Run prompt paraphrases and candidate-order permutations to separate visual
   interaction from forced-choice prompt calibration.
3. Add full-vector cache contrasts at the stable Qwen/Gemma candidate loci.
4. Rebuild intervention logic only on a verified multimodal suffix path, with
   prefix and cache-length assertions required before any swap.
5. Add a fourth architecture and natural/geometric matched controls under the
   same direct protocol.

## Tracked Summary

```text
examples/research_notes/0027_cache_prefix_audit_and_direct_full_vocab/summary.json
```
