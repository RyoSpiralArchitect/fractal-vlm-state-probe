# Research Note 0076: Does Feeding Back The Native Suffix Reproduce The Cache?

2026-11-04. **Qualified for own-suffix replay on these FastVLM development
conditions.** Starting with a fresh image/prompt prefill and feeding the
original two IDs individually reproduced the effective native cache at
prefill and after each token. This establishes a measurement method. The
replayed IDs came from the native trajectory; this is not a test of what
would happen under another answer or of whether the cache caused the answer.

## Which Native Cache?

[Note 0072](0072_prompt_only_source_calibration.md) had saved L1 keys, L12
keys and L23 values, not all-layer historical K/V. For those three targets
we checked the actual old bytes. For the full 24 layers we acquired a fresh
native reference on the same image/prompt, model snapshot and runtime.
The untraced language model supplied a terminal reference; a second native
pass recorded the actual states at all three times. All 4,608 plain/traced
terminal tensor pairs were identical. Traced prefixes also matched the
corresponding untraced terminal prefixes. The forced pass then had a
step-by-step native reference, not just a final-array slice to imitate.

## Paired Results

| Actual decoder boundary | Effective positions | Exact native/forced tensor pairs | Exact bf16 scalar pairs |
| --- | ---: | ---: | ---: |
| Multimodal prefill, before any suffix ID | 395 | 4,608/4,608 | 232,980,480 |
| After the first supplied ID | 396 | 4,608/4,608 | 233,570,304 |
| After the second supplied ID | 397 | 4,608/4,608 | 234,160,128 |
| **Total across checkpoints** | | **13,824/13,824** | **700,710,912** |

Every checked word matched, including the original prefix at later calls.
This is the effective sequence in keys and values, across **all heads and
channels in all 24 layers**. Unused capacity of allocated buffers, hidden
states, direct first-step distributions and timing were not endpoints. The
counts enumerate dependent implementation checks, not independent samples
or evidence of population-level equivalence.

All 96 Note 0072 rows were reacquired in three fresh arms (288 contexts):
native plain, native traced, forced own. There were 32 distinct images
and **64 distinct image/prompt combinations**; the closing ACK was a deliberate
repeat. All 864 historical selected-target checks (288 x 3), prepared inputs,
both visual representations, natural suffixes and role/offset checks passed.
The two decoder calls in each forced arm were observed directly: 192
single-token inputs in total, not 192 response strings or streaming notifications.

| Native IDs actually supplied | Native text | Rows including closing ACK |
| --- | --- | ---: |
| [785,2168] | `The image` | 66 (32 opening ACK, 32 closing ACK, 2 YES exceptions) |
| [9454,151645] | `Yes` | 20 (includes the end-of-turn ID) |
| [9454,11] | `Yes,` | 10 (includes the comma ID) |

## Registration And Verification

The [v2 registration](0075_native_suffix_v2_registration.md) at
`a5c2c1002760e6b47c904ed20fdcf0ba33e09c33` retained all 79 frozen v1
files and added five. The [v1 failure](0074_native_suffix_replay_entry_failure.md)
remains unqualified: two native arms, zero forced calls. It is neither
erased nor counted in the v2 denominators. v2 changed only how an omitted,
inactive stream keyword is passed to the old stepper. Nine synthetic dispatch
fixtures exercised three suffix forms and the omitted, explicit-null and
rejected non-null cases. Acquisitions used one v2 attempt and no replacements.

The [verified summary](../../examples/research_notes/0076_native_suffix_replay_parity/summary.json),
[288 arm receipts](../../examples/research_notes/0076_native_suffix_replay_parity/arms.jsonl)
and [288 boundary receipts](../../examples/research_notes/0076_native_suffix_replay_parity/boundaries.jsonl)
were reproduced byte-for-byte in two fresh verification directories. These
checks reloaded files and did not run the model or processor.
An additional [NumPy-only reader](../../scripts/audit_native_suffix_replay_words.py)
matched the stored additions for both traced arms to the untraced terminal
words, recomputed every complete prefix and per-tensor digest, and joined
the actual supplied IDs back to the Note 0072 rows. Its
[receipt](../../examples/research_notes/0076_native_suffix_replay_parity/independent_word_check.json)
is a file-format cross-check, not a second VLM implementation.
Full native-word arrays remain local in `runs/native_suffix_replay_calibration_v2`;
the tracked receipts alone cannot reconstruct the arrays.

Verified summary SHA-256: `6fc71b836429bcc080bbe2221d391cfec0d840c2b494b83f75d863cebb9b44f4`.
Capture receipt: `6302892632b6b3f8fbe4025ba813dfd560106dd1bff02b06979a2007b45629f4`; preflight receipt: `837c30e2a5b6da9057a7c337cc5a4f6427a14b7d9ed321e0c595ec4077f9cbc8`.

All 590 local tests passed, including public-only checks of every decoder-ID
join, layer/role denominator, and retained v1 failure. All 84 frozen hashes
and 427 local links in the affected notes/navigation checked out. These are
local validations, not CI or an independent model replication.

**Next boundary.** The previous prompt edit changed the natural IDs in
30/32 paired images. A separately registered comparison could now give
both prompts the same two IDs (for example [785,2168]) on all 32, making
the generated positions token-aligned without selecting only the two natural
matches. v2 tested only each prompt's own IDs, not that crossed condition.
No new cache factorial, direct readout, causal intervention result or
untouched confirmation is reported here.
