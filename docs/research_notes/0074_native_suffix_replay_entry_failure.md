# Research Note 0074: Own-Suffix Replay Stopped Before Decoding

2026-11-04. **v1 status: not qualified; native/forced cache equality is
unmeasured.** The [registered calibration](0073_native_suffix_replay_registration.md)
stopped on its first forced context. This is an adapter call-signature
failure. Neither cache agreement nor a numerical mismatch was observed on
the forced path.

| Gate | Result |
| --- | --- |
| Historical inputs and registration | 96/96 rows passed offline preflight; no weights loaded; 79 frozen files |
| Contexts | 3/288 attempted, 2/288 completed: native plain and native traced, first ACK image only |
| Control on that one image | All 48 terminal K/V tensors equal between native paths; all traced checkpoints at lengths 395/396/397 match the corresponding plain terminal prefixes |
| Historical selected targets | Both completed arms matched all three historical targets (6 comparisons) |
| Forced arm | Input prepared; **0 forced decoder calls**, 0 forced suffix tokens consumed |
| Calibration endpoint | **0/96 completed triplets; 0/13,824 possible native/forced tensor comparisons.** No pass rate, equality or inequality estimate |

The row was `geometry_checker_hex_r1/mm/ack_before`, assigned native IDs
[785,2168]. Raw comparisons on the two controls include all 24 layers'
effective keys and values, not allocated slack. The two other native suffix
forms (end-of-turn or comma) were not reached. No other image was acquired.

## What Failed

In the pinned MLX-VLM stream the disabled thinking-budget path does not pass
`thinking_budget_criteria` to `generate_step`. The replacement function
required this keyword, even though it accepted `None` as its value. Python
argument binding raised `TypeError` before the replacement ran prefill. The
first fixture checked explicit `None` and missed the real omitted-key case.
There is a processor input receipt for the failed arm, but no decoder state,
checkpoint or completed source output. Do not count the native matches as
evidence for the forced path.

The [zero-model audit](../../scripts/audit_native_suffix_v1_entry_failure.py)
re-read the retained arrays and independently compared native prefix words
and all 48 per-tensor hashes at each length. A fixture ran the installed
outer stream to the dispatch boundary with a fake processor/cache and no
model forward. The observed keyword names were `image_sizes`, `max_tokens`,
`prompt_cache` and `temperature`. Binding these to the frozen v1 signature
reproduced the missing argument; explicit `None` binds. This fixture
diagnoses the interface only; it cannot qualify real replay.

## Receipts And Decision

- Pre-outcome Git commit: `2cdc19600d0fc6ea6a285e5d6e13fbcb5f96cb4b`.
- Preflight SHA-256: `d08c8a2b2646ed507c49064d4de05ef08527f39d657b7371f92f558b3dcf03c3`.
- Original failure SHA-256: `f14262ad86faf6dfaf18492212ab4aeda134803f1300a4df9b6298aa09bbe804`.
- [Failure audit](../../examples/research_notes/0074_native_suffix_replay_entry_failure/summary.json):
  `7ba8bcf827ce5fa2b1047cb79529e4c237b95490ff496631a6832c7d67729ac8`.
  A second no-model audit produced the identical file.
- [Two accepted arm receipts](../../examples/research_notes/0074_native_suffix_replay_entry_failure/arms.jsonl);
  all partial native arrays and the original failure remain at
  `runs/native_suffix_replay_calibration_v1`.

All 578 local tests pass, including the check that the published failure
cannot be counted as a forced-cache comparison. All 79 frozen code/config
and registration hashes remain unchanged. This is not a passing model gate.

The v1 run is **closed, not resumed or replaced**. Any v2 must have its
own registration and fresh outputs, accept an omitted or explicitly null
criterion as disabled, reject an active criterion, and test the actual
outer-stream dispatch without weights before a real forward. Retain exactly
the same inputs, schedule, endpoints, all-word criteria and stopping rule.
No cross-prompt suffix, direct readout or confirmation measurement here.
