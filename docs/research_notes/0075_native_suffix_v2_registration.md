# Research Note 0075: Releasing Own-Suffix Calibration v2

2026-11-04. Separately authorized after the [retained v1 entry failure](0074_native_suffix_replay_entry_failure.md),
before any v2 model forward. v1 reached zero forced decoder calls and has
no parity result. Its failed attempt, two completed native arms, commit
`771f785`, raw failure digest and zero-model audit stay fixed; **no part
of v1 counts toward v2**.

## Single Interface Change

The outer MLX-VLM stream omits `thinking_budget_criteria` when inactive.
The new entry wrapper passes that omission to the original stepper as
explicit `None`, exactly as it handles an explicitly null argument.
Reject every non-null value, including false-like values. Preserve all
other arguments and all original decoder, capture, comparison, missingness
and stopping code. The [v2 config](../../configs/native_suffix_replay_calibration_v2.json)
is checked against v1 field by field, allowing only release identity, fresh
paths and the retained failure/entry contract to differ.

Use the *installed outer stream* in a weight-free dispatch fixture before
real acquisition. Run all three suffix forms with an omitted, null or
invalid non-null criterion. For the six valid cases check the actual fake
decoder inputs, notification pattern and terminal cache IDs, even when the
fake logits do not select the given IDs. All three invalid cases must stop
before fake decoding. This is an integration fixture, not scientific data.

## Unchanged Measurement And Stop

The complete [Note 0073 contract](0073_native_suffix_replay_registration.md)
applies. Run every one of the 96 Note 0072 historical rows anew, in the
same order and on one model instance. For each row: native plain, native
traced, then forced own. Before releasing the third arm, require all 48
native tensors at the three traced boundaries to match the plain terminal
prefixes. Actual forced decoder inputs must be the two registered IDs,
one call each, including end-of-turn. Compare all 24 layers x K/V at
395, 396 and 397 effective positions bit-for-bit. Keep the three historical
targets distinct from newly acquired full native cache.

288 new source contexts, 192 explicit single-token calls, 13,824 paired
native/forced tensors. No tolerance, exclusions, crossing suffixes,
retry/resume, direct-vocabulary probe or confirmation. Preserve and stop
at the first gate failure. Two fresh no-model verification outputs must
agree before interpreting the complete panel. Failure need not imply cache
inequality. Commit this version, registration, fixtures and launchers first.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_native_suffix_replay_v2.py preflight --output-root runs/native_suffix_replay_preflight_v2
python3 scripts/run_native_suffix_replay_v2.py capture --output-root runs/native_suffix_replay_calibration_v2
python3 scripts/run_native_suffix_replay_v2.py verify --output-root runs/native_suffix_replay_verified_v2
python3 scripts/run_native_suffix_replay_v2.py verify --output-root runs/native_suffix_replay_verified_replay_v2
```
