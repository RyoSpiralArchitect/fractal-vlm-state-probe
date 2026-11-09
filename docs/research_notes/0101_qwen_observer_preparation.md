# Research Note 0101: Capacity Recheck And Qwen Observer Software Preparation

2026-11-09. After the operator freed space, the read-only capacity check
observed **55,492,743,168 bytes free, about 51.682 GiB**. The earlier
FastVLM-only 44 GiB threshold is met at this snapshot. The joint FastVLM plus
Qwen threshold remains unknown, not satisfied by implication. This is not a
runtime launch preflight. No cleanup or deletion is performed here; Note 0099's
earlier insufficient-space snapshot remains intact.

## Software Preparation

The [new observation helper](../../src/fractal_vlm_state_probe/qwen_representation_observation.py)
implements the candidate Qwen boundaries identified in Note 0100 without
importing MLX, loading weights or calling a processor/model. It retains exact
existing Python objects at merger input and vision-tower return. The existing
merger return is also retained as an auxiliary object for future window-order
checks, not as an additional primary endpoint.

The helper follows the existing class-call wrapping pattern, forwards the
normal calls unchanged and restores both class methods on success or error.
The target vision signature is two positional arguments with only
`output_hidden_states=False`; the target merger accepts one positional
argument. Other instances are forwarded without recording. It allows exactly
one target attempt and successful record per site, rejects nested/mixed
observers and rejects inherited call definitions that would leave a new
class-dictionary entry after restoration. It does not evaluate tensors,
inspect their shapes, cast, copy, detach or add a forward. Concurrent
observation is not qualified.

## What The Tests Establish

Python stand-ins exercise OFF/ON/OFF call counts, exact argument/return object
identity, lazy objects that reject materialization/shape inspection, other
instances, missing/duplicate calls, signature drift, caught and uncaught
failures, nested instrumentation and restoration. A caught first native-like
failure cannot be used to retry the target inside the same observer context.
The initial 77 focused observer, existing observation, joint-scope, cohort
preparation, historical scope and example checks pass; lint passes.

These are software tests, **not native MLX OFF/ON/OFF parity**, and do not
establish bit-identical cache/logits, actual tensor shapes/dtypes, processor
matching, window-index interpretation or execution/memory neutrality on Qwen.
Native scheduling or graph-retention effects remain to be checked. The
[software/capacity receipt](../../examples/research_notes/0101_qwen_observer_preparation/summary.json)
pins code hashes and retains each unqualified gate explicitly.

## Remaining Gates

The next preparation is a complete Qwen native observation/input calibration
registration on development/synthetic material, with fixed jobs, budgets,
input metadata, cache/logit comparisons, shapes/dtypes, shams and independent
saved-data checks. No such runtime runner or launch approval is supplied by
this helper. Both processor input gates, the new map construction, joint
precision and the complete storage budget still need qualification before
confirmation generation.

The two-model commitment remains unchanged. The cohort has no new source
images, maps, processor outputs, model contexts or confirmation scores. The
local additions had not been committed, independently reviewed or pushed at
that checkpoint. [Note 0102](0102_qwen_calibration_registration.md) records
subsequent preparation-only review/integration, without native execution.
