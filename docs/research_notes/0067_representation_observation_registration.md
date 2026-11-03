# Research Note 0067: Representation Observation Registration

Date: 2026-11-03

Status: historical calibration implementation and acceptance criteria only.
Commit this complete measurement/verifier path before the first model replay.
No encoder/projector correspondence panel or confirmation cohort is released.

Execution report: [Note 0068](0068_representation_observation_calibration.md).
Registration commit: `7ad7d8219e4ff4ad3c760c3e508e746ea2c16ffd`.

## Why This Gate Comes First

[PR #5](https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/5)
merged Notes 0064-0066 at `dff3101`. The non-learned panel shows that raw
8 x 8 mean pooling can already produce positive zero-overlap correspondence.
It does not reproduce the full cache profile or isolate a learning effect.
The next bounded task is to observe existing intermediate representations
without silently replacing the qualified inference path.

The existing 51 frozen measurement/configuration files stay unchanged.
[This registration](../../configs/representation_observation_calibration_v1.json)
adds a separate observer, runner, verifier and tests. The previous baseline
publication hash is pinned. All six non-learned paths, both starts and their
negative/unavailable cases remain part of the later comparison surface.

## Exact Sites And Numerical Convention

The qualified MLX-VLM FastVLM path casts pixels to model dtype, runs the
vision tower, takes its spatial output, reshapes H/W to tokens, and calls
`mm_projector`. Observe only that existing call:

| Site | Object Held | Expected Layout |
| --- | --- | --- |
| encoder | Projector positional input 0, after the existing tower-output reshape | `[1,256,3072]`, batch/token/channel |
| projector | Exact return object from that projector call, before multimodal insertion | `[1,256,896]`, batch/token/channel |

These are the final encoder-output and full projector-output sites, not a
sweep across hidden encoder layers. Native dtype must be bfloat16. The
projector is the installed `CallableModuleList` containing Linear/GELU/Linear;
the relevant FastVLM implementation, vision and configuration sources are
hash-pinned. The encoder's 16 x 16 ordering follows the existing H/W reshape;
no cache-token-to-2D-patch localization claim is added.

The wrapper delegates the original arguments exactly once and returns the
same result object. It holds only input/output references. There is no
evaluation, copy, cast, array inspection, pooling or extra model computation
inside the wrapper. The class-level method wrapper is identity-scoped to the
one target projector; other instances pass through unchanged. A `finally`
block restores the exact original method on success, missing/duplicate calls,
nested observation or exceptions. Fixtures check identity and non-evaluation.

Only after the original replay, generation and selected-cache persistence
return are held arrays expanded losslessly to float32 and saved. Retain the
complete tensor, all 256 tokens and channels. Do not crop padding, center,
whiten, pool or normalize. Independently reloaded float32 arrays must be
finite, have the fixed shape and have the exact bfloat16 expansion bit layout.
Native graph identity is qualified by frozen code/fixtures and path parity,
not by a second independent encoder implementation.

For a future separately released panel, the provisional fixed convention is
full-tensor C-order flattening, float64 `(JJ-MJ)-(JM-MM)`, then complete-vector
L2 normalization for cosine. No alignment between encoder, projector, cache
or baseline coordinates is assumed. This calibration computes none of those
contrasts, scores, bootstrap ranges or retrieval results. A future release
still needs its endpoint family, budget, sealing, analysis and verifier.

## Fixed Historical Budget

Use all 128 qualified historical source cells, sorted by existing pair ID
and then MM/JJ/MJ/JM. For **each** cell, execute three fresh source contexts
with one shared qualified runtime:

1. OFF before: the unchanged original path.
2. ON: observe the two sites without changing the returned computation.
3. OFF after: prove the original method is restored before another execution.

This is 384 fresh source contexts, 1,152 historical selected-cache tensor
comparisons, and 768 direct ON-versus-OFF tensor comparisons. The term
`new_model_forwards` counts fresh source contexts as in previous repo
calibrations, not internal autoregressive generation steps. There are 128
observed cells and 256 full representation arrays. No replacement, retry,
resume, tolerance relaxation, new map or new experimental source is allowed.

Every context must match the historical native/effective prepared inputs,
pixel bytes and `image_sizes`, prompt, frame hashes, token partition, response
`The image`, actual generated IDs `[785,2168,2168]`, source seed and runtime.
L1 keys, L12 keys and L23 values must match exactly over their complete
`[1,2,397,64]` tensors, including pre-image/image/post-image positions.
ON must also match both OFF brackets byte-for-byte, with exact restoration
and exactly one target projector call. No forced suffix or alternate
processor path is introduced.

Require both offline flags and 8 GiB free disk. First execute no-model-load,
no-observer preflight: check historical evidence, all 896 input gates and
3,584 cells, the complete existing code hashes, installed observation sources,
model snapshot and committed new code. A failure stops and preserves partial
outputs, with no experimental release. Calibration success is bounded to
this model, runtime, source set and suffix, not every input or hidden tensor.

## Independent Verification And Release Boundary

Commit the publisher before model calls as well. It makes no model/processor
calls, reloads all 1,152 cache sidecars, verifies every input/suffix/layout
and repeats the 768 direct ON/OFF byte comparisons. It also reloads all 256
representation arrays and checks hashes/layouts/dtypes/finiteness. Publish
compact cell, historical-comparison, ON/OFF-comparison and representation
receipt ledgers. Full calibration arrays and cache tensors stay local;
public receipts alone do not reproduce the model computation.

Preflight, run and publication use fresh roots:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_representation_observation_calibration.py preflight \
  --output-root runs/representation_observation_preflight_committed_v1

PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/run_representation_observation_calibration.py calibrate \
  --preflight-root runs/representation_observation_preflight_committed_v1 \
  --output-root runs/representation_observation_calibration_v1

PYTHONPATH=src python3 scripts/summarize_representation_observation_calibration.py \
  --run-root runs/representation_observation_calibration_v1 \
  --output-root runs/representation_observation_verified_v1
```

Note 0068 will report the actual success/failure denominators. A passing
result qualifies an observation path only. Repeat calibration in the same
process before a separately registered representation panel. Neither training
attribution, semantic invariance, causal gain, equivalence nor confirmation
is established by exact instrumentation parity. Experimental cache/direct
counts remain unchanged; this calibration has its own denominator.

## Pre-Registration Operational Record

Before this registration commit existed, a commit command failed because the
local Git author was unset. An erroneously issued no-model preflight then
rejected the uncommitted implementation at the commit gate. Its untouched
`runs/representation_observation_preflight_v1/execution_failure.json` records
zero attempted/completed source contexts. No model was loaded, no calibration
was attempted and no acceptance rule or input was changed from that attempt.
Retain it as an unreleased operational failure, not a passing preflight.
After successful registration, the one committed-design preflight uses the
distinct fresh root above. The no-retry/no-resume rule applies to the released
calibration; no failed model replay may be replaced by this distinction.
