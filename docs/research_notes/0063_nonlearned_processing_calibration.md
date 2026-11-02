# Research Note 0063: Non-Learned Processing Calibration

Date: 2026-11-02

Status: PASS for registered calibration and independent publication checks.
No real-image baseline correspondence panel, model forward, encoder/projector
extraction or confirmation cohort was added.

## Registration And Preflight

[Note 0062](0062_nonlearned_processing_registration.md) and
[`nonlearned_processing_phase3_v1.json`](../../configs/nonlearned_processing_phase3_v1.json)
were committed at `5b44429c52e4519a5328692bc2bc8b53c6bd747c` before the run.
All 480 repository tests passed at registration. The 38 prior measurement/config
files remained unchanged; five new files bring the frozen set to 43. The
experiment used the registered definitions, fixtures, seeds and tolerances
without retry, resume or replacement.

The no-processor/no-model preflight passed: all 28 maps, 896 four-cell gates,
3,584 input cells, source hashes, split, runtime and prior publication ledgers
qualified. Processor calls and model forwards were both zero at this step;
neither a processor instance nor model weights were loaded. The frozen source
inventory has 7,442 file hashes, not 7,442 independent observations.

Frozen specification SHA-256:
`5057400cd93ad66a289ea7a4d07b08fde4af0df96c37baa3466387cd553c4b7c`.

## Synthetic Calibration Results

Six synthetic patterns at 32 x 32 plus full-resolution noise on each input
start produce seven cases per start, each through all six paths. These
**84 feature/oracle comparisons** all pass the fixed `1e-10` maximum absolute
tolerance. Every saved feature is float64. The separate implementation uses
1D convolution and explicit grid slices, not cumulative sums/reshape pooling.

| Feature | Raw: maximum absolute error over 7 cases | Processor-shaped synthetic: maximum over 7 cases |
| --- | ---: | ---: |
| global_moments | 0 | 1.0700e-12 |
| grid_mean | 1.7764e-15 | 0 |
| small_mean | 1.8874e-15 | 1.5543e-15 |
| small_energy | 1.1657e-15 | 4.4409e-15 |
| large_mean | 1.7764e-15 | 2.8866e-15 |
| large_energy | 1.7764e-15 | 1.0214e-14 |

The largest difference is a global floating-reduction order difference, not
an experimental correspondence effect. Processor-shaped synthetic values
exercise signed normalized channels; they do not simulate the actual processor.

The six linear contrast-commutation checks pass, with maximum residual
`1.1241e-15`. All four nonlinear-order witnesses have absolute discrepancy
**0.625**, agreeing with the closed-form constant-cell calculation. Applying
the nonlinear feature to an interaction cannot replace applying it to each
cell first. These are deterministic implementation checks, not model evidence.

The raw joint-RGB-permutation global-moment control cancels **exactly**.
Its direction is explicitly unavailable and is not given a cosine score of
zero. The retained 85th synthetic array is this six-dimensional zero vector.
No numerical threshold was tuned from real-image feature norms.

## Historical Processor Calibration

All **128 historical processor cells** reproduce the original native
`[1,3,1024,1024]` float32 pixel bytes by SHA-256, and all non-pixel payloads
match exactly. This uses the unchanged qualified processor/runtime and local
snapshot. There were 128 processor calls and **zero model forwards**. No
baseline feature was computed on these source images. The model weights and
an image-conditioned cache were never loaded.

These are processor replays, not the 384 cache-tensor replays from Phase 2.
There are no new experimental source-cache cells, target tensors, direct
probes or baseline correspondence vectors. Experimental cache totals remain
3,816 cells / 11,304 tensors / 2,826 target analyses; direct-probe totals remain
104 factorials / 416 cells / 1,664 full-vocabulary sidecars.

## Independent Verification And Artifacts

The separate publisher reloads all 85 saved synthetic arrays, reconstructs
the 84 nonzero-or-zero feature cases with the independent oracle, repeats
the six linear checks and checks all four nonlinear witnesses against 0.625.
It verifies preflight, code/source hashes and all 128 historical comparison
receipts against the frozen jobs. This adds no processor call or model forward.

Historical processor tensors are **not saved or independently reloaded** by
the publisher; the original calibration compares their actual bytes by hash.
Receipt verification is not an independent replay of those pixel tensors.
The small synthetic arrays are included and can be regenerated without model
weights or the original research images.

- [Summary and frozen provenance](../../examples/research_notes/0063_nonlearned_processing_calibration/summary.json)
- [84 independently checked feature records](../../examples/research_notes/0063_nonlearned_processing_calibration/feature_checks.jsonl)
- [85 synthetic feature arrays](../../examples/research_notes/0063_nonlearned_processing_calibration/synthetic_features.npz)
- [128 processor comparison receipts](../../examples/research_notes/0063_nonlearned_processing_calibration/processor_checks.jsonl)

Publication summary SHA-256:
`d78a58db2470f71239c515e684db120ca8a066c8c9a5460e219e5db7c6c5affc`.
Synthetic arrays SHA-256:
`97ad9037f77bbffb9f0853b1ea64596ab52ce7a71432df3a73dc499759de2a42`.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python3 scripts/summarize_nonlearned_processing_calibration.py \
  --run-root runs/nonlearned_processing_calibration_v1 \
  --output-root /tmp/nonlearned_processing_calibration_verified_copy
```

Use a fresh publication directory. The source run contains actual processor
comparison receipts and the full frozen source inventory; a clean checkout
does not necessarily contain the original research image paths.

All **484 repository tests pass**, including injected processor-failure
retention, no-resume guards, undefined directions, corrupted publication
receipts and portable synthetic-array reconstruction. Ruff and whitespace
checks pass. A second publication run reproduces all **four public files
byte-for-byte**, without another processor call. All 43 frozen measurement
and config hashes remain unchanged.

## Decision Boundary

The first Phase 3 calibration step is complete. It establishes an implemented,
tested feature definition and a currently exact processor input path, **not**
baseline performance or sufficiency. No cache/input margin comparison follows.
The original config retains `not_implemented_not_released` for the planned
real-image panel, as a historical registration field.

Next, separately release the committed/tested full-panel capture, reference
seal, analysis and independent publication paths, with fresh processor
calibration before any real-image feature score. Keep all twelve views and
24 endpoints, including the unavailable raw global-moment direction. Do not
remove negative families, retune scales on cache comparisons or introduce an
equivalence/semantic conclusion. Encoder/projector paths remain unqualified;
confirmation remains not generated and not sealed.
