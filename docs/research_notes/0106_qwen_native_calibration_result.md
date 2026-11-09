# Research Note 0106: Native Qwen Observation Calibration Result

2026-11-10. The single acquisition authorized in [Note 0105](0105_qwen_calibration_execution_release.md)
completed under the unchanged [Note 0102](0102_qwen_calibration_registration.md)
plan. Both ordered independent saved-data checks passed and their complete
output trees are byte-identical. No retry, replacement or limit change occurred.

## Result

All 34 existing cases, including both shams, completed OFF/ON/OFF:
102 fresh model contexts, 204 decoder calls and 136 combined processor
preparations. Each verification checked all 8,266 capture files other than
the top-level manifest, whose digest is retained separately. All registered
input/cache/logit/score word, dtype, metadata, restoration, layout and
inverse-window gates passed. The comparison is exact, not tolerance-based.

Observed layout is 484 patches with a native grid of `[1, 22, 22]`,
261 prefix IDs and a full 36-layer active cache of 262 positions.
Every K/V tensor is `[1, 2, 262, 128]`. The native first-yield prefetch
therefore remains explicitly prefix plus one, not a prefix-only cache.
Both raw last-position logits and the first-yield scores retain all 151,936
coordinates; their values are not used as a semantic endpoint here.

The ON arrays are complete: premerger `[484, 1280]`, auxiliary merger
`[121, 2048]` and final vision `[121, 2048]`. The independent padded-window
inverse reproduces final-return words exactly in every ON case. Native pixels
are float32; K/V, raw logits, scores and all three visual sites are float16,
saved as lossless float32 expansions. Native IDs/masks/grids are int64.
Source dtypes are recorded, not inferred from the saved float32 container.

The saved tree occupies 2,633,447,156 bytes, below the calibration-only caps.
The 18 GiB launch gate is not a joint-cohort storage qualification.

## Evidence

The [result receipt](../../examples/research_notes/0106_qwen_native_calibration_result/summary.json)
binds the release, preflight, source/capture manifest and both verification
outputs. The [metadata bundle](../../examples/research_notes/0106_qwen_native_calibration_result/capture_metadata.tar.gz)
retains exact capture/frozen manifests, all row/processor receipts and the
complete attempted/returned operation trace. The full lossless NumPy arrays
remain in the local canonical capture root; they are not in the metadata
bundle. Public receipt replay is not a substitute for native-array verification.

Both canonical verification roots contain the same `summary.json`, with SHA
`9ed394ad992196426456b464c126d9f0c6f2e522bcb23c4f0751767fe575554c`.
Capture SHA is
`351bdda6fc9f1db8a0ebf386d3bfc00374cead1d2ed8f77a6367ef339c51e6e6`.
The first capture remains untouched and all raw sources retain their original
file and decoded-word receipts. No model call was added during verification.

## Claim Ceiling And Next Gate

This establishes observation parity and saved-layout validity on these 34
existing, fixed-size development cases only. The verifier's broader
`native_runtime_qualified` flag remains false; no arbitrary-input or general
runtime qualification is claimed. No new visual-direction/correspondence,
semantic, learned or causal result follows from these calibration gates.

The next bounded task is a separately registered native Qwen input calibration
for the proposed map rule and all twelve input-feature views. It is not
authorized here. Untouched FastVLM plus Qwen cohort generation and acquisition
remain withheld, together with joint precision/storage qualification.
