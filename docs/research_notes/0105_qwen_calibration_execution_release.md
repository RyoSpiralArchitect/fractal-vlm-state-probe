# Research Note 0105: Qwen Calibration Execution Release

2026-11-10. Following the passing [offline preflight](0104_qwen_offline_preflight.md),
the operator explicitly approved the unchanged registered calibration and two
ordered saved-data checks, plus independent read-only review and PR/main
integration of the records and results.

The [separate release](../../examples/qwen_native_execution_release_v1.json)
binds the exact Note 0102 plan, complete preflight digest and all 185 committed
software hashes. Acquisition rechecks runtime/model hashes and the calibration
launch-space gate before constructing the backend. The capture/verifier code
remains the implementation merged in Note 0103; no adjustment or retry is
authorized by this release.

## Authorized Scope

- Model: the pinned local `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` revision.
- Sources: the same 32 development reference cells and two existing shams.
- One fresh calibration root, 34 cases in original order, OFF/ON/OFF.
- At most 102 contexts, 204 decoder calls and 136 processor preparations.
- Full registered inputs, cache, logits/scores and ON visual arrays retained.
- First unexpected failure stops all remaining stages and replay; no retry,
  replacement, dtype/cap relaxation or adaptive selection.
- Only successful capture permits two ordered fresh saved-data verifications,
  with zero extra inference and byte-identical complete output trees required.

## Withheld Scope

No cohort images/maps, new-map matching, new confirmation references, test
acquisition or joint-budget qualification is released. Passing calibration
does not by itself qualify these tasks or imply a visual/semantic result.

The execution result is recorded separately after the run. This release is
authorization, not an assertion that the native path or parity has passed.
