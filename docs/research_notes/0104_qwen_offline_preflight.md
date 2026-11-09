# Research Note 0104: Qwen Offline Preflight

2026-11-10. The offline preflight for the unchanged [Note 0102
registration](0102_qwen_calibration_registration.md) passed using the
[Note 0103 implementation](0103_qwen_capture_verifier_preparation.md), merged
at `8d04c36c3f0ea511bfa19d54aec8e0ba02fa1ffa`.
This is admission evidence, not native calibration or confirmation.

## Observed Preflight

The [receipt](../../examples/research_notes/0104_qwen_offline_preflight/summary.json)
records an exclusive fresh `runs/qwen_observation_preflight_v1` root and the
complete preflight digest
`61a69ec3357a61c4f643d737d90c27b057506b8dc79718e30a424834b2b3535a`.
The [lossless compressed manifest](../../examples/research_notes/0104_qwen_offline_preflight/preflight.json.gz)
retains all file-level hashes. Decompression reproduces the exact original
preflight bytes and digest; the small receipt summarizes, not replaces, it.

- Immutable plan, registered sources and all 34 existing raw-file hashes match.
- 185 tracked observation software files match committed contents.
- The pinned local Qwen revision has 11 files, including weights, hashed by
  streaming bytes. No model weights were loaded into a runtime.
- Installed versions and 3,509 Python/native-library files across seven
  packages were frozen; the installed generation source matches registration.
- Free space is 52,421,210,112 bytes (48.821 GiB), exceeding the 18 GiB
  calibration-only launch gate. Capacity must be rechecked at capture admission.
- Offline environment flags were set. No download, processor instantiation,
  native context, new source image, cohort map or confirmation score occurred.

## Launch Boundary

The preflight status is `OFFLINE_PREFLIGHT_ONLY` and its execution-release
field remains false. No `examples/qwen_native_execution_release_v1.json`
was created at this checkpoint. A separate explicit calibration-only approval
must precede that release and the model backend constructor.

The proposed next action remains exactly 34 existing cases, each OFF/ON/OFF:
102 fresh contexts, at most 204 decoder calls including native prefetch,
and 136 combined processor calls. Persist full registered arrays, stop on the
first unexpected failure, and do not retry, replace cases or relax limits.
Only after successful acquisition may two ordered fresh saved-data
verifications run, without inference; complete trees must be byte-identical.

Neither launch approval nor a passing calibration releases cohort generation.
The joint FastVLM plus Qwen storage/precision budget, new-map matching and
all twelve native Qwen input-feature gates remain unqualified. The current
free-space observation is not a joint-budget qualification.

The subsequent explicit calibration-only approval is recorded separately in
[Note 0105](0105_qwen_calibration_execution_release.md). This preflight record
retains its original unreleased status.
