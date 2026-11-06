# Research Note 0083: Same Readout Panel, Separate v2 Release

2026-11-06. The operator chose to recover internal-disk capacity and use a
separate release after the [zero-forward v1 stop](0082_first_token_prompt_preflight_stop.md).
No v1 calibration or experimental context ran; its failed namespace and
null effect remain fixed. Retained commit:
`bdebbd6e63831b8588d9706e08dc18bc227730ae`.

## Exactly What Changes

The [v2 config](../../configs/first_token_prompt_panel_v2.json) changes
release identity, fresh canonical output paths, retained v1 provenance
and a hash-bound storage-recovery receipt only. A field-by-field check rejects any changed scientific
condition, probability definition, endpoint, aggregation, stopping policy
or storage allocation. The [complete Note 0081 design](0081_first_token_prompt_panel_registration.md)
applies without modification. All its 103 frozen files stay byte-identical;
new code only binds the unchanged implementation to this release profile.

Preserve the original failure SHA-256
`41425d1837b6fbe219dba834a6b0d8e8571813b183313ca855645f55107b051e`
and [audit](../../examples/research_notes/0082_first_token_prompt_preflight_stop/summary.json)
`73c423d005654b3c24648c7f043d805f8e2232687fa563c484eebdfacb8128a5`.
Never reuse `runs/first_token_prompt_preflight_v1`, overwrite its failure,
or treat that attempt as calibration data.

## Unchanged Gates And Interpretation

Before preflight, sufficient capacity must exist on the internal target
filesystem. Keep **8 GiB reserved**, the complete **3,451,830,272-byte**
new-data/overhead allocation, and thus **12,041,764,864 bytes** minimum
free at launch. Capacity recovery may remove only specifically approved
disposable files, not prior runs, models, installed code, research evidence
or user files. Record any removal separately; installed/runtime/source
hashes must still match qualification. Space fluctuations do not qualify
a failed attempt, and a later shortage still stops v2.

The operator explicitly approved removal of **pip HTTP download caches only**.
The [bounded recovery receipt](../../examples/research_notes/0083_first_token_prompt_panel_v2/storage_recovery.json)
records 1,511,444,189 logical bytes removed (1,517,973,504 allocated bytes),
with all 13 local build wheels and their 25 total artifacts hash-identical.
Full wheel filenames remain in the local receipt, not the public artifact.
No installed package, model cache or scientific run was removed. The receipt
does not attribute the entire free-space change to cleanup; other processes
can change capacity. Qualification still checks actual runtime/source identity.

On one fresh model/processor instance, redo 96 rows x three calibration
paths (288 contexts), requiring the same whole-vector readout and cache
identities as Note 0080. Seal all calibration rows, then acquire all 64
suffix-free experimental conditions in the original ACK/YES order for
32 development images. Do not inspect or substitute effects from the old
calibration arrays. No sample, family, native-score support or model-output
coordinate is removed.

Primary: mean paired full-vocabulary TV across all 32 images, with
float64 renormalization of the native scores over all 151,936 coordinates.
Keep every paired result, source/cell summary, JS divergence in nats,
Hellinger, delta norms, argmax sets and the predefined top-20 diagnostics.
Publish the lossless complete score archive and require two identical
zero-model verification outputs. One fresh v2 attempt only; no retry/resume,
replacement, p-value, CI, equivalence or untouched-confirmation claim.

Commit this release, launcher, fixture and note before the v2 preflight.
Passing software tests alone does not pass the disk/model gates.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_first_token_prompt_panel_v2.py preflight --output-root runs/first_token_prompt_preflight_v2
python3 scripts/run_first_token_prompt_panel_v2.py capture --output-root runs/first_token_prompt_panel_v2
python3 scripts/run_first_token_prompt_panel_v2.py verify --output-root runs/first_token_prompt_verified_v2
python3 scripts/run_first_token_prompt_panel_v2.py verify --output-root runs/first_token_prompt_verified_replay_v2
```
