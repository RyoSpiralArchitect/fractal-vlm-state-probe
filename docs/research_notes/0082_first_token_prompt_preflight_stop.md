# Research Note 0082: Readout Panel Stopped At The Storage Gate

2026-11-06. **Status: NOT RUN.** The
[registered 32-image panel](0081_first_token_prompt_panel_registration.md)
did not start calibration or experimental acquisition. No paired TV, JS
or other readout effect was calculated. The missing result is **null**,
not zero, absence, cache mismatch or a negative scientific finding.

## Exact Boundary Reached

The implementation/configuration was frozen before preflight at
`8363ae08a2e6c332712278a6dad1849c1cffb808`, with all 656 local tests
passing. The first registered check compared free disk with the complete
uncompressed allocation. It rejected the available space **before**
qualification, model loading or any model/processor forward.

| Stage | Completed |
| --- | ---: |
| Fresh native/prefix calibration contexts | 0/288 |
| Experimental prefix contexts | 0/64 |
| Complete paired readout effects | 0/32 |
| Model weights loaded | No |
| Portable panel score archive | Not created |

The failed preflight directory contains only `failure.json`; no model
snapshot, registration receipt or passing preflight was produced. The
acquisition directory `runs/first_token_prompt_panel_v1` was not created.
The registered failure and current 103 code/config/registration hashes
remain unchanged. A passing unit suite is not a passing disk or model gate.

## Storage Evidence

The registered launch requirement is **12,041,764,864 bytes (11.215 GiB)**:
2,396,504,064 calibration bytes, 518,455,296 panel bytes, 536,870,912 bytes
of metadata/portable-score/scratch allowance, plus 8,589,934,592 bytes of
reserved free space. No ZIP compression savings were assumed.

The original exception did **not** include a free-byte measurement, so
`free_at_failure_bytes` stays null. A separately retained observation
after the stop measured **9,137,606,656 bytes (8.510 GiB)** free, still
2,904,158,208 bytes below the requirement. Disk availability changed during
the turn; this later observation must not be presented as the exact value
at the failing check or as a current promise about future capacity.

No old run, user file or cache was deleted/moved, and neither the reserve
nor the sample size was reduced. No same-attempt restart, alternate
acquisition namespace or computation from earlier calibration scores
was used to bypass the gate.

## Receipts And Next Decision

- [Failure audit and null endpoint](../../examples/research_notes/0082_first_token_prompt_preflight_stop/summary.json)
- [Timestamped post-failure disk observation](../../examples/research_notes/0082_first_token_prompt_preflight_stop/storage_snapshot.json)
- [File-only audit implementation](../../scripts/audit_first_token_preflight_failure.py)

Original failure SHA-256:
`41425d1837b6fbe219dba834a6b0d8e8571813b183313ca855645f55107b051e`.
The audit verifies all 103 frozen Git blobs and zero attempted contexts
without reading score vectors or calling a model. Replaying the **same**
recorded storage observation reproduces the audit, not a new disk sample.
Audit summary SHA-256:
`73c423d005654b3c24648c7f043d805f8e2232687fa563c484eebdfacb8128a5`.
All 658 local tests pass, including public-only checks that missing effects
cannot be counted as zero. Scoped Ruff, whitespace and 453 local links
pass. These are software/document checks, not a passing launch gate or CI.

Resume requires a storage choice: make sufficient space available without
losing evidence, or name a different target filesystem. A separately
registered release should retain this failed namespace and the same
probability definition, endpoints, source order, denominators and reserve.
No new model acquisition is authorized merely because space later changes.
The existing Note 0080 measurement-path qualification remains valid within
its earlier scope; this stopped panel adds no new scientific effect.
