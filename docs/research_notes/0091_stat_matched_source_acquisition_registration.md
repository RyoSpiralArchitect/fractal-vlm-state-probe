# Research Note 0091: Matched-Source Acquisition Plan, Start Withheld

2026-11-08. Register the acquisition plan and permit zero-call preparation
only. **No source generation, native processor call, weight load or model
context is authorized now.** The operator approved registration, independent
read-only review, PR/main integration and zero-call preparation, explicitly
holding the scientific acquisition for a later decision.

## Fixed Implementation And Authority

The [registration](../../configs/stat_matched_source_acquisition_release_v1.json)
binds all **135 selected files**: the 126 preceding preserved files plus all
nine Note 0090 implementation/profile/test/note files. Their hashes match the
tracked tree at `45ad7db162194ebe7cb596526ec1953605af6700`, published by
[PR 11](https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/11).
The [saved publication receipt](../../examples/research_notes/0091_stat_matched_source_acquisition_registration/code_freeze_pr11.json)
is separately hash-bound. No frozen scientific or implementation file is
changed by this registration. Current software evidence is the 810-test
clean-archive and merged-main checks recorded on PR 11, not native calibration.
The receipt preserves the queried GitHub fields and adds only this repository's
`schema_version` and `analysis_kind` metadata; it is not raw API-response bytes.

The canonical file uses the future acquisition-release ID, but
`approved_for_execution=false`. Its filename and public merge are not start
permission. Passing it to the existing native execution gate is deliberately
rejected before qualification. No field in the Note 0090 preparation profile
is changed to bypass that gate.

## Registered Plan, Not Observed Counts

The unchanged [Note 0088 design](0088_stat_matched_source_registration.md)
allocates 64 donor frames, 665 processor-only calls and 920 model source
contexts. All 256 reference cells, full arrays and fixed family references
must be independently checked and sealed at context 664 before any test
source is generated. The first test model context is 665. Decoder calls are
recorded separately; none of these planned counts is an observation here.

Keep all twelve baseline views, both full visual sites, three learned
amplitude floors, unavailable directions and family exceptions. No pair,
feature, reference, tolerance, seed, map or stopping rule is reselected.
The 28 GiB launch gate, eight GiB reserve, bounded payload and no-retry policy
remain unchanged. Space shortages do not authorize deletion or smaller gates.

## Zero-Call Preparation

After public registration, run only this file/fingerprint preparation:

```bash
env PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 scripts/run_stat_matched_source.py preflight \
  --output-root runs/stat_matched_source_prepare_v1
```

There is intentionally **no `--release` argument**: the current registration
does not approve execution. The existing CLI uses the separate preparation
root, instantiates no processor, loads no weights and makes no model call.
It checks selected files, historical artifacts, installed-source provenance,
model snapshot metadata, available source inventory and free disk space.
Reading/hashing local files is not native pipeline calibration.

Success is `PREPARED_NOT_RELEASED`, never `PASS_FOR_APPROVED_RELEASE` or a
scientific result. Preserve the full local frozen specification and receipt;
publish only a compact count/hash receipt rather than private path inventories.
If preparation fails, preserve its first failure and fresh-root state. Do not
retry, change a gate or modify a frozen implementation in this registration.

## Later Start Decision

Report the preparation outcome before asking for a separate start approval.
Even a successful preparation cannot be supplied as the executable preflight:
its canonical root and status differ and it has no approved-release binding.
A future start requires separately committed/publicly merged execution
authorization and a fresh matching zero-call approved preflight, followed by
the unchanged in-process processor/model calibration gates. No acquisition
command is run or released by this step.

Acquisition results remain unmeasured. No claim of learned specificity,
semantic steering, mediation, equivalence or untouched confirmation follows
from registration or file preparation. Confirmation is not generated or sealed.
