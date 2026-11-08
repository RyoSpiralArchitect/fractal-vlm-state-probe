# Research Note 0089: Matched-Source Construction And Analysis Core

2026-11-08. **Software implementation and toy verification only.** This is
a bounded core for the unchanged [Note 0088 design](0088_stat_matched_source_registration.md),
not an acquisition runner or a panel release. No allocated source, registered
320px candidate, new registered coordinate map, native processor call, model load or
representation observation is created by this work.

## Fixed Boundary

The [core profile](../../configs/stat_matched_source_core_v1.json) pins all
three registration files at `5e5dab4fdb55161609015e9c9bfb888f51e9f08d`.
The inherited 118 files and existing input/feature/observer implementations
remain unchanged. Passing software checks cannot release acquisition:
`acquisition_released=false`, with no execution runner. A public freeze
and separate execution approval remain required before registered sources.

## Implemented Components

[The core](../../src/fractal_vlm_state_probe/stat_matched_source.py) provides:

- Frame-zero RNG-initializer enumeration and collisions against supplied
  historical metadata. All 104 planned initializers are distinct from each
  other and known r1-r4. No RNG is drawn. The supplied historical inventory
  is explicitly incomplete, not proof of untouched sources.
- Collision reporting for supplied decoded-RGB hashes. This checks neither
  container hashes nor the provenance of a caller-supplied digest.
- Payload extraction in registered C order, the existing stable luminance
  rank matcher, embedding, and rearrangement via a supplied uint32 bijection.
  Palette matching precedes embedding/shifting. Full raw palette equality
  remains separate from within-pool/native matching.
- Exactly 512 ordered cell identities, all 256 references before tests.
  A metadata-only reference seal checks identities, ordinals and required
  digests; it explicitly does not independently verify artifact bytes.
- Full float64 factorial contrasts, fixed low-norm policies, two-replicate
  references, cancellation handling and lazy loading of complete cells.
- Fixed-reference scores, four eight-by-eight matrices per view, paired
  margin changes, direction/norm diagnostics and unavailable rows. No
  complete-case endpoint substitution, refit or threshold winner selection.
- An enumerated uncompressed storage plan within the 16 GiB payload
  allowance, with overhead/reserve additional and launch still 28 GiB.

The enumerated payload is 12,165,746,688 bytes (about 11.3 GiB), including
native arrays, packed sites, selected caches, optional complete interactions
and scratch. This is a prospective plan, not measured runner storage use.

`qualify_core` checks pinned registration/publication files and the inherited
118 hashes only. It returns `runtime_qualified=false` and
`acquisition_released=false`, not a native/runtime preflight. Likewise,
`rearrange` checks a supplied map's bijection, not its qualified identity;
the future runner must join its digest to the pinned four-map ledger.

## Independent Software Verification

The [separate oracle](../../src/fractal_vlm_state_probe/stat_matched_source_oracle.py)
uses explicit pixel traversal, Python stable sorting and scalar assignments,
without importing the production constructor. Its numerical path uses scalar
contrasts, `math.hypot` norms and `math.fsum` dot products/aggregation rather
than the production NumPy path. Only specified formulas/tie behavior are shared.

[Tests](../../tests/test_stat_matched_source.py) use 80 x 80 software fixtures,
2 x 2 pools, 576 payload pixels and analytic three-coordinate vectors.
They do not call a registered generator. A pointwise fake native tensor
checks feature plumbing, not actual processor behavior.

Checks cover all four arrangements, stable ties, full raw palette equality,
within-pool RGB multisets, all twelve toy feature paths, expected raw-global
zero, word-identical shams and a deliberately failing palette sentinel.
Scalar/NumPy scores and paired endpoints agree within absolute 1e-12,
relative zero, on the software fixtures at all three learned floors.
Missing references/tests remain in fixed denominators. Invalid maps/shapes,
nonfinite/overflow/underflow values, altered registration bytes and incomplete
reference metadata fail closed. These are software observations only.

## Remaining Acquisition Gates

This is not the 920-context experiment. Still implement and freeze bounded
capture orchestration, complete historical inventory, actual decoded source
audits, qualified-map joins, every native matching/oracle gate, fresh
historical OFF/ON/OFF calibration and shams on the same retained model load.
Back the metadata reference seal with independently verified input/site
arrays before generating test sources.

The independent file-only verifier, immutable first-failure/partial-capture
receipts, runtime/model fingerprints, exact call accounting and two
byte-identical complete replays also remain future work. Spectral diagnostics
and full capture verification are not supplied by this core. No native
matching or learned-representation conclusion follows from toy agreement.
Confirmation remains not generated and not sealed.

## Local Validation

All 26 new software checks and all 731 local tests pass. Scoped Ruff,
whitespace and 500 local link targets in the six touched Markdown documents
pass. File qualification confirms all 118 inherited and three registration
files unchanged. These are local software checks, not CI, a native processor
calibration, independent capture replay or an acquisition release.
