# Research Note 0093: Matched-Input Encoder/Projector Correspondence

2026-11-08. The one bounded acquisition authorized in
[Note 0092](0092_stat_matched_source_execution_release.md) completed without
retry, replacement or implementation change. Two fresh native saved-data
verifications returned `VERIFIED` with byte-identical output trees. The
fixed analysis returned `DESCRIPTIVE_PROFILES_VERIFIED`, with every score,
margin and paired change checked against the separate scalar scorer.

## Primary Result

All twelve fixed input-feature views satisfy the registered matching gates
across arrangements, but the full encoder/projector interaction directions
and their fixed-reference correspondence change.

| Full site | Mean identity margin | Mean nonidentity margin | Paired change D |
| --- | ---: | ---: | ---: |
| Encoder | 0.222122 | 0.126465 | -0.095657 |
| Projector | 0.224646 | 0.129592 | -0.095054 |

The exact primary changes are **-0.09565712588343249** and
**-0.09505422162164956**. Both use all **16/16 test pairs**, equally weighted
within **8 known pairing families**. There is no complete-case filtering.
The primary relative amplitude floor is 1e-6. All three registered floors
(1e-8, 1e-6, 1e-4) give identical availability, scores and changes for each site;
these are sensitivity checks on the same data, not replications.

The endpoint is the average of shifts 1/2/3 minus shift 0, using the same
shift-0 r5/r6 reference directions. Each margin compares the own-family cosine
with the mean of the other three families in its broad class. Thus negative D
means reduced **relative known-family correspondence to those fixed
references**, not loss of every correspondence or an absolute representational
distance. Average margins remain positive after rearrangement.

## Heterogeneity And Vector Geometry

| Known pairing family | Encoder D | Projector D |
| --- | ---: | ---: |
| Voronoi / quasicrystal | -0.065158 | -0.074642 |
| Checkerboard / hex tiling | -0.354761 | -0.356681 |
| Square / triangle tiling | -0.252540 | -0.255467 |
| Voronoi / hex tiling | -0.090559 | -0.089357 |
| White / blue noise | -0.052967 | -0.052408 |
| White noise / random dots | +0.018599 | +0.030406 |
| Blue noise / random dots | -0.014497 | -0.018708 |
| Sparse / dense dots | +0.046626 | +0.056425 |

Six of eight family changes are negative at each site; two are positive.
At the individual test-pair level, changes are negative on 12/16 encoder
pairs and 11/16 projector pairs. No family or opposite-sign case is omitted.

Across the 48 test-pair/nonidentity comparisons, the mean identity-versus-
rearranged **interaction-direction cosine** is 0.182170 at encoder and
0.179556 at projector. Interaction norm ratios range from 0.947668 to
1.097644 and 0.947426 to 1.099038 respectively. These are full factorial
interaction vectors, not single-cell activations. Substantial directional
change coexists with broadly retained vector lengths; neither a small
norm ratio nor direction cosine is itself the primary margin endpoint.

## Actual Input Qualification

All 512 candidate cells use raw RGB shape 320 x 320 x 3 and the unchanged
native processor tensor shape 1 x 3 x 1024 x 1024. The payload-first quartet,
fixed gray canvas and four qualified guarded-tile arrangements are unchanged.
These rearrange the registered tile support, not the entire scene or all
content coordinates. No semantic-destruction claim is made.

- All 4,608 matched feature-vector comparisons pass; maximum absolute
  residual is **3.4638958368304884e-14**, below 1e-10 with relative tolerance 0.
- All 6,144 candidate feature/oracle pairs pass; maximum oracle error is
  **3.0253577421035516e-14**.
- All **49,920** full/per-pool raw/native joint-word comparisons across
  arrangements are exact. All **256** candidate raw palette comparisons are
  exact. This does not claim preservation of the full original donor images.
- Historical processor parity, the 25 synthetic cases, the expected
  nonmatching sentinel and zero/constant model shams all pass their fixed gates.

All eleven available baseline profiles have paired changes on the numerical
scale: maximum absolute D **1.628829449790148e-14**. The raw global-moment
direction remains unavailable, with 0/16 available test pairs, and is retained
as unavailable rather than zero or removed. The full twelve-view profiles,
including missingness, are published. Numerical matching is not a statistical
equivalence test or a sufficient-statistics claim about every input feature.

## Execution And Reference Seal

The public execution authorization merged as
`f036a0822bae2d4c4f1eea1d6679b73a6d858c68`
in [PR 13](https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/13).
A fresh approved zero-call preflight passed before capture; the earlier
`PREPARED_NOT_RELEASED` root was not reused as the executable preflight.
All 135 selected frozen files remain unchanged at the PR 11 implementation.

| Actual unit | Count |
| --- | ---: |
| Acquired donor frames | 64 |
| Processor-only calls | 665 |
| Historical model source contexts | 384 |
| Zero/constant sham source contexts | 24 |
| Experimental source contexts | 512 |
| Total model source contexts | 920 |
| Separately recorded decoder calls | 2,760 |

Every acquisition operation has equal attempted and returned counts.
Processor-only counts exclude the separately audited input preparation inside
model contexts. Contexts and decoder calls are not interchangeable denominators.

All 256 reference cells and fixed family directions were independently checked
before the seal at **context 664**, with **32 sources / 409 processor-only
calls**. Only after the seal were test donors generated; the first test model
context was **665**. The 1,649 operations / 3,299 ledger entries retain this order
in the hash chain. No scores were used to reselect references or conditions.

Fresh historical calibration independently checks 1,152 historical cache
tensor pairs (58,540,032 scalar pairs), 768 OFF/ON cache pairs and 256 full
site arrays; maximum historical cache difference is 0. This is bounded
calibration, not equality of all unmeasured hidden state on every new image.
The selected L1 keys / L12 keys / L23 values are retained; this note's new
primary endpoints are the encoder/projector profiles, not new cache margins.

## Verification And Public Artifacts

The immutable capture manifest remains `COMPLETE_UNVERIFIED`; qualification
is recorded by the separate verification and analysis receipts, not by
rewriting acquisition history. Each of two file-only checks validates all
**27,838 receipted files**, including **24,794 binary artifacts**, and returns
`VERIFIED`. Their complete output trees are byte-identical. Neither verification
nor analysis adds a model or processor call; inference is not independently rerun.

The [public summary](../../examples/research_notes/0093_stat_matched_source_results/summary.json),
[all eighteen profiles](../../examples/research_notes/0093_stat_matched_source_results/profiles.jsonl),
[512 spectral records](../../examples/research_notes/0093_stat_matched_source_results/spectra.jsonl)
and [verification receipts](../../examples/research_notes/0093_stat_matched_source_results/verification_receipts.json)
retain complete score matrices, family differences, amplitude/availability
diagnostics and hashes. Public profiles/spectra are byte-identical copies
of the qualified analysis outputs.

Full raw/native inputs, packed bfloat16 visual sites, selected cache tensors,
source manifests and original sidecars stay local under the registered run
roots. Portable scores permit independent reconstruction of all published
aggregates and missingness, **not** native-word reverification without those
full local artifacts. Receipt/Git checks are auditable provenance, not
cryptographic proof that inference occurred.

## Reading And Claim Ceiling

The bounded reading is: **under these registered rearrangements, correspondence
at both full visual sites changes while twelve selected baseline views remain
matched**. The eleven available baseline profiles do not show this
paired change; this is a conditional separation from those fixed views, not
proof that the model introduces information absent from the full inputs.

No p-values, confidence intervals or equivalence claims are registered here.
Cells, shifts, floors and the two sites are dependent, not independent samples.
This does not identify training, learned specificity, semantic steering,
cache-to-answer mediation, a causal processing gain, unseen-family transfer
or invariance to arbitrary transformations. Available-history collision checks
do not establish pretraining novelty. Confirmation is not generated or sealed.

The one authorized acquisition is consumed. Its retained canonical root and
no-resume guards prohibit another acquisition under this release. Any new
cohort, comparator or follow-up requires separate registration and approval.
