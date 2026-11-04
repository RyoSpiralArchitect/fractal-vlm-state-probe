# Research Note 0069: Full Encoder/Projector Correspondence Registration

Date: 2026-11-03

Status: one development panel. Freeze the complete capture, analysis,
independent verifier, fixtures and configuration in Git before any new model
replay. This registration is not a result or an untouched confirmation.

Registration commit: `996656c4263a716af3ac7e48eb84ae3c10c39371`, with
544 local tests passing. All 58 previously frozen files stayed unchanged.

## Question And Fixed Comparison

[PR #6](https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/6)
merged the observation registration and historical calibration at `61b0e16`.
[Note 0068](0068_representation_observation_calibration.md) qualifies two
sites for the historical inputs/suffix. It did not measure correspondence.
Where do zero-overlap known-family margins, overlap differences and changes
in vector direction appear in these **two complete visual representations**?
This is a descriptive comparison with existing input, fixed-processing and
selected-cache profiles, not a test of whether training causes a difference.

The [configuration](../../configs/representation_correspondence_panel_v1.json)
retains the 32 reused source pairs, eight families, four reference maps,
24 test maps, three shared fixed-site levels (0%, 6.25%, 25%) and two
destinations per level/mask. Shared moved-coordinate coincidence is zero;
the same test-source IDs recur across q, while reference and test IDs are
disjoint. Every map fixes 25% of its own
content. Spectra need not match between levels. No new images/maps are drawn.

Observe exactly the Note 0067 sites on the same qualified FastVLM path:

| Site | Native tensor | Complete contrast dimension |
| --- | --- | ---: |
| Encoder, existing input to projector | bfloat16 `[1,256,3072]` | 786,432 |
| Projector, existing return before image-token insertion | bfloat16 `[1,256,896]` | 229,376 |

No extra encoder/projector invocation, pooling, cropping, token mask,
centering, whitening, dimension selection or cross-site alignment. The
existing positional input and identical return object are held, never
evaluated inside the observer. Save after the source generation and cache
capture finish. Keep both sites irrespective of score direction.

## Execution Gates

1. Offline no-model preflight: verify the previous 58 frozen files, qualified
   input gates, source/snapshot/runtime, Note 0068 publication and local
   observations, Note 0060 capture registry, all 2,048 anchors and 6,144
   cache sidecar hashes. Verify the new committed implementation, lossless
   codec and initial free disk. Read model files only for fingerprints.
2. Run the unmodified registered Note 0067 calibration in a fresh root:
   OFF/ON/OFF for every historical cell, 384 source contexts. Retain the
   exact return of its normal single model load for the panel. No alternate
   model/processor, reinitialization or second load. Independently verify the
   calibration; also require the newly captured 256 full site arrays to be
   byte-identical to the original Note 0068 arrays before panel acquisition.
3. Acquire 256 reference cells first (r1/r2, four reference maps), then seal
   their identities, order, packed-shard and source receipts. The seal is after
   total ordinal 640; the first test is 641. Acquire 1,792 r3/r4 cells, including
   the same-map controls on the four reference maps. No scores during capture.
4. At **every** panel cell require the original method restored, one observed
   call, received native dtype/shape, finite values and lossless roundtrip.
   Against the **same** Note 0060 map/source/cell check prepared inputs,
   image/frame, prompt, runtime, token layout, actual `The image` and step IDs
   `[785,2168,2168]`. Reacquire all three selected cache tensors and require
   exact bytes. These are 6,144 path-parity comparisons, not a new cache
   correspondence sample. Do not force a suffix, retry, resume or replace cells.
5. Withhold primary claims until complete capture, analysis and independent
   verification. Preserve the first failure, partial primary artifacts and
   attempted/completed counts. No tolerance change, endpoint removal or
   outcome-dependent release of another condition.

Budget: 384 calibration + 2,048 experimental = 2,432 fresh source contexts
(not internal autoregressive steps); 4,096 experimental full site arrays,
512 four-cell blocks and 1,024 distinct full interaction vectors. No direct
probe, prompt variant, training intervention or confirmation source.

## Storage Without Loss Of Coordinates

The qualified float32 expansion of a finite bfloat16 value has zero low
16 bits. Persist its **upper 16 bits verbatim** as little-endian uint16 words
in an uncompressed two-array NPZ per cell. Verify expansion back to the source
float32 bytes for every array, including signed zero. No rounding or numerical
quantization is permitted. Preflight checks all 65,280 finite bfloat16 bit
patterns and all 256 historical site arrays. A separate decoder writes the
high half-word of each float explicitly during verification.

Keep all primary cell arrays, source/audit outputs and parity caches locally.
Reconstruct complete float64 interactions from the four cells as needed.
Retain their byte hashes, dimensions, norms and availability; do not keep a
second permanent 3.875 GiB vector copy. Temporary NPYs are scoped to one site,
mask and destination: at most 32 same-map + 16 zero + 16 other vectors.
Their removal does not remove the original data or change scoring coordinates.

| Launch allocation | Bytes |
| --- | ---: |
| Full experimental packed cells | 4,160,749,568 |
| Experimental parity caches before compression | 1,248,854,016 |
| Fresh calibration arrays and caches before compression | 754,253,824 |
| Maximum float64 working vectors | 402,653,184 |
| Metadata/container/verification allowance | 1,073,741,824 |
| Reserved free space | 8,589,934,592 |
| **Minimum free at launch** | **16,230,187,008 (15.116 GiB)** |

Do not assume any ZIP compression savings, delete old runs or move user data
to meet the gate. Recheck reserve during capture and reserve plus scratch
before analysis groups. The launch estimate cannot guarantee that other
processes will not consume free disk. A shortage stops this release.

## Estimands And Verification

Flatten all token/channel elements in C order, then form
`z = (JJ - MJ) - (JM - MM)` in float64. Normalize the **complete interaction**,
not individual cell features. For each family use `unit(unit(r1)+unit(r2))`
on the fixed matching reference map. Test r3/r4 against all eight family
references. Margin = own-family cosine minus mean of the other three in its
broad class. Average score matrices over four masks and two destinations;
equal-weight the two test seeds per family and the eight families. References
are fixed. No fitting or averaging of vectors across different map bases.

Two sites times two primary endpoints: **M(0)** and **D=M(25%)-M(0)**.
Retain all four endpoints, six overlap aggregates, 56 score matrices (eight
same-map controls), all family/mask values, ties and negative values.
An exact zero interaction or a summed unit-reference norm <= 1e-12 makes
the affected score unavailable, not zero. Any missing required zero/high
matrix withholds both endpoints for that site. A missing middle or diagonal
matrix alone does not remove otherwise complete primary endpoints.

The 10,000 paired crossed bootstrap draws use seed 2026110301. Resample
four masks and eight families, two seeds within each drawn family and two
destinations within each drawn mask, shared across sites/q. Use percentile
tails 0.025/4 and 1-0.025/4, without shrinking the family if a site is
unavailable. Exploratory ranges have no guaranteed coverage; no p-values,
equivalence width or absence test. Secondary: 512 within-site q0-to-q1/q4
direction records (cosine, norm ratio, relative L2); do not align sites.

Independently regenerate all 1,024 full vectors from packed primary cells
and compare float64 bytes. Use separate scalar-product, family-margin and
nested-draw implementations for scores, retrieval, signs, aggregates and all
primary ranges; check directions from separate reductions. Absolute tolerance
1e-12 for scalar scores/ranges; for large norms relative 1e-11 plus absolute
1e-12. No tolerance for source bytes, reference identity or vector digests.
Independently reload every parity cache and rerun the fresh calibration
publisher without any model/processor calls. Public ledgers support primary
replay **from score matrices**; full vectors/activations require local primary
sidecars. Retain all raw, processor, non-learned and cache endpoints as context.

## Commands

Use the same offline environment for all four commands:

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_representation_panel.py preflight --output-root runs/representation_panel_preflight_v1
python3 scripts/run_representation_panel.py run --preflight-root runs/representation_panel_preflight_v1 --output-root runs/representation_panel_v1
python3 scripts/analyze_representation_panel.py --run-root runs/representation_panel_v1
python3 scripts/summarize_representation_panel.py --run-root runs/representation_panel_v1 --output-root runs/representation_panel_verified_v1
```

Each output root is fresh. A passing panel would describe these development
sources and this runtime. It cannot establish semantic invariance, causal
learning gain, unique emergence, prompt robustness or untouched replication.

Execution report: [Note 0070](0070_representation_correspondence.md) records
the completed capture, all four endpoints, independent full-tensor replay and
the boundary before prompt/suffix controls.
