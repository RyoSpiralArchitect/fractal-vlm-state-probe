# Research Note 0072: Where A One-Token Prompt Edit First Appears

Date: 2026-11-04. **Verified development calibration.** On the same 32
FastVLM images, changing `Return ACK only.` to `Return YES only.` left the
two measured visual representations and the entire selected cache prefix
unchanged. The first cache difference was at the edited token itself in
**32/32 images at each of three targets**. Free generation changed on 30
images; the other two are informative, not failed trials.

## What Stayed Fixed

[Note 0071](0071_prompt_only_source_registration.md) was committed at
`2ca36328caed9afd2569e50d5e2380719d046a9e` before acquisition. The
offline preflight matched the 32 historical anchors, original snapshot and
73 frozen files without loading weights. One unchanged model/processor instance
then ran 32 A-before/B/A-after triplets, with a fresh empty cache each time:
96/96 completed, no retry or replacement. Inputs were r1, mask 0 reference,
all four cells of eight reused families. These are not 96 independent images.

All 32 A/B pairs had the same frame, prepared pixel bytes (by SHA-256),
image sizes, attention mask and 139/140 prompt IDs. The only changed ID
was 53763 to 14080 at processor position 132. The single image marker at
42 expands to 256 cache positions, so the edit maps to effective-cache
position **387**, and the 140-token input ends at 395 (exclusive). The
realized histories happened to have 397 cache positions in every context;
this was checked from the actual IDs/offsets, not assumed for the variant.

Both full visual sites (encoder features entering the projector and its output)
were bit-identical across the edit in 64/64 site pairs. All 192 captured site
arrays also matched their historical Note 0070 anchors. All 192 A-to-history
and 96 closing-A-to-opening-A selected cache tensor comparisons were exact.
These are checks on two visual sites and three cache targets, not all model states.

## Where The Cache Changed

Median per-image RMS difference in the complete region, in that target's
own units. Coordinates are zero-based and half-open. The final row has a
different denominator; it must not be read as a 32-image median.

| Effective cache positions | Comparable images | L1 keys | L12 keys | L23 values |
| --- | ---: | ---: | ---: | ---: |
| [0,387): pre-image, image, and post-image text before edit | 32/32 | 0 (exact) | 0 (exact) | 0 (exact) |
| [387,388): edited prompt token | 32/32 | 1.709370 | 0.884815 | 1.548442 |
| [388,395): seven unchanged prompt/template IDs | 32/32 | 0.059565 | 0.303570 | 0.520578 |
| [395,397): same naturally generated IDs only | **2/32** | 0.013089 | 0.072069 | 0.169686 |

An independent byte scan of the stored arrays found all 4,755,456 prefix
scalar pairs exact. At every target in every image, each of the seven
unchanged prompt-tail positions had at least one changed scalar. Thus the
cache difference persists into identical subsequent **prompt tokens before
any generated token**. That localization is consistent with a causal decoder
processing a late prompt edit; it does not show how a particular key or value
caused the answer. Magnitudes across targets are not on a calibrated common scale.

## Natural Responses And The Two Exceptions

The opening and closing A prompts each produced `The image` on all 32.
With B, the requested word and the visible response should not be confused:

| Visible B response | Actual suffix IDs retained by cache | B stream notifications | Images |
| --- | --- | --- | ---: |
| `Yes` | [9454, 151645] | [9454, 151645] | 20 |
| `Yes,` | [9454, 11] | [9454, 11, 11] | 10 |
| `The image` | [785, 2168] | [785, 2168, 2168] | 2 |

The end-of-turn ID 151645 supplies a second ID without a second visible
word. Conversely, a repeated final stream notification is not a third
cached suffix ID. All 32 suffix spans have the **same length**, but only
two have the **same actual IDs** under A and B. The other 30 are unavailable
under the registered matched-suffix rule (90/96 target comparisons);
their responses and all their prompt-region measurements are retained.

The two naturally matched exceptions were `stochastic_blue_dots_r1/mj` and
`stochastic_sparse_dense_r1/jj`. In **both**, both generated positions
differed in all three selected caches despite identical text **and** suffix
IDs. This is a within-pair contextual difference, not an estimate for the
other 30, a separate family effect, or a substitute for experimentally fixing
the suffix. No inference across sampled families, equivalence claim or new
correspondence endpoint is made.

## Verification And Next Experiment

The [original verifier summary (gzipped JSON)](../../examples/research_notes/0072_prompt_only_source_calibration/verified_summary.json.gz),
[96 context receipts with actual IDs](../../examples/research_notes/0072_prompt_only_source_calibration/contexts.jsonl),
[32 paired responses](../../examples/research_notes/0072_prompt_only_source_calibration/pairs.jsonl)
and [576 regional comparisons](../../examples/research_notes/0072_prompt_only_source_calibration/regions.jsonl)
reproduced byte-for-byte in a fresh second verification without another
model or processor forward. Full native-word sites and selected cache arrays
remain under `runs/prompt_only_source_v1`; the tracked bundle supports
receipt/aggregate replay, not reconstruction of the tensors on its own.
Original summary SHA-256: `afdc6d94330505f485aaade7f1a3321a9a5bb182085251415571858118d9ed23`.
Capture: `1400a15fa3b4847a7d4c418f7f7dc5b63e475e4979135a92a3922d84a069d8e3`.

The repository's example-schema test initially rejected the original summary
because it lacked a catalog type, not because a measurement differed.
The [readable catalog summary](../../examples/research_notes/0072_prompt_only_source_calibration/summary.json)
adds only `analysis_kind` and the original summary's SHA-256. Its own hash is
`dab7430a937b9ca8373101e0aad7fa37dd5d1cc716a81779441768990cfc33a4`.
The [packager](../../scripts/package_prompt_only_source.py) takes the two verified
directories and a fresh destination; it refuses differing ledgers/summaries.
No frozen code, numerical result, threshold or capture was changed.
All 563 local repository tests pass, including packaged-only checks on the
original verifier bytes, the 32 token/anchor joins and all regional summaries.
The 73 registered file hashes, scoped Ruff, whitespace and 395 local links
in the navigation/notes were checked. These are local checks, not CI.

**Next:** register a token-by-token teacher-forcing **parity** test first:
replay each prompt's own native suffix, including end-of-turn and punctuation,
and require its native cache bytes, actual token roles and offsets to match.
Only a separately released comparison should then impose the same suffix
on different prompts. A fresh full-vocabulary first-step readout would answer
a different question before any suffix is generated; do not read it from this
source cache. Neither experiment was run here. The earlier cache factorial and
direct-probe denominators are unchanged; confirmation is still ungenerated.
