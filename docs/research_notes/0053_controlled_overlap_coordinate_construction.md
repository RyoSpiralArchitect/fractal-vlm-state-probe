# Research Note 0053: Controlled Overlap Coordinate Construction

Date: 2026-10-29

Status: construction implemented and unit-tested; full-size receipt is produced
only after committing the measurement specification. No source-image scores
are used to construct maps.

## Construction

For N content coordinates divisible by sixteen, shuffle them using each
registered mask seed. The first N/4 form the reference fixed set F; the
remainder form an ordered outside set O. The reference map is one uniformly
shuffled directed cycle on the complement of F, fixing F exactly.

For q in 0, 1/16, 1/4, use the first qN members of F and the first N/4-qN
members of O as the test fixed set. These coupled sets have exactly N/4
fixed sites and qN shared fixed sites. They are not spatially contiguous masks.
Each mask receives two separately seeded destination cycles. Keep the same
reference map across all levels and both destination draws.

Generate a single cycle on each test mask's complement. Reject a candidate
only when an output site maps to the same moved source coordinate as in the
reference map. Use at most 128 candidates, preserving the rejection counts;
stop on exhaustion without replacement seeds. Single cycles cannot introduce
incidental fixed points. The accepted maps therefore have exactly qN total
coincidences against the reference, all fixed. q = 0 has zero total content
coincidence. Padding is outside this denominator.

This is a distribution over single-cycle derangements conditional on the
coordinate constraint, not uniform sampling over all pixel permutations.
The rejection criterion uses no images, processor statistics or scores.
The two test destination maps need not have zero coincidence with each other;
the scored contrasts are reference-to-test. At q = 1/4 the maps share their
fixed mask but not their moved destinations. Same-map controls remain separate.

## Verification

The independent oracle constructs successor dictionaries by explicit site
assignment, reconstructs the seeded draws and rejection sequence, and checks
the saved map arrays. Unit tests exercise sizes 16, 64 and 320, exact overlap,
same-mask destination replication, malformed maps, corrupted receipts and
exhausted construction. Full-size validation is required before image scoring.

The constructor and oracle share the specified NumPy PCG64 RNG, not the cycle
assignment implementation. This checks implementation agreement under that
RNG, not independence from its distribution. Inputs, seeds, retry rules and
the development budget are specified in [Note 0054](0054_controlled_overlap_phase1_registration.md)
and [the registered config](../../configs/controlled_overlap_phase1_v1.json).

There are four mask replicates, each with one reference plus six test maps:
28 maps in total. At N = 76,800, every map has 19,200 fixed sites, with
0 / 4,800 / 19,200 shared fixed sites against its reference. All four cells
of a source factorial receive the same map. No source-image observations are
added by this coordinate-only stage.
