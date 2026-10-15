# Research Note 0049: Fixed-Fraction Map-Seed Transfer

Date: 2026-10-16

Status: input audit completed; cache comparison registered, not yet executed.

Following [Note 0048](0048_graded_cache_correspondence_and_direction.md), fix
the selected-site fraction at 3/4 and vary the common map seed. The
[Stage 4F registration](../pairing_validation_protocol.md#stage-4f-fixed-fraction-map-seed-input-audit)
first audits the historical anchor and exactly three new maps. No replacement
seeds or relaxed gates are permitted. A cache comparison is conditional on
four complete 32-block panels and requires a separate registration.

## Input Gate

The input implementation and seed grid were committed at `0e6a748` before
measurement. All four maps accept 32/32 blocks and all eight four-seed
families. Independent checks reproduce 512 pixel/inverse/padding/joint-RGB
controls, all 128 four-cell gates, and 128 exact historical-anchor cell checks.
The integer-moment oracle's maximum mean/variance errors are `8.19e-13` /
`1.15e-13` (rounded upward), with zero cells above `1e-12`.

The new maps use seeds 20261016, 20261017 and 20261018. Each seed independently
generates the selected-site order and full permutation through two PCG64 child
streams, then restricts its cycles to exactly 57,600 selected positions.
Both the selected set and destinations change; their separate effects are
not identified. These maps retain the same expanded RGB counts and padding,
not necessarily the same processor spectrum.

Input artifacts: [summary](../../examples/research_notes/0049_map_seed_input/summary.json)
and [512-cell ledger](../../examples/research_notes/0049_map_seed_input/cells.jsonl).
No model weights or cache forwards were used for eligibility.

## Cache Registration

[Stage 4G](../pairing_validation_protocol.md#stage-4g-same-map-and-cross-map-cache-correspondence)
freezes all reference captures before test captures, excludes the historical
anchor from primary inference, and compares same-map versus cross-map scores
across the three new maps. The twelve primary tests share one Holm family;
complete family blocks move synchronously across maps in 576 assignments.
All 96 map-matrix views and 1,152 directional comparisons remain descriptive.
