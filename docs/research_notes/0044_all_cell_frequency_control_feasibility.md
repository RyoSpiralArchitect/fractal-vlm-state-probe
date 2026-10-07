# Research Note 0044: All-Cell Frequency Control Feasibility

Date: 2026-10-07

Status: registered input-only experiment; new results not yet measured.

Following [Note 0043](0043_fastvlm_pairing_holdout_and_input_baseline.md), test
whether all four transformed cells can pass the registered processor-space
frequency gates while preserving each palette donor's joint RGB multiset.

The [stage-4 execution addendum](../pairing_validation_protocol.md#stage-4-input-execution-addendum)
freezes a bounded low-pass rank-field sweep and a separate common pixel
permutation comparator before either input scoring or cache forwards.
The latter destroys geometry and preserves raw full-vector cosine relations
by coordinate permutation; it does not identify frequency alone.

Partial input acceptance is retained as feasibility evidence. A full
eight-family cache study is eligible only with 32/32 accepted source-pair
blocks, exact calibration and the unchanged source/layout contract.
