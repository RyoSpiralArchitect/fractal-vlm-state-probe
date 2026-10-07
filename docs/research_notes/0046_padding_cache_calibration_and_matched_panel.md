# Research Note 0046: Padding Cache Calibration And Matched Panel

Date: 2026-10-07

Status: registered; cache measurements pending.

The [Stage 4C design](../pairing_validation_protocol.md#stage-4c-padding-cache-calibration-and-matched-panel)
first rechecks the qualified historical tensors, tests black-square pixel
shams and crosses only `image_sizes` in a fixed anchor. The complete
permutation/mean panel from [Note 0045](0045_padding_policy_and_explicit_marginals.md)
is measured only after those model-side gates pass, alongside its fixed
unmatched permutation/black comparator. Twelve primary tests share one Holm
family; the existing images are not a new independent or untouched cohort.
