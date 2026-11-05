# Research Note 0080: Full First-Step Readout Without A Suffix

2026-11-05. **Qualified on the registered development inputs.** A fresh
multimodal prefill with no suffix decoder call reproduced the native
first-step score vector at all **151,936 vocabulary coordinates**, in
96/96 rows. This is measurement-path calibration, not an ACK-versus-YES
effect analysis or a completed confirmation cohort.

## Three Fresh Paths Agree

[Note 0079](0079_first_step_readout_registration.md) froze the method,
numerical definition and stopping rule at
`a3679a446d87e91de9ecd4b678d669827e9bbb40`. The 96 historical rows
include 32 images, 64 unique image/prompt conditions and repeated closing
ACKs. One model/processor instance ran all **288 fresh contexts**:
192 natural-generation contexts and 96 independent prefill-only contexts.
No retry, replacement, new prompt, supplied suffix or inter-prompt metric.

| Check | Complete comparisons | Result |
| --- | ---: | --- |
| Native stream-only first scores vs decoder-observed first scores | 96 x 151,936 values | Every byte identical |
| Native first scores vs independent prefill-only scores | 96 x 151,936 values | Every byte identical |
| Observed native prefill raw logits vs independent prefill raw logits | 96 x 151,936 values | Every byte identical |
| Independent prefix vs native prefill K/V, all 24 layers | 4,608 tensors | Every native word identical |
| Two native paths' terminal K/V, all 24 layers | 4,608 tensors | Every native word identical |

All prepared input fields, actual prompt/frame, both full visual sites,
historical native responses/IDs and three historical selected cache targets
matched their original sources. Observed native checkpoints at lengths
395/396/397 also matched the prior all-layer qualification. The original
token/top-logprob records were unchanged by the decoder observer.

The independent path had **one actual decoder call** per row, original
140 prepared IDs, embeddings [1,395,896], effective cache length **395**,
and **zero supplied/generated suffix inputs**. It used its own empty
cache, not a text-only continuation or a branch from the natural source.
Source stream notification timing alone does not identify a score's origin:
the equality to a truly suffix-free prefill checks that boundary explicitly.

Every score was saved as its exact float32 expansion from native bf16.
There are 480 complete local vectors: 288 native score arrays and 192
raw-logit arrays. No top-k truncation, candidate selection, vocabulary
filter or nonfinite replacement. All score coordinates were finite here.
Two of the 96 rows had a two-ID maximum tie; both were retained, and the
actual native first ID belonged to its whole-vector maximum set.

## A Numerical Calibration Result, Not A Prompt Effect

The native runtime computes `logits - logsumexp(logits)` in bf16.
Exponentiating those saved native scores does **not** give exact unit mass:

| Diagnostic over all 96 rows | Observed range |
| --- | ---: |
| Sum of exp(saved native score) | 0.9398161045 to 1.0634184369 |
| Sum after registered float64 renormalization | 0.9999999999999996 to 1.0000000000000004 |
| TV: renormalized native scores vs float64 softmax of saved raw logits | 0.0002806115 to 0.0004435607 |

The largest raw mass deviation from one was about **6.34%**. This did not
fail the gate: exact native *score identity* was the criterion, and native
unit probability mass was explicitly not assumed at registration. No
threshold was changed after observing these values.

For a later prompt comparison, the frozen probability definition is
`p = exp(s - logsumexp64(s))` over all model-output coordinates. This
re-normalizes the saved native score vector in float64. It does not undo
all bf16 rounding or replace the vector by a higher-precision model run.
The raw-logit softmax is a retained numerical diagnostic, not an alternative
chosen after seeing effects. The TV range above is **within a condition,
between two numerical definitions**, not the ACK-versus-YES effect.

This qualifies next-token scores, not a complete distribution over natural
language answers. Neither equal argmax tokens nor exact cache recovery
establishes semantic invariance, cache-to-output mediation, multi-turn
persistence or a training effect.

## Receipts And Verification

- [Verified summary](../../examples/research_notes/0080_first_step_readout_calibration/summary.json)
- [288 arm receipts and complete-vector digests](../../examples/research_notes/0080_first_step_readout_calibration/arms.jsonl)
- [96 identity/normalization checks](../../examples/research_notes/0080_first_step_readout_calibration/comparisons.jsonl)
- [96 actual prefix decoder-boundary receipts](../../examples/research_notes/0080_first_step_readout_calibration/prefix_boundaries.jsonl)
- [Independent file-only check](../../examples/research_notes/0080_first_step_readout_calibration/independent_file_check.json)
  from a [NumPy reader](../../scripts/audit_first_step_readout_files.py)

Two fresh verifications produced the three registered publication files
byte-for-byte. The independent reader rechecked all vocabulary/NPY words,
29,171,712 score-coordinate pairs (observer and prefix comparisons),
14,585,856 raw-logit pairs, complete prefix K/V and scalar normalization
calculations. No model or processor forward was used by those file checks.
Complete score/logit/cache arrays remain local under
`runs/first_step_readout_calibration_v1`. The tracked receipts support
join/summary checks but do not reconstruct full distributions on their own.

Summary SHA-256: `bddeced0371297b702e2accbf5d762ddb39e84bfbdadde85ba8f6f939f70562b`.
Capture: `59b0e9da95c2f174f53143356eab60c16e4b67cf4886db3f0df9e52503ec3c48`.
Preflight: `23824deed2c4b00586201fdbd84d0939c914b68a90095c2a54dad5c9e81a2322`.

All 637 local tests pass, including a weight-free stub of the real prefix
adapter and public-only ID/normalization/denominator checks. The original
90 frozen files and all 97 files in this registration remain unchanged.
Scoped Ruff, whitespace and 444 local links in affected documentation
pass. Both independent file-audit outputs are also byte-identical. These
are local checks, not GitHub CI or a second model implementation.

## Next Boundary

The source-prompt first-token measurement path is qualified. A separate
32-image/64-condition readout panel still needs committed effect endpoints,
paired aggregation/dependence policy, fresh calibration and stopping rules
before release. Do not inspect calibration ACK/YES distances as if they
were preregistered panel outcomes. No inter-prompt TV, JS, p-value,
equivalence claim or new cache/direct factorial is reported here.
The earlier v1 failure remains unqualified and separately visible.
Untouched confirmation is still neither generated nor sealed: reaching
Note 0080 is a record of calibration work, not a scientific completion gate.
