# Research Note 0077: Fixed Common-Suffix Prompt Comparison

2026-11-05. Register before new forwards. This is a development control
on the 32 reused Note 0072 images, not untouched confirmation. Base commit:
`7b7d9edcee2157f25c0c82d9949dd0ea44e0807c`. Keep every one of the 84
previously frozen files byte-identical; do not overwrite the v1 failure,
the successful own-suffix calibration, or unrelated user changes.

## Question And Intervention

[Note 0076](0076_native_suffix_replay_parity.md) qualified explicit replay
of each prompt's *own* two native IDs. Here both the ACK and YES prompts
receive **[785,2168]** (`The image`), explicitly one ID per decoder call.
No new model, image, generator, mask or prompt wording. The sole prompt
change is historical processor position 132 (53763 to 14080); its effective
cache position is 387. All other prepared fields and both full visual sites
must match the corresponding historical source.

This compares prompt-conditioned cache with the realized suffix held fixed,
not natural response rates, a cache swap, a full-vocabulary readout or a
test of which K/V causes output. A forced visible string is a protocol
receipt, never evidence of generated-answer invariance or semantic compliance.
The different continuation is off the native greedy trajectory for 30 YES
conditions; acknowledge that limitation. Do not select only the two YES
conditions with naturally matched suffixes.

## Gates And Fixed Budget

1. Offline preflight: verify the previous 96 own-suffix rows, 288 arm
   receipts, all saved full native/forced arrays at all three boundaries,
   84 frozen files, published ledgers, original input/cache qualification,
   model snapshot, installed source hashes and weight-free stream fixtures.
   Freeze new capture, analysis, verifier, fixtures and config in Git.
   Require 8 GiB reserved plus 4 GiB launch allowance; do not remove old runs.
2. Load the original FastVLM model and processor **once**. First reacquire
   all 96 historical rows in original order, with native plain / native
   traced / forced own (288 contexts), using unchanged Note 0075 routines.
   Require exact original inputs, two visual tensors, three historical cache
   targets, and all 48 new native/forced tensors at lengths 395/396/397.
   All full-word hashes and actual decoder-ID receipts must also match
   Note 0076. Check each native observer sham before its forced arm.
   Seal all 96 rows before releasing any common-suffix context.
3. Then acquire **32 images x 2 prompts = 64** new contexts in the same
   process/model instance: opening ACK then YES for each historical
   family/cell. Retain all eight families and MM/JJ/MJ/JM. No closing ACK
   repeat is counted as a new experimental pair. Every cache starts empty;
   no reused image features, prefix trim, cache substitution or extra vision
   call. Use the qualified v2 stepper and read-only observers unchanged.
4. Capture all 24 layers' complete effective keys and values, with dtype,
   axes, actual decoder inputs and offsets at each of the three boundaries.
   Two uint32 singleton inputs must equal the chosen IDs, including final
   consumption before stream exhaustion. Observe one [1,395,896] multimodal
   prefill and one target projector call, actual 140+2 cached IDs and the
   qualified 256-position image expansion. Every original source/audit is
   retained. Save complete bf16 words losslessly, never pooled or rounded.
5. For each panel context, require the **whole prefill checkpoint** to
   equal its matched same-process forced-own checkpoint. For all 32 ACKs
   and the two naturally matched YESs, require all three checkpoints to
   equal their own-suffix replay (34 x 3 x 48 word-exact tensor pairs).
   These are shams, not exclusions or outcomes of the paired prompt test.
   Stop on the first admission failure; keep partial data/attempt counts.
   Never change tolerance, suffix, sample count or output namespace to retry.

Total: **352 source contexts** (288 calibration + 64 panel). All acquisition
roots are canonical and fresh. Re-execution is not allowed by this release.
There are 32 paired development images, not 64/352 independent replicates;
four related cells per family share source construction and a single r1 seed.

## Frozen Comparisons

Use YES minus ACK float64 differences, after re-expanding full bf16 words.
Primary targets remain L1 keys, L12 keys and L23 values. Their primary
descriptive comparison is the complete fixed-suffix region [395,397).
Report all 32 per-image RMS values with median, minimum and maximum;
equal weight per image (hence balanced families/cells). Retain per-family
and per-cell records; no fitting, bootstrap, p-value, equivalence threshold,
post-hoc layer selection, normalization or cosine-based filtering.

Also enumerate **all 48 targets**, regardless of equality/direction, over:

| Region | Effective positions (half-open) |
| --- | --- |
| pre-image | [0,42) |
| image | [42,298) |
| post-image prompt before edit | [298,387) |
| changed instruction ID | [387,388) |
| unchanged prompt/template tail | [388,395) |
| complete fixed suffix | [395,397) |
| first supplied ID | [395,396) |
| second supplied ID | [396,397) |

Publish **12,288** paired target/region records. Regions with the two
singleton IDs overlap the full suffix and are not independent endpoints.
Record word-exact changed scalars and absolute changed positions (including
signed zero), full-cache first changed position, RMS, max absolute change,
each condition's L2 norm, delta L2, relative L2 and cosine. Zero norms make
the affected normalized quantity null, not zero; do not drop the row.
Keep equality anywhere, including layer-zero suffixes. Cross-layer norm
units are not a common causal scale. The image/pre-edit equality is an
observation, not a gate that forces the hypothesized localization to hold.

No scores are computed until all 64 contexts pass capture. Reload all
primary arrays, rederive the 13,824 fresh calibration comparisons and
all panel shams, then compute the metrics. Independently scan each token's
bytes and use scalar float64 reductions to check consequential norms/RMS/
cosine (rtol 1e-12, atol 1e-14; zero tolerance on stored words). Require
two fresh verification outputs to reproduce all public receipts exactly.
Public JSONL supports aggregate replay; full tensor reconstruction requires
local NPYs, not the tracked digests alone.

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_common_suffix_prompt.py preflight --output-root runs/common_suffix_prompt_preflight_v1
python3 scripts/run_common_suffix_prompt.py capture --output-root runs/common_suffix_prompt_control_v1
python3 scripts/run_common_suffix_prompt.py verify --output-root runs/common_suffix_prompt_verified_v1
python3 scripts/run_common_suffix_prompt.py verify --output-root runs/common_suffix_prompt_verified_replay_v1
```
