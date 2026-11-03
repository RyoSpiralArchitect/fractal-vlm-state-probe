# Research Note 0071: A Single Post-Image Prompt Token

Date: 2026-11-04. Status at registration: development path calibration, not
a correspondence experiment. Freeze this document, config, capture/verifier
and fixtures in Git before any new source forward. Base: Note 0070,
`64a94efd4d8ed3f521b6b0f411c95165af73cb82`. No change to its 67 frozen files.

## Question

On the [qualified FastVLM path](0070_representation_correspondence.md), if
one **post-image prompt token** changes on the same image, are the actual
processor pixels, encoder return into the projector and projector output
bit-identical? Where do the three selected source-cache tensors first differ,
relative to the edit and to the beginning of free generation? This is not a
test of family correspondence, semantic instruction-following, or mediation.

## Fixed Inputs And Intervention

Take map `m0_reference`, r1 only, all eight named development pairs and
their MM/JJ/MJ/JM cells: 32 distinct existing frame and processor-pixel
hashes. There is no new image, mask, seed, held-out set or factorial score.
Use [the registered IDs, order and budget](../../configs/prompt_only_source_calibration_v1.json).
One process, one model/processor load. For each image, in fixed order:

| Stage | Last instruction | Purpose |
| --- | --- | --- |
| A before | Return ACK only. | Exact replay of that Note 0070 cell |
| B | Return YES only. | Natural response to a single lexical edit |
| A after | Return ACK only. | Exact return-to-baseline / carryover check |

The last word to be requested is **not** injected into the model's answer.
At temperature 0, seed 20260604, max 2 generated tokens, each stage uses a
fresh empty prompt cache and runs the existing source-only generation path.
Exactly 96 source contexts; no full-vocabulary direct probes or forced suffix.
The system prompt, chat template, image and its position, frame description,
processor, sampler and all other prompt tokens are fixed. The snapshot tokenizer
on historical inputs identifies token 132 (zero-based): 53763 (` ACK`) to
14080 (` YES`). Both prepared prompts must have 140 tokens. Never search
another prompt or replace a source if this gate fails.

## Admission And Readout

1. Offline preflight with no model/processor forward: pin the Note 0070
   summary and cell ledger; check all 32 source runs, audits, manifests,
   frames, packed representation files and 96 selected cache sidecar files.
   Check inherited and new Git blobs, installed execution sources, complete
   model snapshot and cached tokenizer. Keep at least 8 GiB free; launch
   requires a further 512 MiB for new sidecars and metadata.
2. At every forward observe actual prepared fields: `input_ids` must be
   the prescribed 140 IDs, and mask, complete pixel fingerprint, image sizes
   and other fields must equal the same historical cell. No metadata override.
   Capture the two complete native bf16 sites once, using the existing
   read-only observer. Persist lossless packed words. Reload and compare
   full bytes with the matched historical arrays for all A/B/A captures.
3. Require the actual prompt, frame, runtime and restored observer. Retain
   the actual fresh PromptCacheState's IDs *after* the stream is exhausted.
   Split the suffix only after verifying the full prepared-ID prefix. Check
   actual generation counts, notification IDs and effective cache offsets.
   The stream's final notification may repeat the last ID; never deduplicate
   real tokens or infer the suffix from visible text. Either A must also
   match historical suffix, stream notifications, token layout and every byte
   of L1 keys, L12 keys and L23 values. A after must recover A before exactly.
4. Pre-register six half-open **effective-cache** regions per target/pair.
   Validate the single processor image marker (42) and actual 256-position
   expansion each time. For this edit the intended spans are pre-image
   [0,42), image [42,298), post-image prompt before edit [298,387), edited
   token [387,388), unchanged prompt tail [388,395), and realized suffix
   [395,actual end). The unchanged tail includes chat-template tokens.
   Do not reuse a total length of 397 when generation lengths change.
5. Compare A before to B by complete stored float32 words (exact expansion
   of bf16). Record absolute changed positions, changed scalar count, RMS,
   maximum absolute change and the first differing **comparable** position
   per target. Signed zero counts as a byte change even if RMS is zero.
   Cache equality *before the edit* is an observation, not a pass criterion.
   If the whole realized suffix ID sequence differs, retain both suffixes,
   mark that regional comparison unavailable, and keep all prompt comparisons.
   A naturally matching suffix is not a teacher-forced intervention.
6. Withhold summaries on a failed admission, partial panel or failed replay.
   Save partial primary data and attempted/completed counts; no alternate
   output root for a second acquisition, retry, restoration by regeneration,
   changed tolerance or denominator reduction. With all 96 admitted, replay
   all primary files without a model call, independently scan positions for
   byte changes, and cross-check floating RMS (relative tolerance 1e-12).
   Publish 96 context receipts, 32 pairs, 576 regional records and summary;
   repeat verification to a fresh directory and compare bytes.

The unit is a paired development image; A/B/A are ordered observations, not
independent replicates. This first gate does not estimate a population effect,
prompt robustness, full-vocabulary readout, causal mediation, or persistence
across conversations. No automatic release of a fixed-suffix path, a direct
readout panel, or the still ungenerated/unsealed confirmation cohort.

## Commands

```bash
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 scripts/run_prompt_only_source.py preflight --output-root runs/prompt_only_source_preflight_v1
python3 scripts/run_prompt_only_source.py capture --output-root runs/prompt_only_source_v1
python3 scripts/run_prompt_only_source.py verify --output-root runs/prompt_only_source_verified_v1
python3 scripts/run_prompt_only_source.py verify --output-root runs/prompt_only_source_verified_replay_v1
```
