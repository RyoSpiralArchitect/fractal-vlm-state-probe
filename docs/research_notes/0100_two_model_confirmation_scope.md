# Research Note 0100: Two Model Confirmation Scope Resolution

2026-11-09. The operator chose to **maintain FastVLM plus Qwen2.5-VL
confirmation**, not make Qwen an optional later extension. The
[decision record](../../examples/research_notes/0100_two_model_confirmation_scope/decision.json)
resolves the pending choice in Note 0099 without altering that preparation's
hashed plan, seed metadata or precision report. Cohort source/map generation
remains on hold until complete sample-size and precision registration and
validation. No processor or model call is authorized by this scope decision.

## Common Cohort And Joint Release

The candidate retains the same 16 reference pairs and 64 test pairs in eight
known families, not separate outcome-selected source panels for each model.
If adopted, 160 original sources and 1,280 raw candidate cells are shared
across both models. Each has its own native processor tensors and provenance.
Two models do not turn 64 paired test units into 128 independent observations.

**Both models must complete reference input, representation and provenance
seals before any test source is generated.** Sequential model loading for
resource reasons is compatible with that barrier; inspecting one model's
confirmation outcomes before finalizing the other's targets or analysis is
not. The sample count remains a budget candidate, not a precision-justified
joint sample size. No cohort content has been generated or sealed.

## What Is And Is Not Qualified

FastVLM's complete encoder/projector observation path was qualified on
development inputs in Notes 0067-0068. The new cell-dependent cycle maps are
not yet qualified even there. Qwen's historical source-cache experiments in
[Note 0023](0023_qwen_cross_model_factorial_pilot.md) do not qualify full visual
sites, the new input construction or a new cohort. Its withdrawn branched
readout is not reinstated.

Read-only inspection of the installed Qwen implementation and cached snapshot
configuration found the following architectural candidates:

| Candidate site | Static evidence | Unresolved runtime contract |
| --- | --- | --- |
| Full input to `vision_tower.merger`, before its normalization | Last vision-block states in window-index order; configured width 1280 | Full token count, dtype, single-call capture, native parity and restoration |
| Full `vision_tower` return, after reverse window indexing and before multimodal insertion | Merger output restored by `argsort(window_index)`; configured width 2048 | Full merged-token count, dtype, input-grid consistency and native parity |

The cached model ID is `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`, snapshot
directory `46d4cf06a06ffc1a766c214174f9cbed2f45bcab`. Local config and source
hashes are retained in the decision. This is file inspection, not integrity
qualification of all weights, a processor run, a model load or a forward.
Symbolic shapes involving `image_grid_thw` are not observed tensor shapes.

Qwen has a spatial patch-merging factor of two and a patch/grid processor
contract rather than FastVLM's qualified NCHW 1 x 3 x 1024 x 1024 contract.
The two sites are operational boundaries, not claims of homologous layers
or aligned vector coordinates across architectures. Window/patch order and
its inverse need explicit calibration; no cross-model vector alignment is
fitted.

The same raw construction must pass the separately specified input gates
on **both** native processors before confirmation use. Twelve FastVLM feature
gates, resize phase arguments and shape/dtype assumptions cannot be copied
to Qwen. A native patch-to-spatial interpretation would itself need an exact
oracle before applying spatial summary features. A construction change, if
needed, requires a new pre-generation specification and calibration, not a
quiet change after one model's cohort outcomes. No joint matching claim is
currently qualified.

## Four Endpoint Candidate And Precision Consequences

If both Qwen sites qualify, the candidate primary family contains four
reference-difference endpoints: two FastVLM sites and two Qwen sites. Each
uses the same within-family source-pair estimand and all arrangements.
Bootstrap indices must keep whole source pairs together across models,
sites, cells and maps. The named families, frozen references and one common
map orbit remain conditioning variables.

For nominal familywise alpha 0.05 across four endpoints, the candidate
two-sided percentile intervals have **98.75%** per-endpoint coverage and
quantiles **0.00625 / 0.99375**. The corresponding normal planning critical
value is 2.4977054744123737, not Note 0099's two-endpoint value. This is
multiplicity arithmetic, not calibrated finite-sample bootstrap coverage.
The executable interval procedure remains unimplemented and unregistered.

The old FastVLM-only half-width simulation cannot justify Qwen precision or
the four-endpoint design. Qwen development variance and achieved precision
are unknown. Neither substitute FastVLM variance nor increase sample count
until significance. Resolve the joint precision/budget design before any
confirmation content is generated; retain all prior preparation versions.

A joint positive-reference-effect claim would require all four qualified
endpoints available and their lower bounds above zero. If a required model
or site fails, retain that missing/negative result; do not silently downgrade
to a one-model confirmation or select the successful sites after outcomes.
Separate descriptive results remain reportable. This does not establish
universal architecture generality, semantic preservation or learning effects.

## Resource And Qualification Sequence

There would be 2,560 experimental model contexts if the shared 1,280-cell
candidate is adopted for both models. This is **not** a complete joint budget:
Qwen historical/sham calibration counts, native token/array sizes, decoder
call counts, storage and a joint launch threshold are unknown. FastVLM's
1,688-context estimate and 44 GiB threshold apply only to its earlier
candidate. Neither copying those numbers nor doubling them qualifies Qwen.

Next preparations are a Qwen observation and native-input calibration
registration on development/synthetic material, a jointly qualified map/input
recipe, and complete two-model precision, storage, acquisition and independent
verification contracts. OFF/ON/OFF input/cache parity and exact observer
restoration must pass before trusting new visual observations. Actual cohort
generation and all new processor/model execution remain separately gated.

## Scope Checks

The 47 focused joint-scope, preparation, portable publication, historical
scope and example checks pass; lint passes. They pin the unchanged Note 0099
plan/seed metadata, prevent double-counted samples, distinguish provisional
sites from native qualification, check the four-endpoint multiplicity
arithmetic, and retain unknown joint budgets rather than fill them by
doubling FastVLM. This is metadata/contract validation, not native calibration.
There are no new images, maps, processor calls or model contexts. These local
scope additions had not been committed, independently reviewed or pushed at
that checkpoint. [Note 0102](0102_qwen_calibration_registration.md) records
subsequent preparation-only review/integration, without native execution.
