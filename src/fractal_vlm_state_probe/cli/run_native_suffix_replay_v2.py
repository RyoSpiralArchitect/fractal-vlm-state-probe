"""Second, separately sealed release: normalize only the inactive budget argument."""

from __future__ import annotations

import argparse
import importlib
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace

from fractal_vlm_state_probe.cli import run_native_suffix_replay as base
from fractal_vlm_state_probe.native_suffix_replay import forced_stepper, replaced

CFG = "configs/native_suffix_replay_calibration_v2.json"
V1_CFG, V1_CODE, V1_QUALIFY = base.CFG, base.CODE, base.qualify
CODE = (
    *V1_CODE,
    CFG,
    "src/fractal_vlm_state_probe/cli/run_native_suffix_replay_v2.py",
    "scripts/run_native_suffix_replay_v2.py",
    "tests/test_native_suffix_replay_v2.py",
    "docs/research_notes/0075_native_suffix_v2_registration.md",
)


def inactive_entry(step):
    def generate(*args, thinking_budget_criteria=None, **kwargs):
        if thinking_budget_criteria is not None:
            raise ValueError("active thinking criterion is outside the replay contract")
        return step(*args, thinking_budget_criteria=None, **kwargs)

    return generate


def stepper(*args):
    return inactive_entry(forced_stepper(*args))


def stream_fixture():
    """Use the installed outer stream, real replacement and fake decoder; no weights."""
    import mlx.core as mx

    g = importlib.import_module("mlx_vlm.generate")
    records = []
    for mode in ("omitted", "explicit_none", "active"):
        for suffix in ((785, 2168), (9454, 151645), (9454, 11)):
            decoder_inputs, invocations, dispatch = [], [], []

            def language(inputs, **kwargs):
                decoder_inputs.append(inputs.tolist()[0])
                # Uniform toy logits have argmax 0, never one of the requested IDs.
                return SimpleNamespace(logits=mx.zeros((1, 1, 151936)))

            features = SimpleNamespace(
                inputs_embeds=SimpleNamespace(shape=(1, 395, 896)), to_dict=lambda: {}
            )
            model = SimpleNamespace(
                config=SimpleNamespace(
                    model_type="llava_qwen2", image_token_index=-200
                ),
                language_model=language,
                get_input_embeddings=lambda *a, **kw: features,
            )
            processor = SimpleNamespace(
                tokenizer=SimpleNamespace(stopping_criteria=lambda t: t == 151645),
                detokenizer=SimpleNamespace(
                    reset=lambda: None,
                    add_token=lambda *a, **k: None,
                    finalize=lambda: None,
                    last_segment="",
                ),
            )
            ids = [-200, 3]
            payload = {
                "input_ids": mx.array([ids]),
                "attention_mask": mx.ones((1, 2)),
                "pixel_values": mx.ones(1),
                "image_sizes": [[320, 320]],
            }
            handler = stepper(suffix, g, model, invocations)

            def entered(*args, **kwargs):
                dispatch.append(sorted(kwargs))
                return handler(*args, **kwargs)

            state = g.PromptCacheState()
            options = (
                {}
                if mode == "omitted"
                else {
                    "thinking_budget_criteria": None
                    if mode == "explicit_none"
                    else False
                }
            )
            with ExitStack() as stack:
                for owner, name, fn in (
                    (g, "prepare_inputs", lambda *a, **kw: payload),
                    (
                        g.cache,
                        "make_prompt_cache",
                        lambda *a, **kw: [
                            SimpleNamespace(offset=0, keys=None, values=None)
                        ],
                    ),
                    (g, "wired_limit", lambda *a, **kw: nullcontext()),
                    (g, "generate_step", entered),
                ):
                    stack.enter_context(replaced(owner, name, fn))
                try:
                    chunks = list(
                        g.stream_generate(
                            model,
                            processor,
                            "fixture",
                            image=["unused"],
                            max_tokens=2,
                            temperature=0,
                            prompt_cache_state=state,
                            **options,
                        )
                    )
                except ValueError as exc:
                    if (
                        mode != "active"
                        or str(exc)
                        != "active thinking criterion is outside the replay contract"
                    ):
                        raise
                    chunks = []
                else:
                    if mode == "active":
                        raise ValueError("accepted a non-null criterion")
            expected_keys = sorted(
                ["image_sizes", "max_tokens", "prompt_cache", "temperature", *options]
            )
            if dispatch != [expected_keys]:
                raise ValueError(
                    "installed stream no longer supplies the qualified keyword set"
                )
            if mode == "active":
                if decoder_inputs or invocations or state.token_ids is not None:
                    raise ValueError("invalid criterion reached decoder")
            else:
                expected_stream = (
                    list(suffix) if suffix[-1] == 151645 else [*suffix, suffix[-1]]
                )
                if (
                    decoder_inputs != [ids, [suffix[0]], [suffix[1]]]
                    or state.token_ids != ids + list(suffix)
                    or [x.token for x in chunks] != expected_stream
                    or invocations != [suffix]
                ):
                    raise ValueError(
                        "outer stream/actual forced decoder toy integration differs"
                    )
            records.append(
                {
                    "mode": mode,
                    "suffix": list(suffix),
                    "dispatch_keys": dispatch[0],
                    "actual_stub_decoder_inputs": decoder_inputs,
                    "cached_ids": state.token_ids,
                    "notifications": [x.token for x in chunks],
                }
            )
    return records


def same_design(v1, v2):
    allowed = {
        "study_id",
        "preflight_root",
        "capture_root",
        "retained_v1_commit",
        "retained_v1_failure_sha256",
        "retained_v1_audit_sha256",
        "entry_contract",
    }
    if (
        {k: v for k, v in v1.items() if k not in allowed}
        != {k: v for k, v in v2.items() if k not in allowed}
        or v2["study_id"] != "native_suffix_replay_calibration_v2"
        or v2["capture_root"] != "runs/native_suffix_replay_calibration_v2"
        or v2["preflight_root"] != "runs/native_suffix_replay_preflight_v2"
        or v2["entry_contract"]
        != "missing_and_explicit_null_are_disabled_reject_every_non_null"
    ):
        raise ValueError("v2 changed a scientific condition or attempt namespace")


def qualify(root):
    old = base._read(base.REPO / V1_CFG)
    cfg = base._read(base.REPO / CFG)
    same_design(old, cfg)
    failed = base.prior.checked(
        base.REPO / old["capture_root"] / "failure.json",
        cfg["retained_v1_failure_sha256"],
    )
    audit_path = (
        base.REPO
        / "examples/research_notes/0074_native_suffix_replay_entry_failure/summary.json"
    )
    audit = base._read(base.prior.checked(audit_path, cfg["retained_v1_audit_sha256"]))
    if audit["failure"] != base._read(failed) or audit["parity_result"] is not None:
        raise ValueError("retained v1 status changed")
    spec, prior, anchors = V1_QUALIFY(root)
    base.prior.git(
        "merge-base",
        "--is-ancestor",
        cfg["retained_v1_commit"],
        spec["registered_commit"],
    )
    pre = base.REPO / old["preflight_root"]
    v1_spec = base._read(
        base.prior.checked(
            pre / "registration.json",
            base._read(pre / "preflight.json")["registration_sha256"],
        )
    )
    if any(spec["code_sha256"].get(k) != v for k, v in v1_spec["code_sha256"].items()):
        raise ValueError("v1 frozen files changed")
    spec["original_failed_release_commit"] = audit["registered_commit"]
    spec["no_weight_stream_fixture"] = stream_fixture()
    return spec, prior, anchors


def run(mode, root):
    # A single-process release profile: reuse every old gate and verifier unchanged.
    with ExitStack() as stack:
        for key, value in (
            ("CFG", CFG),
            ("CODE", CODE),
            ("qualify", qualify),
            ("forced_stepper", stepper),
        ):
            stack.enter_context(replaced(base, key, value))
        return base.run(mode, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "capture", "verify"))
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.mode, args.output_root)
    print(f"{args.mode}: {result['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
