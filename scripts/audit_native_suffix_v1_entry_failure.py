"""Audit the retained v1 failure and the installed stream's call signature, without a model."""

import argparse
import hashlib
import importlib
import inspect
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fractal_vlm_state_probe.cli import run_native_suffix_replay as reg
from fractal_vlm_state_probe.native_suffix_replay import forced_stepper


def inactive_stream_entry():
    """Exercise the installed outer stream up to, but not inside, model dispatch."""
    g = importlib.import_module("mlx_vlm.generate")
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="llava_qwen2", image_token_index=-200),
        language_model=object(),
    )
    processor = SimpleNamespace(
        detokenizer=SimpleNamespace(reset=lambda: None),
        tokenizer=SimpleNamespace(thinking_budget_criteria=None),
    )
    got = {}

    class AtStep(Exception):
        pass

    def spy(*args, **kwargs):
        got.update(kwargs)
        raise AtStep

    payload = {
        "input_ids": np.array([[1, -200]], dtype=np.int64),
        "pixel_values": np.ones(1),
        "attention_mask": np.ones((1, 2)),
        "image_sizes": [[320, 320]],
    }
    with ExitStack() as stack:
        for owner, name, fn in (
            (g, "prepare_inputs", lambda *a, **k: payload),
            (g.cache, "make_prompt_cache", lambda *a, **k: []),
            (g, "wired_limit", lambda *a, **k: nullcontext()),
            (g, "generate_step", spy),
        ):
            stack.enter_context(patch.object(owner, name, fn))
        try:
            next(
                g.stream_generate(
                    model,
                    processor,
                    "fixture",
                    image=["unused"],
                    max_tokens=2,
                    temperature=0,
                    prompt_cache_state=g.PromptCacheState(),
                )
            )
        except AtStep:
            return got
    raise ValueError("did not observe installed generator entry")


def audit():
    reg.prior.require_offline()
    cfg = reg._read(reg.REPO / reg.CFG)
    root = reg.REPO / cfg["capture_root"]
    fp = reg.prior.checked(
        root / "failure.json",
        "f14262ad86faf6dfaf18492212ab4aeda134803f1300a4df9b6298aa09bbe804",
    )
    failure = reg._read(fp)
    pre = reg.REPO / cfg["preflight_root"]
    frozen = reg._read(
        reg.prior.checked(
            pre / "registration.json",
            reg._read(pre / "preflight.json")["registration_sha256"],
        )
    )
    sealed = reg.prior.read_rows(
        reg.prior.checked(
            root / "arms.jsonl",
            "52bb6241b5b6d732cb4a9b291fd45680c292acbef9513297415c797516d71a20",
        )
    )
    if (
        failure["type"] != "TypeError"
        or failure["attempted"] != 3
        or failure["completed"] != 2
        or failure["matched_rows"] != 0
        or (root / "capture.json").exists()
        or [r["arm"] for r in sealed] != ["native_plain", "native_traced"]
    ):
        raise ValueError("not the retained v1 entry failure")
    anchor = reg.prior.read_rows(reg.REPO / cfg["public_root"] / "contexts.jsonl")[0]
    d = root / "contexts" / "001"
    if [p.name for p in (d / "forced_own").iterdir()] != ["input.json"]:
        raise ValueError("unexpected forced-arm artifacts")
    for arm in reg.ARMS:
        if reg._read(d / arm / "input.json") != reg._read(
            d / "native_plain" / "input.json"
        ):
            raise ValueError("actual prepared input differs")
    plain = np.load(d / "native_plain" / "kv_0.npy", allow_pickle=False)
    (plain_receipt,) = reg._read(
        reg.prior.checked(
            d / "native_plain" / "checkpoints.json", sealed[0]["checkpoints_sha256"]
        )
    )
    if (
        plain.shape != (24, 2, 1, 2, 397, 64)
        or plain.dtype != np.dtype("<u2")
        or hashlib.sha256(plain.tobytes()).hexdigest() != plain_receipt["sha256"]
    ):
        raise ValueError("native-plain primary data differs")
    trace = reg._read(
        reg.prior.checked(
            d / "native_traced" / "checkpoints.json", sealed[1]["checkpoints_sha256"]
        )
    )
    checks = []
    for i, (r, end) in enumerate(zip(trace, (395, 396, 397), strict=True)):
        start = 0 if i == 0 else end - 1
        part = np.load(d / "native_traced" / f"kv_{i}.npy", allow_pickle=False)
        if (
            part.dtype != plain.dtype
            or part.tobytes() != plain[..., start:end, :].tobytes()
        ):
            raise ValueError("native append words differ from untraced terminal")
        prefix = plain[..., :end, :]
        digest = hashlib.sha256(prefix.tobytes()).hexdigest()
        expected_ids = (
            anchor["prepared_ids"] if not i else anchor["suffix_ids"][i - 1 : i]
        )
        if (
            digest != r["sha256"]
            or r["input_ids"] != expected_ids
            or r["offset"] != end
        ):
            raise ValueError("traced checkpoint/IDs differ")
        for layer in range(24):
            for k, name in enumerate(("keys", "values")):
                if (
                    hashlib.sha256(prefix[layer, k].tobytes()).hexdigest()
                    != r["tensor_sha256"][f"{layer}:{name}"]
                ):
                    raise ValueError("native tensor digest differs")
        checks.append({"offset": end, "matched_tensors": 48, "whole_sha256": digest})
    g = importlib.import_module("mlx_vlm.generate")
    kwargs = inactive_stream_entry()
    signature = inspect.signature(forced_stepper([785, 2168], g, None, []))
    try:
        signature.bind(None, None, None, None, **kwargs)
    except TypeError as exc:
        if (
            "thinking_budget_criteria" not in str(exc)
            or "thinking_budget_criteria" in kwargs
        ):
            raise
        binding_error = str(exc)
    else:
        raise ValueError("v1 missing-key failure did not reproduce")
    signature.bind(None, None, None, None, **kwargs, thinking_budget_criteria=None)
    return {
        "schema_version": 1,
        "analysis_kind": "native_suffix_replay_v1_failure_audit",
        "status": "NOT_QUALIFIED",
        "registered_commit": frozen["registered_commit"],
        "preflight_sha256": reg._sha(pre / "preflight.json"),
        "failure_sha256": reg._sha(fp),
        "failure": failure,
        "partial_arms_sha256": reg._sha(root / "arms.jsonl"),
        "native_controls_one_row": checks,
        "native_terminal_exact_pairs": 48,
        "historical_selected_pairs_in_two_admitted_arms": 6,
        "planned_anchor_rows": 96,
        "completed_native_forced_comparisons": 0,
        "forced_decoder_calls": 0,
        "parity_result": None,
        "retry": False,
        "audit_model_calls": 0,
        "actual_outer_stream_step_kwargs_in_synthetic_fixture": sorted(kwargs),
        "signature_bind_error": binding_error,
        "explicit_none_binds": True,
        "forced_input_audit_sha256": reg._sha(d / "forced_own" / "input.json"),
        "generator_source_sha256": reg._sha(Path(g.__file__)),
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--output-root", type=Path, required=True)
    dest = args.parse_args().output_root
    dest.mkdir(parents=True, exist_ok=False)
    result = audit()
    reg.write_json(dest / "summary.json", result)
    print(result["status"], dest)
