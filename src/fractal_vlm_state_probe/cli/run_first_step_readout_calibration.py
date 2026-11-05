from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli import run_common_suffix_prompt as common
from fractal_vlm_state_probe.first_step_readout import (
    diagnostics,
    native_score_observation,
    readout_identity,
    reload_scores,
    reload_vectors,
    save_scores,
    save_vectors,
    stream_score_observation,
)
from fractal_vlm_state_probe.native_suffix_replay import (
    Checkpoints,
    compare_words,
    replaced,
    sha_words,
    trace_call,
)
from fractal_vlm_state_probe.mlx_stream import (
    set_global_seed,
    summarize_prompt_cache_token_layout,
)
from fractal_vlm_state_probe.padding_cache import prepared_input_audit
from fractal_vlm_state_probe.prompts import SYSTEM_PROMPT
from fractal_vlm_state_probe.representation_observation import observe_projector

base, prior, REPO = common.base, common.prior, common.REPO
CFG = "configs/first_step_readout_calibration_v1.json"
CODE = (
    CFG,
    "src/fractal_vlm_state_probe/first_step_readout.py",
    "src/fractal_vlm_state_probe/cli/run_first_step_readout_calibration.py",
    "scripts/run_first_step_readout_calibration.py",
    "tests/test_first_step_readout.py",
    "docs/research_notes/0079_first_step_readout_registration.md",
    "src/fractal_vlm_state_probe/full_vocab_readout.py",
)


def require_config(cfg):
    expected = {
        "schema_version": 1,
        "study_id": "first_step_readout_calibration_v1",
        "rows": 96,
        "arms": ["native_stream_only", "native_decoder_observed", "prefix_only"],
        "source_generation_contexts": 192,
        "independent_prefix_contexts": 96,
        "total_contexts": 288,
        "vocab_size": 151936,
        "native_dtype": "mlx.core.bfloat16",
        "saved_dtype": "float32",
        "prefix_offset": 395,
        "prefix_processor_ids": 140,
        "prefix_decoder_calls": 1,
        "prefix_suffix_decoder_calls": 0,
        "interprompt_effects_computed": 0,
        "experimental_readout_panel_released": False,
        "retry_or_resume": False,
        "probability_mass_tolerance": 1e-12,
        "oracle_rtol": 1e-12,
        "oracle_atol": 1e-14,
        "reserve_bytes": 8589934592,
        "launch_allowance_bytes": 3221225472,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("readout calibration contract differs")


def qualify(root):
    cfg = base._read(REPO / CFG)
    require_config(cfg)
    inherited, old, anchors, known = common.qualify(root)
    public = REPO / cfg["qualification_root"]
    qualified = base._read(
        prior.checked(public / "summary.json", cfg["qualification_summary_sha256"])
    )
    if (
        qualified["status"] != "VERIFIED"
        or qualified["panel_contexts"] != 64
        or qualified["code_sha256"] != inherited["code_sha256"]
    ):
        raise ValueError("preceding control not qualified or changed")
    for name, meta in qualified["ledgers"].items():
        prior.checked(public / name, meta["sha256"])
    commit = prior.git("log", "-1", "--format=%H", "--", *CODE).decode().strip()
    prior.git("merge-base", "--is-ancestor", cfg["parent_commit"], commit)
    hashes = {**inherited["code_sha256"], **{n: base._sha(REPO / n) for n in CODE}}
    for name, sha in hashes.items():
        if hashlib.sha256(prior.git("show", f"{commit}:{name}")).hexdigest() != sha:
            raise ValueError(f"commit readout config/capture/verifier first: {name}")
    return (
        {
            "registered_commit": commit,
            "config": cfg,
            "code_sha256": hashes,
            "inherited_qualification": inherited,
            "preceding_summary_sha256": cfg["qualification_summary_sha256"],
        },
        old,
        anchors,
        known,
    )


def location(root, anchor, arm):
    return root / "contexts" / f"{anchor['ordinal']:03d}" / arm


def prefix_layout(state, model_config):
    layout = summarize_prompt_cache_token_layout(state, model_config)
    kv = layout["cache_position_layout"]
    if (
        layout["token_count"] != 140
        or layout["marker_positions"] != {"image_token_index": [42]}
        or not kv["available"]
        or not kv["image_partition_available"]
        or kv["strategy"] != "llava_qwen2_single_image_run_replacement"
        or kv["cache_sequence_lengths"] != [395]
        or kv["cache_sequence_length"] != 395
        or kv["image_position_runs"] != [{"start": 42, "end": 297, "length": 256}]
    ):
        raise ValueError("unknown prefix-only image/cache layout")
    return layout


def prefix_source(root, anchor, old, cfg):
    history = prior.context_dir(
        REPO / old["config"]["capture_root"], anchor, anchor["stage"]
    )
    source = base._read(
        prior.checked(history / "source.json", anchor["files"]["source.json"])
    )
    before = base._read(
        prior.checked(history / "input.json", anchor["files"]["input.json"])
    )
    (event,) = source["stream_events"]
    audit = base._read(root / "input.json")
    state = base._read(root / "prefix.json")
    manifest = base._read(Path(source["manifest_path"]))
    (frame,) = manifest["frames"]
    if (
        audit != before
        or state["actual_input_ids"] != anchor["prepared_ids"]
        or state["suffix_ids"] != []
        or state["decoder_calls"] != 1
        or state["fresh_cache"] is not True
        or state["effective_length"] != 395
        or state["frame_sha256"] != frame["sha256"]
        or event["frame_sha256s"] != [frame["sha256"]]
        or state["prompt"] != event["prompt"]
    ):
        raise ValueError(
            "prefix-only source, actual decoder IDs or suffix boundary changed"
        )
    native_sites = history / "sites.npz"
    prior.checked(native_sites, anchor["files"]["sites.npz"])
    with (
        np.load(root / "sites.npz", allow_pickle=False) as a,
        np.load(native_sites, allow_pickle=False) as b,
    ):
        if (
            set(a.files) != {"encoder", "projector"}
            or set(b.files) != set(a.files)
            or any(a[k].tobytes() != b[k].tobytes() for k in a.files)
        ):
            raise ValueError("prefix-only complete visual-site bytes differ")
    layout = state["layout"]
    if layout["token_count"] != 140 or layout["cache_position_layout"][
        "cache_sequence_lengths"
    ] != [395]:
        raise ValueError("prefix-only cached sequence differs")
    return {
        n: base._sha(root / n)
        for n in ("input.json", "prefix.json", "sites.npz", "checkpoints.json")
    }


def acquire_prefix(root, anchor, runtime, old, cfg):
    import mlx.core as mx

    from fractal_vlm_state_probe.cache_tensor_artifact import _to_float32_numpy
    from fractal_vlm_state_probe.representation_panel import pack

    root.mkdir(parents=True, exist_ok=False)
    history = prior.context_dir(
        REPO / old["config"]["capture_root"], anchor, anchor["stage"]
    )
    source = base._read(history / "source.json")
    (event,) = source["stream_events"]
    manifest = Path(source["manifest_path"])
    (frame,) = base._read(manifest)["frames"]
    image = prior.checked(manifest.parent / frame["path"], frame["sha256"])
    model, processor = runtime["model"], runtime["processor"]
    generator = importlib.import_module("mlx_vlm.generate")
    set_global_seed(old["config"]["source_seed"], include_mlx=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": event["prompt"]},
    ]
    prompt = runtime["apply_chat_template"](
        processor, runtime["model_config"], messages, num_images=1
    )
    tracked = generator.PromptCacheState()
    checkpoints = Checkpoints(root)
    native_prepare = generator.prepare_inputs
    original_call = type(model.language_model).__call__
    projector = model.mm_projector
    projector_call = type(projector).__call__
    with prepared_input_audit(
        generator,
        audit_path=root / "input.json",
        context={
            "pair_id": anchor["pair_id"],
            "cell": anchor["cell"],
            "stage": anchor["stage"],
        },
        expected_pixel_sha256=anchor["input_fingerprints"]["pixel_values"]["sha256"],
        expected_sizes=[[320, 320]],
    ):
        inputs = generator.prepare_inputs(
            processor,
            images=[str(image)],
            audio=None,
            prompts=prompt,
            image_token_index=model.config.image_token_index,
            resize_shape=None,
            add_special_tokens=True,
            max_tokens=2,
            temperature=0,
            prompt_cache_state=tracked,
        )
    audit = base._read(root / "input.json")
    ids = inputs["input_ids"]
    if (
        audit != base._read(history / "input.json")
        or ids.tolist() != [anchor["prepared_ids"]]
        or tracked.cache is not None
        or tracked.token_ids is not None
    ):
        raise ValueError("fresh prefix inputs differ before decoder entry")
    cache = generator.cache.make_prompt_cache(model.language_model)
    with (
        replaced(
            type(model.language_model),
            "__call__",
            trace_call(model.language_model, anchor, checkpoints),
        ),
        observe_projector(projector) as sites,
    ):
        with (
            generator.wired_limit(model, [generator.generation_stream]),
            mx.stream(generator.generation_stream),
        ):
            features = model.get_input_embeddings(
                ids,
                inputs["pixel_values"],
                mask=inputs["attention_mask"],
                image_sizes=inputs["image_sizes"],
            )
            if tuple(features.inputs_embeds.shape) != (1, 395, 896) or any(
                k != "inputs_embeds" and v is not None
                for k, v in features.to_dict().items()
            ):
                raise ValueError("prefix embedding insertion differs")
            output = model.language_model(
                ids,
                inputs_embeds=features.inputs_embeds,
                cache=cache,
                image_sizes=inputs["image_sizes"],
            )
            logits = output.logits[:, -1, :]
            scores = (logits - mx.logsumexp(logits)).squeeze(0)
    if (
        len(checkpoints.rows) != 1
        or len(sites) != 1
        or generator.prepare_inputs is not native_prepare
        or type(model.language_model).__call__ is not original_call
        or type(projector).__call__ is not projector_call
    ):
        raise ValueError("prefix observation or restoration failed")
    tracked.update(anchor["prepared_ids"], cache)
    layout = prefix_layout(tracked, runtime["model_config"])
    base.write_json(root / "checkpoints.json", checkpoints.rows)
    base.write_json(
        root / "prefix.json",
        {
            "actual_input_ids": checkpoints.rows[0]["input_ids"],
            "suffix_ids": [],
            "decoder_calls": len(checkpoints.rows),
            "fresh_cache": True,
            "effective_length": 395,
            "frame_sha256": frame["sha256"],
            "prompt": event["prompt"],
            "layout": layout,
        },
    )
    words = {}
    for key, value in zip(("encoder", "projector"), sites[0], strict=True):
        if str(value.dtype) != "mlx.core.bfloat16":
            raise ValueError("prefix native visual dtype differs")
        words[key] = pack(_to_float32_numpy(value))
    np.savez(root / "sites.npz", **words)
    vocab = save_vectors(root, logits, scores, cfg)
    receipt = {
        **readout_identity(anchor),
        "arm": "prefix_only",
        "source_files": prefix_source(root, anchor, old, cfg),
        "prefix_word_sha256": checkpoints.rows[0]["sha256"],
        "vocabulary": vocab,
        "logits_file_sha256": base._sha(root / "logits.npz"),
    }
    return receipt


def acquire_native(root, anchor, runtime, old, cfg, arm):
    job = old["jobs"][(anchor["ordinal"] - 1) // 3]
    if arm == "native_stream_only":
        with stream_score_observation(runtime) as (observed, scores):
            native = base.acquire_arm(root, "native_plain", job, anchor, observed, old)
        return {
            **readout_identity(anchor),
            "arm": arm,
            "source": native,
            "vocabulary": save_scores(root, scores[0], cfg),
        }
    with native_score_observation(base, runtime) as (observed, raw, scores):
        native = base.acquire_arm(root, "native_traced", job, anchor, observed, old)
    vocab = save_vectors(root, raw[0], scores[0], cfg)
    return {
        **readout_identity(anchor),
        "arm": arm,
        "source": native,
        "vocabulary": vocab,
        "logits_file_sha256": base._sha(root / "logits.npz"),
    }


def compare_row(root, anchor, old, cfg, group, known):
    job = old["jobs"][(anchor["ordinal"] - 1) // 3]
    plain, observed, prefix = (location(root, anchor, k) for k in cfg["arms"])
    native, _ = base.reload_arm(
        plain, "native_plain", job, anchor, old, group[0]["source"]
    )
    sampled, trace = base.reload_arm(
        observed, "native_traced", job, anchor, old, group[1]["source"]
    )
    if (
        len(group) != 3
        or [r["arm"] for r in group] != cfg["arms"]
        or any(
            any(r[k] != anchor[k] for k in ("ordinal", "pair_id", "cell", "stage"))
            for r in group
        )
    ):
        raise ValueError("calibration arm identity/order differs")
    compare_words(native[0], sampled[-1], "native full-vocab observer terminal sham")
    before_steps = base._read(plain / "source.json")["stream_events"][0]["generation"][
        "steps"
    ]
    observed_steps = base._read(observed / "source.json")["stream_events"][0][
        "generation"
    ]["steps"]
    if before_steps != observed_steps:
        raise ValueError(
            "native score observer changed existing generation/top-logprob records"
        )
    for i, end in enumerate((395, 396, 397)):
        compare_words(
            native[0][..., :end, :],
            sampled[i],
            "native full-vocab observer prefix sham",
        )
        if (
            trace[i]["sha256"] != known[i]["native_sha256"]
            or trace[i]["tensor_sha256"] != known[i]["tensor_sha256"]
        ):
            raise ValueError(
                "native observed cache differs from historical all-layer qualification"
            )
    src = prefix_source(prefix, anchor, old, cfg)
    if (
        src != group[2]["source_files"]
        or base._sha(prefix / "logits.npz") != group[2]["logits_file_sha256"]
    ):
        raise ValueError("prefix-only receipts changed")
    (r,) = base._read(prefix / "checkpoints.json")
    prefix_words = np.load(prefix / "kv_0.npy", allow_pickle=False)
    if (
        r != trace[0]
        or sha_words(prefix_words) != r["sha256"]
        or group[2]["prefix_word_sha256"] != r["sha256"]
    ):
        raise ValueError("prefix-only actual boundary receipt differs")
    compare_words(prefix_words, sampled[0], "independent prefix-only vs native prefill")
    actual = []
    for path, record in ((observed, group[1]), (prefix, group[2])):
        prior.checked(path / "logits.npz", record["logits_file_sha256"])
        actual.append(reload_vectors(path, record["vocabulary"], cfg))
    for idx, name in enumerate(("raw_logits", "actual_first_step_scores")):
        if actual[0][idx].tobytes() != actual[1][idx].tobytes():
            raise ValueError(
                f"independent prefix-only {name} differs from native generation"
            )
    original_scores = reload_scores(plain, group[0]["vocabulary"], cfg)
    if original_scores.tobytes() != actual[0][1].tobytes():
        raise ValueError("decoder observer changed complete native first-step scores")
    logits, scores = actual[0]
    source = base._read(observed / "source.json")
    first = source["stream_events"][0]["generation"]["steps"][0]
    diagnostics_ = diagnostics(logits, scores)
    if (
        first["token_id"] not in diagnostics_["argmax_ids"]
        or first["token_id"] != anchor["suffix_ids"][0]
    ):
        raise ValueError(
            "native sampled first ID not in the whole-vocabulary argmax set"
        )
    for token in first["top_logprobs"]:
        if float(scores[token["token_id"]]) != token["logprob"]:
            raise ValueError(
                "existing native top-logprob report differs from full saved vector"
            )
    if (
        abs(diagnostics_["renormalized_probability_sum"] - 1)
        > cfg["probability_mass_tolerance"]
    ):
        raise ValueError("declared readout probability normalization failed")
    return {
        **readout_identity(anchor),
        "vocab_size": cfg["vocab_size"],
        "native_dtype": cfg["native_dtype"],
        "saved_dtype": cfg["saved_dtype"],
        "logits_exact": True,
        "scores_exact": True,
        "logits_array_sha256": sha_words(logits),
        "score_array_sha256": sha_words(scores),
        "prefix_cache_tensors_exact": 48,
        "native_observer_terminal_tensors_exact": 48,
        "actual_prefix_decoder_calls": 1,
        "actual_prefix_suffix_calls": 0,
        "prefix_effective_length": 395,
        "natural_first_token_id": first["token_id"],
        "prefix_word_sha256": r["sha256"],
        **diagnostics_,
    }


def capture(root, spec, old, anchors, known, gate_sha, progress):
    cfg = spec["config"]
    runtime = prior._load_mlx_runtime(old["cache_config"]["model_id"])
    identity = prior.runtime_identity(runtime)
    if (
        identity
        != spec["inherited_qualification"]["qualification_spec"]["runtime_identity"]
    ):
        raise ValueError("qualified runtime changed")
    base.write_json(root / "runtime.json", identity)
    for i, a in enumerate(anchors):
        group = []
        for arm in cfg["arms"]:
            prior.check_space(root, cfg)
            progress.update(
                attempted=progress["attempted"] + 1, anchor=a["ordinal"], arm=arm
            )
            base.write_json(root / "progress.json", progress)
            path = location(root, a, arm)
            row = (
                acquire_prefix(path, a, runtime, old, cfg)
                if arm == "prefix_only"
                else acquire_native(path, a, runtime, old, cfg, arm)
            )
            prior.append(root / "arms.jsonl", row)
            group.append(row)
            progress["completed"] += 1
        compare_row(root, a, old, cfg, group, known[3 * i : 3 * i + 3])
        progress["matched_rows"] = i + 1
        base.write_json(root / "progress.json", progress)
        if (i + 1) % 8 == 0:
            print(
                f"first-step parity {i + 1}/96 rows ({progress['completed']}/288 contexts)",
                flush=True,
            )
    if progress["completed"] != 288 or qualify(root)[0] != spec:
        raise ValueError("incomplete readout calibration or changed provenance")
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "registered_commit": spec["registered_commit"],
        "preflight_sha256": gate_sha,
        "contexts": 288,
        "matched_rows": 96,
        "arms_sha256": base._sha(root / "arms.jsonl"),
        "runtime_sha256": base._sha(root / "runtime.json"),
    }
    base.write_json(root / "capture.json", result)
    return result


def verify(root, spec, old, anchors, known, gate_sha):
    cfg = spec["config"]
    cap = REPO / cfg["capture_root"]
    seal = base._read(cap / "capture.json")
    if (
        (cap / "failure.json").exists()
        or seal["status"] != "COMPLETE"
        or seal["contexts"] != 288
        or seal["matched_rows"] != 96
        or seal["registered_commit"] != spec["registered_commit"]
        or seal["preflight_sha256"] != gate_sha
        or base._read(prior.checked(cap / "runtime.json", seal["runtime_sha256"]))
        != spec["inherited_qualification"]["qualification_spec"]["runtime_identity"]
    ):
        raise ValueError("complete unchanged capture required")
    arms = prior.read_rows(prior.checked(cap / "arms.jsonl", seal["arms_sha256"]))
    if len(arms) != 288:
        raise ValueError("incomplete arm denominator")
    checks = []
    for i, a in enumerate(anchors):
        check = compare_row(
            cap, a, old, cfg, arms[3 * i : 3 * i + 3], known[3 * i : 3 * i + 3]
        )
        checks.append(check)
        prior.append(root / "comparisons.jsonl", check)
    shutil.copyfile(cap / "arms.jsonl", root / "arms.jsonl")
    result = {
        "schema_version": 1,
        "analysis_kind": cfg["study_id"],
        "status": "VERIFIED",
        "registered_commit": spec["registered_commit"],
        "code_sha256": spec["code_sha256"],
        "config_sha256": base._sha(REPO / CFG),
        "preflight_sha256": gate_sha,
        "capture_sha256": base._sha(cap / "capture.json"),
        "source_contexts": 288,
        "natural_source_contexts": 192,
        "independent_prefix_contexts": 96,
        "rows": 96,
        "unique_image_prompt_conditions": 64,
        "first_step_score_exact_pairs": 96,
        "first_step_score_coordinate_pairs": 96 * cfg["vocab_size"],
        "native_stream_vs_decoder_observer_score_pairs": 96,
        "raw_logit_exact_pairs": 96,
        "prefix_cache_exact_tensor_pairs": 96 * 48,
        "native_observer_terminal_exact_tensor_pairs": 96 * 48,
        "prefix_decoder_calls": 96,
        "prefix_suffix_calls": 0,
        "native_score_probability_sum_range": [
            min(r["native_score_probability_sum"] for r in checks),
            max(r["native_score_probability_sum"] for r in checks),
        ],
        "renormalized_probability_sum_range": [
            min(r["renormalized_probability_sum"] for r in checks),
            max(r["renormalized_probability_sum"] for r in checks),
        ],
        "native_vs_raw_logits_softmax_tv_range": [
            min(r["native_vs_raw_logits_softmax_tv"] for r in checks),
            max(r["native_vs_raw_logits_softmax_tv"] for r in checks),
        ],
        "total_negative_infinity_scores": sum(
            r["negative_infinity_count"] for r in checks
        ),
        "ledgers": {
            n: {"rows": count, "sha256": base._sha(root / n)}
            for n, count in (("arms.jsonl", 288), ("comparisons.jsonl", 96))
        },
        "verification_model_calls": 0,
        "interprompt_effects_computed": 0,
        "experimental_readout_panel_released": False,
        "confirmation_cohort": "not_generated_not_sealed",
    }
    base.write_json(root / "summary.json", result)
    return result


def run(mode, root):
    prior.require_offline()
    cfg = base._read(REPO / CFG)
    require_config(cfg)
    root = root.resolve()
    if mode not in ("preflight", "capture", "verify"):
        raise ValueError("invalid mode")
    if mode != "verify" and root != (REPO / cfg[mode + "_root"]).resolve():
        raise ValueError("canonical single-attempt root required")
    root.mkdir(parents=True, exist_ok=False)
    progress = {"attempted": 0, "completed": 0, "matched_rows": 0}
    try:
        prior.check_space(root, cfg, launch=mode != "verify")
        spec, old, anchors, known = qualify(root)
        if mode == "preflight":
            base.write_json(root / "registration.json", spec)
            result = {
                "status": "PASS",
                "source_contexts": 0,
                "weights_loaded": False,
                "rows": len(anchors),
                "registration_sha256": base._sha(root / "registration.json"),
                "free_bytes": shutil.disk_usage(root).free,
            }
            base.write_json(root / "preflight.json", result)
            return result
        pre = REPO / cfg["preflight_root"]
        gate = base._read(pre / "preflight.json")
        if (
            (pre / "failure.json").exists()
            or gate["status"] != "PASS"
            or gate["weights_loaded"]
            or gate["source_contexts"]
            or base._read(
                prior.checked(pre / "registration.json", gate["registration_sha256"])
            )
            != spec
        ):
            raise ValueError("passing identical preflight required")
        sha = base._sha(pre / "preflight.json")
        return (
            capture(root, spec, old, anchors, known, sha, progress)
            if mode == "capture"
            else verify(root, spec, old, anchors, known, sha)
        )
    except Exception as exc:
        base.write_json(
            root / "failure.json",
            {
                "status": "FAILED",
                "mode": mode,
                **progress,
                "type": type(exc).__name__,
                "message": str(exc),
                "retry": False,
            },
        )
        raise


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate native first-step full-vocabulary scores against independent prefill."
    )
    ap.add_argument("mode", choices=("preflight", "capture", "verify"))
    ap.add_argument("--output-root", required=True, type=Path)
    args = ap.parse_args()
    result = run(args.mode, args.output_root)
    print(f"{args.mode}: {result['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
