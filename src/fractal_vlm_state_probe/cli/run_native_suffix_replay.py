from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import shutil
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from fractal_vlm_state_probe.cli import run_prompt_only_source as prior
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.native_suffix_replay import (
    ARMS,
    NAMES,
    OFFSETS,
    Checkpoints,
    compare_words,
    forced_stepper,
    read_live_cache,
    reconstruct,
    replaced,
    sha_words,
    trace_call,
)
from fractal_vlm_state_probe.representation_panel import oracle_expand
from fractal_vlm_state_probe.stimulus import write_json

CFG = "configs/native_suffix_replay_calibration_v1.json"
CODE = (
    CFG,
    "src/fractal_vlm_state_probe/native_suffix_replay.py",
    "src/fractal_vlm_state_probe/cli/run_native_suffix_replay.py",
    "scripts/run_native_suffix_replay.py",
    "tests/test_native_suffix_replay.py",
    "docs/research_notes/0073_native_suffix_replay_registration.md",
)


def anchor_order(rows, jobs):
    expected = [
        (j["pair_id"], j["cell"], stage) for j in jobs for stage in prior.STAGES
    ]
    actual = [(r["pair_id"], r["cell"], r["stage"]) for r in rows]
    if (
        len(set(expected)) != 96
        or actual != expected
        or [r["ordinal"] for r in rows] != list(range(1, 97))
        or Counter(tuple(r["suffix_ids"]) for r in rows)
        != {(785, 2168): 66, (9454, 151645): 20, (9454, 11): 10}
    ):
        raise ValueError("Note 72 anchor order, suffix census or coverage differs")


def qualify(root):
    cfg, pcfg = _read(REPO / CFG), _read(REPO / prior.CFG)
    if (
        cfg["arms"] != list(ARMS)
        or cfg["offsets"] != list(OFFSETS)
        or cfg["kv"] != ["keys", "values"]
        or cfg["layers"] != 24
        or cfg["historical_rows"] != 96
        or cfg["source_contexts"] != 288
        or cfg["retry_or_resume"]
        or cfg["cross_prompt_suffixes"]
        or cfg["direct_readout_contexts"]
        or cfg["dtype"] != "mlx.core.bfloat16"
        or cfg["cache_class"] != "mlx_lm.models.cache.KVCache"
    ):
        raise ValueError("unsupported replay calibration scope")
    old, gate = prior.load_gate(pcfg, root)
    public = REPO / cfg["public_root"]
    catalog = _read(prior.checked(public / "summary.json", cfg["catalog_sha256"]))
    raw = gzip.decompress((public / "verified_summary.json.gz").read_bytes())
    if (
        catalog.pop("analysis_kind") != "prompt_only_source_calibration"
        or catalog.pop("verified_summary_sha256") != cfg["original_summary_sha256"]
        or hashlib.sha256(raw).hexdigest() != cfg["original_summary_sha256"]
        or catalog != json.loads(raw)
        or catalog["status"] != "VERIFIED"
        or catalog["code_sha256"] != old["code_sha256"]
    ):
        raise ValueError("original summary/catalog or inherited registration differs")
    cap = REPO / pcfg["capture_root"]
    seal = _read(prior.checked(cap / "capture.json", cfg["capture_receipt_sha256"]))
    if (
        (cap / "failure.json").exists()
        or seal["status"] != "COMPLETE"
        or any(gate[k] != seal[k] for k in gate)
        or seal["source_contexts"] != 96
    ):
        raise ValueError("prior capture incomplete")
    for name, meta in catalog["ledgers"].items():
        prior.checked(public / name, meta["sha256"])
    rows = prior.read_rows(
        prior.checked(
            cap / "contexts.jsonl", catalog["ledgers"]["contexts.jsonl"]["sha256"]
        )
    )
    anchor_order(rows, old["jobs"])
    for j, r in enumerate(rows):
        path = prior.context_dir(cap, r, r["stage"])
        observed, _ = prior.inspect_one(path, old["jobs"][j // 3], r["stage"], old)
        if {**observed, "ordinal": j + 1} != r:
            raise ValueError("local primary source does not reproduce historical row")
    commit = prior.git("log", "-1", "--format=%H", "--", *CODE).decode().strip()
    prior.git("merge-base", "--is-ancestor", cfg["parent_merge"], commit)
    hashes = {**old["code_sha256"], **{name: _sha(REPO / name) for name in CODE}}
    for name, value in hashes.items():
        if hashlib.sha256(prior.git("show", f"{commit}:{name}")).hexdigest() != value:
            raise ValueError(f"commit the measurement contract first: {name}")
    if importlib.metadata.version("mlx-lm") != cfg["mlx_lm_version"]:
        raise ValueError("mlx-lm version changed")
    for name, value in cfg["installed_sha256"].items():
        prior.checked(Path(importlib.util.find_spec(name).origin), value)
    spec = {
        "schema_version": 1,
        "registered_commit": commit,
        "config": cfg,
        "code_sha256": hashes,
        "prior_gate": gate,
        "prior_contexts_sha256": seal["contexts_sha256"],
        "snapshot_sha256": old["snapshot_sha256"],
        "runtime_identity": old["runtime_identity"],
    }
    return spec, old, rows


def receipt_without_files(row):
    return {k: v for k, v in row.items() if k not in ("files", "ordinal")}


def read_selected(root):
    path = root / "source.json"
    (event,) = _read(path)["stream_events"]
    return prior.read_caches(path, event, 397, 142)


def check_source(root, job, anchor, old, words):
    source, caches = prior.inspect_one(root, job, anchor["stage"], old)
    if receipt_without_files(source) != receipt_without_files(anchor):
        raise ValueError(
            "prepared inputs, image/visual bytes, native suffix, layout or selected cache differs"
        )
    earlier = prior.context_dir(
        REPO / old["config"]["capture_root"], anchor, anchor["stage"]
    )
    historical = read_selected(earlier)
    for target in prior.TARGETS:
        layer, kv = target.split(":")
        expanded = oracle_expand(words[int(layer), 0 if kv == "keys" else 1])
        if not (
            expanded.tobytes()
            == caches[target].tobytes()
            == historical[target].tobytes()
        ):
            raise ValueError(
                f"all-cache terminal disagrees with historical or new selected target: {target}"
            )
    return source["files"]


def acquire_arm(root, arm, job, anchor, runtime, old):
    generator = importlib.import_module("mlx_vlm.generate")
    model = runtime["model"]
    language = model.language_model
    call, step = type(language).__call__, generator.generate_step
    snapshots, states, injected = Checkpoints(root), [], []

    def tracked(*args, **kwargs):
        state = kwargs["prompt_cache_state"]
        if states or state.cache is not None or state.token_ids is not None:
            raise ValueError("not a fresh source context")
        states.append(state)
        return runtime["stream_generate"](*args, **kwargs)

    with ExitStack() as stack:
        if arm != "native_plain":
            stack.enter_context(
                replaced(
                    type(language), "__call__", trace_call(language, anchor, snapshots)
                )
            )
        if arm == "forced_own":
            stack.enter_context(
                replaced(
                    generator,
                    "generate_step",
                    forced_stepper(
                        tuple(anchor["suffix_ids"]), generator, model, injected
                    ),
                )
            )
        prior.capture_one(
            root, job, anchor["stage"], {**runtime, "stream_generate": tracked}, old
        )
    if (
        type(language).__call__ is not call
        or generator.generate_step is not step
        or injected != ([tuple(anchor["suffix_ids"])] if arm == "forced_own" else [])
        or len(states) != 1
        or states[0].token_ids != anchor["cache_state_ids"]
    ):
        raise ValueError(
            "decoder restoration, forcing invocation or actual cached history differs"
        )
    terminal = read_live_cache(states[0].cache, 397)
    if arm == "native_plain":
        snapshots.save(terminal, {"step": "terminal_untraced"})
    else:
        if len(snapshots.rows) != 3:
            raise ValueError("expected three observed decoder calls")
        compare_words(
            terminal,
            snapshots.last,
            "exhausted native/forced stream vs last live checkpoint",
        )
    write_json(root / "checkpoints.json", snapshots.rows)
    return arm_receipt(root, arm, job, anchor, old, terminal)


def arm_receipt(root, arm, job, anchor, old, terminal):
    return {
        "ordinal": anchor["ordinal"],
        "pair_id": anchor["pair_id"],
        "cell": anchor["cell"],
        "stage": anchor["stage"],
        "arm": arm,
        "actual_suffix_ids": anchor["suffix_ids"],
        "source_files": check_source(root, job, anchor, old, terminal),
        "checkpoints_sha256": _sha(root / "checkpoints.json"),
        "terminal_sha256": sha_words(terminal),
    }


def reload_arm(root, arm, job, anchor, old, sealed):
    trace = _read(
        prior.checked(root / "checkpoints.json", sealed["checkpoints_sha256"])
    )
    arrays = list(reconstruct(root, trace, arm, anchor))
    if arm_receipt(root, arm, job, anchor, old, arrays[-1]) != sealed:
        raise ValueError("sealed source/terminal receipts differ")
    return arrays, trace


def location(root, anchor, arm):
    return root / "contexts" / f"{anchor['ordinal']:03d}" / arm


def compare_group(root, job, anchor, old, sealed):
    groups, traces = {}, {}
    for arm, row in zip(ARMS, sealed, strict=True):
        groups[arm], traces[arm] = reload_arm(
            location(root, anchor, arm), arm, job, anchor, old, row
        )
    (plain,) = groups["native_plain"]
    compare_words(plain, groups["native_traced"][-1], "observer vs untraced terminal")
    results = []
    for i, end in enumerate(OFFSETS):
        native, forced = groups["native_traced"][i], groups["forced_own"][i]
        compare_words(
            plain[..., :end, :],
            native,
            "untraced terminal prefix vs actual native checkpoint",
        )
        compare_words(native, forced, f"natural vs forced at effective length {end}")
        r, f = traces["native_traced"][i], traces["forced_own"][i]
        if r != f:
            raise ValueError("native/forced actual decoder receipts differ")
        results.append(
            {
                "ordinal": anchor["ordinal"],
                "pair_id": anchor["pair_id"],
                "cell": anchor["cell"],
                "stage": anchor["stage"],
                "suffix_ids": anchor["suffix_ids"],
                "step": i,
                "offset": end,
                "actual_input_ids": r["input_ids"],
                "input_dtype": r["input_dtype"],
                "embedding_shape": r["embedding_shape"],
                "native_sha256": r["sha256"],
                "forced_sha256": f["sha256"],
                "tensor_sha256": r["tensor_sha256"],
                "bitwise_exact_tensors": len(NAMES),
                "scalar_pairs": int(native.size),
            }
        )
    return results


def acquire(root, frozen, old, anchors, gate_sha, progress):
    cfg = frozen["config"]
    runtime = prior._load_mlx_runtime(old["cache_config"]["model_id"])
    identity = prior.runtime_identity(runtime)
    if identity != frozen["runtime_identity"]:
        raise ValueError("runtime implementation drift")
    write_json(root / "runtime.json", identity)
    for i, anchor in enumerate(anchors):
        job, group = old["jobs"][i // 3], []
        for arm in ARMS:
            prior.check_space(root, cfg)
            progress.update(
                attempted=progress["attempted"] + 1,
                ordinal=i + 1,
                arm=arm,
                stage=anchor["stage"],
                pair_id=anchor["pair_id"],
                cell=anchor["cell"],
            )
            write_json(root / "progress.json", progress)
            row = acquire_arm(
                location(root, anchor, arm), arm, job, anchor, runtime, old
            )
            prior.append(root / "arms.jsonl", row)
            group.append(row)
            progress["completed"] += 1
            if arm == "native_traced":
                plain, _ = reload_arm(
                    location(root, anchor, ARMS[0]), ARMS[0], job, anchor, old, group[0]
                )
                traced, _ = reload_arm(
                    location(root, anchor, arm), arm, job, anchor, old, row
                )
                for end, observed in zip(OFFSETS, traced, strict=True):
                    compare_words(
                        plain[0][..., :end, :],
                        observed,
                        "native sham before releasing own-suffix arm",
                    )
        compare_group(root, job, anchor, old, group)
        progress["matched_rows"] = i + 1
        write_json(root / "progress.json", progress)
        if (i + 1) % 4 == 0:
            print(
                f"matched {i + 1}/96 anchor rows ({progress['completed']}/288 contexts)",
                flush=True,
            )
    if progress["completed"] != 288 or qualify(root)[0] != frozen:
        raise ValueError("incomplete run or post-acquisition provenance drift")
    result = {
        "status": "COMPLETE",
        "registered_commit": frozen["registered_commit"],
        "preflight_sha256": gate_sha,
        "contexts": 288,
        "matched_rows": 96,
        "arms_sha256": _sha(root / "arms.jsonl"),
        "runtime_sha256": _sha(root / "runtime.json"),
    }
    write_json(root / "capture.json", result)
    return result


def verify(root, spec, old, anchors, gate_sha):
    cap = REPO / spec["config"]["capture_root"]
    capseal = _read(cap / "capture.json")
    if (
        (cap / "failure.json").exists()
        or capseal["status"] != "COMPLETE"
        or capseal["contexts"] != 288
        or capseal["matched_rows"] != 96
        or capseal["preflight_sha256"] != gate_sha
        or capseal["registered_commit"] != spec["registered_commit"]
        or _read(prior.checked(cap / "runtime.json", capseal["runtime_sha256"]))
        != spec["runtime_identity"]
    ):
        raise ValueError("no complete matching capture")
    sealed = prior.read_rows(prior.checked(cap / "arms.jsonl", capseal["arms_sha256"]))
    if len(sealed) != 288:
        raise ValueError("wrong capture denominator")
    for i, a in enumerate(anchors):
        group = sealed[3 * i : 3 * i + 3]
        results = compare_group(cap, old["jobs"][i // 3], a, old, group)
        for row in group:
            prior.append(root / "arms.jsonl", row)
        for result in results:
            prior.append(root / "boundaries.jsonl", result)
    result = {
        "schema_version": 1,
        "analysis_kind": "native_suffix_replay_calibration",
        "status": "VERIFIED",
        "registered_commit": spec["registered_commit"],
        "preflight_sha256": gate_sha,
        "capture_sha256": _sha(cap / "capture.json"),
        "source_summary_sha256": spec["config"]["original_summary_sha256"],
        "code_sha256": spec["code_sha256"],
        "snapshot_sha256": spec["snapshot_sha256"],
        "historical_rows": len(anchors),
        "unique_prompt_image_conditions": 64,
        "source_contexts": len(sealed),
        "native_plain": 96,
        "native_traced": 96,
        "forced_own": 96,
        "suffix_counts_including_ack_repeats": {
            ",".join(map(str, k)): v
            for k, v in Counter(tuple(a["suffix_ids"]) for a in anchors).items()
        },
        "actual_forced_single_token_calls": 192,
        "effective_offsets": list(OFFSETS),
        "tensor_names": list(NAMES),
        "native_observer_terminal_exact_pairs": 96 * 48,
        "historical_selected_exact_pairs": 288 * 3,
        "matched_boundary_rows": 96 * 3,
        "native_forced_exact_tensor_pairs": 96 * 3 * 48,
        "native_forced_exact_scalar_pairs": 96 * 48 * 128 * sum(OFFSETS),
        "ledgers": {
            n: {"rows": 288, "sha256": _sha(root / n)}
            for n in ("arms.jsonl", "boundaries.jsonl")
        },
        "verification_model_calls": 0,
        "cross_prompt_suffixes": 0,
        "direct_readout_contexts": 0,
        "confirmation_cohort": "not_generated_not_sealed",
    }
    write_json(root / "summary.json", result)
    return result


def run(mode, root):
    prior.require_offline()
    cfg, root = _read(REPO / CFG), root.resolve()
    if mode not in ("preflight", "capture", "verify"):
        raise ValueError("invalid mode")
    if mode != "verify" and root != (REPO / cfg[mode + "_root"]).resolve():
        raise ValueError("canonical single-attempt output required")
    root.mkdir(parents=True, exist_ok=False)
    progress = {"attempted": 0, "completed": 0, "matched_rows": 0}
    try:
        prior.check_space(root, cfg, launch=mode != "verify")
        spec, old, anchors = qualify(root)
        pre = REPO / cfg["preflight_root"]
        if mode == "preflight":
            write_json(root / "registration.json", spec)
            result = {
                "status": "PASS",
                "weights_loaded": False,
                "source_contexts": 0,
                "historical_rows": len(anchors),
                "free_bytes": shutil.disk_usage(root).free,
                "registration_sha256": _sha(root / "registration.json"),
            }
            write_json(root / "preflight.json", result)
            return result
        gate = _read(pre / "preflight.json")
        if (
            (pre / "failure.json").exists()
            or gate["status"] != "PASS"
            or gate["weights_loaded"]
            or gate["source_contexts"]
            or _read(
                prior.checked(pre / "registration.json", gate["registration_sha256"])
            )
            != spec
        ):
            raise ValueError("passing unchanged preflight required")
        sha = _sha(pre / "preflight.json")
        return (
            acquire(root, spec, old, anchors, sha, progress)
            if mode == "capture"
            else verify(root, spec, old, anchors, sha)
        )
    except Exception as exc:
        write_json(
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
        description="Single-token native-suffix replay parity; no cross-prompt intervention."
    )
    ap.add_argument("mode", choices=("preflight", "capture", "verify"))
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    r = run(args.mode, args.output_root)
    print(f"{args.mode}: {r['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
