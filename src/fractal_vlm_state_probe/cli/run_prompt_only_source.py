from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import subprocess
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe import mlx_cumulative_replay as replay
from fractal_vlm_state_probe.cache_tensor_artifact import (
    _to_float32_numpy,
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.cli.run_controlled_overlap import REPO
from fractal_vlm_state_probe.cli.run_controlled_overlap_cache_calibration import (
    require_offline,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _freeze_model_snapshot,
    _read,
    _sha,
)
from fractal_vlm_state_probe.cli.run_representation_observation_calibration import (
    runtime_identity,
)
from fractal_vlm_state_probe.controlled_overlap_cache import (
    CELLS,
    TARGETS,
    validate_capture,
)
from fractal_vlm_state_probe.mlx_stream import _load_mlx_runtime
from fractal_vlm_state_probe.padding_cache import prepared_input_audit
from fractal_vlm_state_probe.prompt_only import (
    ROLES,
    STAGES,
    aggregate,
    check_inputs,
    region_rows,
    source_prompt,
    token_roles,
)
from fractal_vlm_state_probe.representation_observation import (
    array_receipts,
    observe_projector,
)
from fractal_vlm_state_probe.representation_panel import (
    SITES,
    digest,
    oracle_expand,
    pack,
)
from fractal_vlm_state_probe.stimulus import write_json

CFG = "configs/prompt_only_source_calibration_v1.json"
CODE = (
    CFG,
    "src/fractal_vlm_state_probe/prompt_only.py",
    "src/fractal_vlm_state_probe/cli/run_prompt_only_source.py",
    "scripts/run_prompt_only_source.py",
    "tests/test_prompt_only_source.py",
    "docs/research_notes/0071_prompt_only_source_registration.md",
)


def read_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def checked(path, sha):
    if _sha(path) != sha:
        raise ValueError(f"file digest differs: {path}")
    return path


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def select_jobs(rows, cfg):
    keys = [(p, c) for p in cfg["pair_ids"] for c in CELLS]
    selected = [
        r
        for r in rows
        if r["phase"] == cfg["phase"]
        and r["map"] == cfg["map"]
        and r["pair_id"] in cfg["pair_ids"]
    ]
    lookup = {(r["pair_id"], r["cell"]): r for r in selected}
    if (
        len(selected) != 32
        or len(set(keys)) != 32
        or set(lookup) != set(keys)
        or cfg["cells"] != list(CELLS)
        or cfg["stages"] != list(STAGES)
        or cfg["targets"] != list(TARGETS)
        or cfg["source_contexts"] != 96
        or cfg["image_pairs"] != 32
    ):
        raise ValueError("incomplete or duplicate registered anchor identities")
    return [lookup[k] for k in keys]


def qualify(cfg, root):
    from huggingface_hub import try_to_load_from_cache
    from tokenizers import Tokenizer

    pub = _read(checked(REPO / cfg["publication"], cfg["publication_sha256"]))
    old = pub["frozen_specification"]
    ledger = (REPO / cfg["publication"]).with_name("cells.jsonl")
    rows = read_rows(checked(ledger, pub["ledgers"]["cells.jsonl"]["sha256"]))
    jobs = select_jobs(rows, cfg)
    commit = git("log", "-1", "--format=%H", "--", *CODE).decode().strip()
    git("merge-base", "--is-ancestor", cfg["parent_commit"], commit)
    hashes = dict(old["code_sha256"])
    hashes.update({name: _sha(REPO / name) for name in CODE})
    for name, sha in hashes.items():
        checked(REPO / name, sha)
        if hashlib.sha256(git("show", f"{commit}:{name}")).hexdigest() != sha:
            raise ValueError(f"registration must be committed before execution: {name}")
    for source in (
        *old["runtime_identity"]["modules"].values(),
        *old["installed_observation_sources"].values(),
    ):
        checked(Path(source["path"]), source["sha256"])
    _freeze_model_snapshot(old["config"]["model_id"], root / "model_snapshot.json")
    if _read(root / "model_snapshot.json") != old["model_snapshot"]:
        raise ValueError("model snapshot differs from qualified Note 70 runtime")
    tokenizer = Tokenizer.from_file(
        try_to_load_from_cache(old["config"]["model_id"], "tokenizer.json")
    )
    edit = cfg["token_edit"]
    if (
        tokenizer.encode(" ACK", add_special_tokens=False).ids != [edit["old_id"]]
        or tokenizer.encode(" YES", add_special_tokens=False).ids != [edit["new_id"]]
        or cfg["aligned_suffix"]
        != "require_identical_whole_realized_suffix_ids_otherwise_unavailable"
        or cfg["retry_or_resume"]
        or cfg["teacher_forced_contexts"]
        or cfg["direct_readout_contexts"]
    ):
        raise ValueError("tokenizer edit or release boundary differs")

    anchors = REPO / cfg["anchor_root"]
    for job in jobs:
        path = checked(anchors / job["path"], job["run_sha256"])
        run = _read(path)
        audit = _read(
            checked(path.with_suffix(".input_audit.json"), job["input_audit_sha256"])
        )
        manifest = checked(
            Path(run["manifest_path"]), audit["context"]["manifest_sha256"]
        )
        (frame,) = _read(manifest)["frames"]
        checked(manifest.parent / frame["path"], frame["sha256"])
        (event,) = run["stream_events"]
        source_prompt(event["prompt"], cfg, "yes")
        check_inputs(audit, audit["effective"], edit, "ack_before")
        if (
            event["frame_sha256s"] != [frame["sha256"]]
            or audit["native"] != audit["effective"]
            or audit["effective"]["pixel_values"]["sha256"]
            != job["pixel_values_sha256"]
            or run["runtime"] != old["runtime_identity"]["runtime"]
            or event["cache_tensor_artifacts"] != job["tensor_artifacts"]
        ):
            raise ValueError("historical anchor chain differs")
        for artifact in job["tensor_artifacts"]:
            checked(path.parent / artifact["path"], artifact["sha256"])
        checked(
            anchors / job["representation"]["path"], job["representation"]["sha256"]
        )
    return {
        "registered_commit": commit,
        "code_sha256": hashes,
        "publication_sha256": cfg["publication_sha256"],
        "source_ledger_sha256": _sha(ledger),
        "config": cfg,
        "jobs": jobs,
        "runtime_identity": old["runtime_identity"],
        "snapshot_sha256": _sha(root / "model_snapshot.json"),
        "snapshot_revision": old["model_snapshot"]["revision"],
        "cache_config": old["config"],
        "observation_config": old["observation_config"],
    }


def check_space(path, cfg, launch=False):
    free = shutil.disk_usage(path).free
    if free < cfg["reserve_bytes"] + (cfg["launch_allowance_bytes"] if launch else 0):
        raise ValueError("insufficient registered disk reserve")
    return free


def preflight(cfg, root):
    free = check_space(root, cfg, launch=True)
    frozen = qualify(cfg, root)
    write_json(root / "registration.json", frozen)
    result = {
        "status": "PASS",
        "source_contexts": 0,
        "weights_loaded": False,
        "free_bytes": free,
        "anchors": len(frozen["jobs"]),
        "registration_sha256": _sha(root / "registration.json"),
    }
    write_json(root / "preflight.json", result)
    return result


@contextmanager
def prompt_condition(cfg, stage):
    original = replay.cumulative_replay_prompt
    if stage == "yes":
        replay.cumulative_replay_prompt = lambda frames: source_prompt(
            original(frames), cfg, stage
        )
    try:
        yield
    finally:
        replay.cumulative_replay_prompt = original


def capture_one(root, job, stage, runtime, frozen):
    cfg, oc = frozen["config"], frozen["observation_config"]
    anchor = REPO / cfg["anchor_root"] / job["path"]
    old = _read(checked(anchor, job["run_sha256"]))
    manifest = Path(old["manifest_path"])
    gen = importlib.import_module("mlx_vlm.generate")
    proj = runtime["model"].mm_projector
    cls = type(proj)
    original_call, prepare, prompt = (
        cls.__call__,
        gen.prepare_inputs,
        replay.cumulative_replay_prompt,
    )
    if (
        f"{cls.__module__}.{cls.__name__}" != oc["projector_class"]
        or [type(m).__name__ for m in proj] != oc["projector_components"]
    ):
        raise ValueError("unqualified observation site")
    states = []

    def tracked(*args, **kwargs):
        state = kwargs["prompt_cache_state"]
        if states or state.cache is not None or state.token_ids is not None:
            raise ValueError("exactly one fresh source stream required")
        states.append(state)
        return runtime["stream_generate"](*args, **kwargs)

    root.mkdir(parents=True, exist_ok=False)
    path = root / "source.json"
    with (
        prompt_condition(cfg, stage),
        prepared_input_audit(
            gen,
            audit_path=root / "input.json",
            context={"pair_id": job["pair_id"], "cell": job["cell"], "stage": stage},
            expected_pixel_sha256=job["pixel_values_sha256"],
            expected_sizes=[[320, 320]],
        ),
        observe_projector(proj) as sites,
    ):
        replay.run_cumulative_replay_probe(
            replay.CumulativeReplayRunConfig(
                manifest_path=manifest,
                output_path=path,
                model_id=old["model_id"],
                max_frames=1,
                max_tokens=cfg["max_tokens"],
                seed=cfg["source_seed"],
                temperature=cfg["temperature"],
                probe_temperature=0,
                probe_preset="default",
                cache_summary_max_layers=None,
                source_cache_only=True,
                include_frame_artifacts=False,
                cache_tensor_captures=tuple(
                    map(parse_cache_tensor_capture_spec, TARGETS)
                ),
            ),
            mlx_runtime={**runtime, "stream_generate": tracked},
        )
    if (
        len(states) != 1
        or len(sites) != 1
        or cls.__call__ is not original_call
        or gen.prepare_inputs is not prepare
        or replay.cumulative_replay_prompt is not prompt
    ):
        raise ValueError("observation or restoration failed")
    write_json(
        root / "state.json",
        {
            "cache_state_ids": list(states[0].token_ids),
            "fresh_state": True,
            "projector_calls": len(sites),
            "restored": True,
        },
    )
    words = {}
    for name, value in zip(SITES, sites[0], strict=True):
        if str(value.dtype) != oc["source_dtype"]:
            raise ValueError("unexpected native site dtype")
        words[name] = pack(_to_float32_numpy(value))
    np.savez(root / "sites.npz", **words)


def read_caches(path, event, total, token_count):
    out = {}
    for meta in event["cache_tensor_artifacts"]:
        key = f"{meta['layer_index']}:{meta['tensor']}"
        if (
            key in out
            or meta["source_dtype"] != "mlx.core.bfloat16"
            or meta["dtype"] != "float32"
            or meta["shape"] != [1, 2, total, 64]
            or meta["effective_sequence_length"] != total
            or meta["cache_offset"] != total
            or meta["token_count"] != token_count
        ):
            raise ValueError("target identity, dtype or actual layout differs")
        with np.load(
            checked(path.parent / meta["path"], meta["sha256"]), allow_pickle=False
        ) as archive:
            a = archive["tensor"]
        if (
            a.dtype != np.float32
            or list(a.shape) != meta["shape"]
            or not np.isfinite(a).all()
            or np.any(a.view(np.uint32) & 65535)
        ):
            raise ValueError("invalid stored cache array")
        out[key] = a
    if set(out) != set(TARGETS):
        raise ValueError("selected cache coverage differs")
    return out


def inspect_one(root, job, stage, frozen):
    cfg = frozen["config"]
    anchor_root = REPO / cfg["anchor_root"]
    anchor = checked(anchor_root / job["path"], job["run_sha256"])
    old = _read(anchor)
    old_inputs = _read(
        checked(anchor.with_suffix(".input_audit.json"), job["input_audit_sha256"])
    )["effective"]
    run, audit, state = (
        _read(root / n) for n in ("source.json", "input.json", "state.json")
    )
    (event,) = run["stream_events"]
    (prior,) = old["stream_events"]
    identity = {"pair_id": job["pair_id"], "cell": job["cell"], "stage": stage}
    if (
        run["dry_run"]
        or run["model_id"] != frozen["cache_config"]["model_id"]
        or any(
            run[k] != old[k]
            for k in (
                "manifest_path",
                "context_policy",
                "runtime",
                "probes",
                "stimulus_delivery",
                "reproducibility",
            )
        )
        or event["prompt"] != source_prompt(prior["prompt"], cfg, stage)
        or event["frame_sha256s"] != prior["frame_sha256s"]
        or event["frame_paths"] != prior["frame_paths"]
        or event["image_count"] != 1
        or audit["context"] != identity
        or not state["fresh_state"]
        or state["projector_calls"] != 1
        or not state["restored"]
    ):
        raise ValueError("source path, condition or prompt differs")
    prepared = check_inputs(audit, old_inputs, cfg["token_edit"], stage)
    roles = token_roles(prepared, state["cache_state_ids"], event, cfg["token_edit"])
    is_ack = stage != "yes"
    if is_ack:
        validate_capture(run, audit, old, old_inputs, frozen["cache_config"])
    sites = job["representation"]
    with (
        np.load(root / "sites.npz", allow_pickle=False) as now,
        np.load(
            checked(anchor_root / sites["path"], sites["sha256"]), allow_pickle=False
        ) as before,
    ):
        if set(now.files) != set(SITES) or set(before.files) != set(SITES):
            raise ValueError("site coverage differs")
        arrays = {s: oracle_expand(now[s]) for s in SITES}
        receipts = array_receipts(arrays, frozen["observation_config"])
        if receipts != sites["arrays"] or any(
            arrays[s].tobytes() != oracle_expand(before[s]).tobytes() for s in SITES
        ):
            raise ValueError(
                "visual-site bytes differ; withhold matched-prompt interpretation"
            )
    caches = read_caches(
        root / "source.json",
        event,
        roles["spans"]["suffix"][1],
        len(state["cache_state_ids"]),
    )
    sums = {t: digest(v) for t, v in caches.items()}
    if is_ack:
        originals = read_caches(
            anchor, prior, roles["spans"]["suffix"][1], len(state["cache_state_ids"])
        )
        if sums != job["cache_array_sha256"] or any(
            caches[t].tobytes() != originals[t].tobytes() for t in TARGETS
        ):
            raise ValueError("ACK path failed historical cache parity")
    return {
        **identity,
        **roles,
        "frame_sha256": event["frame_sha256s"][0],
        "input_fingerprints": {
            k: {x: v[x] for x in ("shape", "dtype", "sha256")}
            for k, v in audit["effective"].items()
        },
        "image_sizes": audit["effective"]["image_sizes"]["values"],
        "site_array_sha256": {s: receipts[s]["array_sha256"] for s in SITES},
        "cache_array_sha256": sums,
        "anchor_path": job["path"],
        "historical_cache_checked": is_ack,
        "files": {
            n: _sha(root / n)
            for n in ("source.json", "input.json", "state.json", "sites.npz")
        },
    }, caches


def context_dir(root, job, stage):
    return root / "contexts" / job["pair_id"] / job["cell"] / stage


def load_gate(cfg, root):
    pre = REPO / cfg["preflight_root"]
    if (pre / "failure.json").exists():
        raise ValueError("failed preflight; no release")
    gate = _read(pre / "preflight.json")
    frozen = qualify(cfg, root)
    if (
        gate["status"] != "PASS"
        or gate["source_contexts"] != 0
        or gate["weights_loaded"]
        or _read(checked(pre / "registration.json", gate["registration_sha256"]))
        != frozen
    ):
        raise ValueError("passing unchanged preflight required")
    checked(pre / "model_snapshot.json", frozen["snapshot_sha256"])
    return frozen, {
        "preflight_sha256": _sha(pre / "preflight.json"),
        "registration_sha256": gate["registration_sha256"],
    }


def capture(cfg, root, progress):
    check_space(root, cfg, launch=True)
    frozen, gate = load_gate(cfg, root)
    runtime = _load_mlx_runtime(frozen["cache_config"]["model_id"])
    identity = runtime_identity(runtime)
    write_json(root / "runtime.json", identity)
    if identity != frozen["runtime_identity"]:
        raise ValueError("loaded runtime/processor differs from Note 70")
    for job in frozen["jobs"]:
        for stage in STAGES:
            check_space(root, cfg)
            path = context_dir(root, job, stage)
            progress.update(
                attempted=progress["attempted"] + 1,
                pair_id=job["pair_id"],
                cell=job["cell"],
                stage=stage,
            )
            write_json(root / "progress.json", progress)
            capture_one(path, job, stage, runtime, frozen)
            row, _ = inspect_one(path, job, stage, frozen)
            append(root / "contexts.jsonl", {**row, "ordinal": progress["attempted"]})
            progress["completed"] += 1
            write_json(root / "progress.json", progress)
        print(f"accepted {progress['completed']}/96 fresh source contexts", flush=True)
    if progress["completed"] != cfg["source_contexts"] or qualify(cfg, root) != frozen:
        raise ValueError("incomplete capture or source drift")
    result = {
        "status": "COMPLETE",
        "registered_commit": frozen["registered_commit"],
        **gate,
        "source_contexts": progress["completed"],
        "direct_readout_contexts": 0,
        "teacher_forced_contexts": 0,
        "contexts_sha256": _sha(root / "contexts.jsonl"),
        "runtime_sha256": _sha(root / "runtime.json"),
        "snapshot_sha256": frozen["snapshot_sha256"],
    }
    write_json(root / "capture.json", result)
    return result


def verify_region(row, a, b, left, right):
    # An independent per-position byte scan checks the vectorized change mask.
    role = row["role"]
    if role == "suffix" and left["suffix_ids"] != right["suffix_ids"]:
        if (
            row["available"]
            or row["rms"] is not None
            or row["changed_positions"] is not None
        ):
            raise ValueError("unmatched suffix scored")
        return
    start, end = row["before_span"]
    positions = [
        i
        for i in range(start, end)
        if a[:, :, i, :].tobytes() != b[:, :, i, :].tobytes()
    ]
    x = a[:, :, start:end, :].astype(np.float64)
    delta = (x - b[:, :, start:end, :]).ravel()
    rms = float(np.linalg.norm(delta) / np.sqrt(delta.size))
    if positions != row["changed_positions"] or not np.isclose(
        rms, row["rms"], rtol=1e-12, atol=0
    ):
        raise ValueError("independent region check differs")


def compare_contexts(job, bundle):
    (left, a), (right, b), (back, c) = bundle
    for k in (
        "prepared_ids",
        "cache_state_ids",
        "stream_notification_ids",
        "assistant_text",
        "spans",
        "site_array_sha256",
        "input_fingerprints",
    ):
        if left[k] != back[k]:
            raise ValueError("closing ACK did not recover the opening ACK")
    if (
        any(a[t].tobytes() != c[t].tobytes() for t in TARGETS)
        or left["site_array_sha256"] != right["site_array_sha256"]
    ):
        raise ValueError("same-image visual or return-to-baseline parity failed")
    identity = {"pair_id": job["pair_id"], "cell": job["cell"]}
    pair = {
        **identity,
        "ack_text": left["assistant_text"],
        "edited_text": right["assistant_text"],
        "ack_suffix_ids": left["suffix_ids"],
        "edited_suffix_ids": right["suffix_ids"],
        "suffix_comparable": left["suffix_ids"] == right["suffix_ids"],
        "first_changed_comparable_position": {},
    }
    rows = []
    for target in TARGETS:
        found = []
        for row in region_rows(a[target], b[target], left, right):
            verify_region(row, a[target], b[target], left, right)
            found.extend(row["changed_positions"] or [])
            rows.append({**identity, "target": target, **row})
        pair["first_changed_comparable_position"][target] = min(found, default=None)
    return pair, rows


def verify(cfg, root):
    cap = REPO / cfg["capture_root"]
    if (cap / "failure.json").exists():
        raise ValueError("failed or partial capture; withhold summary")
    frozen, gate = load_gate(cfg, root)
    receipt = _read(cap / "capture.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["source_contexts"] != 96
        or receipt["registered_commit"] != frozen["registered_commit"]
        or any(receipt[k] != val for k, val in gate.items())
        or _read(checked(cap / "runtime.json", receipt["runtime_sha256"]))
        != frozen["runtime_identity"]
    ):
        raise ValueError("capture seal differs")
    checked(cap / "model_snapshot.json", frozen["snapshot_sha256"])
    saved = read_rows(checked(cap / "contexts.jsonl", receipt["contexts_sha256"]))
    pairs, regions = [], []
    ordinal = 0
    for job in frozen["jobs"]:
        bundle = []
        for stage in STAGES:
            row, caches = inspect_one(context_dir(cap, job, stage), job, stage, frozen)
            ordinal += 1
            row["ordinal"] = ordinal
            if row != saved[ordinal - 1]:
                raise ValueError(
                    "reloaded primary artifacts do not match sealed context"
                )
            append(root / "contexts.jsonl", row)
            bundle.append((row, caches))
        pair, rr = compare_contexts(job, bundle)
        pairs.append(pair)
        regions.extend(rr)
    if (
        ordinal != len(saved)
        or len(pairs) != 32
        or len(regions) != 32 * len(TARGETS) * len(ROLES)
    ):
        raise ValueError("incomplete verified ledgers")
    for name, rows in (("pairs.jsonl", pairs), ("regions.jsonl", regions)):
        for row in rows:
            append(root / name, row)
    result = {
        "schema_version": 1,
        "study_id": cfg["study_id"],
        "status": "VERIFIED",
        "registered_commit": frozen["registered_commit"],
        **gate,
        "capture_sha256": _sha(cap / "capture.json"),
        "source_publication_sha256": cfg["publication_sha256"],
        "config_sha256": _sha(REPO / CFG),
        "code_sha256": frozen["code_sha256"],
        "model_revision": frozen["snapshot_revision"],
        "source_contexts": ordinal,
        "verification_model_calls": 0,
        "image_pairs": len(pairs),
        "paired_pixel_and_geometry_exact": len(pairs),
        "paired_word_edit_only": len(pairs),
        "site_arrays_checked_against_anchor": ordinal * len(SITES),
        "paired_visual_sites_exact": len(pairs) * len(SITES),
        "historical_ack_cache_pairs_exact": 2 * len(pairs) * len(TARGETS),
        "return_ack_cache_pairs_exact": len(pairs) * len(TARGETS),
        "ack_response_counts": dict(Counter(p["ack_text"] for p in pairs)),
        "edited_response_counts": dict(Counter(p["edited_text"] for p in pairs)),
        "aligned_suffix_pairs": sum(p["suffix_comparable"] for p in pairs),
        "cache_regions": aggregate(regions, pairs, TARGETS),
        "ledgers": {
            n: {"rows": num, "sha256": _sha(root / n)}
            for n, num in (
                ("contexts.jsonl", ordinal),
                ("pairs.jsonl", len(pairs)),
                ("regions.jsonl", len(regions)),
            )
        },
        "direct_readout_contexts": 0,
        "teacher_forced_contexts": 0,
        "correspondence_scores": 0,
        "confirmation_cohort": cfg["confirmation_cohort"],
    }
    write_json(root / "summary.json", result)
    return result


def run(mode, output_root):
    require_offline()
    cfg, root = _read(REPO / CFG), output_root.resolve()
    if mode not in ("preflight", "capture", "verify"):
        raise ValueError("unknown stage")
    if mode != "verify" and root != (REPO / cfg[mode + "_root"]).resolve():
        raise ValueError("fixed single-attempt acquisition/preflight root required")
    root.mkdir(parents=True, exist_ok=False)
    progress = {"attempted": 0, "completed": 0}
    try:
        if mode == "preflight":
            return preflight(cfg, root)
        if mode == "capture":
            return capture(cfg, root, progress)
        return verify(cfg, root)
    except Exception as exc:
        write_json(
            root / "failure.json",
            {
                "status": "FAILED",
                "mode": mode,
                **progress,
                "type": type(exc).__name__,
                "message": str(exc),
                "retry_performed": False,
            },
        )
        raise


def main():
    p = argparse.ArgumentParser(
        description="Preregistered same-image, single-token source-prompt gate."
    )
    p.add_argument("mode", choices=("preflight", "capture", "verify"))
    p.add_argument("--output-root", required=True, type=Path)
    args = p.parse_args()
    r = run(args.mode, args.output_root)
    print(f"{args.mode}: {r['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
