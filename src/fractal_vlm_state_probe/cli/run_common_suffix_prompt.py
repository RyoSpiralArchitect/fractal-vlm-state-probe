from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli import run_native_suffix_replay_v2 as qualified
from fractal_vlm_state_probe.common_suffix import (
    aggregate,
    compare_pair,
    expected_anchor,
    panel_anchors,
)
from fractal_vlm_state_probe.native_suffix_replay import (
    ARMS,
    OFFSETS,
    Checkpoints,
    compare_words,
    read_live_cache,
    reconstruct,
    replaced,
    sha_words,
    trace_call,
)
from fractal_vlm_state_probe.prompt_only import token_roles
from fractal_vlm_state_probe.representation_observation import array_receipts
from fractal_vlm_state_probe.representation_panel import oracle_expand

base = qualified.base
prior = base.prior
REPO = base.REPO
CFG = "configs/common_suffix_prompt_control_v1.json"
CODE = (
    CFG,
    "src/fractal_vlm_state_probe/common_suffix.py",
    "src/fractal_vlm_state_probe/cli/run_common_suffix_prompt.py",
    "scripts/run_common_suffix_prompt.py",
    "tests/test_common_suffix_prompt.py",
    "docs/research_notes/0077_common_suffix_prompt_registration.md",
)


def qualification(root):
    # Reuse the old qualification without editing its immutable release profile.
    with ExitStack() as stack:
        for key, value in (("CFG", qualified.CFG), ("CODE", qualified.CODE)):
            stack.enter_context(replaced(base, key, value))
        return qualified.qualify(root)


def require_config(cfg):
    expected = {
        "schema_version": 1,
        "study_id": "common_suffix_prompt_control_v1",
        "common_suffix_ids": [785, 2168],
        "stages": ["ack_before", "yes"],
        "calibration_rows": 96,
        "calibration_contexts": 288,
        "panel_images": 32,
        "panel_contexts": 64,
        "total_contexts": 352,
        "metric_rows": 12288,
        "retry_or_resume": False,
        "p_values": 0,
        "equivalence_tests": 0,
        "direct_readout_contexts": 0,
        "numeric_oracle_rtol": 1e-12,
        "numeric_oracle_atol": 1e-14,
        "reserve_bytes": 8589934592,
        "launch_allowance_bytes": 4294967296,
    }
    regions = {
        "pre_image": [0, 42],
        "image": [42, 298],
        "post_image_before_edit": [298, 387],
        "edit": [387, 388],
        "prompt_tail": [388, 395],
        "suffix": [395, 397],
        "suffix_first": [395, 396],
        "suffix_second": [396, 397],
    }
    if (
        any(cfg.get(k) != v for k, v in expected.items())
        or cfg["regions"] != regions
        or cfg["primary_targets"] != list(prior.TARGETS)
    ):
        raise ValueError("common-suffix design differs")


def qualify(root):
    cfg = base._read(REPO / CFG)
    require_config(cfg)
    oldspec, old, anchors = qualification(root)
    public = REPO / cfg["qualification_root"]
    summary = base._read(
        prior.checked(public / "summary.json", cfg["qualification_summary_sha256"])
    )
    cap = REPO / cfg["qualification_capture_root"]
    seal = base._read(prior.checked(cap / "capture.json", summary["capture_sha256"]))
    if (
        (cap / "failure.json").exists()
        or seal["status"] != "COMPLETE"
        or summary["status"] != "VERIFIED"
        or summary["code_sha256"] != oldspec["code_sha256"]
        or summary["source_contexts"] != 288
        or summary["native_forced_exact_tensor_pairs"] != 13824
    ):
        raise ValueError("incomplete or changed preceding parity qualification")
    for name, meta in summary["ledgers"].items():
        prior.checked(public / name, meta["sha256"])
    arms = prior.read_rows(
        prior.checked(cap / "arms.jsonl", summary["ledgers"]["arms.jsonl"]["sha256"])
    )
    boundaries = prior.read_rows(public / "boundaries.jsonl")
    if len(arms) != 288 or len(boundaries) != 288:
        raise ValueError("prior parity ledger coverage differs")
    for i, a in enumerate(anchors):
        check = base.compare_group(
            cap, old["jobs"][i // 3], a, old, arms[3 * i : 3 * i + 3]
        )
        if check != boundaries[3 * i : 3 * i + 3]:
            raise ValueError(
                "previous all-layer checkpoint arrays do not reproduce qualification"
            )
    commit = prior.git("log", "-1", "--format=%H", "--", *CODE).decode().strip()
    prior.git("merge-base", "--is-ancestor", cfg["parent_commit"], commit)
    hashes = {**oldspec["code_sha256"], **{n: base._sha(REPO / n) for n in CODE}}
    for name, sha in hashes.items():
        if hashlib.sha256(prior.git("show", f"{commit}:{name}")).hexdigest() != sha:
            raise ValueError(f"uncommitted measurement contract: {name}")
    return (
        {
            "registered_commit": commit,
            "config": cfg,
            "code_sha256": hashes,
            "qualification_spec": oldspec,
            "qualification_capture_sha256": summary["capture_sha256"],
            "qualification_summary_sha256": cfg["qualification_summary_sha256"],
        },
        old,
        anchors,
        boundaries,
    )


def native_contexts(root, runtime, old, anchors, previous, cfg, progress):
    cal = root / "recalibration"
    with replaced(base, "forced_stepper", qualified.stepper):
        for i, anchor in enumerate(anchors):
            job = old["jobs"][i // 3]
            group = []
            for arm in ARMS:
                prior.check_space(root, cfg)
                progress.update(
                    attempted=progress["attempted"] + 1,
                    phase="recalibration",
                    anchor=anchor["ordinal"],
                    arm=arm,
                )
                base.write_json(root / "progress.json", progress)
                row = base.acquire_arm(
                    base.location(cal, anchor, arm), arm, job, anchor, runtime, old
                )
                prior.append(root / "calibration_arms.jsonl", row)
                group.append(row)
                progress["completed"] += 1
                if arm == "native_traced":
                    plain, _ = base.reload_arm(
                        base.location(cal, anchor, "native_plain"),
                        "native_plain",
                        job,
                        anchor,
                        old,
                        group[0],
                    )
                    traced, _ = base.reload_arm(
                        base.location(cal, anchor, arm), arm, job, anchor, old, row
                    )
                    for n, a in zip(OFFSETS, traced, strict=True):
                        compare_words(
                            plain[0][..., :n, :],
                            a,
                            "fresh observer sham before own-suffix replay",
                        )
            result = base.compare_group(cal, job, anchor, old, group)
            if result != previous[3 * i : 3 * i + 3]:
                raise ValueError(
                    "same-process own-suffix parity differs from full preceding qualification"
                )
            for row in result:
                prior.append(root / "calibration_boundaries.jsonl", row)
            progress["calibrated_rows"] = i + 1
            base.write_json(root / "progress.json", progress)
            if (i + 1) % 8 == 0:
                print(
                    f"recalibration {i + 1}/96 rows ({progress['completed']}/352 contexts)",
                    flush=True,
                )
    base.write_json(
        root / "panel_release.json",
        {
            "calibrated_rows": 96,
            "completed_contexts": 288,
            "panel_contexts_started": 0,
            "calibration_arms_sha256": base._sha(root / "calibration_arms.jsonl"),
            "calibration_boundaries_sha256": base._sha(
                root / "calibration_boundaries.jsonl"
            ),
        },
    )


def panel_location(root, anchor):
    return root / "panel" / f"{anchor['ordinal']:03d}"


def common_source(root, anchor, old, terminal, suffix):
    history = prior.context_dir(
        REPO / old["config"]["capture_root"], anchor, anchor["stage"]
    )
    actual = base._read(root / "source.json")
    historical = base._read(
        prior.checked(history / "source.json", anchor["files"]["source.json"])
    )
    audit = base._read(root / "input.json")
    before = base._read(
        prior.checked(history / "input.json", anchor["files"]["input.json"])
    )
    state = base._read(root / "state.json")
    (event,) = actual["stream_events"]
    (expected,) = historical["stream_events"]
    for key in (
        "model_id",
        "dry_run",
        "manifest_path",
        "context_policy",
        "runtime",
        "probes",
        "stimulus_delivery",
        "reproducibility",
    ):
        if actual[key] != historical[key]:
            raise ValueError(f"common-suffix protocol drift: {key}")
    if (
        audit != before
        or not state["fresh_state"]
        or not state["restored"]
        or state["projector_calls"] != 1
        or state["cache_state_ids"] != anchor["prepared_ids"] + suffix
        or any(
            event[k] != expected[k]
            for k in ("prompt", "frame_paths", "frame_sha256s", "image_count")
        )
        or event["assistant_text"] != "The image"
    ):
        raise ValueError(
            "actual image/prompt, prepared fields or forced stream differs"
        )
    roles = token_roles(
        anchor["prepared_ids"],
        state["cache_state_ids"],
        event,
        old["config"]["token_edit"],
    )
    if roles["suffix_ids"] != suffix or roles["spans"] != anchor["spans"]:
        raise ValueError("common suffix/layout not actually realized")
    with (
        np.load(root / "sites.npz", allow_pickle=False) as a,
        np.load(
            prior.checked(history / "sites.npz", anchor["files"]["sites.npz"]),
            allow_pickle=False,
        ) as b,
    ):
        if set(a.files) != {"encoder", "projector"} or set(b.files) != {
            "encoder",
            "projector",
        }:
            raise ValueError("missing visual sites")
        arrays = {s: oracle_expand(a[s]) for s in ("encoder", "projector")}
        receipts = array_receipts(arrays, old["observation_config"])
        if (
            any(a[s].tobytes() != b[s].tobytes() for s in arrays)
            or {s: r["array_sha256"] for s, r in receipts.items()}
            != anchor["site_array_sha256"]
        ):
            raise ValueError("complete visual-site bytes changed")
    selected = base.read_selected(root)
    for target in prior.TARGETS:
        layer, kv = target.split(":")
        if (
            oracle_expand(terminal[int(layer), int(kv == "values")]).tobytes()
            != selected[target].tobytes()
        ):
            raise ValueError("selected cache artifact differs from all-layer words")
    return {
        "anchor_ordinal": anchor["ordinal"],
        "pair_id": anchor["pair_id"],
        "cell": anchor["cell"],
        "stage": anchor["stage"],
        "supplied_suffix_ids": suffix,
        "natural_suffix_ids": anchor["suffix_ids"],
        "prepared_ids": roles["prepared_ids"],
        "actual_cache_ids": roles["cache_state_ids"],
        "spans": roles["spans"],
        "site_array_sha256": anchor["site_array_sha256"],
        "pixel_values_sha256": anchor["input_fingerprints"]["pixel_values"]["sha256"],
        "terminal_sha256": sha_words(terminal),
        "forced_text_not_readout": event["assistant_text"],
        "source_files": {
            n: base._sha(root / n)
            for n in ("source.json", "input.json", "state.json", "sites.npz")
        },
    }


def compare_own_sham(root, anchor, cfg, own_trace, terminal):
    own_path = base.location(root.parents[1] / "recalibration", anchor, "forced_own")
    for i, own in enumerate(reconstruct(own_path, own_trace, "forced_own", anchor)):
        if i == 0 or anchor["suffix_ids"] == cfg["common_suffix_ids"]:
            compare_words(
                terminal[..., : OFFSETS[i], :],
                own,
                "full-word same-process own-suffix sham",
            )


def acquire_common(root, anchor, job, old, runtime, cfg, own_trace):
    generator = importlib.import_module("mlx_vlm.generate")
    model = runtime["model"]
    language = model.language_model
    checkpoints, invocations, states = Checkpoints(root), [], []
    modified = expected_anchor(anchor, cfg["common_suffix_ids"])
    original_call, original_step = type(language).__call__, generator.generate_step

    def tracked(*args, **kw):
        state = kw["prompt_cache_state"]
        if states or state.cache is not None or state.token_ids is not None:
            raise ValueError("panel cache is not fresh")
        states.append(state)
        return runtime["stream_generate"](*args, **kw)

    with (
        replaced(
            type(language), "__call__", trace_call(language, modified, checkpoints)
        ),
        replaced(
            generator,
            "generate_step",
            qualified.stepper(
                tuple(cfg["common_suffix_ids"]), generator, model, invocations
            ),
        ),
    ):
        prior.capture_one(
            root, job, anchor["stage"], {**runtime, "stream_generate": tracked}, old
        )
    if (
        type(language).__call__ is not original_call
        or generator.generate_step is not original_step
        or invocations != [tuple(cfg["common_suffix_ids"])]
        or len(states) != 1
        or len(checkpoints.rows) != 3
    ):
        raise ValueError("panel invocation, boundary count or restoration failed")
    terminal = read_live_cache(states[0].cache, 397)
    compare_words(
        terminal,
        checkpoints.last,
        "panel exhausted stream vs actual last decoder checkpoint",
    )
    if checkpoints.rows[0] != own_trace[0] or (
        anchor["suffix_ids"] == cfg["common_suffix_ids"]
        and checkpoints.rows != own_trace
    ):
        raise ValueError(
            "prefill or matching-own-suffix sham changed under new panel path"
        )
    compare_own_sham(root, anchor, cfg, own_trace, terminal)
    base.write_json(root / "checkpoints.json", checkpoints.rows)
    row = common_source(root, anchor, old, terminal, cfg["common_suffix_ids"])
    return {**row, "checkpoints_sha256": base._sha(root / "checkpoints.json")}


def reload_common(root, anchor, old, cfg, sealed, own_trace):
    trace = base._read(
        prior.checked(root / "checkpoints.json", sealed["checkpoints_sha256"])
    )
    if trace[0] != own_trace[0] or (
        anchor["suffix_ids"] == cfg["common_suffix_ids"] and trace != own_trace
    ):
        raise ValueError("stored same-prefill/own-suffix sham differs")
    arrays = list(
        reconstruct(
            root, trace, "forced_own", expected_anchor(anchor, cfg["common_suffix_ids"])
        )
    )
    compare_own_sham(root, anchor, cfg, own_trace, arrays[-1])
    row = {
        **common_source(root, anchor, old, arrays[-1], cfg["common_suffix_ids"]),
        "checkpoints_sha256": base._sha(root / "checkpoints.json"),
    }
    if row != sealed:
        raise ValueError("common-suffix sources no longer match sealed capture")
    return arrays[-1]


def acquire(root, spec, old, anchors, previous, progress, gate_sha):
    cfg = spec["config"]
    runtime = prior._load_mlx_runtime(old["cache_config"]["model_id"])
    identity = prior.runtime_identity(runtime)
    if identity != spec["qualification_spec"]["runtime_identity"]:
        raise ValueError("loaded runtime differs from qualified one")
    base.write_json(root / "runtime.json", identity)
    native_contexts(root, runtime, old, anchors, previous, cfg, progress)
    for i, anchor in enumerate(panel_anchors(anchors)):
        prior.check_space(root, cfg)
        progress.update(
            attempted=progress["attempted"] + 1,
            phase="common_suffix",
            anchor=anchor["ordinal"],
        )
        base.write_json(root / "progress.json", progress)
        own = base._read(
            base.location(root / "recalibration", anchor, "forced_own")
            / "checkpoints.json"
        )
        row = acquire_common(
            panel_location(root, anchor),
            anchor,
            old["jobs"][(anchor["ordinal"] - 1) // 3],
            old,
            runtime,
            cfg,
            own,
        )
        prior.append(root / "panel_cells.jsonl", row)
        progress["completed"] += 1
        progress["panel_contexts"] = i + 1
        base.write_json(root / "progress.json", progress)
        if (i + 1) % 8 == 0:
            print(
                f"common suffix {i + 1}/64 cells ({progress['completed']}/352 contexts)",
                flush=True,
            )
    if progress["completed"] != 352 or qualify(root)[0] != spec:
        raise ValueError("incomplete capture or provenance changed")
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "registered_commit": spec["registered_commit"],
        "preflight_sha256": gate_sha,
        "source_contexts": 352,
        "calibration_contexts": 288,
        "panel_contexts": 64,
        "receipts": {
            n: base._sha(root / n)
            for n in (
                "panel_release.json",
                "calibration_arms.jsonl",
                "calibration_boundaries.jsonl",
                "panel_cells.jsonl",
                "runtime.json",
            )
        },
    }
    base.write_json(root / "capture.json", result)
    return result


def verify(root, spec, old, anchors, previous, gate_sha):
    cfg = spec["config"]
    cap = REPO / cfg["capture_root"]
    seal = base._read(cap / "capture.json")
    if (
        (cap / "failure.json").exists()
        or seal["status"] != "COMPLETE"
        or seal["source_contexts"] != 352
        or seal["preflight_sha256"] != gate_sha
        or seal["registered_commit"] != spec["registered_commit"]
    ):
        raise ValueError("no complete unchanged panel")
    for name, sha in seal["receipts"].items():
        prior.checked(cap / name, sha)
    release = base._read(cap / "panel_release.json")
    calarms = prior.read_rows(cap / "calibration_arms.jsonl")
    calboundaries = prior.read_rows(cap / "calibration_boundaries.jsonl")
    cells = prior.read_rows(cap / "panel_cells.jsonl")
    if (
        len(calarms) != 288
        or calboundaries != previous
        or len(cells) != 64
        or release
        != {
            "calibrated_rows": 96,
            "completed_contexts": 288,
            "panel_contexts_started": 0,
            "calibration_arms_sha256": seal["receipts"]["calibration_arms.jsonl"],
            "calibration_boundaries_sha256": seal["receipts"][
                "calibration_boundaries.jsonl"
            ],
        }
        or base._read(cap / "runtime.json")
        != spec["qualification_spec"]["runtime_identity"]
    ):
        raise ValueError("release, calibration or denominator differs")
    for i, a in enumerate(anchors):
        results = base.compare_group(
            cap / "recalibration",
            old["jobs"][i // 3],
            a,
            old,
            calarms[3 * i : 3 * i + 3],
        )
        if results != previous[3 * i : 3 * i + 3]:
            raise ValueError(
                "fresh same-process calibration did not reproduce qualified bytes"
            )
    metrics = []
    panel = panel_anchors(anchors)
    for i in range(0, 64, 2):
        caches = []
        for idx in (i, i + 1):
            a = panel[idx]
            own = base._read(
                base.location(cap / "recalibration", a, "forced_own")
                / "checkpoints.json"
            )
            cache = reload_common(panel_location(cap, a), a, old, cfg, cells[idx], own)
            caches.append(cache)
            prior.append(root / "panel_cells.jsonl", cells[idx])
        records = list(compare_pair(*caches, panel[i], cfg))
        metrics.extend(records)
        for row in records:
            prior.append(root / "regions.jsonl", row)
    if len(metrics) != cfg["metric_rows"]:
        raise ValueError("incomplete descriptive comparisons")
    for name in ("calibration_arms.jsonl", "calibration_boundaries.jsonl"):
        shutil.copyfile(cap / name, root / name)
    result = {
        "schema_version": 1,
        "analysis_kind": cfg["study_id"],
        "status": "VERIFIED",
        "registered_commit": spec["registered_commit"],
        "config_sha256": base._sha(REPO / CFG),
        "code_sha256": spec["code_sha256"],
        "preflight_sha256": gate_sha,
        "capture_sha256": base._sha(cap / "capture.json"),
        "preceding_qualification_summary_sha256": cfg["qualification_summary_sha256"],
        "source_contexts": 352,
        "same_model_instance": True,
        "calibration_contexts": 288,
        "calibration_native_forced_exact_tensor_pairs": 13824,
        "panel_images": 32,
        "panel_contexts": 64,
        "common_suffix_ids": cfg["common_suffix_ids"],
        "primary_targets": cfg["primary_targets"],
        "prefix_sham_exact_tensor_pairs": 64 * 48,
        "full_own_suffix_sham_contexts": 34,
        "full_own_suffix_sham_tensor_pairs": 34 * 3 * 48,
        "summaries": aggregate(metrics, cfg),
        "ledgers": {
            n: {"rows": count, "sha256": base._sha(root / n)}
            for n, count in (
                ("panel_cells.jsonl", 64),
                ("regions.jsonl", cfg["metric_rows"]),
                ("calibration_arms.jsonl", 288),
                ("calibration_boundaries.jsonl", 288),
            )
        },
        "new_p_values": 0,
        "equivalence_tests": 0,
        "direct_readout_contexts": 0,
        "verification_model_calls": 0,
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
        raise ValueError("canonical single attempt required")
    root.mkdir(parents=True, exist_ok=False)
    progress = {
        "attempted": 0,
        "completed": 0,
        "calibrated_rows": 0,
        "panel_contexts": 0,
    }
    try:
        prior.check_space(root, cfg, launch=mode != "verify")
        spec, old, anchors, previous = qualify(root)
        if mode == "preflight":
            base.write_json(root / "registration.json", spec)
            result = {
                "status": "PASS",
                "source_contexts": 0,
                "weights_loaded": False,
                "registration_sha256": base._sha(root / "registration.json"),
                "free_bytes": shutil.disk_usage(root).free,
                "qualified_rows": len(anchors),
            }
            base.write_json(root / "preflight.json", result)
            return result
        pre = REPO / cfg["preflight_root"]
        gate = base._read(pre / "preflight.json")
        if (
            (pre / "failure.json").exists()
            or gate["status"] != "PASS"
            or gate["source_contexts"]
            or gate["weights_loaded"]
            or base._read(
                prior.checked(pre / "registration.json", gate["registration_sha256"])
            )
            != spec
        ):
            raise ValueError("passing identical preflight required")
        sha = base._sha(pre / "preflight.json")
        return (
            acquire(root, spec, old, anchors, previous, progress, sha)
            if mode == "capture"
            else verify(root, spec, old, anchors, previous, sha)
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
    p = argparse.ArgumentParser(
        description="Frozen common-suffix prompt comparison; no readout or cache swap."
    )
    p.add_argument("mode", choices=("preflight", "capture", "verify"))
    p.add_argument("--output-root", required=True, type=Path)
    args = p.parse_args()
    r = run(args.mode, args.output_root)
    print(f"{args.mode}: {r['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
