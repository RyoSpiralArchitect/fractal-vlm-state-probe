from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli import (
    run_first_step_readout_calibration as calibration,
)
from fractal_vlm_state_probe.common_suffix import panel_anchors
from fractal_vlm_state_probe.first_step_readout import reload_vectors
from fractal_vlm_state_probe.first_token_panel import (
    aggregate,
    check_metrics,
    oracle_expand_scores,
    pack_scores,
    paired_metrics,
)
from fractal_vlm_state_probe.native_suffix_replay import compare_words, sha_words

base, prior, REPO = calibration.base, calibration.prior, calibration.REPO
CFG = "configs/first_token_prompt_panel_v1.json"
CODE = (
    CFG,
    "src/fractal_vlm_state_probe/first_token_panel.py",
    "src/fractal_vlm_state_probe/cli/run_first_token_prompt_panel.py",
    "scripts/run_first_token_prompt_panel.py",
    "tests/test_first_token_prompt_panel.py",
    "docs/research_notes/0081_first_token_prompt_panel_registration.md",
)


def storage_budget():
    native_pair = (
        2 * (48 * 128 * 397 * 2 + 1015808 * 2 + 3 * 1 * 2 * 397 * 64 * 4)
        + 3 * 151936 * 4
    )
    prefix = 48 * 128 * 395 * 2 + 1015808 * 2 + 2 * 151936 * 4
    return {
        "calibration_payload_bytes": 96 * (native_pair + prefix),
        "panel_payload_bytes": 64 * prefix,
        "overhead_allowance_bytes": 512 * 2**20,
        "reserve_bytes": 8 * 2**30,
    }


def require_config(cfg):
    values = {
        "study_id": "first_token_prompt_panel_v1",
        "calibration_rows": 96,
        "calibration_contexts": 288,
        "panel_images": 32,
        "panel_contexts": 64,
        "total_contexts": 352,
        "vocab_size": 151936,
        "panel_stages": ["ack_before", "yes"],
        "primary_endpoint": "equal_image_mean_total_variation_over_32_pairs",
        "top_delta_coordinates": 20,
        "numeric_oracle_rtol": 1e-12,
        "numeric_oracle_atol": 1e-14,
        "retry_or_resume": False,
        "p_values": 0,
        "confidence_intervals": 0,
        "equivalence_tests": 0,
        "new_confirmation_inputs": 0,
        **storage_budget(),
    }
    values.update(
        probability_definition="float64_exp(saved_native_logprobs_minus_float64_logsumexp)_no_vocab_filter",
        secondary_metrics=[
            "jensen_shannon_nats",
            "hellinger",
            "delta_l2",
            "max_abs_delta",
            "full_argmax_sets",
        ],
        portable_scores="64_by_151936_uint16_native_bfloat16_words_no_loss",
        top_delta_order="descending_abs_delta_then_ascending_token_id_display_only",
    )
    allocation = sum(
        values[k]
        for k in (
            "calibration_payload_bytes",
            "panel_payload_bytes",
            "overhead_allowance_bytes",
        )
    )
    if (
        any(cfg.get(k) != v for k, v in values.items())
        or cfg["launch_allowance_bytes"] != allocation
    ):
        raise ValueError(
            "paired readout panel design or complete storage allocation differs"
        )


def qualify(root):
    cfg = base._read(REPO / CFG)
    require_config(cfg)
    previous, old, anchors, known = calibration.qualify(root)
    public = REPO / cfg["qualification_root"]
    summary = base._read(
        prior.checked(public / "summary.json", cfg["qualification_summary_sha256"])
    )
    cap = REPO / cfg["qualification_capture_root"]
    seal = base._read(prior.checked(cap / "capture.json", summary["capture_sha256"]))
    if (
        summary["status"] != "VERIFIED"
        or summary["code_sha256"] != previous["code_sha256"]
        or (cap / "failure.json").exists()
        or seal["status"] != "COMPLETE"
        or seal["contexts"] != 288
    ):
        raise ValueError("prior suffix-free qualification incomplete or changed")
    for name, meta in summary["ledgers"].items():
        prior.checked(public / name, meta["sha256"])
    arms = prior.read_rows(
        prior.checked(cap / "arms.jsonl", summary["ledgers"]["arms.jsonl"]["sha256"])
    )
    checks = prior.read_rows(public / "comparisons.jsonl")
    if len(arms) != 288 or len(checks) != 96:
        raise ValueError("prior readout denominator differs")
    for i, a in enumerate(anchors):
        if (
            calibration.compare_row(
                cap,
                a,
                old,
                previous["config"],
                arms[3 * i : 3 * i + 3],
                known[3 * i : 3 * i + 3],
            )
            != checks[i]
        ):
            raise ValueError(
                "historical complete raw logits/scores/prefix arrays do not reproduce qualification"
            )
    commit = prior.git("log", "-1", "--format=%H", "--", *CODE).decode().strip()
    prior.git("merge-base", "--is-ancestor", cfg["parent_commit"], commit)
    hashes = {**previous["code_sha256"], **{n: base._sha(REPO / n) for n in CODE}}
    for name, value in hashes.items():
        if hashlib.sha256(prior.git("show", f"{commit}:{name}")).hexdigest() != value:
            raise ValueError(f"freeze the readout panel before new acquisition: {name}")
    return (
        {
            "registered_commit": commit,
            "config": cfg,
            "code_sha256": hashes,
            "calibration_spec": previous,
            "qualification_summary_sha256": cfg["qualification_summary_sha256"],
        },
        old,
        anchors,
        known,
        checks,
    )


def recalibrate(root, spec, old, anchors, known, previous, runtime, progress):
    cfg, calcfg = spec["config"], spec["calibration_spec"]["config"]
    dest = root / "recalibration"
    for i, a in enumerate(anchors):
        group = []
        for arm in calcfg["arms"]:
            prior.check_space(root, cfg)
            progress.update(
                attempted=progress["attempted"] + 1,
                phase="calibration",
                anchor=a["ordinal"],
                arm=arm,
            )
            base.write_json(root / "progress.json", progress)
            path = calibration.location(dest, a, arm)
            row = (
                calibration.acquire_prefix(path, a, runtime, old, calcfg)
                if arm == "prefix_only"
                else calibration.acquire_native(path, a, runtime, old, calcfg, arm)
            )
            group.append(row)
            prior.append(root / "calibration_arms.jsonl", row)
            progress["completed"] += 1
        result = calibration.compare_row(
            dest, a, old, calcfg, group, known[3 * i : 3 * i + 3]
        )
        if result != previous[i]:
            raise ValueError(
                "fresh in-process full readout does not reproduce Note 80 identity/normalization"
            )
        prior.append(root / "calibration_comparisons.jsonl", result)
        progress["calibrated_rows"] = i + 1
        base.write_json(root / "progress.json", progress)
        if (i + 1) % 8 == 0:
            print(
                f"readout recalibration {i + 1}/96 ({progress['completed']}/352 contexts)",
                flush=True,
            )
    base.write_json(
        root / "panel_release.json",
        {
            "calibrated_rows": 96,
            "completed_contexts": 288,
            "experimental_contexts_started": 0,
            "arms_sha256": base._sha(root / "calibration_arms.jsonl"),
            "comparisons_sha256": base._sha(root / "calibration_comparisons.jsonl"),
        },
    )


def panel_location(root, a):
    return root / "panel" / f"{a['ordinal']:03d}"


def reload_panel(root, a, old, calcfg, row, calrow):
    path = panel_location(root, a)
    if calibration.prefix_source(path, a, old, calcfg) != row["source_files"]:
        raise ValueError(
            "panel prepared inputs, actual decoder/suffix boundary or visual arrays changed"
        )
    prior.checked(path / "logits.npz", row["logits_file_sha256"])
    values = reload_vectors(path, row["vocabulary"], calcfg)
    calpath = calibration.location(root / "recalibration", a, "prefix_only")
    prior.checked(calpath / "logits.npz", calrow["logits_file_sha256"])
    baseline = reload_vectors(calpath, calrow["vocabulary"], calcfg)
    if any(x.tobytes() != y.tobytes() for x, y in zip(values, baseline, strict=True)):
        raise ValueError(
            "experimental prefix logits/scores differ from matched same-process calibration"
        )
    (checkpoint,) = base._read(path / "checkpoints.json")
    (native,) = base._read(calpath / "checkpoints.json")
    words = np.load(path / "kv_0.npy", allow_pickle=False)
    other = np.load(calpath / "kv_0.npy", allow_pickle=False)
    compare_words(words, other, "experimental prefix vs same-process calibration")
    if (
        checkpoint != native
        or checkpoint["sha256"] != row["prefix_word_sha256"]
        or sha_words(words) != checkpoint["sha256"]
    ):
        raise ValueError("experimental all-layer prefix/decoder receipt changed")
    if any(row[k] != a[k] for k in ("ordinal", "pair_id", "cell", "stage")):
        raise ValueError("experimental row identity differs")
    return values[1]


def acquire(root, spec, old, anchors, known, previous, gate_sha, progress):
    cfg = spec["config"]
    runtime = prior._load_mlx_runtime(old["cache_config"]["model_id"])
    identity = prior.runtime_identity(runtime)
    if (
        identity
        != spec["calibration_spec"]["inherited_qualification"]["qualification_spec"][
            "runtime_identity"
        ]
    ):
        raise ValueError("original qualified runtime changed")
    base.write_json(root / "runtime.json", identity)
    recalibrate(root, spec, old, anchors, known, previous, runtime, progress)
    calcfg = spec["calibration_spec"]["config"]
    calrows = prior.read_rows(root / "calibration_arms.jsonl")
    for i, a in enumerate(panel_anchors(anchors)):
        prior.check_space(root, cfg)
        progress.update(
            attempted=progress["attempted"] + 1,
            phase="panel",
            anchor=a["ordinal"],
            arm="prefix_only",
        )
        base.write_json(root / "progress.json", progress)
        row = {
            **calibration.acquire_prefix(
                panel_location(root, a), a, runtime, old, calcfg
            ),
            "panel_ordinal": i + 1,
        }
        reload_panel(root, a, old, calcfg, row, calrows[3 * (a["ordinal"] - 1) + 2])
        prior.append(root / "panel_cells.jsonl", row)
        progress["completed"] += 1
        progress["panel_contexts"] = i + 1
        base.write_json(root / "progress.json", progress)
        if (i + 1) % 8 == 0:
            print(
                f"first-token panel {i + 1}/64 ({progress['completed']}/352 contexts)",
                flush=True,
            )
    if progress["completed"] != 352 or qualify(root)[0] != spec:
        raise ValueError("incomplete panel or provenance drift")
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "registered_commit": spec["registered_commit"],
        "preflight_sha256": gate_sha,
        "contexts": 352,
        "calibration_contexts": 288,
        "panel_contexts": 64,
        "receipts": {
            n: base._sha(root / n)
            for n in (
                "runtime.json",
                "panel_release.json",
                "calibration_arms.jsonl",
                "calibration_comparisons.jsonl",
                "panel_cells.jsonl",
            )
        },
    }
    base.write_json(root / "capture.json", result)
    return result


def verify(root, spec, old, anchors, known, previous, gate_sha):
    cfg, calcfg = spec["config"], spec["calibration_spec"]["config"]
    cap = REPO / cfg["capture_root"]
    seal = base._read(cap / "capture.json")
    if (
        (cap / "failure.json").exists()
        or seal["status"] != "COMPLETE"
        or seal["contexts"] != 352
        or seal["panel_contexts"] != 64
        or seal["calibration_contexts"] != 288
        or seal["registered_commit"] != spec["registered_commit"]
        or seal["preflight_sha256"] != gate_sha
    ):
        raise ValueError("complete matching panel capture required")
    for name, value in seal["receipts"].items():
        prior.checked(cap / name, value)
    if (
        base._read(cap / "runtime.json")
        != spec["calibration_spec"]["inherited_qualification"]["qualification_spec"][
            "runtime_identity"
        ]
    ):
        raise ValueError("capture runtime identity differs")
    cal = prior.read_rows(cap / "calibration_arms.jsonl")
    cells = prior.read_rows(cap / "panel_cells.jsonl")
    checked = prior.read_rows(cap / "calibration_comparisons.jsonl")
    release = base._read(cap / "panel_release.json")
    if (
        len(cal) != 288
        or len(cells) != 64
        or checked != previous
        or release
        != {
            "calibrated_rows": 96,
            "completed_contexts": 288,
            "experimental_contexts_started": 0,
            "arms_sha256": seal["receipts"]["calibration_arms.jsonl"],
            "comparisons_sha256": seal["receipts"]["calibration_comparisons.jsonl"],
        }
    ):
        raise ValueError(
            "calibration, release seal or complete paired denominator differs"
        )
    for i, a in enumerate(anchors):
        if (
            calibration.compare_row(
                cap / "recalibration",
                a,
                old,
                calcfg,
                cal[3 * i : 3 * i + 3],
                known[3 * i : 3 * i + 3],
            )
            != previous[i]
        ):
            raise ValueError(
                "in-process calibration no longer reproduces its accepted data"
            )
    selected = panel_anchors(anchors)
    arrays = []
    for i, (a, row) in enumerate(zip(selected, cells, strict=True)):
        if row["panel_ordinal"] != i + 1:
            raise ValueError("panel order differs")
        arrays.append(
            reload_panel(cap, a, old, calcfg, row, cal[3 * (a["ordinal"] - 1) + 2])
        )
    matrix = np.stack([pack_scores(x) for x in arrays])
    np.savez_compressed(
        root / "native_scores.npz",
        native_score_words=matrix,
        anchor_ordinals=np.array([a["ordinal"] for a in selected], dtype="<i2"),
    )
    # Re-expand through another bit-level path before any paired analysis.
    with np.load(root / "native_scores.npz", allow_pickle=False) as archive:
        words = archive["native_score_words"]
        if words.shape != (64, 151936):
            raise ValueError("portable vocabulary matrix incomplete")
        reloaded = oracle_expand_scores(words)
        if any(reloaded[i].tobytes() != arrays[i].tobytes() for i in range(64)):
            raise ValueError("lossless public score archive failed roundtrip")
    pairs = []
    for i in range(0, 64, 2):
        a, b = selected[i : i + 2]
        if (a["pair_id"], a["cell"]) != (b["pair_id"], b["cell"]) or (
            a["stage"],
            b["stage"],
        ) != tuple(cfg["panel_stages"]):
            raise ValueError("mispaired experimental images/prompts")
        m = paired_metrics(arrays[i], arrays[i + 1], cfg["top_delta_coordinates"])
        check_metrics(
            reloaded[i],
            reloaded[i + 1],
            m,
            cfg["numeric_oracle_rtol"],
            cfg["numeric_oracle_atol"],
        )
        row = {
            "pair_id": a["pair_id"],
            "cell": a["cell"],
            "ack_panel_ordinal": i + 1,
            "yes_panel_ordinal": i + 2,
            "ack_anchor_ordinal": a["ordinal"],
            "yes_anchor_ordinal": b["ordinal"],
            "ack_score_array_sha256": sha_words(arrays[i]),
            "yes_score_array_sha256": sha_words(arrays[i + 1]),
            **m,
        }
        pairs.append(row)
        prior.append(root / "pairs.jsonl", row)
    for name in (
        "panel_cells.jsonl",
        "calibration_arms.jsonl",
        "calibration_comparisons.jsonl",
    ):
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
        "qualification_summary_sha256": cfg["qualification_summary_sha256"],
        "fresh_contexts": 352,
        "calibration_contexts": 288,
        "panel_contexts": 64,
        "panel_prefix_suffix_calls": 0,
        "paired_images": 32,
        "same_model_instance": True,
        "calibration_first_step_score_exact_pairs": 96,
        "calibration_raw_logit_exact_pairs": 96,
        "calibration_prefix_cache_exact_tensors": 4608,
        "panel_prefix_sham_exact_tensors": 64 * 48,
        "probability_definition": cfg["probability_definition"],
        "result": aggregate(pairs),
        "archive": {
            "file": "native_scores.npz",
            "sha256": base._sha(root / "native_scores.npz"),
            "dtype": "<u2",
            "shape": [64, 151936],
            "value_kind": "native_bfloat16_score_words_not_probabilities",
        },
        "ledgers": {
            n: {"rows": count, "sha256": base._sha(root / n)}
            for n, count in (
                ("pairs.jsonl", 32),
                ("panel_cells.jsonl", 64),
                ("calibration_arms.jsonl", 288),
                ("calibration_comparisons.jsonl", 96),
            )
        },
        "p_values": 0,
        "confidence_intervals": 0,
        "equivalence_tests": 0,
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
        raise ValueError("invalid stage")
    if mode != "verify" and root != (REPO / cfg[mode + "_root"]).resolve():
        raise ValueError("canonical single acquisition root required")
    root.mkdir(parents=True, exist_ok=False)
    progress = {
        "attempted": 0,
        "completed": 0,
        "calibrated_rows": 0,
        "panel_contexts": 0,
    }
    try:
        prior.check_space(root, cfg, launch=mode != "verify")
        spec, old, anchors, known, previous = qualify(root)
        if mode == "preflight":
            base.write_json(root / "registration.json", spec)
            result = {
                "status": "PASS",
                "weights_loaded": False,
                "contexts": 0,
                "rows": len(anchors),
                "registration_sha256": base._sha(root / "registration.json"),
                "free_bytes": shutil.disk_usage(root).free,
                "storage_budget": storage_budget(),
            }
            base.write_json(root / "preflight.json", result)
            return result
        pre = REPO / cfg["preflight_root"]
        gate = base._read(pre / "preflight.json")
        if (
            (pre / "failure.json").exists()
            or gate["status"] != "PASS"
            or gate["contexts"]
            or gate["weights_loaded"]
            or base._read(
                prior.checked(pre / "registration.json", gate["registration_sha256"])
            )
            != spec
        ):
            raise ValueError("unchanged passing no-model preflight required")
        sha = base._sha(pre / "preflight.json")
        return (
            acquire(root, spec, old, anchors, known, previous, sha, progress)
            if mode == "capture"
            else verify(root, spec, old, anchors, known, previous, sha)
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
        description="Registered paired first-token whole-vocabulary prompt panel."
    )
    p.add_argument("mode", choices=("preflight", "capture", "verify"))
    p.add_argument("--output-root", required=True, type=Path)
    args = p.parse_args()
    result = run(args.mode, args.output_root)
    print(f"{args.mode}: {result['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
