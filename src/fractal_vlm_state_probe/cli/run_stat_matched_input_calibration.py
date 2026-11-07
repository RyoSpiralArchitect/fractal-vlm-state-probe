"""Frozen guarded-tile calibration, including retained-failure file audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from fractal_vlm_state_probe import stat_matched_input as m
from fractal_vlm_state_probe import stat_matched_input_oracle as oracle
from fractal_vlm_state_probe.cli import run_nonlearned_processing_calibration as old
from fractal_vlm_state_probe.stimulus import sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
DESIGN = "configs/stat_matched_input_calibration_v1.json"
RELEASE = "configs/stat_matched_input_release_v1.json"
CODE = (
    RELEASE,
    "src/fractal_vlm_state_probe/stat_matched_input.py",
    "src/fractal_vlm_state_probe/stat_matched_input_oracle.py",
    "src/fractal_vlm_state_probe/cli/run_stat_matched_input_calibration.py",
    "scripts/run_stat_matched_input_calibration.py",
    "tests/test_stat_matched_input.py",
    "docs/research_notes/0086_stat_matched_input_calibration_release.md",
)
LEDGERS = (
    "coordinate_maps.jsonl",
    "raw_inputs.jsonl",
    "historical_processor.jsonl",
    "synthetic_processor.jsonl",
    "cases.jsonl",
)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return (
        [json.loads(x) for x in path.read_bytes().splitlines()] if path.exists() else []
    )


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def check_space(root, cfg, launch=False):
    needed = cfg["required_launch_free_bytes"] if launch else cfg["reserve_bytes"]
    if shutil.disk_usage(root).free < needed:
        raise ValueError("insufficient registered disk reserve")


def inventory(root):
    return {
        str(p.relative_to(root)): sha256_file(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in ("capture.json", "failure.json")
    }


def qualify():
    release = read(REPO / RELEASE)
    if (
        not release["calibration_only"]
        or not release["execution_authorized"]
        or release["model_forwards"] != 0
        or release["retry_or_resume"]
    ):
        raise ValueError("calibration-only release contract differs")
    cfg = read(REPO / DESIGN)
    if sha256_file(REPO / DESIGN) != release["design_config_sha256"]:
        raise ValueError("registered recipe changed")
    for name in cfg["registration_files"]:
        if hashlib.sha256(
            git("show", f"{release['design_commit']}:{name}")
        ).hexdigest() != sha256_file(REPO / name):
            raise ValueError(f"registration changed: {name}")
    for name, digest in cfg["pinned_artifacts"].items():
        if sha256_file(REPO / name) != digest:
            raise ValueError(f"pinned artifact changed: {name}")
    previous = read(REPO / cfg["preserve_inherited_frozen_hashes_from"])
    inherited = previous["code_sha256"]
    if len(inherited) != cfg["inherited_frozen_files"]:
        raise ValueError("inherited file denominator differs")
    code = {
        **inherited,
        **{n: sha256_file(REPO / n) for n in (*cfg["registration_files"], *CODE)},
    }
    commit = (
        git("log", "-1", "--format=%H", "--", *cfg["registration_files"], *CODE)
        .decode()
        .strip()
    )
    git("merge-base", "--is-ancestor", release["design_commit"], commit)
    for name, digest in code.items():
        if (
            sha256_file(REPO / name) != digest
            or hashlib.sha256(git("show", f"{commit}:{name}")).hexdigest() != digest
        ):
            raise ValueError(f"freeze implementation before acquisition: {name}")
    legacy = old.specification()
    # The legacy qualifier labels current HEAD, although its selected hashes are unchanged.
    legacy.pop("registered_design_commit")
    historical = read(
        REPO
        / "examples/research_notes/0063_nonlearned_processing_calibration/summary.json"
    )
    if (
        legacy["processor_jobs"] != historical["frozen_specification"]["processor_jobs"]
        or legacy["runtime"] != historical["frozen_specification"]["runtime"]
        or legacy["processor_provenance"]
        != historical["frozen_specification"]["processor_provenance"]
    ):
        raise ValueError("historical processor qualification changed")
    return {
        "registered_commit": commit,
        "config": cfg,
        "release": release,
        "code_sha256": code,
        "legacy": legacy,
    }


def assert_frozen(spec):
    for name, digest in spec["code_sha256"].items():
        if sha256_file(REPO / name) != digest:
            raise ValueError(f"frozen code changed: {name}")
    old.assert_hashes(spec["legacy"])


def progress(root, state, **updates):
    state.update(updates)
    write_json(root / "progress.json", state)


def prepare_raw(root, spec, state):
    cfg, maps = spec["config"], {}
    for shift in cfg["construction"]["arrangement_shifts"]:
        progress(root, state, phase="coordinates", current_shift=shift)
        a = m.coordinate_map(cfg, shift)
        p = root / f"map_{shift}.npy"
        np.save(p, a, allow_pickle=False)
        row = m.map_audit(a, oracle.coordinate_map(cfg, shift), cfg, shift)
        append(
            root / LEDGERS[0], {**row, "file": p.name, "file_sha256": sha256_file(p)}
        )
        maps[shift] = a
    for key, name, shift in m.cases(cfg):
        check_space(root, cfg["future_calibration"])
        progress(root, state, phase="raw_construction", current_case=key)
        path = root / "cases" / key
        path.mkdir(parents=True)
        raw = m.fixture(cfg, name, shift, maps[shift], key == "sentinel")
        np.save(path / "raw.npy", raw, allow_pickle=False)
        Image.fromarray(raw).save(path / "raw.png")
        expected = oracle.fixture(cfg, name, shift, key == "sentinel")
        if raw.tobytes() != expected.tobytes() or not np.array_equal(
            np.asarray(Image.open(path / "raw.png")), raw
        ):
            raise ValueError(f"raw constructor/oracle/PNG mismatch: {key}")
        append(
            root / LEDGERS[1],
            {
                "key": key,
                "fixture": name,
                "shift": shift,
                "word_sha256": m.word_sha(raw),
                "npy_sha256": sha256_file(path / "raw.npy"),
                "png_sha256": sha256_file(path / "raw.png"),
            },
        )
    write_json(
        root / "raw_seal.json",
        {
            "maps": len(maps),
            "raw_images": len(m.cases(cfg)),
            "processor_calls_started": 0,
            "ledger_sha256": {n: sha256_file(root / n) for n in LEDGERS[:2]},
        },
    )


def load_processor():
    return old.qualified_processor(old.read(old.BASELINE))[0]


def native_payload(processor, rgb):
    payload = processor(
        images=Image.fromarray(rgb), text=["<image>"], return_tensors="np"
    )
    # Canonical C-order storage preserves native words and replay reduction order.
    return np.ascontiguousarray(payload["pixel_values"]), {
        k: np.asarray(v).tolist() for k, v in payload.items() if k != "pixel_values"
    }


def acquire(root, spec, state):
    cfg, cal = spec["config"], spec["config"]["future_calibration"]
    prepare_raw(root, spec, state)
    processor = load_processor()
    for i, job in enumerate(spec["legacy"]["processor_jobs"]):
        check_space(root, cal)
        progress(
            root,
            state,
            phase="historical_processor",
            current_job=i + 1,
            attempted_processor_calls=state["attempted_processor_calls"] + 1,
        )
        rgb, _ = old._checked_artifact(job["artifact"])
        pixels, other = native_payload(processor, rgb)
        state["completed_processor_calls"] += 1
        exact = (
            pixels.dtype == np.float32
            and list(pixels.shape) == cal["processor_shape_nchw"]
            and np.isfinite(pixels).all()
            and m.word_sha(pixels) == job["pixel_values_sha256"]
            and other == job["processor_non_pixel_payload"]
        )
        row = {
            "ordinal": i + 1,
            "pair_id": job["pair_id"],
            "cell": job["cell"],
            "pixel_word_sha256": m.word_sha(pixels),
            "non_pixel_payload": other,
            "shape": list(pixels.shape),
            "dtype": str(pixels.dtype),
            "exact": bool(exact),
        }
        append(root / LEDGERS[2], row)
        progress(root, state)
        if not exact:
            enough = (
                shutil.disk_usage(root).free - pixels.nbytes >= cal["reserve_bytes"]
            )
            state["unexpected_historical_array_saved"] = enough
            if enough:
                np.save(
                    root / "unexpected_historical_pixels.npy",
                    pixels,
                    allow_pickle=False,
                )
            raise ValueError(f"historical processor parity failed: {i + 1}")
        if (i + 1) % 16 == 0:
            print(
                f"historical processor {i + 1}/{cal['historical_processor_checks']}",
                flush=True,
            )
    refs = {}
    feature_config = read(REPO / cfg["inherited_feature_config"])
    for key, name, shift in m.cases(cfg):
        check_space(root, cal)
        progress(
            root,
            state,
            phase="synthetic_processor",
            current_case=key,
            attempted_processor_calls=state["attempted_processor_calls"] + 1,
        )
        p = root / "cases" / key
        raw = np.load(p / "raw.npy", allow_pickle=False)
        pixels, other = native_payload(processor, raw)
        state["completed_processor_calls"] += 1
        if shutil.disk_usage(root).free - pixels.nbytes < cal["reserve_bytes"]:
            raise ValueError("native payload would breach reserved disk space")
        np.save(p / "pixels.npy", pixels, allow_pickle=False)
        append(
            root / LEDGERS[3],
            {
                "key": key,
                "pixel_word_sha256": m.word_sha(pixels),
                "file_sha256": sha256_file(p / "pixels.npy"),
                "non_pixel_payload": other,
                "shape": list(pixels.shape),
                "dtype": str(pixels.dtype),
            },
        )
        progress(root, state)
        m.checked_pixels(pixels, cfg)
        if other != cal["synthetic_non_pixel_expected"]:
            raise ValueError(f"synthetic non-pixel payload differs: {key}")
        arrays, errors = m.feature_arrays(raw, pixels, feature_config)
        np.savez(p / "features.npz", **arrays)
        reference = refs.get(name) if shift or key == "sentinel" else None
        row, residuals = m.case_audit(
            key,
            name,
            shift,
            raw,
            pixels,
            arrays,
            errors,
            reference,
            feature_config,
            cfg,
        )
        np.savez(p / "residuals.npz", **residuals)
        row.update(
            features_sha256=sha256_file(p / "features.npz"),
            residuals_sha256=sha256_file(p / "residuals.npz"),
        )
        append(root / LEDGERS[4], row)
        if reference is None:
            refs[name] = {
                start: (values, arrays, row["joint_multisets"][start])
                for start, values in zip(m.STARTS, (raw, pixels), strict=True)
            }
        print(f"synthetic {key}: {'FAIL' if row['failures'] else 'PASS'}", flush=True)
        if row["failures"]:
            raise ValueError(f"calibration gate failed at {key}: {row['failures'][0]}")
    counts = audit_counts(root)
    require_complete(counts, cfg)
    assert_frozen(spec)
    seal = {
        "status": "PASS",
        "registered_commit": spec["registered_commit"],
        "counts": counts,
        "progress": state,
        "receipts": inventory(root),
    }
    write_json(root / "capture.json", seal)
    return seal


def audit_counts(root):
    maps, raw, hist, synth, audits = [rows(root / n) for n in LEDGERS]
    matched = [r for r in audits if r["shift"] and r["key"] != "sentinel"]
    return {
        "coordinate_maps": len(maps),
        "raw_images": len(raw),
        "historical_processor_checks": len(hist),
        "historical_exact": sum(r["exact"] for r in hist),
        "synthetic_processor_calls": len(synth),
        "processor_return_receipts": len(hist) + len(synth),
        "audited_cases": len(audits),
        "feature_oracle_vector_pairs": sum(len(r["oracle_checks"]) for r in audits),
        "matched_pairs": len(matched),
        "matched_feature_vector_comparisons": sum(
            len(r["comparisons"]) for r in matched
        ),
        "joint_multiset_comparisons": sum(
            len(w["matched_groups"]) for r in matched for w in r["word_checks"].values()
        ),
        "nonconstant_changed_array_checks": sum(
            not w["word_equal"]
            for r in matched
            if r["fixture"] not in ("zero", "constant")
            for w in r["word_checks"].values()
        ),
        "constant_or_zero_word_exact_shams": sum(
            w["word_equal"]
            for r in matched
            if r["fixture"] in ("zero", "constant")
            for w in r["word_checks"].values()
        ),
        "unexpected_gate_failures": sum(len(r["failures"]) for r in audits),
        "sentinel_audited": any(r["key"] == "sentinel" for r in audits),
    }


def require_complete(counts, cfg):
    cal = cfg["future_calibration"]
    expected = {
        k: cal[k]
        for k in (
            "historical_processor_checks",
            "synthetic_processor_calls",
            "feature_oracle_vector_pairs",
            "matched_pairs",
            "matched_feature_vector_comparisons",
            "joint_multiset_comparisons",
            "nonconstant_changed_array_checks",
            "constant_or_zero_word_exact_shams",
        )
    }
    expected.update(
        coordinate_maps=4,
        raw_images=25,
        historical_exact=cal["historical_processor_checks"],
        processor_return_receipts=cal["total_processor_calls"],
        audited_cases=25,
        unexpected_gate_failures=0,
        sentinel_audited=True,
    )
    if counts != expected:
        raise ValueError("complete fixed calibration denominator/gates not satisfied")


def checked_npz(path, expected):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(expected) or any(
            z[k].dtype != v.dtype
            or z[k].shape != v.shape
            or z[k].tobytes() != v.tobytes()
            for k, v in expected.items()
        ):
            raise ValueError(f"saved numerical evidence changed: {path.name}")


def verify(root, spec, gate_sha):
    cfg, cal = spec["config"], spec["config"]["future_calibration"]
    cap = REPO / spec["release"]["capture_root"]
    failed = (cap / "failure.json").exists()
    seal_name = "failure.json" if failed else "capture.json"
    seal = read(cap / seal_name)
    if read(cap / "registration.json") != spec or inventory(cap) != seal["receipts"]:
        raise ValueError("capture inventory or registration drift")
    if not failed and seal["status"] != "PASS":
        raise ValueError("capture not accepted")
    maps, raw_rows, history, synthetic, audits = [rows(cap / n) for n in LEDGERS]
    if [r["shift"] for r in maps] != cfg["construction"]["arrangement_shifts"][
        : len(maps)
    ]:
        raise ValueError("coordinate map order differs")
    for row in maps:
        a = np.load(cap / row["file"], allow_pickle=False)
        audit = m.map_audit(
            a, oracle.coordinate_map(cfg, row["shift"]), cfg, row["shift"]
        )
        if (
            any(row[k] != v for k, v in audit.items())
            or sha256_file(cap / row["file"]) != row["file_sha256"]
        ):
            raise ValueError("coordinate receipt changed")
    ordered = m.cases(cfg)
    if [r["key"] for r in raw_rows] != [k for k, _, _ in ordered[: len(raw_rows)]]:
        raise ValueError("raw input order differs")
    for row, (key, name, shift) in zip(raw_rows, ordered, strict=False):
        if (row["fixture"], row["shift"]) != (name, shift):
            raise ValueError("raw fixture metadata differs")
        p = cap / "cases" / key
        a = np.load(p / "raw.npy", allow_pickle=False)
        b = oracle.fixture(cfg, name, shift, key == "sentinel")
        if (
            a.dtype != np.uint8
            or list(a.shape) != cfg["construction"]["raw_shape_hwc"]
            or a.tobytes() != b.tobytes()
            or m.word_sha(a) != row["word_sha256"]
            or sha256_file(p / "raw.npy") != row["npy_sha256"]
            or sha256_file(p / "raw.png") != row["png_sha256"]
            or not np.array_equal(np.asarray(Image.open(p / "raw.png")), a)
        ):
            raise ValueError("raw image/oracle/receipt differs")
    if (cap / "raw_seal.json").exists():
        rs = read(cap / "raw_seal.json")
        if rs != {
            "maps": len(maps),
            "raw_images": len(raw_rows),
            "processor_calls_started": 0,
            "ledger_sha256": {n: sha256_file(cap / n) for n in LEDGERS[:2]},
        }:
            raise ValueError("raw seal changed")
    elif not failed:
        raise ValueError("missing raw seal")
    if len(history) > len(spec["legacy"]["processor_jobs"]):
        raise ValueError("extra historical calls")
    for i, row in enumerate(history):
        j = spec["legacy"]["processor_jobs"][i]
        exact = (
            row["pixel_word_sha256"] == j["pixel_values_sha256"]
            and row["non_pixel_payload"] == j["processor_non_pixel_payload"]
            and row["shape"] == cal["processor_shape_nchw"]
            and row["dtype"] == "float32"
        )
        if (
            row["ordinal"] != i + 1
            or (row["pair_id"], row["cell"]) != (j["pair_id"], j["cell"])
            or row["exact"] != exact
        ):
            raise ValueError("historical receipt differs")
    unexpected = cap / "unexpected_historical_pixels.npy"
    if unexpected.exists():
        a = np.load(unexpected, allow_pickle=False)
        if (
            not history
            or history[-1]["exact"]
            or m.word_sha(a) != history[-1]["pixel_word_sha256"]
        ):
            raise ValueError("unexpected historical native array differs from receipt")
    if [r["key"] for r in synthetic] != [k for k, _, _ in ordered[: len(synthetic)]]:
        raise ValueError("synthetic call order differs")
    for row in synthetic:
        p = cap / "cases" / row["key"] / "pixels.npy"
        a = np.load(p, allow_pickle=False)
        if (
            sha256_file(p) != row["file_sha256"]
            or m.word_sha(a) != row["pixel_word_sha256"]
            or list(a.shape) != row["shape"]
            or str(a.dtype) != row["dtype"]
        ):
            raise ValueError("native array receipt differs")
        if not failed:
            m.checked_pixels(a, cfg)
            if row["non_pixel_payload"] != cal["synthetic_non_pixel_expected"]:
                raise ValueError("synthetic metadata differs")
    if [r["key"] for r in audits] != [k for k, _, _ in ordered[: len(audits)]]:
        raise ValueError("case audit order differs")
    refs, feature_archive, residual_archive = {}, {}, {}
    feature_config = read(REPO / cfg["inherited_feature_config"])
    for row, (key, name, shift) in zip(audits, ordered, strict=False):
        p = cap / "cases" / key
        raw, pix = (
            np.load(p / n, allow_pickle=False) for n in ("raw.npy", "pixels.npy")
        )
        m.checked_pixels(pix, cfg)
        arrays, errors = m.feature_arrays(raw, pix, feature_config)
        reference = refs.get(name) if shift or key == "sentinel" else None
        expected, residuals = m.case_audit(
            key, name, shift, raw, pix, arrays, errors, reference, feature_config, cfg
        )
        checked_npz(p / "features.npz", arrays)
        checked_npz(p / "residuals.npz", residuals)
        expected.update(
            features_sha256=sha256_file(p / "features.npz"),
            residuals_sha256=sha256_file(p / "residuals.npz"),
        )
        if expected != row:
            raise ValueError("complete case audit does not reproduce")
        if reference is None:
            refs[name] = {
                start: (values, arrays, row["joint_multisets"][start])
                for start, values in zip(m.STARTS, (raw, pix), strict=True)
            }
        feature_archive.update({f"{key}__{k}": v for k, v in arrays.items()})
        residual_archive.update({f"{key}__{k}": v for k, v in residuals.items()})
    counts = audit_counts(cap)
    if not failed:
        require_complete(counts, cfg)
        if counts != seal["counts"]:
            raise ValueError("capture count seal differs")
    np.savez_compressed(root / "features.npz", **feature_archive)
    np.savez_compressed(root / "residuals.npz", **residual_archive)
    for name in LEDGERS:
        if (cap / name).exists():
            shutil.copyfile(cap / name, root / name)
        else:
            (root / name).touch(exist_ok=False)
    result = {
        "schema_version": 1,
        "analysis_kind": cfg["study_id"],
        "status": "FAILED_CALIBRATION_AUDITED" if failed else "PASS",
        "calibration_passed": not failed,
        "registered_commit": spec["registered_commit"],
        "design_config_sha256": sha256_file(REPO / DESIGN),
        "code_sha256": spec["code_sha256"],
        "capture_or_failure_sha256": sha256_file(cap / seal_name),
        "preflight_sha256": gate_sha,
        "actual_counts": counts,
        "planned_processor_calls": cal["total_processor_calls"],
        "attempted_processor_calls": seal["progress"]["attempted_processor_calls"],
        "completed_processor_calls": seal["progress"]["completed_processor_calls"],
        "acquisition_progress": seal["progress"],
        "first_failure": {k: seal[k] for k in ("type", "message")} if failed else None,
        "model_forwards": 0,
        "verification_processor_calls": 0,
        "confirmation_cohort": "not_generated_not_sealed",
        "measurement_panel_released": False,
        "archives": {
            n: {"sha256": sha256_file(root / n), "arrays": len(a)}
            for n, a in (
                ("features.npz", feature_archive),
                ("residuals.npz", residual_archive),
            )
        },
        "ledgers": {
            n: {"sha256": sha256_file(root / n), "rows": len(rows(root / n))}
            for n in LEDGERS
        },
    }
    assert_frozen(spec)
    write_json(root / "summary.json", result)
    return result


def run(mode, root):
    if any(
        os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        raise ValueError("offline flags required")
    release = read(REPO / RELEASE)
    root = root.resolve()
    if mode not in ("preflight", "calibrate", "verify"):
        raise ValueError("invalid mode")
    key = "preflight_root" if mode == "preflight" else "capture_root"
    if mode != "verify" and root != (REPO / release[key]).resolve():
        raise ValueError("canonical fresh acquisition root required")
    root.mkdir(parents=True, exist_ok=False)
    state = {
        "attempted_processor_calls": 0,
        "completed_processor_calls": 0,
        "phase": mode,
    }
    try:
        cfg = read(REPO / DESIGN)
        check_space(root, cfg["future_calibration"], launch=mode != "verify")
        spec = qualify()
        write_json(root / "registration.json", spec)
        if mode == "preflight":
            result = {
                "status": "PASS",
                "processor_instantiated": False,
                "model_weights_loaded": False,
                "new_images": 0,
                "new_maps": 0,
                "processor_calls": 0,
                "registration_sha256": sha256_file(root / "registration.json"),
                "free_bytes": shutil.disk_usage(root).free,
            }
            write_json(root / "preflight.json", result)
            return result
        pre = REPO / release["preflight_root"]
        gate = read(pre / "preflight.json")
        if (
            (pre / "failure.json").exists()
            or gate["status"] != "PASS"
            or gate["processor_instantiated"]
            or gate["model_weights_loaded"]
            or gate["processor_calls"]
            or sha256_file(pre / "registration.json") != gate["registration_sha256"]
            or read(pre / "registration.json") != spec
        ):
            raise ValueError("unchanged passing preflight required")
        return (
            acquire(root, spec, state)
            if mode == "calibrate"
            else verify(root, spec, sha256_file(pre / "preflight.json"))
        )
    except Exception as error:
        write_json(
            root / "failure.json",
            {
                "status": "FAILED",
                "mode": mode,
                "type": type(error).__name__,
                "message": str(error),
                "progress": state,
                "retry": False,
                "receipts": inventory(root),
            },
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preflight", "calibrate", "verify"))
    p.add_argument("--output-root", required=True, type=Path)
    args = p.parse_args()
    result = run(args.mode, args.output_root)
    print(f"{args.mode}: {result['status']} -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
