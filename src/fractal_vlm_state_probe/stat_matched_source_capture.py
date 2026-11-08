"""Bounded capture engine. Native execution is separately release-gated by CLI."""

from pathlib import Path
import shutil

import numpy as np

from . import stat_matched_input as matching
from . import stat_matched_source as core
from .representation_panel import expand, pack
from .image_stats import _luminance, _spectral_stats as raw_spectra
from .processor_image_stats import _spectral_stats as native_spectra, _tensor_view
from .stat_matched_source_storage import (
    Recorder,
    append,
    inventory,
    ledger,
    load_array,
    save_array,
    sha,
    write,
)


def candidate_audit(raw, pixels, arrays, errors, reference, features):
    failures = [
        f"oracle:{e['start']}:{e['feature']}" for e in errors if e["max_abs"] > 1e-10
    ]
    groups = {
        start: matching.joint_multisets(a, start, features["grid"])
        for start, a in (("raw_rgb", raw), ("processor_pixel_values", pixels))
    }
    residuals, comparisons, changes, value_changes, pixel_changes = {}, [], {}, {}, {}
    if reference is not None:
        rr, pp, aa, gg = reference
        for start, a, b in (
            ("raw_rgb", raw, rr),
            ("processor_pixel_values", pixels, pp),
        ):
            value_changes[start] = int(np.count_nonzero(a != b))
            aw = (
                a
                if start == "raw_rgb"
                else np.ascontiguousarray(np.moveaxis(a[0], 0, -1)).view(np.uint32)
            )
            bw = (
                b
                if start == "raw_rgb"
                else np.ascontiguousarray(np.moveaxis(b[0], 0, -1)).view(np.uint32)
            )
            changes[start] = int(np.count_nonzero(aw != bw))
            pixel_changes[start] = int(np.count_nonzero(np.any(aw != bw, axis=-1)))
            if groups[start] != gg[start]:
                failures.append("joint_words:" + start)
            for feature in matching.FEATURES:
                key = f"actual__{start}__{feature}"
                delta = arrays[key] - aa[key]
                residuals[start + "__" + feature] = delta
                maximum = float(np.max(np.abs(delta)))
                comparisons.append(
                    {
                        "start": start,
                        "feature": feature,
                        "max_abs": maximum,
                        "l2": float(np.linalg.norm(delta)),
                    }
                )
                if maximum > 1e-10:
                    failures.append("matched_feature:" + start + "/" + feature)
    return {
        "failures": failures,
        "groups": groups,
        "oracle_errors": errors,
        "comparisons": comparisons,
        "changed_words": changes,
        "changed_values": value_changes,
        "changed_pixels": pixel_changes,
    }, residuals


def save_features(root, folder, arrays, reserve):
    return {
        k: save_array(root, folder + "/" + k + ".npy", v, reserve=reserve)
        for k, v in arrays.items()
    }


def counted_backend(recorder, kind, identity, function):
    calls = 0

    def invoke(operation):
        nonlocal calls
        if calls:
            raise ValueError("extra backend call refused before execution")
        calls += 1
        return recorder.call(kind, identity, operation)

    result = function(invoke)
    if calls != 1:
        raise ValueError("backend did not perform exactly one counted call")
    return result


def capture_input(
    root,
    folder,
    identity,
    raw,
    frozen,
    backend,
    recorder,
    reference=None,
    synthetic=None,
):
    reserve = frozen["spec"]["budget"]["reserve_bytes"]
    raw_receipt = save_array(root, folder + "/raw.npy", raw, reserve=reserve)
    pixels, other = counted_backend(
        recorder, "processor", identity, lambda invoke: backend.process(raw, invoke)
    )
    pixels = np.ascontiguousarray(pixels)
    pixel_receipt = save_array(root, folder + "/pixels.npy", pixels, reserve=reserve)
    row = {
        "identity": identity,
        "folder": folder,
        "raw": raw_receipt,
        "pixels": pixel_receipt,
        "non_pixel": other,
    }
    append(root, "processor_returns.jsonl", row)
    if (
        list(pixels.shape) != frozen["spec"]["input_gates"]["expected_native_shape"]
        or pixels.dtype != np.float32
        or not np.isfinite(pixels).all()
        or other != frozen["processor_non_pixel"]
    ):
        raise ValueError("native input shape/dtype/metadata/finiteness differs")
    arrays, errors = matching.feature_arrays(raw, pixels, frozen["features"])
    if synthetic is None:
        audit, residuals = candidate_audit(
            raw, pixels, arrays, errors, reference, frozen["features"]
        )
    else:
        key, name, shift = synthetic
        refs = (
            None
            if reference is None
            else {
                start: (v, reference[2], reference[3][start])
                for start, v in (
                    ("raw_rgb", reference[0]),
                    ("processor_pixel_values", reference[1]),
                )
            }
        )
        audit, residuals = matching.case_audit(
            key,
            name,
            shift,
            raw,
            pixels,
            arrays,
            errors,
            refs,
            frozen["features"],
            frozen["geometry"],
        )
        audit["groups"] = audit["joint_multisets"]
    row.update(
        spectra={
            "raw_rgb": raw_spectra(_luminance(raw.astype(np.float64) / 255)),
            "processor_pixel_values": native_spectra(
                _tensor_view(pixels[0]), patch_size=None
            ),
        },
        features=save_features(root, folder + "/features", arrays, reserve),
        residuals=save_features(root, folder + "/residuals", residuals, reserve),
        audit=audit,
    )
    append(root, "inputs.jsonl", row)
    if audit["failures"]:
        raise ValueError("input gate failed: " + audit["failures"][0])
    return row, (raw, pixels, arrays, audit["groups"])


def capture_model(root, job, case, frozen, backend, recorder, observed=True):
    reserve = frozen["spec"]["budget"]["reserve_bytes"]
    folder = "observations/" + job["identity"]
    values = counted_backend(
        recorder,
        "model",
        job["identity"],
        lambda invoke: backend.observe(root, folder, case, observed, invoke),
    )
    prepared = values["prepared"]
    internal = values["internal_calls"]
    if (
        internal["prepared_inputs"] != 1
        or internal["recorded_generation_steps"] != len(values["suffix"])
        or internal["language_model_call"]["attempted"]
        != internal["language_model_call"]["returned"]
        or internal["language_model_call"]["returned"] < 1
    ):
        raise ValueError("internal preparation/decoder call accounting differs")
    if (
        prepared["pixel_word_sha256"] != case["pixels"]["word_sha256"]
        or prepared["non_pixel"] != frozen["model_non_pixel"]
        or not values["restored"]
        or values["observer_calls"] != int(observed)
    ):
        raise ValueError("observed native path parity differs")
    if len(values["suffix"]) > 3 or not values["suffix"]:
        raise ValueError("natural suffix exceeds the registered generation bound")
    sites = values["sites"]
    if set(sites) != (
        set(frozen["spec"]["observation"]["sites"]) if observed else set()
    ):
        raise ValueError("observed site coverage differs")
    row = {
        **job,
        "prepared": prepared,
        "suffix": values["suffix"],
        "layout": values["layout"],
        "evidence": values["evidence"],
        "observer_calls": values["observer_calls"],
        "restored": values["restored"],
        "internal_calls": values["internal_calls"],
    }
    row["sites"] = {}
    for site, a in sites.items():
        if (
            a.dtype != np.float32
            or list(a.shape) != frozen["spec"]["observation"]["sites"][site]
        ):
            raise ValueError("full native site layout differs")
        words = pack(np.ascontiguousarray(a, dtype="<f4"))
        row["sites"][site] = save_array(
            root, folder + "/" + site + ".npy", words, reserve=reserve
        )
        row["sites"][site]["expanded_word_sha256"] = matching.word_sha(a)
    if set(values["caches"]) != {"1:keys", "12:keys", "23:values"}:
        raise ValueError("selected cache coverage differs")
    row["caches"] = {}
    for target, a in values["caches"].items():
        expected = frozen["cache_shape_max"]
        if (
            a.dtype != np.float32
            or a.ndim != 4
            or list(a.shape[:2]) != expected[:2]
            or a.shape[2] < 1
            or a.shape[2] > expected[2]
            or a.shape[3] != expected[3]
            or not np.isfinite(a).all()
            or np.any(a.view(np.uint32) & np.uint32(65535))
        ):
            raise ValueError("selected cache shape/dtype/finiteness differs")
        row["caches"][target] = save_array(
            root, folder + "/" + target.replace(":", "_") + ".npy", a, reserve=reserve
        )
    row["representation_sha256"] = row["sites"].get("encoder", row["caches"]["1:keys"])[
        "sha256"
    ]
    row["prepared_pixel_sha256"] = prepared["pixel_word_sha256"]
    if not frozen["software_test_only"]:
        from .stat_matched_source_verify import _native_evidence

        _native_evidence(root, row, case, frozen)
    append(root, "models.jsonl", row)
    return row


def seal_family_references(root, frozen, models, inputs):
    reserve = frozen["spec"]["budget"]["reserve_bytes"]
    spec = frozen["spec"]
    pairs = spec["sources"]["pairs"]
    by_model = {(r["pair_id"], r["shift"], r["cell"]): r for r in models}
    by_input = {
        (r["pair_id"], r["shift"], r["cell"]): r for r in inputs if "pair_id" in r
    }
    profiles = [
        (v, f)
        for v in spec["analysis"]["views"]
        for f in (core.FLOORS if v in ("encoder", "projector") else (1e-6,))
    ]
    rows = []
    for vi, (view, floor) in enumerate(profiles):
        for fi, family in enumerate(dict.fromkeys(p["pairing_family"] for p in pairs)):
            units, diagnostics = [], []
            for pair in [
                p
                for p in pairs
                if p["pairing_family"] == family and p["split"] == "reference"
            ]:
                cells = {}
                for cell in core.CELLS:
                    key = pair["pair_id"], 0, cell
                    receipt = (
                        by_model[key]["sites"][view]
                        if view in ("encoder", "projector")
                        else by_input[key]["features"][
                            "actual__" + view.replace("/", "__")
                        ]
                    )
                    a = load_array(root, receipt)
                    cells[cell] = expand(a) if view in ("encoder", "projector") else a
                _, unit, info = core.contrast_direction(
                    cells,
                    learned=view in ("encoder", "projector"),
                    relative_floor=floor,
                )
                units.append(unit)
                diagnostics.append(info)
            unit, reason = core.family_reference(units)
            rows.append(
                {
                    "view": view,
                    "relative_floor": floor,
                    "family": family,
                    "reason": reason,
                    "interactions": diagnostics,
                    "unit": None
                    if unit is None
                    else save_array(
                        root, f"references/v{vi}_f{fi}.npy", unit, reserve=reserve
                    ),
                }
            )
    write(root / "family_references.json", rows)


def acquire(root, frozen, backend):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    recorder = Recorder(root, frozen["spec"]["budget"]["reserve_bytes"])
    reserve = frozen["spec"]["budget"]["reserve_bytes"]
    stage = "qualification"
    try:
        write(root / "frozen.json", frozen)
        if shutil.disk_usage(root).free < frozen["spec"]["budget"]["required_launch_free_bytes"]:
            raise ValueError("registered launch disk threshold not met immediately before capture")
        backend.check_frozen(frozen)
        maps = {s: backend.map(s) for s in range(4)}
        for shift, a in maps.items():
            row = save_array(root, f"maps/s{shift}.npy", a, reserve=reserve)
            if row["word_sha256"] != frozen["map_word_sha256"][str(shift)]:
                raise ValueError("qualified map digest differs")
        stage = "processor_calibration"
        for i, expected in enumerate(frozen["historical_processor"]):
            raw = backend.historical_raw(i)
            row = {
                "identity": f"historical_{i}",
                "raw": save_array(
                    root, f"historical/{i}/raw.npy", raw, reserve=reserve
                ),
            }
            pixels, other = counted_backend(
                recorder,
                "processor",
                row["identity"],
                lambda invoke, raw=raw: backend.process(raw, invoke),
            )
            row.update(
                pixels=save_array(
                    root,
                    f"historical/{i}/pixels.npy",
                    np.ascontiguousarray(pixels),
                    reserve=reserve,
                ),
                non_pixel=other,
            )
            append(root, "historical.jsonl", row)
            if (
                row["pixels"]["word_sha256"] != expected["pixel_values_sha256"]
                or other != expected["processor_non_pixel_payload"]
                or pixels.dtype != np.float32
                or list(pixels.shape)
                != frozen["spec"]["input_gates"]["expected_native_shape"]
                or not np.isfinite(pixels).all()
            ):
                raise ValueError("historical native processor parity differs")
        synth_refs, synthetic_cases = {}, {}
        for key, name, shift in matching.cases(frozen["geometry"]):
            raw = backend.synthetic_raw(key)
            row, ref = capture_input(
                root,
                "synthetic/" + key,
                key,
                raw,
                frozen,
                backend,
                recorder,
                synth_refs.get(name) if shift or key == "sentinel" else None,
                (key, name, shift),
            )
            row.update(fixture=name, shift=shift)
            append(root, "synthetic.jsonl", row)
            synthetic_cases[key] = row
            if shift == 0 and key != "sentinel":
                synth_refs[name] = ref
        source_hashes = list(frozen["historical_source_hashes"])
        source_count = 0

        def prepare(split):
            nonlocal source_count
            for pair in [
                p for p in frozen["spec"]["sources"]["pairs"] if p["split"] == split
            ]:
                donors = []
                for role in ("source_a", "source_b"):
                    identity = pair["pair_id"] + "/" + role
                    raw, provenance = recorder.call(
                        "source", identity, lambda: backend.render(root, pair, role)
                    )
                    receipt = save_array(
                        root, "sources/" + identity + "/raw.npy", raw, reserve=reserve
                    )
                    row = {
                        "identity": identity,
                        "pair_id": pair["pair_id"],
                        "role": role,
                        "split": split,
                        "raw": receipt,
                        "provenance": provenance,
                        "decoded_rgb_sha256": receipt["word_sha256"],
                    }
                    append(root, "sources.jsonl", row)
                    source_count += 1
                    if core.decoded_hash_audit([row], source_hashes)["collisions"]:
                        raise ValueError("decoded source collision; no replacement")
                    source_hashes.append(row)
                    donors.append(raw)
                quartet = core.identity_quartet(
                    *donors, frozen["spec"], frozen["geometry"]
                )
                refs = {}
                for shift in range(4):
                    images = {
                        cell: core.rearrange(raw, maps[shift])
                        for cell, raw in quartet.items()
                    }
                    palette = core.palette_audit(images)
                    append(
                        root,
                        "palettes.jsonl",
                        {"pair_id": pair["pair_id"], "shift": shift, **palette},
                    )
                    if not all(palette["matches"]):
                        raise ValueError("raw palette marginal gate differs")
                    for cell, raw in images.items():
                        identity = f"{pair['pair_id']}/s{shift}/{cell}"
                        row, ref = capture_input(
                            root,
                            "inputs/" + identity,
                            identity,
                            raw,
                            frozen,
                            backend,
                            recorder,
                            refs.get(cell),
                        )
                        row.update(
                            pair_id=pair["pair_id"], split=split, shift=shift, cell=cell
                        )
                        append(root, "candidates.jsonl", row)
                        backend.manifest(root, row)
                        if shift == 0:
                            refs[cell] = ref
            write(
                root / (split + "_input_seal.json"),
                {
                    "source_count": source_count,
                    "processor_calls": recorder.counts["processor"],
                    "inputs_sha256": sha(root / "candidates.jsonl"),
                },
            )

        stage = "reference_inputs"
        prepare("reference")
        stage = "historical_model_calibration"

        def calibrated_call(function):
            i = recorder.counts["model"]["attempted"]
            if i >= 384:
                raise ValueError("historical model context budget exceeded")
            return recorder.call("model", f"calibration_{i}", function)

        backend.calibrate(root, calibrated_call)
        if recorder.counts["model"] != {"attempted": 384, "returned": 384}:
            raise ValueError("historical model calibration coverage differs")
        stage = "shams"
        for name in ("zero", "constant"):
            first = None
            for shift in range(4):
                on = None
                for observed, label in (
                    (False, "off_before"),
                    (True, "on"),
                    (False, "off_after"),
                ):
                    row = capture_model(
                        root,
                        {
                            "identity": f"sham/{name}_s{shift}/{label}",
                            "phase": "sham",
                            "fixture": name,
                            "shift": shift,
                            "stage": label,
                        },
                        synthetic_cases[f"{name}_s{shift}"],
                        frozen,
                        backend,
                        recorder,
                        observed,
                    )
                    signature = {k: row[k] for k in ("prepared", "suffix", "layout")}
                    signature["caches"] = {
                        k: r["word_sha256"] for k, r in row["caches"].items()
                    }
                    if first is not None and signature != first:
                        raise ValueError(
                            "sham OFF/ON/OFF or cross-shift parity differs"
                        )
                    first = signature
                    if observed:
                        on = {k: r["word_sha256"] for k, r in row["sites"].items()}
                if shift == 0:
                    on0 = on
                elif on != on0:
                    raise ValueError("sham cross-shift representation words differ")
        jobs = core.ordered_jobs(frozen["spec"])
        for split in ("reference", "test"):
            if split == "test":
                stage = "test_inputs"
                prepare(split)
            stage = split + "_model"
            by_case = {r["identity"]: r for r in ledger(root, "candidates.jsonl")}
            rows = []
            for job in [j for j in jobs if j["split"] == split]:
                identity = f"{job['pair_id']}/s{job['shift']}/{job['cell']}"
                if recorder.counts["model"]["attempted"] + 1 != job["ordinal"]:
                    raise ValueError("source context ordinal differs")
                rows.append(
                    capture_model(
                        root,
                        {**job, "identity": identity, "phase": split},
                        by_case[identity],
                        frozen,
                        backend,
                        recorder,
                    )
                )
            if split == "reference":
                from .stat_matched_source_verify import (
                    verify_prefix,
                    verify_reference_vectors,
                )

                proof = verify_prefix(root, frozen)
                seal_family_references(
                    root, frozen, rows, ledger(root, "candidates.jsonl")
                )
                verify_reference_vectors(root, frozen)
                seal = core.seal_reference_metadata(rows, frozen["spec"])
                seal.update(
                    artifact_bytes_independently_verified=True,
                    proof=proof,
                    family_references_sha256=sha(root / "family_references.json"),
                    operation_sequence=recorder.sequence,
                    counts=recorder.counts,
                )
                write(root / "reference_seal.json", seal)
                recorder.event(
                    "reference_seal", sha256=sha(root / "reference_seal.json")
                )
        stage = "final_checks"
        backend.finish(root)
        backend.check_frozen(frozen)
        expected = {"processor": 665, "model": 920, "source": 64}
        if any(
            recorder.counts[k] != {"attempted": v, "returned": v}
            for k, v in expected.items()
        ):
            raise ValueError("complete acquisition denominator differs")
        result = {
            "status": "COMPLETE_UNVERIFIED",
            "counts": recorder.counts,
            "last_operation_digest": recorder.chain,
            "receipts": inventory(root),
            "primary_results_withheld": True,
        }
        write(root / "capture.json", result)
        return result
    except BaseException as error:
        write(
            root / "failure.json",
            {
                "status": "FAILED_RETAINED",
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "cause": repr(error.__context__)
                if error.__context__ is not None
                else None,
                "counts": recorder.counts,
                "last_operation_digest": recorder.chain,
                "receipts": inventory(root),
                "primary_results_withheld": True,
                "retry_or_resume": False,
            },
        )
        raise
