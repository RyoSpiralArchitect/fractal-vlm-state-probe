"""Independent reconstruction and complete saved-data checks; no inference."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from . import stat_matched_input as matching
from . import stat_matched_input_oracle as map_oracle
from . import stat_matched_source as core
from . import stat_matched_source_oracle as source_oracle
from .nonlearned_processing_calibration import oracle_features
from .nonlearned_processing import extract_features
from .image_stats import _luminance, _spectral_stats as raw_spectra
from .processor_image_stats import _spectral_stats as native_spectra, _tensor_view
from .representation_panel import oracle_expand
from .stat_matched_source_storage import (
    inside,
    inventory,
    ledger,
    load_array,
    read,
    sha,
    write,
)


def validate_preflight_binding(frozen):
    pre = frozen["outer_preflight"]
    receipt_bytes = pre["receipt_text"].encode("utf-8")
    receipt = json.loads(receipt_bytes)
    prior = {k: v for k, v in frozen.items() if k != "outer_preflight"}
    if (
        hashlib.sha256(receipt_bytes).hexdigest() != pre["receipt_sha256"]
        or receipt["status"] != "PASS_FOR_APPROVED_RELEASE"
        or receipt["model_weights_loaded"]
        or receipt["new_processor_calls"] != 0
        or receipt["new_model_contexts"] != 0
        or receipt["available_bytes"]
        < frozen["spec"]["budget"]["required_launch_free_bytes"]
        or receipt["frozen_sha256"]
        != hashlib.sha256(pre["frozen_text"].encode("utf-8")).hexdigest()
        or json.loads(pre["frozen_text"]) != prior
    ):
        raise ValueError("saved zero-call approved preflight binding differs")


def anchored_summary(repo, name, digest):
    path = repo / name
    if sha(path) != digest:
        raise ValueError("trusted historical summary hash differs")
    return read(path)


def verify_source_inventory(repo, capture_root, frozen):
    """Recount the pre-acquisition scope, excluding only this acquisition tree."""
    if capture_root is None:
        raise ValueError("capture root required for historical inventory coverage")
    capture_root = capture_root.resolve()
    configs = sorted((repo / "configs").rglob("*.json"))
    manifests = {
        p.resolve()
        for p in (repo / "runs").rglob("manifest.json")
        if "sources" in p.relative_to(repo / "runs").parts[:-1]
        and not p.resolve().is_relative_to(capture_root)
    }
    metadata, gaps, hashes, decoded = [], [], {}, []

    def inspect(value, origin):
        if isinstance(value, list):
            for item in value:
                inspect(item, origin)
        elif isinstance(value, dict):
            pairs = value.get("pairs")
            if isinstance(pairs, list):
                defaults = value.get("source_defaults", value.get("defaults", {}))
                for pair in pairs:
                    if not isinstance(pair, dict):
                        continue
                    for role in ("source_a", "source_b"):
                        source = pair.get(role)
                        if not isinstance(source, dict) or "kind" not in source:
                            continue
                        if (
                            source["kind"] in core.RNG_RULES
                            and "seed" in source
                            and {**defaults, **source}.get("total_frames") == 1
                        ):
                            metadata.append(
                                {
                                    "identity": origin
                                    + "/"
                                    + pair["pair_id"]
                                    + "/"
                                    + role,
                                    "source": source,
                                    "defaults": defaults,
                                }
                            )
                        else:
                            gaps.append(
                                origin
                                + ":unsupported-or-non-single-frame-spec:"
                                + source["kind"]
                            )
            for item in value.values():
                inspect(item, origin)
        elif isinstance(value, str) and value.endswith("manifest.json"):
            path = Path(value).expanduser()
            path = path if path.is_absolute() else repo / path
            if path.is_file():
                manifests.add(path.resolve())
            else:
                gaps.append(str(path))

    for path in configs:
        if (
            path.resolve()
            == (repo / "configs/stat_matched_source_panel_v1.json").resolve()
        ):
            continue
        inspect(read(path), str(path.relative_to(repo)))
        hashes[str(path)] = sha(path)
    for path in sorted(manifests):
        if path.is_relative_to(capture_root):
            raise ValueError("historical configuration references current capture")
        hashes[str(path)] = sha(path)
        for frame in read(path).get("frames", []):
            image_path = (path.parent / frame["path"]).resolve()
            if not image_path.is_file():
                gaps.append(str(image_path))
                continue
            if sha(image_path) != frame["sha256"]:
                raise ValueError("historical inventory image hash differs")
            hashes[str(image_path)] = frame["sha256"]
            with Image.open(image_path) as image:
                decoded.append(
                    {
                        "identity": str(image_path),
                        "decoded_rgb_sha256": matching.word_sha(
                            np.asarray(image.convert("RGB"))
                        ),
                    }
                )
    expected = {
        "scope": "all_configuration_json_and_their_available_manifest_references_plus_all_runs_sources_manifests",
        "metadata": metadata,
        "decoded": decoded,
        "file_sha256": hashes,
        "gaps": sorted(gaps),
        "complete_available_inventory_scanned": True,
        "untouched_claim_released": False,
    }
    recorded = {
        **frozen["source_inventory"],
        "gaps": sorted(frozen["source_inventory"]["gaps"]),
    }
    if recorded != expected or frozen["historical_source_hashes"] != decoded:
        raise ValueError("historical inventory coverage/metadata/fingerprints differ")
    if core.source_spec_audit(frozen["spec"], metadata)["collisions"]:
        raise ValueError("historical inventory RNG collision")


def validate_native_frozen(frozen, capture_root=None):
    repo = Path(__file__).resolve().parents[2]
    core.qualify_core(repo)
    profile_path = repo / "configs/stat_matched_source_capture_preparation_v1.json"
    profile = read(profile_path)
    if frozen.get("preparation_profile_sha256") != sha(profile_path):
        raise ValueError("captured preparation profile binding differs")
    validate_preflight_binding(frozen)
    spec = read(repo / "configs/stat_matched_source_panel_v1.json")
    geometry = read(repo / spec["construction"]["inherited_geometry"])
    if (
        frozen["spec"] != spec
        or frozen["geometry"] != geometry
        or frozen["features"] != read(repo / geometry["inherited_feature_config"])
        or frozen["processor_non_pixel"]
        != geometry["future_calibration"]["synthetic_non_pixel_expected"]
        or frozen["cache_shape_max"] != [1, 2, 397, 64]
        or Path(frozen["repository"]).resolve() != repo
        or Path(frozen["old_input_root"]).resolve()
        != repo / "runs/stat_matched_input_calibration_v1"
    ):
        raise ValueError("capture differs from the registered native contract")
    for name, digest in profile["preserved_core_sha256"].items():
        if sha(repo / name) != digest:
            raise ValueError("reviewed core file changed")
    pub = repo / "examples/research_notes/0087_stat_matched_input_calibration"
    if (
        sha(pub / "coordinate_maps.jsonl")
        != read(pub / "summary.json")["ledgers"]["coordinate_maps.jsonl"]["sha256"]
    ):
        raise ValueError("qualified coordinate-map ledger changed")
    map_rows = ledger(pub, "coordinate_maps.jsonl")
    if frozen["map_word_sha256"] != {
        str(r["shift"]): r["word_sha256"] for r in map_rows
    }:
        raise ValueError("frozen maps differ from qualified publication")
    from .cli.run_stat_matched_source import PROFILE

    release = frozen.get("execution_release_data")
    if (
        not release
        or release.get("approved_for_execution") is not True
        or release["study_id"] != profile["required_separate_release_id"]
        or release["preparation_sha256"] != sha(repo / PROFILE)
    ):
        raise ValueError("saved native execution authorization missing or changed")
    release_path = "configs/" + profile["required_separate_release_id"] + ".json"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo)

    blob = git("show", frozen["execution_release_commit"] + ":" + release_path)
    if (
        hashlib.sha256(blob).hexdigest() != frozen["execution_release_sha256"]
        or json.loads(blob) != release
    ):
        raise ValueError("saved authorization differs from committed release")
    git(
        "merge-base",
        "--is-ancestor",
        release["implementation_commit"],
        frozen["execution_release_commit"],
    )
    previous = read(repo / spec["preserve_inherited_frozen_hashes_from"])
    reviewed = read(repo / "configs/stat_matched_source_core_v1.json")
    required = (
        set(previous["code_sha256"])
        | set(reviewed["registration_sha256"])
        | set(profile["preserved_core_sha256"])
        | set(profile["implementation_files"])
    )
    if (
        set(release["code_sha256"]) != required
        or frozen["implementation_sha256"] != release["code_sha256"]
    ):
        raise ValueError("saved implementation freeze is incomplete")
    for name, digest in release["code_sha256"].items():
        if (
            sha(repo / name) != digest
            or hashlib.sha256(
                git("show", release["implementation_commit"] + ":" + name)
            ).hexdigest()
            != digest
        ):
            raise ValueError("native verifier/capture implementation changed")
    for text_key, expected_digest, ancestor in (
        (
            "public_code_receipt_text",
            release["public_receipt_sha256"],
            release["implementation_commit"],
        ),
        (
            "public_release_receipt_text",
            frozen["public_release_receipt_sha256"],
            frozen["execution_release_commit"],
        ),
    ):
        text = frozen[text_key].encode("utf-8")
        receipt = json.loads(text)
        if (
            hashlib.sha256(text).hexdigest() != expected_digest
            or receipt["state"] != "MERGED"
            or not receipt["url"].startswith(
                "https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/"
            )
        ):
            raise ValueError("saved public receipt binding differs")
        git("merge-base", "--is-ancestor", ancestor, receipt["mergeCommit"]["oid"])
        git("merge-base", "--is-ancestor", receipt["mergeCommit"]["oid"], "HEAD")
    panel = read(repo / "configs/representation_correspondence_panel_v1.json")
    observation = anchored_summary(
        repo, panel["observation_publication"], panel["observation_publication_sha256"]
    )["frozen_specification"]
    for key in ("jobs", "runtime_identity", "model_snapshot", "observation_config"):
        if frozen["observation_qualification"][key] != observation[key]:
            raise ValueError("saved observation qualification anchor differs")
    first = observation["jobs"][0]
    run_path = Path(first["historical_run"])
    audit_path = run_path.with_suffix(".input_audit.json")
    for path in (run_path, audit_path):
        if sha(path) != observation["source_sha256"][str(path)]:
            raise ValueError("trusted historical ACK anchor changed")
    old_run, old_audit = read(run_path), read(audit_path)
    if (
        frozen["model_context_policy"] != old_run["context_policy"]
        or frozen["model_runtime"] != old_run["runtime"]
        or frozen["source_prompt"] != old_run["stream_events"][0]["prompt"]
        or frozen["model_non_pixel"]
        != {k: v for k, v in old_audit["effective"].items() if k != "pixel_values"}
    ):
        raise ValueError("native ACK preparation differs from historical anchor")
    manifest = Path(old_run["manifest_path"])
    if (
        sha(manifest) != observation["manifest_sha256"][str(manifest)]
        or frozen["delivery_condition"] != read(manifest)["stimulus_condition"]
    ):
        raise ValueError("native delivery condition differs from historical anchor")
    history = (
        "examples/research_notes/0063_nonlearned_processing_calibration/summary.json"
    )
    expected_jobs = anchored_summary(
        repo, history, geometry["pinned_artifacts"][history]
    )["frozen_specification"]["processor_jobs"]
    if len(frozen["historical_processor"]) != 128:
        raise ValueError("native historical processor denominator differs")
    from .cli import run_stat_matched_input_calibration as processor_cal

    for actual, expected in zip(
        frozen["historical_processor"], expected_jobs, strict=True
    ):
        raw, _ = processor_cal.old._checked_artifact(expected["artifact"])
        if actual != {
            "pixel_values_sha256": expected["pixel_values_sha256"],
            "processor_non_pixel_payload": expected["processor_non_pixel_payload"],
            "raw_word_sha256": matching.word_sha(raw),
        }:
            raise ValueError("native historical processor anchor differs")
    verify_source_inventory(repo, capture_root, frozen)


def check_operations(root, expected_digest):
    chain, counts, pending = (
        "0" * 64,
        {k: {"attempted": 0, "returned": 0} for k in ("source", "processor", "model")},
        None,
    )
    sealed = False
    for i, saved in enumerate(ledger(root, "operations.jsonl"), 1):
        row = dict(saved)
        digest = row.pop("digest")
        if (
            row["sequence"] != i
            or row["previous"] != chain
            or hashlib.sha256(
                json.dumps(
                    row, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            != digest
        ):
            raise ValueError("operation hash chain differs")
        chain = digest
        if row["kind"] == "reference_seal":
            if (
                sealed
                or pending
                or counts
                != {
                    "source": {"attempted": 32, "returned": 32},
                    "processor": {"attempted": 409, "returned": 409},
                    "model": {"attempted": 664, "returned": 664},
                }
                or row["sha256"] != sha(root / "reference_seal.json")
            ):
                raise ValueError("reference seal boundary differs")
            sealed = True
            continue
        key = row["operation"], row["identity"]
        if row["kind"] == "begin":
            if pending is not None:
                raise ValueError("overlapping operations")
            if (
                row["operation"] == "source"
                and counts["source"]["attempted"] >= 32
                and not sealed
            ):
                raise ValueError("test source generated before reference seal")
            counts[row["operation"]]["attempted"] += 1
            pending = key
        elif row["kind"] == "return" and pending == key:
            counts[row["operation"]]["returned"] += 1
            pending = None
        else:
            raise ValueError("operation order differs")
    if chain != expected_digest:
        raise ValueError("last operation digest differs")
    return counts, pending, sealed


def expected_operations(frozen):
    operations = [("processor", f"historical_{i}") for i in range(128)]
    operations += [
        ("processor", key) for key, _, _ in matching.cases(frozen["geometry"])
    ]
    pairs = frozen["spec"]["sources"]["pairs"]

    def inputs(split):
        result = []
        for p in pairs:
            if p["split"] == split:
                result += [
                    ("source", p["pair_id"] + "/" + role)
                    for role in ("source_a", "source_b")
                ]
                result += [
                    ("processor", f"{p['pair_id']}/s{s}/{c}")
                    for s in range(4)
                    for c in core.CELLS
                ]
        return result

    operations += inputs("reference")
    operations += [("model", f"calibration_{i}") for i in range(384)]
    operations += [
        ("model", f"sham/{n}_s{s}/{t}")
        for n in ("zero", "constant")
        for s in range(4)
        for t in ("off_before", "on", "off_after")
    ]
    jobs = core.ordered_jobs(frozen["spec"])
    operations += [
        ("model", f"{j['pair_id']}/s{j['shift']}/{j['cell']}") for j in jobs[:256]
    ]
    operations += inputs("test")
    operations += [
        ("model", f"{j['pair_id']}/s{j['shift']}/{j['cell']}") for j in jobs[256:]
    ]
    return operations


def check_legacy_cache_members(root):
    for row in ledger(root / "recalibration", "cells.jsonl"):
        run_path = root / "recalibration" / row["path"]
        for artifact in row["tensor_artifacts"]:
            relative = str((run_path.parent / artifact["path"]).relative_to(root))
            with np.load(inside(root, relative), allow_pickle=False) as archive:
                if artifact["array_key"] != "tensor" or archive.files != ["tensor"]:
                    raise ValueError("legacy cache archive member coverage differs")


def binary_coverage(root, frozen):
    known = {f"maps/s{s}.npy" for s in range(4)}

    def collect(value):
        if isinstance(value, dict):
            if {"path", "word_sha256", "shape", "dtype", "sha256"} <= set(value):
                known.add(value["path"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for name in (
        "historical.jsonl",
        "sources.jsonl",
        "synthetic.jsonl",
        "candidates.jsonl",
        "models.jsonl",
    ):
        collect(ledger(root, name))
    collect(read(root / "family_references.json"))
    if not frozen["software_test_only"]:
        check_legacy_cache_members(root)
        for row in ledger(root, "models.jsonl"):
            run_path = inside(root, row["evidence"]["run"])
            for artifact in read(run_path)["stream_events"][0][
                "cache_tensor_artifacts"
            ]:
                known.add(str((run_path.parent / artifact["path"]).relative_to(root)))
        for row in ledger(root / "recalibration", "cells.jsonl"):
            run_path = root / "recalibration" / row["path"]
            for artifact in row["tensor_artifacts"]:
                relative = str((run_path.parent / artifact["path"]).relative_to(root))
                known.add(relative)
            if row.get("representation") is not None:
                known.add("recalibration/" + row["representation"]["path"])
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.suffix in (".npy", ".npz")
    }
    if actual != known:
        raise ValueError("unreferenced or missing saved binary artifacts")
    return len(actual)


def verify_reference_vectors(root, frozen):
    pairs = frozen["spec"]["sources"]["pairs"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    models = {
        (r["pair_id"], r["shift"], r["cell"]): r
        for r in ledger(root, "models.jsonl")
        if r["phase"] != "sham"
    }
    inputs = {
        (r["pair_id"], r["shift"], r["cell"]): r
        for r in ledger(root, "candidates.jsonl")
    }
    profiles = [
        (v, f)
        for v in frozen["spec"]["analysis"]["views"]
        for f in (core.FLOORS if v in ("encoder", "projector") else (1e-6,))
    ]
    rows = read(root / "family_references.json")
    if [(r["view"], r["relative_floor"], r["family"]) for r in rows] != [
        (v, f, family) for v, f in profiles for family in families
    ]:
        raise ValueError("all fixed reference profiles required")
    for row in rows:
        view = row["view"]
        learned = view in ("encoder", "projector")
        units = []
        for p, diagnostics in zip(
            [
                p
                for p in pairs
                if p["split"] == "reference" and p["pairing_family"] == row["family"]
            ],
            row["interactions"],
            strict=True,
        ):
            cells = {}
            for cell in core.CELLS:
                key = p["pair_id"], 0, cell
                a = load_array(
                    root,
                    models[key]["sites"][view]
                    if learned
                    else inputs[key]["features"]["actual__" + view.replace("/", "__")],
                )
                cells[cell] = oracle_expand(a) if learned else a
            units.append(source_oracle.direction(cells, learned, row["relative_floor"]))
            z = (cells["jj"].astype(np.float64) - cells["mj"]) - (
                cells["jm"].astype(np.float64) - cells["mm"]
            )
            norm = math.hypot(*z.ravel().tolist())
            scale = sum(
                math.hypot(*v.astype(float).ravel().tolist()) for v in cells.values()
            )
            threshold = (
                row["relative_floor"] * scale
                if learned
                else 100 * 4e-10 * math.sqrt(z.size)
            )
            reason = (
                None
                if norm > threshold
                else ("exact_zero" if norm == 0 else "below_amplitude_guard")
            )
            if (
                diagnostics["dimension"] != z.size
                or diagnostics["vector_sha256"]
                != matching.word_sha(np.asarray(z.ravel(), dtype="<f8"))
                or diagnostics["available"] != (units[-1] is not None)
                or diagnostics["reason"] != reason
                or any(
                    not math.isclose(diagnostics[k], v, rel_tol=1e-11, abs_tol=0)
                    for k, v in (
                        ("norm", norm),
                        ("cell_scale", scale),
                        ("threshold", threshold),
                    )
                )
            ):
                raise ValueError("saved reference contrast/norm/availability differs")
        expected, reason = None, "unavailable_reference_replicate"
        if all(u is not None for u in units):
            total = np.array([x + y for x, y in zip(*units, strict=True)])
            norm = math.hypot(*total.tolist())
            reason = "cancelling_unit_references" if norm <= 1e-12 else None
            if reason is None:
                expected = total / norm
        if row["reason"] != reason or (row["unit"] is None) != (expected is None):
            raise ValueError("fixed family reference availability differs")
        if expected is not None:
            saved = load_array(root, row["unit"])
            if (
                saved.shape != expected.shape
                or saved.dtype != np.float64
                or not np.isfinite(saved).all()
                or np.max(np.abs(saved - expected)) > 1e-12
            ):
                raise ValueError("fixed family reference differs from scalar oracle")


def _features(root, row, frozen):
    raw, pixels = (load_array(root, row[k]) for k in ("raw", "pixels"))
    spec = frozen["spec"]
    if (
        raw.dtype != np.uint8
        or list(raw.shape) != spec["construction"]["raw_shape_hwc"]
        or pixels.dtype != np.float32
        or list(pixels.shape) != spec["input_gates"]["expected_native_shape"]
        or not np.isfinite(pixels).all()
        or row["non_pixel"] != frozen["processor_non_pixel"]
    ):
        raise ValueError("saved input layout/dtype/nonpixel differs")
    expected_keys = {
        f"{kind}__{s}__{f}"
        for kind in ("actual", "oracle")
        for s in matching.STARTS
        for f in matching.FEATURES
    }
    if set(row["features"]) != expected_keys:
        raise ValueError("saved feature coverage differs")
    arrays = {k: load_array(root, v) for k, v in row["features"].items()}
    errors = []
    spectra = {
        "raw_rgb": raw_spectra(_luminance(raw.astype(np.float64) / 255)),
        "processor_pixel_values": native_spectra(
            _tensor_view(pixels[0]), patch_size=None
        ),
    }
    if spectra != row["spectra"]:
        raise ValueError("saved spectrum diagnostics differ")
    for start, values in (("raw_rgb", raw), ("processor_pixel_values", pixels)):
        expected = oracle_features(values, start, frozen["features"])
        actual = extract_features(values, start, frozen["features"])
        for feature, result in expected.items():
            for kind, recomputed in (("actual", actual[feature]), ("oracle", result)):
                stored = arrays[f"{kind}__{start}__{feature}"]
                if (
                    stored.shape != result.shape
                    or stored.dtype != np.float64
                    or not np.isfinite(stored).all()
                    or np.max(np.abs(stored - result)) > 1e-10
                    or stored.tobytes() != recomputed.tobytes()
                ):
                    raise ValueError("saved features differ from independent oracle")
            errors.append(
                {
                    "start": start,
                    "feature": feature,
                    "max_abs": float(np.max(np.abs(actual[feature] - result))),
                }
            )
    if errors != row["audit"].get("oracle_errors", row["audit"].get("oracle_checks")):
        raise ValueError("saved oracle-error diagnostics differ")
    return raw, pixels, arrays


def _check_models(root, rows, cases, frozen):
    if not frozen["software_test_only"]:
        internal_rows = ledger(root, "source_internal_calls.jsonl")
        if [r["identity"] for r in internal_rows] != [
            r["identity"] for r in rows
        ] or any(
            a["decoder"] != b["internal_calls"]["language_model_call"]
            for a, b in zip(internal_rows, rows, strict=True)
        ):
            raise ValueError("native internal call journal differs")
    for row in rows:
        identity = row["identity"]
        internal = row["internal_calls"]
        if (
            internal["prepared_inputs"] != 1
            or internal["recorded_generation_steps"] != len(row["suffix"])
            or internal["language_model_call"]["attempted"]
            != internal["language_model_call"]["returned"]
            or internal["language_model_call"]["returned"] < 1
        ):
            raise ValueError("recorded internal call accounting differs")
        case = (
            cases[identity]
            if row["phase"] != "sham"
            else cases[row["fixture"] + f"_s{row['shift']}"]
        )
        if (
            row["prepared"]
            != {
                "pixel_word_sha256": case["pixels"]["word_sha256"],
                "non_pixel": frozen["model_non_pixel"],
            }
            or not row["restored"]
            or row["observer_calls"] != int(row.get("stage", "on") == "on")
            or not 0 < len(row["suffix"]) <= 3
        ):
            raise ValueError("model prepared path/observer/suffix receipt differs")
        observed = row.get("stage", "on") == "on"
        if set(row["sites"]) != (
            set(frozen["spec"]["observation"]["sites"]) if observed else set()
        ) or set(row["caches"]) != {"1:keys", "12:keys", "23:values"}:
            raise ValueError("model site/cache coverage differs")
        for site, receipt in row["sites"].items():
            words = load_array(root, receipt)
            values = oracle_expand(words)
            if (
                words.dtype != np.dtype("<u2")
                or list(values.shape) != frozen["spec"]["observation"]["sites"][site]
                or not np.isfinite(values).all()
                or matching.word_sha(values) != receipt["expanded_word_sha256"]
            ):
                raise ValueError("saved full-site bfloat16 layout differs")
        cache_lengths = set()
        for receipt in row["caches"].values():
            a = load_array(root, receipt)
            limit = frozen["cache_shape_max"]
            if (
                a.dtype != np.float32
                or a.ndim != 4
                or list(a.shape[:2]) != limit[:2]
                or not 1 <= a.shape[2] <= limit[2]
                or a.shape[3] != limit[3]
                or not np.isfinite(a).all()
                or np.any(a.view(np.uint32) & np.uint32(65535))
            ):
                raise ValueError("saved cache layout/finiteness differs")
            cache_lengths.add(a.shape[2])
        if len(cache_lengths) != 1:
            raise ValueError("selected cache lengths differ")
        if not frozen["software_test_only"]:
            _native_evidence(root, row, case, frozen)
    shams = [r for r in rows if r["phase"] == "sham"]
    expected = [
        (name, shift, stage)
        for name in ("zero", "constant")
        for shift in range(4)
        for stage in ("off_before", "on", "off_after")
    ]
    if [(r["fixture"], r["shift"], r["stage"]) for r in shams] != expected:
        raise ValueError("all 24 sham contexts required")
    for name in ("zero", "constant"):
        group = [r for r in shams if r["fixture"] == name]
        for key in ("prepared", "suffix", "layout"):
            if any(r[key] != group[0][key] for r in group):
                raise ValueError("saved sham path parity differs")
        for target in group[0]["caches"]:
            if len({r["caches"][target]["word_sha256"] for r in group}) != 1:
                raise ValueError("saved sham cache words differ")
        for site in frozen["spec"]["observation"]["sites"]:
            if (
                len(
                    {
                        r["sites"][site]["word_sha256"]
                        for r in group
                        if r["stage"] == "on"
                    }
                )
                != 1
            ):
                raise ValueError("saved sham site words differ")


def _native_evidence(root, row, case, frozen):
    evidence = row["evidence"]
    if evidence != {
        "run": "observations/" + row["identity"] + "/run.json",
        "audit": "observations/" + row["identity"] + "/run.input_audit.json",
    }:
        raise ValueError("native evidence identity/path differs")
    run = read(inside(root, evidence["run"]))
    audit = read(inside(root, evidence["audit"]))
    if (
        run["model_id"] != frozen["spec"]["observation"]["model_id"]
        or run.get("dry_run")
        or run["context_policy"] != frozen["model_context_policy"]
        or run["runtime"] != frozen["model_runtime"]
        or run["reproducibility"]["seed"] != 20260604
        or len(run["stream_events"]) != 1
    ):
        raise ValueError("native saved source protocol differs")
    event = run["stream_events"][0]
    if (
        event["prompt"] != frozen["source_prompt"]
        or event["frame_sha256s"] != [sha(root / case["folder"] / "raw.png")]
        or event["cache_token_layout"] != row["layout"]
        or [s["token_id"] for s in event["generation"]["steps"]] != row["suffix"]
    ):
        raise ValueError("native frame/prompt/natural suffix/layout differs")
    layout = event["cache_token_layout"]["cache_position_layout"]
    minimum_cache_length = (
        len(frozen["model_non_pixel"]["input_ids"]["values"][0]) + 255
    )
    if (
        not layout["available"]
        or not layout["image_partition_available"]
        or layout["image_position_runs"] != [{"start": 42, "end": 297, "length": 256}]
        or layout["cache_sequence_length"] != row["caches"]["1:keys"]["shape"][2]
        or layout["cache_sequence_length"] < minimum_cache_length
        or layout["image_position_count"] != 256
        or layout["pre_image_position_count"] != 42
        or layout["post_image_position_count"] != layout["cache_sequence_length"] - 298
        or layout["expansion_delta"] != 255
        or layout["processor_token_count"] + 255 != layout["cache_sequence_length"]
    ):
        raise ValueError("native image-token/cache layout differs")
    if (
        audit["override_image_sizes"] is not None
        or audit["native"] != audit["effective"]
        or audit["effective"]["pixel_values"]["sha256"] != case["pixels"]["word_sha256"]
        or {k: v for k, v in audit["effective"].items() if k != "pixel_values"}
        != frozen["model_non_pixel"]
    ):
        raise ValueError("native effective input receipt differs")
    manifest_path = root / case["folder"] / "manifest.json"
    if audit["context"] != {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha(manifest_path),
    } or run["manifest_path"] != str(manifest_path):
        raise ValueError("native manifest context differs")
    artifacts = event["cache_tensor_artifacts"]
    if len(artifacts) != 3 or {
        f"{a['layer_index']}:{a['tensor']}" for a in artifacts
    } != set(row["caches"]):
        raise ValueError("native selected cache artifact coverage differs")
    for a in artifacts:
        p = inside(
            root,
            str((inside(root, evidence["run"]).parent / a["path"]).relative_to(root)),
        )
        if sha(p) != a["sha256"] or a["source_dtype"] != "mlx.core.bfloat16":
            raise ValueError("native cache sidecar receipt differs")
        with np.load(p, allow_pickle=False) as saved:
            if a["array_key"] != "tensor" or saved.files != ["tensor"]:
                raise ValueError("native cache archive member coverage differs")
            values = saved[a["array_key"]]
            target = f"{a['layer_index']}:{a['tensor']}"
            if (
                values.dtype != np.float32
                or a["dtype"] != "float32"
                or values.tobytes() != load_array(root, row["caches"][target]).tobytes()
                or list(values.shape) != a["shape"]
                or values.shape[2] != a["effective_sequence_length"]
            ):
                raise ValueError("native versus canonical cache words differ")
    manifest = read(root / case["folder"] / "manifest.json")
    frame = manifest["frames"][0]
    if len(manifest["frames"]) != 1 or frame["sha256"] != sha(
        root / case["folder"] / "raw.png"
    ):
        raise ValueError("native source manifest differs")
    with Image.open(root / case["folder"] / "raw.png") as image:
        if (
            np.asarray(image.convert("RGB")).tobytes()
            != load_array(root, case["raw"]).tobytes()
        ):
            raise ValueError("native source PNG differs")


def verify_prefix(root, frozen, *, complete=False):
    candidates = ledger(root, "candidates.jsonl")
    expected_jobs = core.ordered_jobs(frozen["spec"])
    jobs = expected_jobs if complete else expected_jobs[:256]
    if [(r["pair_id"], r["shift"], r["cell"]) for r in candidates] != [
        (j["pair_id"], j["shift"], j["cell"]) for j in jobs
    ]:
        raise ValueError("candidate identity/order/coverage differs")
    maps = {
        s: np.load(inside(root, f"maps/s{s}.npy"), allow_pickle=False) for s in range(4)
    }
    for s, a in maps.items():
        expected_map = map_oracle.coordinate_map(frozen["geometry"], s)
        if (
            a.dtype != np.dtype("<u4")
            or a.shape != expected_map.shape
            or matching.word_sha(a) != frozen["map_word_sha256"][str(s)]
            or a.tobytes() != expected_map.tobytes()
        ):
            raise ValueError("map independent reconstruction differs")
    source_rows = ledger(root, "sources.jsonl")
    pairs = [
        p
        for p in frozen["spec"]["sources"]["pairs"]
        if complete or p["split"] == "reference"
    ]
    expected_sources = [
        (p["pair_id"], r)
        for split in ("reference", "test")
        for p in pairs
        if p["split"] == split
        for r in ("source_a", "source_b")
    ]
    if [(r["pair_id"], r["role"]) for r in source_rows] != expected_sources:
        raise ValueError("source identities/order/coverage differ")
    if core.decoded_hash_audit(source_rows, frozen["historical_source_hashes"])[
        "collisions"
    ]:
        raise ValueError("source decoded hash collision")
    sources = {}
    for row in source_rows:
        a = load_array(root, row["raw"])
        pair = next(p for p in pairs if p["pair_id"] == row["pair_id"])
        if frozen["software_test_only"]:
            y, x = np.indices(a.shape[:2])
            seed = pair[row["role"]]["seed"]
            expected = np.stack(
                (
                    (x + seed) % 256,
                    (y * 3 + seed // 256) % 256,
                    (x + y * 7 + seed // 65536) % 256,
                ),
                axis=-1,
            ).astype(np.uint8)
        else:
            from .control_stimulus import GeneratedControlSpec, generate_control_frame

            values = {**frozen["spec"]["sources"]["defaults"], **pair[row["role"]]}
            values["condition_id"] = (
                values.get("condition_id")
                or f"{pair['pair_id']}_{row['role']}_{values['kind']}_seed{values['seed']}"
            )
            expected_spec = GeneratedControlSpec(**values)
            if row["provenance"]["generator_spec"] != expected_spec.to_dict():
                raise ValueError(
                    "source generator specification differs from allocation"
                )
            expected = generate_control_frame(expected_spec, 0)
            manifest_path = inside(root, row["provenance"]["manifest"])
            manifest = read(manifest_path)
            frame_path = (
                manifest_path.parent / manifest["frames"][0]["path"]
            ).resolve()
            if (
                not frame_path.is_relative_to(root.resolve())
                or sha(frame_path) != manifest["frames"][0]["sha256"]
            ):
                raise ValueError("source generator manifest differs")
            with Image.open(frame_path) as image:
                if np.asarray(image.convert("RGB")).tobytes() != a.tobytes():
                    raise ValueError("source generator PNG differs")
        if (
            a.dtype != np.uint8
            or list(a.shape)
            != frozen["spec"]["construction"]["payload_source_shape_hwc"]
            or a.tobytes() != expected.tobytes()
            or row["decoded_rgb_sha256"] != row["raw"]["word_sha256"]
        ):
            raise ValueError("source generation reconstruction differs")
        sources[(row["pair_id"], row["role"])] = a
    refs, cases, palette_rows = {}, {}, ledger(root, "palettes.jsonl")
    palette_expected = []
    expected_images = {}
    for row in candidates:
        pair_id, shift, cell = row["pair_id"], row["shift"], row["cell"]
        if (pair_id, shift) not in expected_images:
            expected_images = {
                (pair_id, shift): source_oracle.quartet(
                    sources[(pair_id, "source_a")],
                    sources[(pair_id, "source_b")],
                    frozen["spec"],
                    frozen["geometry"],
                    shift,
                )
            }
            palette_expected.append(
                {
                    "pair_id": pair_id,
                    "shift": shift,
                    **core.palette_audit(expected_images[(pair_id, shift)]),
                }
            )
        raw, pixels, arrays = _features(root, row, frozen)
        if raw.tobytes() != expected_images[(pair_id, shift)][cell].tobytes():
            raise ValueError("payload-first independent reconstruction differs")
        key = pair_id, cell
        groups = {
            s: matching.joint_multisets(a, s, frozen["features"]["grid"])
            for s, a in (("raw_rgb", raw), ("processor_pixel_values", pixels))
        }
        if shift == 0:
            refs[key] = (arrays, groups, row["raw"], row["pixels"])
            if row["residuals"]:
                raise ValueError("identity input cannot have shift residuals")
        else:
            old, old_groups, ref_raw, ref_pixels = refs[key]
            if groups != old_groups:
                raise ValueError("saved native joint RGB multisets differ")
            if set(row["residuals"]) != {
                s + "__" + f for s in matching.STARTS for f in matching.FEATURES
            }:
                raise ValueError("signed residual coverage differs")
            for start in matching.STARTS:
                for feature in matching.FEATURES:
                    k = f"actual__{start}__{feature}"
                    residual = arrays[k] - old[k]
                    if (
                        np.max(np.abs(residual)) > 1e-10
                        or residual.tobytes()
                        != load_array(
                            root, row["residuals"][start + "__" + feature]
                        ).tobytes()
                    ):
                        raise ValueError("saved signed matching residual differs")
        comparisons, words, values, changed_pixels = [], {}, {}, {}
        if shift:
            for start, a, b in (
                ("raw_rgb", raw, load_array(root, ref_raw)),
                ("processor_pixel_values", pixels, load_array(root, ref_pixels)),
            ):
                values[start] = int(np.count_nonzero(a != b))
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
                words[start] = int(np.count_nonzero(aw != bw))
                changed_pixels[start] = int(np.count_nonzero(np.any(aw != bw, axis=-1)))
                for feature in matching.FEATURES:
                    name = f"actual__{start}__{feature}"
                    delta = arrays[name] - old[name]
                    comparisons.append(
                        {
                            "start": start,
                            "feature": feature,
                            "max_abs": float(np.max(np.abs(delta))),
                            "l2": float(np.linalg.norm(delta)),
                        }
                    )
        expected_audit = {
            "failures": [],
            "groups": groups,
            "oracle_errors": row["audit"]["oracle_errors"],
            "comparisons": comparisons,
            "changed_words": words,
            "changed_values": values,
            "changed_pixels": changed_pixels,
        }
        if row["audit"] != expected_audit:
            raise ValueError("saved complete candidate diagnostics differ")
        cases[row["identity"]] = row
    if palette_rows != palette_expected or any(
        not all(r["matches"]) for r in palette_expected
    ):
        raise ValueError("palette marginal audits differ")
    synthetic = ledger(root, "synthetic.jsonl")
    if [(r["identity"], r["fixture"], r["shift"]) for r in synthetic] != matching.cases(
        frozen["geometry"]
    ):
        raise ValueError("synthetic requalification coverage differs")
    synthetic_refs = {}
    for row in synthetic:
        raw, pixels, arrays = _features(root, row, frozen)
        key, name, shift = row["identity"], row["fixture"], row["shift"]
        if (
            raw.tobytes()
            != map_oracle.fixture(
                frozen["geometry"], name, shift, key == "sentinel"
            ).tobytes()
        ):
            raise ValueError("synthetic raw reconstruction differs")
        if not shift and key != "sentinel":
            synthetic_refs[name] = {
                s: (
                    v,
                    arrays,
                    matching.joint_multisets(v, s, frozen["features"]["grid"]),
                )
                for s, v in (("raw_rgb", raw), ("processor_pixel_values", pixels))
            }
        expected_audit, residuals = matching.case_audit(
            key,
            name,
            shift,
            raw,
            pixels,
            arrays,
            row["audit"]["oracle_checks"],
            synthetic_refs.get(name) if shift or key == "sentinel" else None,
            frozen["features"],
            frozen["geometry"],
        )
        expected_audit["groups"] = expected_audit["joint_multisets"]
        if (
            expected_audit != row["audit"]
            or expected_audit["failures"]
            or set(residuals) != set(row["residuals"])
            or any(
                v.tobytes() != load_array(root, row["residuals"][k]).tobytes()
                for k, v in residuals.items()
            )
        ):
            raise ValueError("synthetic sham/sentinel audit differs")
        cases[key] = row
    historical = ledger(root, "historical.jsonl")
    if len(historical) != 128:
        raise ValueError("historical processor coverage differs")
    for i, row in enumerate(historical):
        raw = load_array(root, row["raw"])
        a = load_array(root, row["pixels"])
        expected = frozen["historical_processor"][i]
        if (
            row["identity"] != f"historical_{i}"
            or a.dtype != np.float32
            or list(a.shape) != frozen["spec"]["input_gates"]["expected_native_shape"]
            or not np.isfinite(a).all()
            or row["pixels"]["word_sha256"] != expected["pixel_values_sha256"]
            or row["non_pixel"] != expected["processor_non_pixel_payload"]
            or matching.word_sha(raw) != expected["raw_word_sha256"]
        ):
            raise ValueError("historical processor receipt parity differs")
    models = ledger(root, "models.jsonl")
    _check_models(root, models, cases, frozen)
    actual_jobs = [r for r in models if r["phase"] != "sham"]
    if len(actual_jobs) != len(jobs) or any(
        any(r[k] != j[k] for k in j) for r, j in zip(actual_jobs, jobs, strict=True)
    ):
        raise ValueError("model source identity/order/ordinal differs")
    generic = ledger(root, "inputs.jsonl")
    specialized = synthetic + candidates
    keys = (
        "identity",
        "folder",
        "raw",
        "pixels",
        "non_pixel",
        "spectra",
        "features",
        "residuals",
        "audit",
    )
    if generic != [{k: r[k] for k in keys} for r in specialized] or ledger(
        root, "processor_returns.jsonl"
    ) != [{k: r[k] for k in keys[:5]} for r in specialized]:
        raise ValueError("processor-return/input ledgers disagree")
    if not frozen["software_test_only"] and not complete:
        from tempfile import TemporaryDirectory
        from .stat_matched_source_native import verify_legacy_calibration

        with TemporaryDirectory(prefix="reference-calibration-verify-") as temporary:
            from pathlib import Path

            verify_legacy_calibration(root, Path(temporary) / "legacy", final=False)
    return {
        "candidate_cells": len(candidates),
        "source_frames": len(source_rows),
        "model_cells": len(models),
        "historical_processor_cells": 128,
        "synthetic_cells": 25,
        "independent_saved_array_checks": True,
        "new_processor_calls": 0,
        "new_model_contexts": 0,
    }


def analyze_verified(root, output, first, second, *, allow_software_fixture=False):
    root, output, first, second = (p.resolve() for p in (root, output, first, second))
    roots = (root, output, first, second)
    if any(
        a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(roots)
        for b in roots[i + 1 :]
    ):
        raise ValueError(
            "two separate verification roots and fresh analysis root required"
        )
    cap = read(root / "capture.json")
    if (root / "failure.json").exists() or inventory(root) != cap["receipts"]:
        raise ValueError("capture changed after verification")
    summaries = [read(p / "summary.json") for p in (first, second)]
    accepted = "VERIFIED_SOFTWARE_FIXTURE" if allow_software_fixture else "VERIFIED"
    if (
        summaries[0] != summaries[1]
        or any(
            s["status"] != accepted or s["capture_sha256"] != sha(root / "capture.json")
            for s in summaries
        )
        or inventory(first) != inventory(second)
        or any((p / "verification_failure.json").exists() for p in (first, second))
    ):
        raise ValueError("two byte-identical complete verifications required")
    frozen = read(root / "frozen.json")
    if frozen["software_test_only"] and not allow_software_fixture:
        raise ValueError("software capture cannot produce native results")
    models = {
        (r["pair_id"], r["shift"], r["cell"]): r
        for r in ledger(root, "models.jsonl")
        if r["phase"] != "sham"
    }
    inputs = {
        (r["pair_id"], r["shift"], r["cell"]): r
        for r in ledger(root, "candidates.jsonl")
    }

    def cells(pair_id, shift, view):
        values = {}
        for cell in core.CELLS:
            key = pair_id, shift, cell
            if view in ("encoder", "projector"):
                values[cell] = oracle_expand(
                    load_array(root, models[key]["sites"][view])
                )
            else:
                values[cell] = load_array(
                    root, inputs[key]["features"]["actual__" + view.replace("/", "__")]
                )
        return values

    output.mkdir(parents=True, exist_ok=False)
    from .stat_matched_source_storage import append

    profiles = []
    for view in frozen["spec"]["analysis"]["views"]:
        for floor in core.FLOORS if view in ("encoder", "projector") else (1e-6,):
            actual = core.analyze_view(
                cells, frozen["spec"], view, relative_floor=floor
            )
            independent = source_oracle.profile(cells, frozen["spec"], view, floor)
            for key in ("scores", "margins", "paired_margin_change"):
                a = np.asarray(actual[key], dtype=np.float64)
                b = np.asarray(independent[key], dtype=np.float64)
                if a.shape != b.shape or not np.allclose(
                    a, b, atol=1e-12, rtol=0, equal_nan=True
                ):
                    raise ValueError("full-vector profile differs from scalar oracle")
            append(output, "profiles.jsonl", actual)
            profiles.append(actual)
    append_rows = [
        {"identity": r["identity"], "spectra": r["spectra"]} for r in inputs.values()
    ]
    for row in append_rows:
        append(output, "spectra.jsonl", row)
    result = {
        "status": "SOFTWARE_PROFILES"
        if frozen["software_test_only"]
        else "DESCRIPTIVE_PROFILES_VERIFIED",
        "capture_sha256": sha(root / "capture.json"),
        "profiles": 18,
        "primary_endpoints": [
            {
                "view": r["view"],
                "available": r["available"],
                "paired_margin_change": r["paired_margin_change"],
            }
            for r in profiles
            if r["primary_learned_profile"]
        ],
        "output_sha256": inventory(output),
        "new_processor_calls": 0,
        "new_model_contexts": 0,
        "p_values": None,
        "equivalence_tested": False,
    }
    write(output / "summary.json", result)
    return result


def verify_capture(root, output, *, allow_software_fixture=False):
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("capture and verification roots must be disjoint")
    output.mkdir(parents=True, exist_ok=False)
    try:
        frozen = read(root / "frozen.json")
        if not frozen["software_test_only"]:
            validate_native_frozen(frozen, root)
        if frozen["software_test_only"] and not allow_software_fixture:
            raise ValueError("software fixture cannot qualify native acquisition")
        failed = (root / "failure.json").exists()
        cap = read(root / ("failure.json" if failed else "capture.json"))
        if inventory(root) != cap["receipts"]:
            raise ValueError("complete saved-file inventory differs")
        counts, pending, sealed = check_operations(root, cap["last_operation_digest"])
        attempted = [
            (r["operation"], r["identity"])
            for r in ledger(root, "operations.jsonl")
            if r["kind"] == "begin"
        ]
        if attempted != expected_operations(frozen)[: len(attempted)]:
            raise ValueError("registered operation identity/order differs")
        if counts != cap["counts"]:
            raise ValueError("attempted/returned counts differ")
        if failed:
            result = {
                "status": "FAILED_CAPTURE_AUDITED",
                "failure": {k: cap[k] for k in ("stage", "type", "message")},
                "counts": counts,
                "partial_semantics": "not_promoted_to_complete_verification",
                "primary_results_withheld": True,
                "new_processor_calls": 0,
                "new_model_contexts": 0,
            }
        else:
            expected = {
                k: {"attempted": v, "returned": v}
                for k, v in (("source", 64), ("processor", 665), ("model", 920))
            }
            if (
                cap["status"] != "COMPLETE_UNVERIFIED"
                or counts != expected
                or pending
                or not sealed
            ):
                raise ValueError("complete capture coverage differs")
            proof = verify_prefix(root, frozen, complete=True)
            seal = read(root / "reference_seal.json")
            reference_models = [
                r for r in ledger(root, "models.jsonl") if r["phase"] == "reference"
            ]
            expected_seal = core.seal_reference_metadata(
                reference_models, frozen["spec"]
            )
            if (
                any(
                    seal[k] != v
                    for k, v in expected_seal.items()
                    if k != "artifact_bytes_independently_verified"
                )
                or seal["proof"]["candidate_cells"] != 256
                or seal["counts"]
                != {
                    "source": {"attempted": 32, "returned": 32},
                    "processor": {"attempted": 409, "returned": 409},
                    "model": {"attempted": 664, "returned": 664},
                }
            ):
                raise ValueError("reference seal identity/count/proof differs")
            if (
                not seal["artifact_bytes_independently_verified"]
                or seal["last_context"] != 664
                or seal["first_test_context"] != 665
                or seal["family_references_sha256"]
                != sha(root / "family_references.json")
            ):
                raise ValueError("artifact-backed reference seal differs")
            verify_reference_vectors(root, frozen)
            proof["saved_binary_artifacts_checked"] = binary_coverage(root, frozen)
            if not frozen["software_test_only"]:
                from .stat_matched_source_native import verify_legacy_calibration

                verify_legacy_calibration(root, output / "legacy_calibration")
            result = {
                "status": "VERIFIED_SOFTWARE_FIXTURE"
                if frozen["software_test_only"]
                else "VERIFIED",
                "counts": counts,
                "proof": proof,
                "capture_sha256": sha(root / "capture.json"),
                "primary_results_withheld": True,
                "new_processor_calls": 0,
                "new_model_contexts": 0,
            }
        write(output / "summary.json", result)
        return result
    except BaseException as error:
        write(
            output / "verification_failure.json",
            {
                "status": "INVALID_CAPTURE",
                "type": type(error).__name__,
                "message": str(error),
                "primary_results_withheld": True,
                "new_processor_calls": 0,
                "new_model_contexts": 0,
            },
        )
        raise
