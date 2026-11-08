"""Native adapters for a separately approved run; importing never loads weights."""

import importlib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import stat_matched_source as core
from .cli import run_representation_observation_calibration as cal
from .cli import run_representation_panel as legacy_panel
from .cli import run_stat_matched_input_calibration as processor_cal
from .cli import summarize_representation_observation_calibration as cal_pub
from .conditions import StimulusCondition
from .external_frames import build_external_frame_manifest
from .generator_pairing_panel import _materialize_source
from .representation_observation import observe_projector
from .stat_matched_source_storage import load_array, read, sha, write


@contextmanager
def count_decoder(model, on_exit=None):
    cls, counts = type(model), {"attempted": 0, "returned": 0}
    original = cls.__call__
    if getattr(original, "_source_decoder_counter", False):
        raise ValueError("nested decoder counters are not permitted")

    def counted(instance, *args, **kwargs):
        if instance is model:
            counts["attempted"] += 1
        result = original(instance, *args, **kwargs)
        if instance is model:
            counts["returned"] += 1
        return result

    counted._source_decoder_counter = True
    cls.__call__ = counted
    try:
        yield counts
    finally:
        cls.__call__ = original
        if on_exit is not None:
            on_exit(counts)


def source_inventory(repo, spec):
    metadata, fingerprints, file_hashes, gaps = [], [], {}, []
    manifests = set((repo / "runs").glob("**/sources/**/manifest.json"))

    def groups(value):
        if isinstance(value, dict):
            if isinstance(value.get("pairs"), list):
                yield value
            for item in value.values():
                yield from groups(item)
        elif isinstance(value, list):
            for item in value:
                yield from groups(item)

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and value.endswith("manifest.json"):
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = repo / path
            if path.is_file():
                manifests.add(path.resolve())
            else:
                gaps.append(str(path))

    for path in sorted((repo / "configs").rglob("*.json")):
        if (
            path.resolve()
            == (repo / "configs/stat_matched_source_panel_v1.json").resolve()
        ):
            continue
        data = read(path)
        collect(data)
        for group in groups(data):
            for pair in group["pairs"]:
                if not isinstance(pair, dict):
                    continue
                for role in ("source_a", "source_b"):
                    source = pair.get(role)
                    if not isinstance(source, dict) or "kind" not in source:
                        continue
                    defaults = group.get("source_defaults", group.get("defaults", {}))
                    if (
                        source["kind"] in core.RNG_RULES
                        and "seed" in source
                        and {**defaults, **source}.get("total_frames") == 1
                    ):
                        metadata.append(
                            {
                                "identity": str(path.relative_to(repo))
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
                            str(path.relative_to(repo))
                            + ":unsupported-or-non-single-frame-spec:"
                            + source["kind"]
                        )
        file_hashes[str(path)] = sha(path)
    for path in sorted(manifests):
        manifest = read(path)
        file_hashes[str(path)] = sha(path)
        for frame in manifest.get("frames", []):
            image_path = (path.parent / frame["path"]).resolve()
            if not image_path.is_file():
                gaps.append(str(image_path))
                continue
            if sha(image_path) != frame["sha256"]:
                raise ValueError("historical source manifest hash differs")
            with Image.open(image_path) as image:
                raw = np.asarray(image.convert("RGB"))
                digest = processor_cal.m.word_sha(raw)
            file_hashes[str(image_path)] = frame["sha256"]
            fingerprints.append(
                {"identity": str(image_path), "decoded_rgb_sha256": digest}
            )
    audit = core.source_spec_audit(spec, metadata)
    if audit["collisions"]:
        raise ValueError("new source metadata collides with historical inventory")
    return {
        "scope": "all_configuration_json_and_their_available_manifest_references_plus_all_runs_sources_manifests",
        "metadata": metadata,
        "decoded": fingerprints,
        "file_sha256": file_hashes,
        "gaps": gaps,
        "complete_available_inventory_scanned": True,
        "untouched_claim_released": False,
    }


def qualification(repo):
    cal.require_offline()
    core.qualify_core(repo)
    spec = read(repo / "configs/stat_matched_source_panel_v1.json")
    profile = read(repo / "configs/stat_matched_source_capture_preparation_v1.json")
    if (
        profile["execution_authorized"]
        or profile["study_id"] != "stat_matched_source_capture_preparation_v1"
    ):
        raise ValueError("preparation-only profile differs")
    for name, expected in profile["preserved_core_sha256"].items():
        if sha(repo / name) != expected:
            raise ValueError("previously reviewed core changed")
    legacy = processor_cal.qualify()
    observation = cal.qualify()
    public = repo / "examples/research_notes/0087_stat_matched_input_calibration"
    summary = read(public / "summary.json")
    for name, receipt in summary["ledgers"].items():
        if sha(public / name) != receipt["sha256"]:
            raise ValueError("published calibration ledger differs")
    old_root = repo / "runs/stat_matched_input_calibration_v1"
    maps = [read_line for read_line in _jsonl(public / "coordinate_maps.jsonl")]
    map_hashes = {}
    for row in maps:
        if sha(old_root / row["file"]) != row["file_sha256"]:
            raise ValueError("qualified map file differs")
        map_hashes[str(row["shift"])] = row["word_sha256"]
    historical = []
    for job in legacy["legacy"]["processor_jobs"]:
        raw, _ = processor_cal.old._checked_artifact(job["artifact"])
        historical.append(
            {
                "pixel_values_sha256": job["pixel_values_sha256"],
                "processor_non_pixel_payload": job["processor_non_pixel_payload"],
                "raw_word_sha256": processor_cal.m.word_sha(raw),
            }
        )
    first = observation["jobs"][0]
    original = read(Path(first["historical_run"]))
    audit = read(Path(first["historical_run"]).with_suffix(".input_audit.json"))
    manifest = read(Path(original["manifest_path"]))
    inv = source_inventory(repo, spec)
    return {
        "schema_version": 1,
        "software_test_only": False,
        "spec": spec,
        "geometry": legacy["config"],
        "features": read(repo / legacy["config"]["inherited_feature_config"]),
        "processor_non_pixel": legacy["config"]["future_calibration"][
            "synthetic_non_pixel_expected"
        ],
        "model_non_pixel": {
            k: v for k, v in audit["effective"].items() if k != "pixel_values"
        },
        "model_context_policy": original["context_policy"],
        "model_runtime": original["runtime"],
        "source_prompt": original["stream_events"][0]["prompt"],
        "delivery_condition": manifest["stimulus_condition"],
        "cache_shape_max": [1, 2, 397, 64],
        "map_word_sha256": map_hashes,
        "historical_processor": historical,
        "historical_source_hashes": inv["decoded"],
        "source_inventory": inv,
        "processor_qualification": legacy,
        "observation_qualification": observation,
        "repository": str(repo),
        "old_input_root": str(old_root),
    }


def _jsonl(path):
    import json

    return [json.loads(line) for line in path.read_bytes().splitlines()]


class NativeBackend:
    def __init__(self, frozen):
        self.frozen = frozen
        self.repo = Path(frozen["repository"])
        self.processor = None
        self.runtime = None

    def check_frozen(self, frozen):
        core.qualify_core(self.repo)
        for name, digest in frozen.get("implementation_sha256", {}).items():
            if sha(self.repo / name) != digest:
                raise ValueError("released implementation changed")
        for name, digest in frozen["source_inventory"]["file_sha256"].items():
            if sha(Path(name)) != digest:
                raise ValueError("historical source inventory changed")
        for name, digest in frozen["processor_qualification"]["code_sha256"].items():
            if sha(self.repo / name) != digest:
                raise ValueError("inherited frozen code changed")

    def map(self, shift):
        return np.load(
            Path(self.frozen["old_input_root"]) / f"map_{shift}.npy", allow_pickle=False
        )

    def historical_raw(self, index):
        job = self.frozen["processor_qualification"]["legacy"]["processor_jobs"][index]
        return processor_cal.old._checked_artifact(job["artifact"])[0]

    def synthetic_raw(self, key):
        return np.load(
            Path(self.frozen["old_input_root"]) / "cases" / key / "raw.npy",
            allow_pickle=False,
        )

    def process(self, raw, invoke):
        if self.processor is None:
            self.processor = processor_cal.load_processor()
        return processor_cal.native_payload(
            lambda **kwargs: invoke(lambda: self.processor(**kwargs)), raw
        )

    def render(self, root, pair, role):
        receipt = _materialize_source(
            pair[role],
            output_root=root / "generated",
            defaults=self.frozen["spec"]["sources"]["defaults"],
            pair_id=pair["pair_id"],
            role=role,
            overwrite=False,
        )
        path = Path(receipt["manifest_path"])
        manifest = read(path)
        frame = path.parent / manifest["frames"][0]["path"]
        with Image.open(frame) as image:
            raw = np.asarray(image.convert("RGB")).copy()
        return raw, {
            "manifest": str(path.relative_to(root)),
            "generator_spec": {
                k: v
                for k, v in manifest["stimulus_config"].items()
                if k != "stimulus_condition"
            },
        }

    def manifest(self, root, row):
        folder = root / row["folder"]
        if (folder / "manifest.json").exists():
            return
        Image.fromarray(load_array(root, row["raw"])).save(folder / "raw.png")
        condition = StimulusCondition.from_dict(self.frozen["delivery_condition"])
        build_external_frame_manifest(
            frames_dir=folder,
            output_path=folder / "manifest.json",
            condition=condition,
            fps=1,
            frame_glob="raw.png",
        )

    def calibrate(self, root, invoke):
        cal.run("preflight", root / "calibration_preflight")
        original = cal.run_cumulative_replay_probe

        def counted(*args, **kwargs):
            runtime = kwargs["mlx_runtime"]
            with count_decoder(runtime["model"].language_model) as internal:
                try:
                    return invoke(lambda: original(*args, **kwargs))
                finally:
                    from .stat_matched_source_storage import append

                    append(
                        root,
                        "calibration_internal_calls.jsonl",
                        {
                            "index": len(
                                _jsonl(root / "calibration_internal_calls.jsonl")
                            )
                            if (root / "calibration_internal_calls.jsonl").exists()
                            else 0,
                            "decoder": internal,
                        },
                    )

        with legacy_panel.retain_loaded_runtime() as loaded:
            with patch.object(cal, "run_cumulative_replay_probe", counted):
                cal.run(
                    "calibrate", root / "recalibration", root / "calibration_preflight"
                )
        self.runtime = loaded[0]
        if (
            cal.runtime_identity(self.runtime)
            != self.frozen["observation_qualification"]["runtime_identity"]
        ):
            raise ValueError("retained runtime differs")
        cal_pub.publish(root / "recalibration", root / "recalibration_verified")
        parity = legacy_panel.recalibration_parity(
            {
                "release": read(
                    self.repo / "configs/representation_correspondence_panel_v1.json"
                )
            },
            root,
        )
        write(root / "calibration_parity.json", parity)
        write(
            root / "runtime_binding.json",
            {
                "model_loads": 1,
                "same_loader_return_retained": True,
                "runtime_identity": cal.runtime_identity(self.runtime),
            },
        )

    def observe(self, root, folder, case, observed, invoke):
        if self.runtime is None:
            raise ValueError("same-process calibration must precede model capture")
        self.manifest(root, case)
        path = root / folder / "run.json"
        path.parent.mkdir(parents=True, exist_ok=False)
        manifest = root / case["folder"] / "manifest.json"
        projector = self.runtime["model"].mm_projector
        original = type(projector).__call__

        def save_decoder(counts):
            from .stat_matched_source_storage import append

            append(
                root,
                "source_internal_calls.jsonl",
                {"identity": folder.removeprefix("observations/"), "decoder": counts},
            )

        with (
            count_decoder(
                self.runtime["model"].language_model, on_exit=save_decoder
            ) as decoder_calls,
            cal.prepared_input_audit(
                importlib.import_module("mlx_vlm.generate"),
                audit_path=path.with_suffix(".input_audit.json"),
                context={
                    "manifest_path": str(manifest),
                    "manifest_sha256": sha(manifest),
                },
                expected_pixel_sha256=case["pixels"]["word_sha256"],
                expected_sizes=[[320, 320]],
            ) as prepared_calls,
        ):
            with (
                observe_projector(projector) if observed else nullcontext([]) as records
            ):
                invoke(
                    lambda: cal.run_cumulative_replay_probe(
                        cal.CumulativeReplayRunConfig(
                            manifest_path=manifest,
                            output_path=path,
                            model_id=self.frozen["spec"]["observation"]["model_id"],
                            max_frames=1,
                            max_tokens=2,
                            seed=20260604,
                            temperature=0,
                            probe_temperature=0,
                            probe_preset="default",
                            cache_summary_max_layers=None,
                            source_cache_only=True,
                            include_frame_artifacts=False,
                            cache_tensor_captures=tuple(
                                cal.parse_cache_tensor_capture_spec(t)
                                for t in cal.TARGETS
                            ),
                        ),
                        mlx_runtime=self.runtime,
                    )
                )
        run, audit = read(path), read(path.with_suffix(".input_audit.json"))
        event = run["stream_events"][0]
        sites = {}
        if observed:
            if len(records) != 1 or any(
                str(v.dtype) != "mlx.core.bfloat16" for v in records[0]
            ):
                raise ValueError("observed source dtype or target-call count differs")
            sites = {
                k: cal._to_float32_numpy(v)
                for k, v in zip(("encoder", "projector"), records[0], strict=True)
            }
        caches = {}
        for receipt in event["cache_tensor_artifacts"]:
            if receipt["source_dtype"] != "mlx.core.bfloat16":
                raise ValueError("source cache native dtype differs")
            with np.load(path.parent / receipt["path"], allow_pickle=False) as saved:
                caches[f"{receipt['layer_index']}:{receipt['tensor']}"] = saved[
                    receipt["array_key"]
                ]
        if (
            audit["native"] != audit["effective"]
            or audit["override_image_sizes"] is not None
        ):
            raise ValueError("native input override detected")
        return {
            "prepared": {
                "pixel_word_sha256": audit["effective"]["pixel_values"]["sha256"],
                "non_pixel": {
                    k: v for k, v in audit["effective"].items() if k != "pixel_values"
                },
            },
            "suffix": [s["token_id"] for s in event["generation"]["steps"]],
            "layout": event["cache_token_layout"],
            "sites": sites,
            "caches": caches,
            "restored": type(projector).__call__ is original,
            "observer_calls": len(records),
            "evidence": {
                "run": str(path.relative_to(root)),
                "audit": str(path.with_suffix(".input_audit.json").relative_to(root)),
            },
            "internal_calls": {
                "prepared_inputs": len(prepared_calls),
                "language_model_call": decoder_calls,
                "recorded_generation_steps": len(event["generation"]["steps"]),
            },
        }

    def finish(self, root):
        cal.assert_frozen(self.frozen["observation_qualification"])
        if (
            cal.runtime_identity(self.runtime)
            != self.frozen["observation_qualification"]["runtime_identity"]
        ):
            raise ValueError("runtime changed during acquisition")
        cal._freeze_model_snapshot(
            self.frozen["spec"]["observation"]["model_id"],
            root / "model_snapshot_after.json",
        )
        if (
            read(root / "model_snapshot_after.json")
            != self.frozen["observation_qualification"]["model_snapshot"]
        ):
            raise ValueError("model snapshot changed")


def verify_legacy_calibration(root, output, *, final=True):
    from .stat_matched_source_verify import check_legacy_cache_members

    check_legacy_cache_members(root)
    cal_pub.publish(root / "recalibration", output)
    stored = root / "recalibration_verified"
    if {
        str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
    } != {str(p.relative_to(stored)): sha(p) for p in stored.rglob("*") if p.is_file()}:
        raise ValueError("legacy calibration independent replay differs")
    frozen = read(root / "frozen.json")
    internal = _jsonl(root / "calibration_internal_calls.jsonl")
    if (
        len(internal) != 384
        or [r["index"] for r in internal] != list(range(384))
        or any(
            r["decoder"]["attempted"] != r["decoder"]["returned"]
            or r["decoder"]["returned"] < 1
            for r in internal
        )
    ):
        raise ValueError("historical decoder call accounting differs")
    parity = legacy_panel.recalibration_parity(
        {
            "release": read(
                Path(frozen["repository"])
                / "configs/representation_correspondence_panel_v1.json"
            )
        },
        root,
    )
    if (
        parity != read(root / "calibration_parity.json")
        or not read(root / "runtime_binding.json")["same_loader_return_retained"]
        or read(root / "runtime_binding.json")["model_loads"] != 1
        or (
            final
            and read(root / "model_snapshot_after.json")
            != frozen["observation_qualification"]["model_snapshot"]
        )
    ):
        raise ValueError("model calibration/binding/final snapshot differs")
