"""80px software-only lifecycle tests; no registered generator or native calls."""

import copy
import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe import stat_matched_input as matching
from fractal_vlm_state_probe import stat_matched_input_oracle as oracle
from fractal_vlm_state_probe import stat_matched_source as core
from fractal_vlm_state_probe import stat_matched_source_capture as capture
from fractal_vlm_state_probe import stat_matched_source_storage as storage
from fractal_vlm_state_probe import stat_matched_source_verify as verifier
from fractal_vlm_state_probe.cli import run_stat_matched_source as cli

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return storage.read(ROOT / name)


def pixels(raw):
    return np.ascontiguousarray(
        np.moveaxis(raw.astype(np.float32) / np.float32(255), -1, 0)[None]
    )


def toy_frozen():
    spec = read("configs/stat_matched_source_panel_v1.json")
    g = read(spec["construction"]["inherited_geometry"])
    g["construction"].update(
        raw_shape_hwc=[80, 80, 3],
        map_shape=[80, 80],
        grid=[8, 8],
        cell_shape_hw=[10, 10],
        tile_shape_hw=[1, 1],
        tile_offsets_yx=[[2, 2], [2, 6], [6, 2], [6, 6]],
        active_grid_rows=[0, 8],
        active_grid_cols=[0, 8],
        content_rows=[0, 80],
    )
    g["fixtures"]["noise_seed"] = 17
    g["fixtures"]["sentinel"]["raw_rows"] = [0, 10]
    g["fixtures"]["sentinel"]["raw_cols"] = [0, 10]
    g["future_calibration"]["processor_shape_nchw"] = [1, 3, 80, 80]
    spec["construction"].update(
        raw_shape_hwc=[80, 80, 3],
        payload_source_shape_hwc=[80, 80, 3],
        packed_shape_hwc=[16, 16, 3],
        payload_pixels=256,
    )
    spec["observation"]["sites"] = {"encoder": [3], "projector": [3]}
    spec["input_gates"]["expected_native_shape"] = [1, 3, 80, 80]
    spec["budget"]["reserve_bytes"] = 0
    spec["budget"]["required_launch_free_bytes"] = 0
    features = read("configs/nonlearned_processing_phase3_v1.json")
    features["windows"] = {s: {"small": 1, "large": 3} for s in matching.STARTS}
    other = {"image_sizes": [[80, 80]], "input_ids": [[-200]], "attention_mask": [[1]]}
    g["future_calibration"]["synthetic_non_pixel_expected"] = other
    history = []
    for i in range(128):
        raw = np.full((80, 80, 3), i, dtype=np.uint8)
        history.append(
            {
                "pixel_values_sha256": matching.word_sha(pixels(raw)),
                "processor_non_pixel_payload": other,
                "raw_word_sha256": matching.word_sha(raw),
            }
        )
    return {
        "software_test_only": True,
        "spec": spec,
        "geometry": g,
        "features": features,
        "processor_non_pixel": other,
        "model_non_pixel": {"image_sizes": [[80, 80]], "tokens": [[1, 2]]},
        "historical_processor": history,
        "historical_source_hashes": [],
        "map_word_sha256": {
            str(s): matching.word_sha(oracle.coordinate_map(g, s)) for s in range(4)
        },
        "cache_shape_max": [1, 2, 4, 2],
    }


class FakeBackend:
    def __init__(self, frozen, fail_processor=None, fail_model=None, double=False):
        self.frozen = frozen
        self.processors = self.models = self.sources = 0
        self.fail_processor, self.fail_model, self.double = (
            fail_processor,
            fail_model,
            double,
        )

    def check_frozen(self, frozen):
        assert frozen["software_test_only"]

    def map(self, shift):
        return oracle.coordinate_map(self.frozen["geometry"], shift)

    def historical_raw(self, index):
        return np.full((80, 80, 3), index, dtype=np.uint8)

    def synthetic_raw(self, key):
        _, name, shift = next(
            x for x in matching.cases(self.frozen["geometry"]) if x[0] == key
        )
        return oracle.fixture(self.frozen["geometry"], name, shift, key == "sentinel")

    def process(self, raw, invoke):
        def operation():
            self.processors += 1
            if self.processors == self.fail_processor:
                raise RuntimeError("toy processor interruption")
            return pixels(raw), copy.deepcopy(self.frozen["processor_non_pixel"])

        result = invoke(operation)
        if self.double:
            invoke(operation)
        return result

    def render(self, root, pair, role):
        self.sources += 1
        y, x = np.indices((80, 80))
        seed = pair[role]["seed"]
        raw = np.stack(
            (
                (x + seed) % 256,
                (y * 3 + seed // 256) % 256,
                (x + y * 7 + seed // 65536) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        return raw, {"software_fixture": True}

    def manifest(self, root, row):
        pass

    def model_operation(self, function):
        self.models += 1
        if self.models == self.fail_model:
            raise RuntimeError("toy model interruption")
        return function()

    def calibrate(self, root, invoke):
        for _ in range(384):
            invoke(lambda: self.model_operation(lambda: None))
        storage.write(
            root / "software_calibration.json",
            {"simulated_contexts": 384, "native_model_calls": 0},
        )

    def observe(self, root, folder, case, observed, invoke):
        def operation():
            value = int(case["raw"]["word_sha256"][:4], 16) % 16
            arrays = (
                {
                    k: np.array([value, value + 1, value + 2], dtype=np.float32)
                    for k in ("encoder", "projector")
                }
                if observed
                else {}
            )
            caches = {
                k: np.full((1, 2, 4, 2), value, dtype=np.float32)
                for k in ("1:keys", "12:keys", "23:values")
            }
            return {
                "prepared": {
                    "pixel_word_sha256": case["pixels"]["word_sha256"],
                    "non_pixel": self.frozen["model_non_pixel"],
                },
                "sites": arrays,
                "caches": caches,
                "suffix": [1, 2, 3],
                "layout": {"sequence": 4},
                "evidence": {"software_fixture": True},
                "restored": True,
                "observer_calls": int(observed),
                "internal_calls": {
                    "prepared_inputs": 1,
                    "language_model_call": {"attempted": 3, "returned": 3},
                    "recorded_generation_steps": 3,
                },
            }

        return invoke(lambda: self.model_operation(operation))

    def finish(self, root):
        storage.write(
            root / "software_final.json",
            {"simulated_contexts": self.models, "native_model_calls": 0},
        )


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    parent = tmp_path_factory.mktemp("source_capture_software")
    frozen = toy_frozen()
    backend = FakeBackend(frozen)
    root = parent / "capture"
    result = capture.acquire(root, frozen, backend)
    assert result["status"] == "COMPLETE_UNVERIFIED"
    assert (backend.processors, backend.models, backend.sources) == (665, 920, 64)
    return root, parent, frozen


def test_complete_capture_two_independent_verifications_are_byte_identical(complete):
    root, parent, _ = complete
    first = verifier.verify_capture(
        root, parent / "verified1", allow_software_fixture=True
    )
    second = verifier.verify_capture(
        root, parent / "verified2", allow_software_fixture=True
    )
    assert first == second and first["status"] == "VERIFIED_SOFTWARE_FIXTURE"
    assert storage.inventory(parent / "verified1") == storage.inventory(
        parent / "verified2"
    )
    assert first["proof"]["candidate_cells"] == 512
    assert first["proof"]["model_cells"] == 536
    seal = storage.read(root / "reference_seal.json")
    assert (
        seal["artifact_bytes_independently_verified"]
        and seal["proof"]["candidate_cells"] == 256
    )
    assert seal["last_context"] == 664 and seal["first_test_context"] == 665
    analysis = verifier.analyze_verified(
        root,
        parent / "analysis",
        parent / "verified1",
        parent / "verified2",
        allow_software_fixture=True,
    )
    assert analysis["status"] == "SOFTWARE_PROFILES" and analysis["profiles"] == 18
    assert len(storage.ledger(parent / "analysis", "profiles.jsonl")) == 18
    assert len(storage.ledger(parent / "analysis", "spectra.jsonl")) == 512
    with pytest.raises(ValueError, match="two separate"):
        verifier.analyze_verified(
            root,
            parent / "bad_analysis",
            parent / "verified1",
            parent / "verified1",
            allow_software_fixture=True,
        )


@pytest.mark.parametrize(
    "processor_at,model_at",
    [(1, None), (130, None), (410, None), (None, 1), (None, 409), (None, 665)],
)
def test_first_failure_retained_no_retry_or_test_leakage(
    tmp_path, processor_at, model_at
):
    frozen = toy_frozen()
    backend = FakeBackend(frozen, fail_processor=processor_at, fail_model=model_at)
    root = tmp_path / "capture"
    with pytest.raises(RuntimeError):
        capture.acquire(root, frozen, backend)
    failure = storage.read(root / "failure.json")
    assert (
        failure["status"] == "FAILED_RETAINED" and failure["primary_results_withheld"]
    )
    assert not (root / "capture.json").exists()
    result = verifier.verify_capture(
        root, tmp_path / "failure_audit", allow_software_fixture=True
    )
    assert result["status"] == "FAILED_CAPTURE_AUDITED"
    with pytest.raises(FileExistsError):
        capture.acquire(root, frozen, backend)
    if processor_at in (1, 130) or model_at in (1, 409):
        assert backend.sources <= 32
    count = failure["counts"]["processor" if processor_at else "model"]
    assert count["attempted"] == (processor_at or model_at)
    assert count["returned"] + 1 == count["attempted"]


def test_extra_processor_call_is_refused_before_execution(tmp_path):
    frozen = toy_frozen()
    backend = FakeBackend(frozen, double=True)
    with pytest.raises(ValueError, match="extra backend call"):
        capture.acquire(tmp_path / "capture", frozen, backend)
    assert backend.processors == 1
    assert storage.read(tmp_path / "capture/failure.json")["counts"]["processor"] == {
        "attempted": 1,
        "returned": 1,
    }


def test_cli_capture_is_closed_without_separate_execution_release(
    tmp_path, monkeypatch
):
    called = []
    monkeypatch.setattr(cli, "qualification", lambda *args: called.append(True))
    with pytest.raises(ValueError, match="separate explicit acquisition release"):
        cli.run("capture", tmp_path / "capture")
    assert not called and not (tmp_path / "capture").exists()


def test_native_verifier_cannot_promote_software_fixture(complete, tmp_path):
    root, _, _ = complete
    with pytest.raises(ValueError, match="software fixture"):
        verifier.verify_capture(root, tmp_path / "verify")
    assert (
        storage.read(tmp_path / "verify/verification_failure.json")["status"]
        == "INVALID_CAPTURE"
    )


def test_safe_storage_paths_and_array_corruption_fail_closed(tmp_path):
    for path in ("../escape.npy", "/tmp/escape.npy"):
        with pytest.raises(ValueError):
            storage.inside(tmp_path, path)
    receipt = storage.save_array(
        tmp_path, "a.npy", np.array([-0.0, 1.0], dtype=np.float32)
    )
    assert np.signbit(storage.load_array(tmp_path, receipt)[0])
    with pytest.raises(FileExistsError):
        storage.save_array(tmp_path, "a.npy", np.ones(2))
    (tmp_path / "a.npy").write_bytes(b"broken")
    with pytest.raises(ValueError, match="array file hash"):
        storage.load_array(tmp_path, receipt)


def test_preserved_126_files_and_no_acquisition_release():
    profile = read("configs/stat_matched_source_capture_preparation_v1.json")
    assert not profile["execution_authorized"]
    assert (
        profile["registered_sources_generated"]
        == profile["native_processor_calls"]
        == profile["model_contexts"]
        == 0
    )
    for name, digest in profile["preserved_core_sha256"].items():
        assert storage.sha(ROOT / name) == digest
    assert core.qualify_core(ROOT)["inherited_files"] == 118


def replace_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def replace_ledger(root, name, rows):
    (root / name).write_text(
        "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )


def recapture_manifest(root, *, rechain=False):
    if rechain:
        chain = "0" * 64
        events = storage.ledger(root, "operations.jsonl")
        for i, row in enumerate(events, 1):
            row.pop("digest")
            row.update(sequence=i, previous=chain)
            if row["kind"] == "reference_seal":
                row["sha256"] = storage.sha(root / "reference_seal.json")
            chain = hashlib.sha256(
                json.dumps(
                    row, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            row["digest"] = chain
        replace_ledger(root, "operations.jsonl", events)
    value = storage.read(root / "capture.json")
    if rechain:
        value["last_operation_digest"] = chain
    value["receipts"] = storage.inventory(root)
    replace_json(root / "capture.json", value)


def change_array(root, receipt, values):
    path = root / receipt["path"]
    np.save(path, values, allow_pickle=False)
    receipt.update(
        sha256=storage.sha(path),
        word_sha256=matching.word_sha(values),
        shape=list(values.shape),
        dtype=str(values.dtype),
    )


@pytest.mark.parametrize(
    "kind",
    [
        "unexpected_file",
        "source",
        "spectrum",
        "reference",
        "operation_order",
        "unreferenced_binary",
        "map_shape",
    ],
)
def test_full_file_verifier_rejects_corruption_even_with_refreshed_manifest(
    complete, tmp_path, kind
):
    original, _, _ = complete
    root = tmp_path / "copy"
    shutil.copytree(original, root)
    if kind == "unexpected_file":
        (root / "unregistered.npy").write_bytes(b"unregistered")
    elif kind == "unreferenced_binary":
        np.save(root / "not_a_registered_artifact.npy", np.ones(2))
        recapture_manifest(root)
    elif kind == "map_shape":
        path = root / "maps/s1.npy"
        np.save(path, np.load(path, allow_pickle=False).reshape(-1))
        recapture_manifest(root)
    elif kind == "source":
        rows = storage.ledger(root, "sources.jsonl")
        raw = storage.load_array(root, rows[0]["raw"])
        raw[0, 0, 0] ^= 1
        change_array(root, rows[0]["raw"], raw)
        rows[0]["decoded_rgb_sha256"] = matching.word_sha(raw)
        replace_ledger(root, "sources.jsonl", rows)
        recapture_manifest(root)
    elif kind == "spectrum":
        rows = storage.ledger(root, "candidates.jsonl")
        rows[0]["spectra"]["raw_rgb"]["spectral_centroid"] += 0.1
        replace_ledger(root, "candidates.jsonl", rows)
        recapture_manifest(root)
    elif kind == "reference":
        rows = storage.read(root / "family_references.json")
        row = next(r for r in rows if r["unit"] is not None)
        change_array(root, row["unit"], -storage.load_array(root, row["unit"]))
        replace_json(root / "family_references.json", rows)
        seal = storage.read(root / "reference_seal.json")
        seal["family_references_sha256"] = storage.sha(root / "family_references.json")
        replace_json(root / "reference_seal.json", seal)
        recapture_manifest(root, rechain=True)
    else:
        rows = storage.ledger(root, "operations.jsonl")
        rows[:4] = rows[2:4] + rows[:2]
        replace_ledger(root, "operations.jsonl", rows)
        recapture_manifest(root, rechain=True)
    with pytest.raises(ValueError):
        verifier.verify_capture(root, tmp_path / "invalid", allow_software_fixture=True)
    assert (
        storage.read(tmp_path / "invalid/verification_failure.json")["status"]
        == "INVALID_CAPTURE"
    )


def test_native_render_adapter_uses_existing_manifest_contract_only(
    tmp_path, monkeypatch
):
    from fractal_vlm_state_probe import stat_matched_source_native as native
    from fractal_vlm_state_probe.control_stimulus import GeneratedControlSpec
    from PIL import Image

    frozen = toy_frozen()
    frozen["spec"]["sources"]["defaults"].update(width=80, height=80)
    pair = copy.deepcopy(frozen["spec"]["sources"]["pairs"][0])
    pair["source_a"]["seed"] = 17
    frozen["repository"] = str(ROOT)
    expected = np.zeros((80, 80, 3), dtype=np.uint8)

    def materialize(source, *, output_root, defaults, pair_id, role, overwrite):
        assert source["seed"] == 17 and not overwrite
        folder = output_root / "toy"
        folder.mkdir(parents=True)
        Image.fromarray(expected).save(folder / "frame.png")
        data = GeneratedControlSpec(**defaults, **source).to_dict()
        data["stimulus_condition"] = {"software_fixture": True}
        storage.write(
            folder / "manifest.json",
            {"frames": [{"path": "frame.png"}], "stimulus_config": data},
        )
        return {"manifest_path": str(folder / "manifest.json")}

    monkeypatch.setattr(native, "_materialize_source", materialize)
    backend = native.NativeBackend(frozen)
    actual, provenance = backend.render(tmp_path, pair, "source_a")
    assert actual.tobytes() == expected.tobytes()
    assert "stimulus_condition" not in provenance["generator_spec"]
    assert provenance["generator_spec"]["seed"] == 17


def test_unsigned_and_wrong_release_fail_before_qualification(tmp_path, monkeypatch):
    path = tmp_path / "release.json"
    storage.write(path, {"approved_for_execution": True})
    invoked = []
    monkeypatch.setattr(cli, "qualification", lambda *args: invoked.append(True))
    with pytest.raises(ValueError, match="approved release binding"):
        cli.run("capture", tmp_path / "capture", release=path)
    assert not invoked


def test_decoder_counter_preserves_exact_return_and_restores_on_failure():
    from fractal_vlm_state_probe.stat_matched_source_native import count_decoder

    class Decoder:
        def __call__(self, result, fail=False):
            if fail:
                raise RuntimeError("toy decoder failed")
            return result

    model, other = Decoder(), Decoder()
    original = Decoder.__call__
    result = object()
    with pytest.raises(RuntimeError):
        with count_decoder(model) as counts:
            assert model(result) is result
            assert other(result) is result
            with pytest.raises(ValueError, match="nested decoder"):
                with count_decoder(model):
                    pass
            model(result, fail=True)
    assert Decoder.__call__ is original
    assert counts == {"attempted": 2, "returned": 1}


def test_reserve_is_checked_before_arrays_and_before_calls(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        storage.shutil, "disk_usage", lambda root: SimpleNamespace(free=0)
    )
    with pytest.raises(ValueError, match="saving array would breach"):
        storage.save_array(tmp_path, "blocked.npy", np.ones(2), reserve=8)
    invoked = []
    recorder = storage.Recorder(tmp_path, 8)
    with pytest.raises(ValueError, match="disk reserve"):
        recorder.call("model", "blocked", lambda: invoked.append(True))
    assert not invoked and recorder.counts["model"]["attempted"] == 0


def test_capture_payload_bound_includes_native_duplicate_evidence_and_references():
    spec = read("configs/stat_matched_source_panel_v1.json")
    profile = read("configs/stat_matched_source_capture_preparation_v1.json")
    base = core.disk_budget(spec)["enumerated_payload_bytes"]
    extra = (
        536 * 3 * 1 * 2 * 397 * 64 * 4
        + 128 * 320 * 320 * 3
        + (512 + 25) * 320 * 320 * 3
        + 64 * 240 * 320 * 3
        + 4 * 320 * 320 * 4
        + 3 * 8 * (256 * 3072 + 256 * 896) * 8
        + 12 * 8 * 192 * 8
    )
    assert (
        base + extra
        < profile["enumerated_payload_bound_bytes"]
        <= spec["budget"]["payload_allowance_bytes"]
    )


def test_native_inventory_records_available_config_manifest_and_missingness(tmp_path):
    from fractal_vlm_state_probe.stat_matched_source_native import source_inventory
    from PIL import Image

    (tmp_path / "configs").mkdir()
    (tmp_path / "external").mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
        tmp_path / "external/a.png"
    )
    storage.write(
        tmp_path / "external/manifest.json",
        {
            "frames": [
                {"path": "a.png", "sha256": storage.sha(tmp_path / "external/a.png")}
            ]
        },
    )
    storage.write(
        tmp_path / "configs/history.json",
        {
            "pairs": 5,
            "manifest": "external/manifest.json",
            "missing": "missing/manifest.json",
        },
    )
    result = source_inventory(
        tmp_path, read("configs/stat_matched_source_panel_v1.json")
    )
    assert len(result["decoded"]) == 1 and result["gaps"]
    assert not result["untouched_claim_released"]


def test_release_gate_requires_committed_code_and_separate_publication_receipts(
    tmp_path,
):
    profile = read("configs/stat_matched_source_capture_preparation_v1.json")
    spec = read("configs/stat_matched_source_panel_v1.json")
    old = read(spec["preserve_inherited_frozen_hashes_from"])
    core_profile = read("configs/stat_matched_source_core_v1.json")
    names = (
        set(old["code_sha256"])
        | set(core_profile["registration_sha256"])
        | set(profile["preserved_core_sha256"])
        | set(profile["implementation_files"])
    )
    copies = names | {spec["preserve_inherited_frozen_hashes_from"]}
    for name in copies:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)

    def git(*args):
        return (
            subprocess.check_output(
                ["git", *args], cwd=tmp_path, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Software Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "software fixture implementation",
    )
    commit = git("rev-parse", "HEAD")
    code_receipt = tmp_path / "software_code_public_receipt.json"
    storage.write(
        code_receipt,
        {
            "state": "MERGED",
            "url": "https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/999999",
            "mergeCommit": {"oid": commit},
        },
    )
    release_path = tmp_path / "configs/stat_matched_source_acquisition_release_v1.json"
    release = {
        "study_id": profile["required_separate_release_id"],
        "approved_for_execution": True,
        "preparation_sha256": storage.sha(tmp_path / cli.PROFILE),
        "code_sha256": {n: storage.sha(tmp_path / n) for n in names},
        "implementation_commit": commit,
        "public_receipt": str(code_receipt),
        "public_receipt_sha256": storage.sha(code_receipt),
    }
    storage.write(release_path, release)
    with pytest.raises(ValueError, match="commit the separate"):
        cli.require_release(tmp_path, release_path)
    git("add", str(release_path.relative_to(tmp_path)))
    git(
        "-c",
        "user.name=Software Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "software fixture approval not an experiment",
    )
    public = tmp_path / "software_release_public_receipt.json"
    storage.write(
        public,
        {
            "state": "MERGED",
            "mergedAt": "software_fixture_not_actual_publication",
            "url": "https://github.com/RyoSpiralArchitect/fractal-vlm-state-probe/pull/999999",
            "mergeCommit": {"oid": git("rev-parse", "HEAD")},
        },
    )
    assert cli.require_release(tmp_path, release_path, public) == release
    changed = tmp_path / "scripts/run_stat_matched_source.py"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="commit the complete"):
        cli.require_release(tmp_path, release_path, public)


@pytest.mark.parametrize("mutation", [None, "bad_length", "extra_member"])
def test_native_saved_evidence_layout_parser_without_runtime(tmp_path, mutation):
    from PIL import Image

    frozen = toy_frozen()
    frozen.update(
        model_context_policy={"source_cache_only": True},
        model_runtime={"software_fixture": True},
        source_prompt="ACK software parser fixture",
        model_non_pixel={"input_ids": {"values": [[0] * 140]}},
    )
    identity = "software/s0/mm"
    folder = "inputs/" + identity
    (tmp_path / folder).mkdir(parents=True)
    raw = np.zeros((80, 80, 3), dtype=np.uint8)
    Image.fromarray(raw).save(tmp_path / folder / "raw.png")
    case = {
        "folder": folder,
        "raw": storage.save_array(tmp_path, folder + "/raw.npy", raw),
        "pixels": {"word_sha256": matching.word_sha(pixels(raw))},
    }
    png_sha = storage.sha(tmp_path / folder / "raw.png")
    manifest_path = tmp_path / folder / "manifest.json"
    storage.write(manifest_path, {"frames": [{"sha256": png_sha}]})
    model_folder = tmp_path / "observations" / identity
    model_folder.mkdir(parents=True)
    layout = {
        "cache_position_layout": {
            "available": True,
            "image_partition_available": True,
            "image_position_runs": [{"start": 42, "end": 297, "length": 256}],
            "cache_sequence_length": 397,
            "image_position_count": 256,
            "pre_image_position_count": 42,
            "post_image_position_count": 99,
            "expansion_delta": 255,
            "processor_token_count": 142,
        }
    }
    model = {
        "identity": identity,
        "evidence": {
            "run": "observations/" + identity + "/run.json",
            "audit": "observations/" + identity + "/run.input_audit.json",
        },
        "caches": {},
        "layout": layout,
        "suffix": [1, 2, 3],
    }
    artifacts = []
    for target in ("1:keys", "12:keys", "23:values"):
        cache = np.zeros((1, 2, 397, 64), dtype=np.float32)
        name = target.replace(":", "_")
        model["caches"][target] = storage.save_array(
            tmp_path, "canonical/" + name + ".npy", cache
        )
        np.savez(model_folder / (name + ".npz"), tensor=cache)
        layer, tensor = target.split(":")
        artifacts.append(
            {
                "path": name + ".npz",
                "sha256": storage.sha(model_folder / (name + ".npz")),
                "layer_index": int(layer),
                "tensor": tensor,
                "array_key": "tensor",
                "source_dtype": "mlx.core.bfloat16",
                "dtype": "float32",
                "shape": list(cache.shape),
                "effective_sequence_length": 397,
            }
        )
    if mutation == "bad_length":
        layout["cache_position_layout"]["cache_sequence_length"] = 4
        model["caches"]["1:keys"]["shape"][2] = 4
    elif mutation == "extra_member":
        path = model_folder / artifacts[0]["path"]
        np.savez(path, tensor=cache, unregistered=np.array([np.nan]))
        artifacts[0]["sha256"] = storage.sha(path)
    event = {
        "prompt": frozen["source_prompt"],
        "frame_sha256s": [png_sha],
        "cache_token_layout": layout,
        "generation": {"steps": [{"token_id": n} for n in model["suffix"]]},
        "cache_tensor_artifacts": artifacts,
    }
    storage.write(
        model_folder / "run.json",
        {
            "model_id": frozen["spec"]["observation"]["model_id"],
            "dry_run": False,
            "context_policy": frozen["model_context_policy"],
            "runtime": frozen["model_runtime"],
            "reproducibility": {"seed": 20260604},
            "manifest_path": str(manifest_path),
            "stream_events": [event],
        },
    )
    effective = {
        **frozen["model_non_pixel"],
        "pixel_values": {"sha256": case["pixels"]["word_sha256"]},
    }
    storage.write(
        model_folder / "run.input_audit.json",
        {
            "context": {
                "manifest_path": str(manifest_path),
                "manifest_sha256": storage.sha(manifest_path),
            },
            "override_image_sizes": None,
            "native": effective,
            "effective": effective,
        },
    )
    if mutation:
        message = (
            "image-token/cache layout"
            if mutation == "bad_length"
            else "archive member coverage"
        )
        with pytest.raises(ValueError, match=message):
            verifier._native_evidence(tmp_path, model, case, frozen)
    else:
        verifier._native_evidence(tmp_path, model, case, frozen)


def test_reference_calibration_does_not_require_future_outer_snapshot(
    tmp_path, monkeypatch
):
    from fractal_vlm_state_probe import stat_matched_source_native as native

    stored = tmp_path / "recalibration_verified"
    stored.mkdir()
    storage.write(stored / "summary.json", {"software_parser_fixture": True})
    monkeypatch.setattr(
        native.cal_pub, "publish", lambda root, output: shutil.copytree(stored, output)
    )
    monkeypatch.setattr(
        native.legacy_panel, "recalibration_parity", lambda *args: {"fixture": True}
    )
    storage.write(
        tmp_path / "frozen.json",
        {
            "repository": str(ROOT),
            "observation_qualification": {"model_snapshot": {"fixture": "final"}},
        },
    )
    storage.write(tmp_path / "calibration_parity.json", {"fixture": True})
    storage.write(
        tmp_path / "runtime_binding.json",
        {"same_loader_return_retained": True, "model_loads": 1},
    )
    for i in range(384):
        storage.append(
            tmp_path,
            "calibration_internal_calls.jsonl",
            {"index": i, "decoder": {"attempted": 3, "returned": 3}},
        )
    native.verify_legacy_calibration(tmp_path, tmp_path / "prefix", final=False)
    with pytest.raises(FileNotFoundError):
        native.verify_legacy_calibration(tmp_path, tmp_path / "missing_final")
    storage.write(tmp_path / "model_snapshot_after.json", {"fixture": "wrong"})
    with pytest.raises(ValueError, match="final snapshot"):
        native.verify_legacy_calibration(tmp_path, tmp_path / "wrong_final")


def test_legacy_cache_archive_rejects_unregistered_members(tmp_path):
    (tmp_path / "recalibration").mkdir()
    storage.write(tmp_path / "family_references.json", [])
    storage.append(
        tmp_path / "recalibration",
        "cells.jsonl",
        {
            "path": "calibration/cell.json",
            "tensor_artifacts": [{"path": "extra.npz", "array_key": "tensor"}],
        },
    )
    folder = tmp_path / "recalibration/calibration"
    folder.mkdir()
    np.savez(folder / "extra.npz", tensor=np.zeros(1), extra=np.array([np.nan]))
    with pytest.raises(ValueError, match="legacy cache archive member coverage"):
        verifier.binary_coverage(tmp_path, {"software_test_only": False})


def test_saved_preflight_binding_checks_zero_calls_and_exact_frozen_contract():
    frozen = toy_frozen()
    prior = json.dumps(frozen, sort_keys=True)
    receipt = {
        "status": "PASS_FOR_APPROVED_RELEASE",
        "model_weights_loaded": False,
        "new_processor_calls": 0,
        "new_model_contexts": 0,
        "available_bytes": 1024,
        "frozen_sha256": hashlib.sha256(prior.encode()).hexdigest(),
    }
    text = json.dumps(receipt)
    frozen["outer_preflight"] = {
        "receipt_text": text,
        "receipt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "frozen_text": prior,
    }
    verifier.validate_preflight_binding(frozen)
    frozen["spec"]["sources"]["pairs"][0]["source_a"]["seed"] += 1
    with pytest.raises(ValueError, match="approved preflight binding"):
        verifier.validate_preflight_binding(frozen)


def test_changing_fixture_flag_cannot_bypass_registered_native_contract():
    frozen = toy_frozen()
    frozen.update(
        software_test_only=False,
        preparation_profile_sha256=storage.sha(ROOT / cli.PROFILE),
    )
    prior = json.dumps(frozen)
    receipt = {
        "status": "PASS_FOR_APPROVED_RELEASE",
        "model_weights_loaded": False,
        "new_processor_calls": 0,
        "new_model_contexts": 0,
        "available_bytes": 100 * 2**30,
        "frozen_sha256": hashlib.sha256(prior.encode()).hexdigest(),
    }
    text = json.dumps(receipt)
    frozen["outer_preflight"] = {
        "receipt_text": text,
        "receipt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "frozen_text": prior,
    }
    with pytest.raises(ValueError, match="registered native contract"):
        verifier.validate_native_frozen(frozen)


def test_preflight_failure_retained_without_native_work(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    (tmp_path / "configs").mkdir()
    for name in (cli.PROFILE, "configs/stat_matched_source_panel_v1.json"):
        shutil.copyfile(ROOT / name, tmp_path / name)
    monkeypatch.setattr(cli, "REPO", tmp_path)
    invoked = []

    def refuse(*args):
        invoked.append(True)
        raise ValueError("software-only qualification failure")

    monkeypatch.setattr(cli, "qualification", refuse)
    import shutil as disk
    from types import SimpleNamespace

    monkeypatch.setattr(
        disk, "disk_usage", lambda *args: SimpleNamespace(free=100 * 2**30)
    )
    output = tmp_path / read(cli.PROFILE)["preparation_root"]
    with pytest.raises(ValueError, match="software-only qualification"):
        cli.run("preflight", output)
    failure = storage.read(output / "preflight_failure.json")
    assert (
        failure["status"] == "FAILED_PREFLIGHT_RETAINED"
        and not failure["model_weights_loaded"]
    )
    assert failure["new_processor_calls"] == failure["new_model_contexts"] == 0
    assert invoked == [True] and not (output / "preflight.json").exists()


@pytest.mark.parametrize("note", ["0063", "0068"])
def test_historical_summary_is_hashed_before_trusting_its_contents(tmp_path, note):
    path = tmp_path / (note + ".json")
    storage.write(path, {"runtime_identity": {"fixture": "original"}})
    digest = storage.sha(path)
    assert verifier.anchored_summary(tmp_path, path.name, digest)[
        "runtime_identity"
    ] == {"fixture": "original"}
    replace_json(path, {"runtime_identity": {"fixture": "substituted"}})
    with pytest.raises(ValueError, match="historical summary hash"):
        verifier.anchored_summary(tmp_path, path.name, digest)


@pytest.mark.parametrize(
    "omission", [None, "decoded", "file_sha256", "metadata", "gaps"]
)
def test_independent_inventory_recounts_coverage_and_ignores_only_current_acquisition(
    tmp_path, omission
):
    from fractal_vlm_state_probe.stat_matched_source_native import source_inventory
    from PIL import Image

    (tmp_path / "configs").mkdir()
    historical = tmp_path / "runs/historical/sources/fixture"
    historical.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(historical / "frame.png")
    storage.write(
        historical / "manifest.json",
        {
            "frames": [
                {"path": "frame.png", "sha256": storage.sha(historical / "frame.png")}
            ]
        },
    )
    storage.write(
        tmp_path / "configs/fixture.json",
        {
            "defaults": {"width": 2, "height": 2, "total_frames": 1},
            "pairs": [
                {
                    "pair_id": "old",
                    "source_a": {"kind": "white_noise", "seed": 17},
                    "source_b": {"kind": "unsupported_fixture", "seed": 17},
                }
            ],
            "missing": "missing/manifest.json",
        },
    )
    spec = read("configs/stat_matched_source_panel_v1.json")
    inv = source_inventory(tmp_path, spec)
    assert inv["decoded"] and inv["file_sha256"] and inv["metadata"] and inv["gaps"]
    frozen = {
        "spec": spec,
        "source_inventory": inv,
        "historical_source_hashes": copy.deepcopy(inv["decoded"]),
    }
    current = tmp_path / "runs/current"
    (current / "generated/sources/current").mkdir(parents=True)
    storage.write(
        current / "generated/sources/current/manifest.json", {"not_historical": True}
    )
    if omission:
        inv[omission] = {} if omission == "file_sha256" else []
        if omission == "decoded":
            frozen["historical_source_hashes"] = []
        with pytest.raises(ValueError, match="inventory coverage"):
            verifier.verify_source_inventory(tmp_path, current, frozen)
    else:
        verifier.verify_source_inventory(tmp_path, current, frozen)
