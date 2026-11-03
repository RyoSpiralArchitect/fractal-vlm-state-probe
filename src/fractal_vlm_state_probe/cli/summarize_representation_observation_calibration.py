from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.cli.run_representation_observation_calibration import (
    assert_frozen,
)
from fractal_vlm_state_probe.cli.summarize_controlled_overlap_cache_calibration import (
    checked_array,
    verify_pair,
)
from fractal_vlm_state_probe.controlled_overlap_cache import (
    TARGETS,
    independent_exact_arrays,
)
from fractal_vlm_state_probe.representation_observation import (
    STAGES,
    array_receipts,
    require_complete,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json


def checked_representations(root, receipt, config):
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or _sha(path) != receipt["sha256"]:
        raise ValueError("representation path/hash differs")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    actual = array_receipts(arrays, config)
    if actual != receipt["arrays"]:
        raise ValueError("representation array receipt differs")
    return actual


def publish(root, output):
    if output.exists():
        raise FileExistsError("fresh publication root required")
    if (root / "execution_failure.json").exists():
        raise ValueError("failed calibration cannot be published as passing")
    summary, frozen = (
        _read(root / n) for n in ("summary.json", "frozen_specification.json")
    )
    expected = {
        "status": "PASS",
        "calibration_source_contexts": 384,
        "historical_source_cells": 128,
        "historical_tensor_comparisons": 1152,
        "on_off_tensor_comparisons": 768,
        "observed_source_cells": 128,
        "representation_arrays": 256,
        "new_model_forwards": 384,
        "new_experimental_panel_cells": 0,
        "new_direct_probe_cells": 0,
        "new_correspondence_scores": 0,
        "experimental_panel_released": False,
        "confirmation_cohort": "not_generated_not_sealed",
    }
    required_receipts = {
        "frozen_specification.json",
        "model_snapshot.json",
        "model_snapshot_after.json",
        "preflight_receipt.json",
        "runtime_identity.json",
        "cells.jsonl",
        "capture_registry.json",
    }
    if (
        any(summary.get(k) != v for k, v in expected.items())
        or set(summary["receipts"]) != required_receipts
    ):
        raise ValueError("calibration completion/count/claim boundary differs")
    for name, digest in summary["receipts"].items():
        if _sha(root / name) != digest:
            raise ValueError("calibration receipt changed")
    assert_frozen(frozen)
    if summary["registered_design_commit"] != frozen["registered_design_commit"]:
        raise ValueError("registration commit differs")
    for name in ("model_snapshot.json", "model_snapshot_after.json"):
        if _read(root / name) != frozen["model_snapshot"]:
            raise ValueError("model snapshot differs")
    if _read(root / "runtime_identity.json") != frozen["runtime_identity"]:
        raise ValueError("runtime identity differs")
    pre_receipt = _read(root / "preflight_receipt.json")
    pre_root = Path(pre_receipt["path"])
    pre = _read(pre_root / "preflight.json")
    if (
        _sha(pre_root / "preflight.json") != pre_receipt["sha256"]
        or (pre_root / "execution_failure.json").exists()
        or _read(pre_root / "frozen_specification.json") != frozen
        or _sha(pre_root / "frozen_specification.json")
        != pre["frozen_specification_sha256"]
        or _sha(pre_root / "model_snapshot.json") != pre["model_snapshot_sha256"]
        or pre["status"] != "PASS"
        or pre["new_model_forwards"] != 0
        or pre["model_weights_loaded"]
        or pre["observer_installed"]
        or pre["experimental_panel_released"]
    ):
        raise ValueError("no-model preflight differs")
    rows = [
        json.loads(line) for line in (root / "cells.jsonl").read_text().splitlines()
    ]
    jobs, config = frozen["jobs"], frozen["observation_config"]
    validate_config(config)
    require_complete(rows, jobs)
    registry = _read(root / "capture_registry.json")
    if len(registry) != 384 or set(registry) != {r["path"] for r in rows}:
        raise ValueError("capture registry coverage differs")
    cells, historical, on_off, representations = [], [], [], []
    scalar_pairs = 0
    for ji, job in enumerate(jobs):
        tensors = {}
        for si, stage in enumerate(STAGES):
            row = rows[ji * 3 + si]
            relative = f"calibration/{job['pair_id']}_{job['cell']}_{stage}.json"
            if (
                row["path"] != relative
                or row["historical_run"] != job["historical_run"]
            ):
                raise ValueError("calibration path/history mapping differs")
            receipt, path = registry[relative], root / relative
            if any(row[k] != v for k, v in receipt.items()):
                raise ValueError("cell and capture registry differ")
            result, audit, response = verify_pair(
                Path(job["historical_run"]),
                path,
                receipt,
                frozen,
                summary["receipts"]["frozen_specification.json"],
            )
            if (
                audit["context"]["manifest_path"] != job["manifest_path"]
                or audit["effective"]["pixel_values"]["sha256"]
                != job["pixel_values_sha256"]
                or response != "The image"
                or row["cache_array_sha256"]
                != {t: a["after_array_sha256"] for t, a in result.items()}
            ):
                raise ValueError("source input or cache receipt mapping differs")
            event = _read(path)["stream_events"][0]
            artifacts = {
                f"{a['layer_index']}:{a['tensor']}": a
                for a in event["cache_tensor_artifacts"]
            }
            tensors[stage] = {t: checked_array(path, artifacts[t]) for t in TARGETS}
            for target, check in result.items():
                historical.append(
                    {
                        "pair_id": job["pair_id"],
                        "cell": job["cell"],
                        "stage": stage,
                        "target": target,
                        **check,
                        "independent_tensor_bytes_exact": True,
                    }
                )
                scalar_pairs += check["scalar_count"]
            if stage == "on":
                actual = checked_representations(root, row["representation"], config)
                if (
                    row["representation"]["path"]
                    != f"representations/{job['pair_id']}_{job['cell']}.npz"
                ):
                    raise ValueError("representation identity mapping differs")
                representations.append(
                    {
                        "pair_id": job["pair_id"],
                        "cell": job["cell"],
                        **row["representation"],
                        "independent_arrays_verified": actual,
                        "native_graph_identity": "capture_code_and_fixture_verified_not_separate_model_replay",
                    }
                )
            cells.append(
                {**row, "independent_input_suffix_layout_and_cache_exact": True}
            )
        for stage in ("off_before", "off_after"):
            for target in TARGETS:
                independent_exact_arrays(tensors[stage][target], tensors["on"][target])
                on_off.append(
                    {
                        "pair_id": job["pair_id"],
                        "cell": job["cell"],
                        "off_stage": stage,
                        "target": target,
                        "independent_tensor_bytes_exact": True,
                        "scalar_count": int(tensors[stage][target].size),
                    }
                )
    if len(historical) != 1152 or len(on_off) != 768 or len(representations) != 128:
        raise ValueError("independent verification denominator differs")
    if (root / "panel").exists():
        raise ValueError("unexpected experimental panel output")
    output.mkdir(parents=True)
    ledgers = {}
    for name, values in (
        ("cells.jsonl", cells),
        ("historical_comparisons.jsonl", historical),
        ("on_off_comparisons.jsonl", on_off),
        ("representations.jsonl", representations),
    ):
        path = output / name
        path.write_text(
            "".join(
                json.dumps(r, separators=(",", ":"), allow_nan=False) + "\n"
                for r in values
            )
        )
        ledgers[name] = {"rows": len(values), "sha256": _sha(path)}
    result = {
        **summary,
        "analysis_kind": "representation_observation_calibration_publication",
        "source_summary_sha256": _sha(root / "summary.json"),
        "frozen_specification": frozen,
        "preflight": pre,
        "ledgers": ledgers,
        "source_response_counts": {"The image": 384},
        "independent_verification": {
            "prepared_input_equalities": 384,
            "historical_tensor_pairs": 1152,
            "historical_scalar_pairs": scalar_pairs,
            "on_off_tensor_pairs": 768,
            "on_off_scalar_pairs": sum(r["scalar_count"] for r in on_off),
            "representation_arrays": 256,
            "representation_scalars": sum(
                a["scalar_count"] for r in representations for a in r["arrays"].values()
            ),
            "maximum_cache_absolute_difference": 0,
            "verification_model_forwards": 0,
            "verification_processor_calls": 0,
            "method": "independent_reload_hash_layout_dtype_finiteness_and_uint8_cache_equality",
        },
        "representation_sidecars_published": False,
        "claim_boundaries": [
            "Historical OFF-ON-OFF calibration only; no new controlled-overlap representation scores.",
            "Saved representations are verified locally; native graph identity follows frozen code and fixtures, not an independent encoder forward.",
            "Exact selected-cache parity is bounded to these inputs, model, runtime, sites and suffix; not proof for every hidden tensor or future input.",
            "No automatic panel release, training attribution, equivalence test or confirmation claim.",
        ],
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result["independent_verification"], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Verify observation calibration without model calls."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    publish(args.run_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    main()
