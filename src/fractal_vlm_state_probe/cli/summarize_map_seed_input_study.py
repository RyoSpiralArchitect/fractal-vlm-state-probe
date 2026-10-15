from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_graded_permutation_study import read
from fractal_vlm_state_probe.cli.run_map_seed_input_study import validate_config
from fractal_vlm_state_probe.cli.summarize_graded_permutation_study import (
    cycle_oracle,
    load_pixels,
    verify_gate,
    verify_pixels,
)
from fractal_vlm_state_probe.frequency_control import CELLS
from fractal_vlm_state_probe.permutation_maps import ANCHOR, MAPS, NEW_SEEDS
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def verify_map_recipe(key, receipt, anchor):
    path = Path(receipt["path"])
    mapping = np.load(path, allow_pickle=False)
    if sha256_file(path) != receipt["sha256"] or mapping.shape != (76800,):
        raise ValueError("map receipt differs")
    if key == ANCHOR:
        expected = anchor
    else:
        seed = int(key.removeprefix("seed_"))
        if seed not in NEW_SEEDS:
            raise ValueError("unregistered map seed")
        children = np.random.SeedSequence(seed).spawn(2)
        order = np.random.Generator(np.random.PCG64(children[0])).permutation(76800)
        full = np.random.Generator(np.random.PCG64(children[1])).permutation(76800)
        for name, value in (("selection", order), ("full", full)):
            item = receipt["ingredients"][name]
            if sha256_file(Path(item["path"])) != item["sha256"] or not np.array_equal(
                np.load(item["path"], allow_pickle=False), value
            ):
                raise ValueError("map RNG ingredient differs")
        expected = cycle_oracle(full, order, 57600)
    if (
        not np.array_equal(mapping, expected)
        or receipt["selected_count"] != 57600
        or receipt["selected_fraction"] != 0.75
    ):
        raise ValueError("independent map recipe differs")
    return mapping


def verify_panel_counts(records, claimed):
    if len(records) != 32 or len({r["pair_id"] for r in records}) != 32:
        raise ValueError("input block denominator differs")
    families = {r["pairing_family"] for r in records}
    if len(families) != 8 or any(
        sorted(r["replicate"] for r in records if r["pairing_family"] == f)
        != [1, 2, 3, 4]
        for f in families
    ):
        raise ValueError("input family structure differs")
    for key in MAPS:
        accepted = [r["pair_id"] for r in records if r["maps"][key]["gate"]["accepted"]]
        rejected = [
            r["pair_id"] for r in records if not r["maps"][key]["gate"]["accepted"]
        ]
        complete = sorted(
            f
            for f in families
            if all(
                r["maps"][key]["gate"]["accepted"]
                for r in records
                if r["pairing_family"] == f
            )
        )
        expected = {
            "accepted_blocks": len(accepted),
            "total_blocks": 32,
            "accepted_pair_ids": accepted,
            "rejected_pair_ids": rejected,
            "complete_four_seed_families": complete,
        }
        if claimed[key] != expected:
            raise ValueError("published map denominator differs")
    return all(claimed[key]["accepted_blocks"] == 32 for key in MAPS)


def publish(source, output):
    study = read(source)
    frozen = study["frozen_specification"]
    validate_config(frozen["config"])
    if (
        read(source.parent / "frozen_specification.json") != frozen
        or sha256_file(source.parent / "frozen_specification.json")
        != study["frozen_specification_sha256"]
        or (source.parent / "execution_failure.json").exists()
    ):
        raise ValueError("input run is incomplete or specification changed")
    repo = Path(__file__).resolve().parents[3]
    for name, digest in frozen["code_sha256"].items():
        if sha256_file(repo / name) != digest:
            raise ValueError(f"registered input code changed: {name}")
    for key in ("graded_receipt", "published_graded"):
        if sha256_file(Path(frozen[f"{key}_path"])) != frozen[f"{key}_sha256"]:
            raise ValueError("historical source changed")
    old = read(frozen["graded_receipt_path"])
    prior = {r["pair_id"]: r for r in old["records"]}
    padding_path = old["frozen_specification"]["source_paths"]["padding_receipt"]
    if (
        sha256_file(Path(padding_path))
        != old["frozen_specification"]["source_sha256"]["padding_receipt"]
    ):
        raise ValueError("padding baseline changed")
    padding = {r["pair_id"]: r for r in read(padding_path)["records"]}
    anchor_receipt = old["frozen_specification"]["maps"]["selected_06_of_08"]
    if sha256_file(Path(anchor_receipt["path"])) != anchor_receipt["sha256"]:
        raise ValueError("historical anchor map changed")
    anchor = np.load(anchor_receipt["path"], allow_pickle=False)
    if set(frozen["maps"]) != set(MAPS) or set(study["maps"]) != set(MAPS):
        raise ValueError("map denominator differs")
    mappings = {k: verify_map_recipe(k, r, anchor) for k, r in frozen["maps"].items()}
    rows, drift, integer_errors, anchor_checks = [], [], [], 0
    for record in study["records"]:
        pair = record["pair_id"]
        if any(
            record[k] != prior[pair][k]
            for k in ("pairing_family", "broad_class", "replicate")
        ) or set(record["maps"]) != set(MAPS):
            raise ValueError("input hierarchy or map coverage differs")
        baseline = padding[pair]["conditions"]["original/palette_mean"]
        baselines = {c: load_pixels(a) for c, a in baseline["artifacts"].items()}
        for key, measured in record["maps"].items():
            if set(measured["cells"]) != set(CELLS) or set(
                measured["artifacts"]
            ) != set(CELLS):
                raise ValueError("all four input cells required")
            verify_gate(measured["cells"], measured["gate"])
            for cell in CELLS:
                row = measured["cells"][cell]
                canvas = load_pixels(measured["artifacts"][cell])
                color = np.array(
                    baseline["cells"][cell]["marginal_audit"]["fill_rgb_u8"]
                )
                integer_errors.append(
                    verify_pixels(canvas, baselines[cell], mappings[key], color, row)
                )
                if (
                    row["processor_non_pixel_payload"]
                    != baseline["cells"][cell]["processor_non_pixel_payload"]
                ):
                    raise ValueError("processor metadata changed")
                if key == ANCHOR:
                    previous = prior[pair]["levels"]["selected_06_of_08"]
                    expected = dict(previous["cells"][cell])
                    expected.pop("endpoint_pixel_and_stats_reproduced", None)
                    if (
                        row != expected
                        or measured["gate"] != previous["gate"]
                        or not np.array_equal(
                            canvas, load_pixels(previous["artifacts"][cell])
                        )
                    ):
                        raise ValueError("historical anchor changed")
                    anchor_checks += 1
                rows.append(
                    {
                        "pair_id": pair,
                        "map": key,
                        "cell": cell,
                        "artifact": measured["artifacts"][cell],
                        **row,
                    }
                )
            anchor_gate = record["maps"][ANCHOR]["gate"]
            drift.append(
                {
                    "pair_id": pair,
                    "map": key,
                    "gate": measured["gate"],
                    "centroid_target_relative_change_from_anchor": measured["gate"][
                        "common_centroid_target"
                    ]
                    / anchor_gate["common_centroid_target"]
                    - 1,
                    "hf_target_change_from_anchor": measured["gate"]["common_hf_target"]
                    - anchor_gate["common_hf_target"],
                }
            )
    eligible = verify_panel_counts(study["records"], study["maps"])
    if (
        study["cache_panel_eligible"] != eligible
        or anchor_checks != 128
        or study["anchor_exact_cell_checks"] != 128
        or len(rows) != 512
        or study["input_cells"] != 512
        or study["new_cache_forwards"] != 0
        or study["model_weights_loaded"]
    ):
        raise ValueError("input completion or eligibility differs")
    if output.exists():
        raise FileExistsError("use a fresh publication directory")
    output.mkdir(parents=True)
    ledger = output / "cells.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
            for r in rows
        )
    )
    summary = {k: v for k, v in study.items() if k != "records"}
    summary.update(
        {
            "analysis_kind": "fixed_fraction_map_seed_input_publication",
            "source_report_sha256": sha256_file(source),
            "exporter_sha256": sha256_file(Path(__file__)),
            "verification_helper_sha256": sha256_file(
                Path(__file__).with_name("summarize_graded_permutation_study.py")
            ),
            "input_context": drift,
            "cell_ledger": {
                "path": "cells.jsonl",
                "rows": 512,
                "sha256": sha256_file(ledger),
            },
            "independent_verification": {
                "map_recipes": 4,
                "pixel_inverse_padding_rgb_audits": 512,
                "four_cell_gates": 128,
                "historical_anchor_cell_checks": 128,
                "integer_moment_oracle_max_mean_error": max(
                    r["mean"] for r in integer_errors
                ),
                "integer_moment_oracle_max_variance_error": max(
                    r["variance"] for r in integer_errors
                ),
                "integer_moment_oracle_cells_above_1e_12": sum(
                    max(r.values()) > 1e-12 for r in integer_errors
                ),
            },
        }
    )
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Independently verify and publish the fixed-fraction map input audit."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.source, args.output_root)
    print(json.dumps(result["independent_verification"], indent=2))
    print(f"cache_panel_eligible={result['cache_panel_eligible']}")


if __name__ == "__main__":
    main()
