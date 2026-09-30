from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cache_direction_holdout import holm_adjust
from fractal_vlm_state_probe.cache_tensor_factorial import cache_tensor_regions
from fractal_vlm_state_probe.stimulus import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a joint Holm correction to registered held-out seed targets."
    )
    parser.add_argument("--execution", required=True, action="append", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--expected-primary-tests", type=int, default=8)
    args = parser.parse_args()
    executions, primary, exploratory, factorials = [], [], [], []
    seen = set()
    for path in args.execution:
        execution = _read(path)
        if execution.get("analysis_kind") != "pairing_seed_validation_execution":
            raise ValueError("expected pairing seed validation executions")
        executions.append(execution)
        for raw in execution["holdout_analyses"]:
            analysis = _read(Path(raw))
            target = (analysis["model_id"], analysis["layer_index"], analysis["tensor"])
            if target in seen:
                raise ValueError(f"duplicate target: {target}")
            seen.add(target)
            for view in analysis["views"]:
                record = {
                    "model_id": analysis["model_id"],
                    "layer_index": analysis["layer_index"],
                    "tensor": analysis["tensor"],
                    "analysis_path": raw,
                    **view,
                }
                (primary if view["primary"] else exploratory).append(record)
            factorials.extend(
                _read(Path(raw_path))
                for label, raw_path in analysis["analysis_paths"].items()
                if analysis["hierarchy_metadata"][label]["replicate"] in (3, 4)
            )
    if len(primary) != args.expected_primary_tests:
        raise ValueError(
            f"registered test family requires {args.expected_primary_tests} primary tests; received {len(primary)}"
        )
    corrected = holm_adjust(
        [r["exact_block_test"]["p_greater"] if r["available"] else 1.0 for r in primary]
    )
    for record, adjusted in zip(primary, corrected):
        record["holm_p_greater"] = adjusted
        record["holm_family_size"] = len(primary)
    image_records = [
        next(r for r in a["regions"] if r["region"] == "image_tokens")
        for a in factorials
    ]
    dominant_axes = [
        max(
            r["balanced_contrast_energy"]["energy_shares"],
            key=r["balanced_contrast_energy"]["energy_shares"].get,
        )
        for r in image_records
    ]
    dominance = {
        name: dominant_axes.count(name)
        for name in ("spatial_contrast", "palette_contrast", "interaction_contrast")
    }
    summary = {
        "schema_version": 1,
        "analysis_kind": "pairing_seed_validation_study_summary",
        "primary_test_count": len(primary),
        "primary_holm_p_below_0_05": sum(
            r["available"] and r["holm_p_greater"] < 0.05 for r in primary
        ),
        "exploratory_view_count": len(exploratory),
        "new_source_cells": sum(e["new_source_cells"] for e in executions),
        "new_tensor_sidecars": sum(e["new_tensor_sidecars"] for e in executions),
        "new_factorial_analyses": sum(e["new_factorial_analyses"] for e in executions),
        "reference_recheck_cells": sum(
            e["reference_recheck_cells"] for e in executions
        ),
        "model_snapshots": [e["model_snapshot"] for e in executions],
        "new_factorial_integrity": {
            "pre_image_zero": sum(
                all(
                    e["l2_norm"] == 0
                    for e in next(
                        r for r in a["regions"] if r["region"] == "pre_image"
                    )["effects"].values()
                )
                for a in factorials
            ),
            "image_argmax": sum(
                next(r for r in a["regions"] if r["region"] == "all_effective")[
                    "effects"
                ]["interaction"]["argmax_sequence_position"]
                in cache_tensor_regions(
                    a["cells"]["mm"]["cache_token_layout"],
                    sequence_length=a["tensor_shape"][-2],
                )["image_tokens"]
                for a in factorials
            ),
            "image_energy_above_0_9": sum(
                a["interaction_partition"]["image_energy_fraction"] > 0.9
                for a in factorials
            ),
            "image_interaction_share_at_or_below_one_third": sum(
                r["balanced_contrast_energy"]["energy_shares"]["interaction_contrast"]
                <= 1 / 3
                for r in image_records
            ),
            "image_dominance": dominance,
        },
        "execution_paths": [str(p) for p in args.execution],
        "primary_tests": primary,
        "exploratory_views": exploratory,
        "claim_boundaries": [
            "References use the two previously observed seeds; test seeds never tune targets or directions.",
            "Inference is conditional on eight observed generator pairing families.",
            "Holm correction covers primary target-region tests only; head and token-band views are exploratory.",
            "New source-only runs do not add direct full-vocabulary factorials or establish persistence or mediation.",
        ],
    }
    write_json(args.output_json, summary)
    lines = [
        "# Pairing Seed Validation: Registered Primary Tests",
        "",
        f"New source cells: {summary['new_source_cells']}; tensor sidecars: {summary['new_tensor_sidecars']}; factorials: {summary['new_factorial_analyses']}.",
        "",
        "| Model | Layer / tensor | Region | Own cosine | Other cosine | Margin | Positive families | Retrieval | Exact p | Holm p |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in primary:
        if not r["available"]:
            lines.append(
                f"| {r['model_id']} | {r['layer_index']} {r['tensor']} | {r['view']} | unavailable: {r['reason']} | | | | | | |"
            )
            continue
        lines.append(
            f"| {r['model_id']} | {r['layer_index']} {r['tensor']} | {r['view']} | "
            f"{r['own_family_cosine_mean']:.5f} | {r['other_family_cosine_mean']:.5f} | {r['mean_family_margin']:.5f} | "
            f"{r['positive_family_margin_count']}/8 | {r['retrieval_correct_count']}/{r['test_seed_count']} | "
            f"{r['exact_block_test']['p_greater']:.6f} | {r['holm_p_greater']:.6f} |"
        )
    lines.extend(["", *summary["claim_boundaries"], ""])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote study summary to {args.output_md}")


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
