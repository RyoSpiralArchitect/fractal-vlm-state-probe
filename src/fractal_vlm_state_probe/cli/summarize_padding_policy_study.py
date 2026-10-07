from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.padding_policy import (
    CONTENT_STATES,
    POLICIES,
    padding_gate,
)
from fractal_vlm_state_probe.stimulus import sha256_file, write_json


def publish(source_path: Path, output_root: Path, *, figures: bool = True) -> dict:
    study = json.loads(source_path.read_text())
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_path = output_root / "cells.jsonl"
    records, all_cells, shams = [], [], []
    with ledger_path.open("w", encoding="utf-8") as handle:
        for record in study["records"]:
            compact = {
                k: record[k]
                for k in ("pair_id", "pairing_family", "broad_class", "replicate")
            }
            compact["conditions"] = {}
            for key, condition in record["conditions"].items():
                if padding_gate(condition["cells"]) != condition["gate"]:
                    raise ValueError("saved padding gate does not reproduce")
                compact["conditions"][key] = {
                    "gate": condition["gate"],
                    "processor_centroid_by_cell": {
                        c: r["processor"]["spectral_centroid"]
                        for c, r in condition["cells"].items()
                    },
                    "processor_hf_ratio_by_cell": {
                        c: r["processor"]["high_frequency_energy_ratio"]
                        for c, r in condition["cells"].items()
                    },
                }
                for cell, measured in condition["cells"].items():
                    row = {
                        "pair_id": record["pair_id"],
                        "condition": key,
                        "cell": cell,
                        **measured,
                        "artifact": condition["artifacts"][cell],
                    }
                    handle.write(
                        json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                    )
                    all_cells.append(measured)
                    if "black_pad_sham" in measured:
                        shams.append(measured["black_pad_sham"])
            records.append(compact)
    condition_stats = {}
    for key, status in study["conditions"].items():
        gates = [r["conditions"][key]["gate"] for r in study["records"]]
        errors = [g["max_centroid_relative_error"] for g in gates]
        if sum(g["accepted"] for g in gates) != status["accepted_blocks"]:
            raise ValueError("condition count differs from all block gates")
        condition_stats[key] = {
            **status,
            "max_centroid_relative_error_min": min(errors),
            "max_centroid_relative_error_median": float(np.median(errors)),
            "max_centroid_relative_error_max": max(errors),
            "max_hf_spread": max(
                g["hf_max_pairwise_absolute_difference"] for g in gates
            ),
        }
    summary = {
        "schema_version": 1,
        "analysis_kind": "padding_policy_input_comparison_snapshot",
        "date": study["date"],
        "registered_design_commit": "a79da87",
        "source_report_sha256": sha256_file(source_path),
        "frozen_specification": study["frozen_specification"],
        "frozen_specification_sha256": study["frozen_specification_sha256"],
        "code_exporter_sha256": sha256_file(Path(__file__)),
        "cell_ledger": {
            "path": ledger_path.name,
            "sha256": sha256_file(ledger_path),
            "rows": len(all_cells),
        },
        "block_count": study["block_count"],
        "conditions": condition_stats,
        "checks": {
            "unchanged_content_cells": sum(
                r["marginal_audit"]["content_pixels_unchanged"] for r in all_cells
            ),
            "exact_fill_cells": sum(
                r["marginal_audit"]["padding_pixels_equal_registered_fill"]
                for r in all_cells
            ),
            "registered_expanded_palette_cells": sum(
                r["registered_expanded_palette_verified"] for r in all_cells
            ),
            "mean_max_abs_error": max(
                r["marginal_audit"]["mean_max_abs_error"] for r in all_cells
            ),
            "variance_max_abs_error": max(
                r["marginal_audit"]["variance_max_abs_error"] for r in all_cells
            ),
            "black_pixel_shams_bitwise_equal": sum(
                r["pixel_values_bitwise_equal"] for r in shams
            ),
            "black_shams_full_processor_payload_equal": sum(
                r["whole_processor_payload_equal"] for r in shams
            ),
            "changed_non_pixel_payload_keys": sorted(
                {k for r in shams for k in r["changed_non_pixel_payload_keys"]}
            ),
            "marginal_total_variation_min": min(
                r["marginal_audit"]["joint_rgb_total_variation_from_content"]
                for r in all_cells
            ),
            "marginal_total_variation_max": max(
                r["marginal_audit"]["joint_rgb_total_variation_from_content"]
                for r in all_cells
            ),
        },
        "new_cache_forwards": 0,
        "model_weights_loaded": False,
        "full_panel_input_eligible_conditions": [
            k for k, s in condition_stats.items() if s["all_input_blocks_accepted"]
        ],
        "model_cache_calibration_status": "NOT_MEASURED",
        "previous_selected_cache_counts_unchanged": {
            "source_cells": 616,
            "tensors": 1704,
            "factorials": 426,
        },
        "records": records,
        "claim_boundaries": study["claim_boundaries"],
    }
    if figures:
        _plot(study, output_root)
        summary["figures"] = {
            name: sha256_file(output_root / name)
            for name in (
                "padding_acceptance.png",
                "padding_acceptance.pdf",
                "padding_example.png",
            )
        }
    write_json(output_root / "summary.json", summary)
    return summary


def _plot(study: dict, output_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    labels = {
        "black": "Black",
        "fixed_gray": "Fixed gray",
        "palette_mean": "Palette mean",
    }
    colors = ["#303D43", "#97A4AD", "#278875"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    x = np.arange(3)
    for index, policy in enumerate(POLICIES):
        counts = [
            study["conditions"][f"{state}/{policy}"]["accepted_blocks"]
            for state in CONTENT_STATES
        ]
        positions = x + (index - 1) * 0.23
        ax.bar(
            positions,
            counts,
            width=0.21,
            color=colors[index],
            label=labels[policy],
            zorder=3,
        )
        for xpos, count in zip(positions, counts):
            ax.text(xpos, count + 0.7, str(count), ha="center", fontsize=11)
    ax.set_xticks(
        x, ["Original content", "Frozen low-pass content", "Common pixel permutation"]
    )
    ax.set_ylim(0, 37)
    ax.set_yticks([0, 8, 16, 24, 32])
    ax.set_ylabel("Accepted source-pair blocks (of 32)")
    ax.set_title(
        "Padding policy changes whole-processor frequency matching",
        loc="left",
        pad=18,
        fontsize=14,
    )
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.grid(axis="y", color="#DCE3E5", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.10,
        0.025,
        "All four cells: centroid within 5%; HF spread <= 0.02. Full-image RGB marginals are explicitly changed.",
        fontsize=9,
        color="#465359",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_root / f"padding_acceptance.{suffix}", dpi=180, bbox_inches="tight"
        )
    plt.close(fig)

    record = next(
        r for r in study["records"] if r["pair_id"] == "geometry_checker_hex_r1"
    )
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for row, cell in enumerate(("mm", "mj")):
        for column, policy in enumerate(POLICIES):
            condition = record["conditions"][f"shared_pixel_permutation/{policy}"]
            manifest_path = Path(condition["artifacts"][cell]["manifest_path"])
            if (
                sha256_file(manifest_path)
                != condition["artifacts"][cell]["manifest_sha256"]
            ):
                raise ValueError("example manifest differs from measured input")
            manifest = json.loads(manifest_path.read_text())
            frame_path = manifest_path.parent / manifest["frames"][0]["path"]
            if sha256_file(frame_path) != condition["artifacts"][cell]["frame_sha256"]:
                raise ValueError("example pixels differ from measured input")
            ax = axes[row, column]
            with Image.open(frame_path) as image:
                ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            centroid = condition["cells"][cell]["processor"]["spectral_centroid"]
            ax.set_xlabel(f"Processor centroid: {centroid:.5f}")
            if row == 0:
                ax.set_title(labels[policy])
            if column == 0:
                ax.set_ylabel("Palette A (MM)" if row == 0 else "Palette B (MJ)")
    fig.suptitle(
        "Identical content pixels within each row; only the padding changes",
        fontsize=13,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.20)
    fig.savefig(output_root / "padding_example.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the padding-policy input study and its scientific figures."
    )
    parser.add_argument("--input-summary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    summary = publish(args.input_summary, args.output_root, figures=not args.no_figures)
    print(
        f"published {summary['cell_ledger']['rows']} cells to {args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
