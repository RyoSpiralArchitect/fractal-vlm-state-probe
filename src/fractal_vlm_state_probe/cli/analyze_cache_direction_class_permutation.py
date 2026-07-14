from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cache_direction_permutation import (
    DEFAULT_REGIONS,
    analyze_cache_direction_class_permutation,
    write_cache_direction_class_permutation_json,
    write_cache_direction_class_permutation_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate cache direction with source-pair-level class-label permutations."
        )
    )
    parser.add_argument("--replication", required=True, type=Path)
    parser.add_argument(
        "--class-label",
        action="append",
        required=True,
        metavar="SOURCE_PAIR=CLASS",
    )
    parser.add_argument("--region", action="append", default=None)
    parser.add_argument(
        "--effect",
        choices=("spatial_main_effect", "palette_main_effect", "interaction"),
        default="interaction",
    )
    parser.add_argument("--exact-limit", type=int, default=100_000)
    parser.add_argument("--monte-carlo-permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    with args.replication.open("r", encoding="utf-8") as handle:
        replication = json.load(handle)
    analysis = analyze_cache_direction_class_permutation(
        replication,
        class_by_label=_parse_class_labels(args.class_label),
        regions=args.region or DEFAULT_REGIONS,
        effect=args.effect,
        exact_limit=args.exact_limit,
        monte_carlo_permutations=args.monte_carlo_permutations,
        seed=args.seed,
    )
    analysis["replication_path"] = str(args.replication)
    write_cache_direction_class_permutation_json(analysis, args.output_json)
    write_cache_direction_class_permutation_markdown(analysis, args.output_md)
    print(f"wrote source-level cache direction permutation to {args.output_md}")


def _parse_class_labels(specs: list[str]) -> dict[str, str]:
    output = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"class label must be SOURCE_PAIR=CLASS: {spec}")
        label, class_label = (value.strip() for value in spec.split("=", 1))
        if not label or not class_label or label in output:
            raise ValueError(f"class labels must be non-empty and unique: {spec}")
        output[label] = class_label
    return output


if __name__ == "__main__":
    main()
