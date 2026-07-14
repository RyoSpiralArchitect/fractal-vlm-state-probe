from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cache_direction_hierarchy import (
    DEFAULT_REGIONS,
    analyze_cache_direction_hierarchy,
    write_cache_direction_hierarchy_json,
    write_cache_direction_hierarchy_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate seed replication from cache-direction transfer across generator pairings."
    )
    parser.add_argument("--replication", required=True, type=Path)
    hierarchy = parser.add_mutually_exclusive_group(required=True)
    hierarchy.add_argument(
        "--hierarchy-label",
        action="append",
        metavar="SOURCE_PAIR=BROAD_CLASS,PAIRING_FAMILY",
    )
    hierarchy.add_argument(
        "--panel-summary",
        type=Path,
        help="Load broad-class and pairing-family labels from a generated panel summary.",
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
    if args.panel_summary is not None:
        broad, pairing = _load_panel_hierarchy(args.panel_summary)
    else:
        broad, pairing = _parse_hierarchy_labels(args.hierarchy_label)
    analysis = analyze_cache_direction_hierarchy(
        replication,
        broad_class_by_label=broad,
        pairing_family_by_label=pairing,
        regions=args.region or DEFAULT_REGIONS,
        effect=args.effect,
        exact_limit=args.exact_limit,
        monte_carlo_permutations=args.monte_carlo_permutations,
        seed=args.seed,
    )
    analysis["replication_path"] = str(args.replication)
    write_cache_direction_hierarchy_json(analysis, args.output_json)
    write_cache_direction_hierarchy_markdown(analysis, args.output_md)
    print(f"wrote cache direction pairing hierarchy to {args.output_md}")


def _parse_hierarchy_labels(
    specs: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    broad = {}
    pairing = {}
    for spec in specs:
        if "=" not in spec or "," not in spec:
            raise ValueError(
                f"hierarchy label must be SOURCE_PAIR=BROAD_CLASS,PAIRING_FAMILY: {spec}"
            )
        label, values = (value.strip() for value in spec.split("=", 1))
        broad_class, pairing_family = (value.strip() for value in values.split(",", 1))
        if not label or not broad_class or not pairing_family or label in broad:
            raise ValueError(f"hierarchy labels must be non-empty and unique: {spec}")
        broad[label] = broad_class
        pairing[label] = pairing_family
    return broad, pairing


def _load_panel_hierarchy(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("analysis_kind") != "generator_pairing_factorial_panel":
        raise ValueError("panel summary must be a generator-pairing factorial panel")
    broad = summary.get("broad_class_by_pair")
    pairing = summary.get("pairing_family_by_pair")
    if not isinstance(broad, dict) or not isinstance(pairing, dict):
        raise ValueError("panel summary is missing hierarchy maps")
    return (
        {str(label): str(value) for label, value in broad.items()},
        {str(label): str(value) for label, value in pairing.items()},
    )


if __name__ == "__main__":
    main()
