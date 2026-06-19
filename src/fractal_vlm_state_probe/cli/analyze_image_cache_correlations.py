from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.image_cache_correlation import (
    analyze_image_cache_correlations,
    write_image_cache_correlation_json,
    write_image_cache_correlation_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate pairwise image-stat deltas with cache-summary distances.")
    parser.add_argument("--image-stats", action="append", required=True, type=Path)
    parser.add_argument("--cache-analysis", action="append", required=True, type=Path)
    parser.add_argument("--batch-summary", action="append", type=Path, default=[])
    parser.add_argument(
        "--condition-alias",
        action="append",
        default=[],
        metavar="ALIAS=CONDITION_ID",
        help="Manual alias for joining cache condition keys to image-stat condition ids.",
    )
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    analysis = analyze_image_cache_correlations(
        image_stats_paths=args.image_stats,
        cache_analysis_paths=args.cache_analysis,
        batch_summary_paths=args.batch_summary,
        condition_aliases=_parse_aliases(args.condition_alias),
        min_samples=args.min_samples,
    )
    write_image_cache_correlation_json(analysis, args.output_json)
    write_image_cache_correlation_markdown(analysis, args.output_md)
    print(f"wrote image/cache correlation report to {args.output_md}")


def _parse_aliases(raw_values: list[str]) -> dict[str, str]:
    aliases = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"condition alias must be ALIAS=CONDITION_ID: {raw}")
        alias, condition_id = raw.split("=", 1)
        aliases[alias.strip()] = condition_id.strip()
    return aliases


if __name__ == "__main__":
    main()
