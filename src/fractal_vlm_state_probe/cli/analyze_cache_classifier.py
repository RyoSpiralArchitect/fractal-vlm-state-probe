from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.cache_classifier import (
    analyze_cache_feature_records,
    discover_run_paths,
    format_classifier_markdown,
    load_run_records,
    write_classifier_json,
    write_classifier_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify stimulus conditions from cache-summary features.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--batch-root", type=Path)
    source.add_argument("--runs", nargs="+", type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    run_paths = discover_run_paths(args.batch_root) if args.batch_root else sorted(args.runs)
    if not run_paths:
        parser.error("no run JSON files found")

    analysis = analyze_cache_feature_records(load_run_records(run_paths))
    if args.output_json:
        write_classifier_json(analysis, args.output_json)
    if args.output_md:
        write_classifier_markdown(analysis, args.output_md)
    if not args.output_json and not args.output_md:
        print(format_classifier_markdown(analysis))


if __name__ == "__main__":
    main()
