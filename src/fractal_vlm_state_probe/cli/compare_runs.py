from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.compare_runs import (
    compare_runs,
    load_run,
    write_comparison_json,
    write_comparison_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two run JSON files.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    comparison = compare_runs(load_run(args.left), load_run(args.right))
    if args.output_json:
        write_comparison_json(comparison, args.output_json)
    if args.output_md:
        write_comparison_markdown(comparison, args.output_md)
    if not args.output_json and not args.output_md:
        from fractal_vlm_state_probe.compare_runs import format_comparison_markdown

        print(format_comparison_markdown(comparison))


if __name__ == "__main__":
    main()

