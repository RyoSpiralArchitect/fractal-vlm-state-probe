from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.compare_runs import load_run
from fractal_vlm_state_probe.factorial_contrast import (
    analyze_factorial_cache_contrast,
    write_factorial_contrast_json,
    write_factorial_contrast_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a 2x2 factorial contrast over saved cache summaries.")
    parser.add_argument("--mm", required=True, type=Path, help="M spatial x M palette run JSON.")
    parser.add_argument("--jj", required=True, type=Path, help="J spatial x J palette run JSON.")
    parser.add_argument("--mj", required=True, type=Path, help="M spatial x J palette run JSON.")
    parser.add_argument("--jm", required=True, type=Path, help="J spatial x M palette run JSON.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    analysis = analyze_factorial_cache_contrast(
        mm=load_run(args.mm),
        jj=load_run(args.jj),
        mj=load_run(args.mj),
        jm=load_run(args.jm),
    )
    write_factorial_contrast_json(analysis, args.output_json)
    write_factorial_contrast_markdown(analysis, args.output_md)
    print(f"wrote factorial cache contrast to {args.output_md}")


if __name__ == "__main__":
    main()
