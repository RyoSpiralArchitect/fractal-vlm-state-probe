from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.factorial_image_contrast import (
    analyze_factorial_image_contrast,
    load_stats,
    write_factorial_image_contrast_json,
    write_factorial_image_contrast_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a 2x2 factorial contrast over raw or processor-space image statistics."
    )
    parser.add_argument("--stats-json", required=True, type=Path)
    parser.add_argument("--mm", required=True, help="M spatial x M palette condition id.")
    parser.add_argument("--jj", required=True, help="J spatial x J palette condition id.")
    parser.add_argument("--mj", required=True, help="M spatial x J palette condition id.")
    parser.add_argument("--jm", required=True, help="J spatial x M palette condition id.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    analysis = analyze_factorial_image_contrast(
        stats=load_stats(args.stats_json),
        mm_condition_id=args.mm,
        jj_condition_id=args.jj,
        mj_condition_id=args.mj,
        jm_condition_id=args.jm,
    )
    write_factorial_image_contrast_json(analysis, args.output_json)
    write_factorial_image_contrast_markdown(analysis, args.output_md)
    print(f"wrote factorial image contrast to {args.output_md}")


if __name__ == "__main__":
    main()
