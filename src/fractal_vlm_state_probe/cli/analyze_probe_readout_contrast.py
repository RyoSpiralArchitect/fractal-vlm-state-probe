from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.compare_runs import load_run
from fractal_vlm_state_probe.probe_readout import (
    analyze_first_token_readout_contrast,
    write_first_token_readout_json,
    write_first_token_readout_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze first-token top-k probe readout over 2x2 cross-palette cells."
    )
    parser.add_argument("--mm", required=True, type=Path, help="M spatial x M palette run JSON.")
    parser.add_argument("--jj", required=True, type=Path, help="J spatial x J palette run JSON.")
    parser.add_argument("--mj", required=True, type=Path, help="M spatial x J palette run JSON.")
    parser.add_argument("--jm", required=True, type=Path, help="J spatial x M palette run JSON.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument(
        "--max-token-effects",
        type=int,
        default=20,
        help="Maximum common top-k token effects to keep in the top-level summary.",
    )
    args = parser.parse_args()

    analysis = analyze_first_token_readout_contrast(
        mm=load_run(args.mm),
        jj=load_run(args.jj),
        mj=load_run(args.mj),
        jm=load_run(args.jm),
        max_token_effects=args.max_token_effects,
    )
    write_first_token_readout_json(analysis, args.output_json)
    write_first_token_readout_markdown(analysis, args.output_md)
    print(f"wrote first-token probe readout contrast to {args.output_md}")


if __name__ == "__main__":
    main()
