from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cache_tensor_factorial import (
    analyze_cache_tensor_factorial,
    write_cache_tensor_factorial_json,
    write_cache_tensor_factorial_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze full source-cache tensor vectors over a 2x2 factorial."
    )
    parser.add_argument("--mm", required=True, type=Path)
    parser.add_argument("--jj", required=True, type=Path)
    parser.add_argument("--mj", required=True, type=Path)
    parser.add_argument("--jm", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--tensor", choices=["keys", "values"], default="values")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    paths = {cell: getattr(args, cell) for cell in ("mm", "jj", "mj", "jm")}
    runs = {}
    for cell, path in paths.items():
        with path.open("r", encoding="utf-8") as handle:
            runs[cell] = json.load(handle)
    analysis = analyze_cache_tensor_factorial(
        runs=runs,
        run_paths=paths,
        layer_index=args.layer,
        tensor=args.tensor,
    )
    write_cache_tensor_factorial_json(analysis, args.output_json)
    write_cache_tensor_factorial_markdown(analysis, args.output_md)
    print(f"wrote source-cache tensor factorial to {args.output_md}")


if __name__ == "__main__":
    main()
