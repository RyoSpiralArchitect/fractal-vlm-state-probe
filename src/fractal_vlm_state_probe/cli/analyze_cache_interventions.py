from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cache_intervention_analysis import (
    analyze_values_swap_interventions,
    write_values_swap_analysis_json,
    write_values_swap_analysis_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one or more MLX cache values-swap intervention runs."
    )
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    runs = [_load_json(path) for path in args.run]
    analysis = analyze_values_swap_interventions(
        runs,
        run_paths=[str(path) for path in args.run],
    )
    write_values_swap_analysis_json(analysis, args.output_json)
    write_values_swap_analysis_markdown(analysis, args.output_md)
    print(f"wrote cache intervention analysis to {args.output_json}")
    print(f"wrote cache intervention analysis to {args.output_md}")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()
