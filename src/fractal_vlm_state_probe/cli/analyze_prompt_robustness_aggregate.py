from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.prompt_robustness_aggregate import (
    analyze_prompt_robustness_aggregate,
    write_prompt_robustness_aggregate_json,
    write_prompt_robustness_aggregate_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate semantic prompt-robustness audits across models."
    )
    parser.add_argument(
        "--analysis", action="append", required=True, metavar="KEY=PATH"
    )
    parser.add_argument(
        "--comparison-axis",
        choices=("model", "source_pair", "condition"),
        default="model",
        help="Meaning of each KEY in --analysis (default: model).",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    analyses: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    for spec in args.analysis:
        if "=" not in spec:
            raise ValueError(f"analysis spec must be KEY=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        path = Path(raw_path).expanduser()
        if not label or label in analyses:
            raise ValueError(f"analysis labels must be non-empty and unique: {label!r}")
        with path.open("r", encoding="utf-8") as handle:
            analyses[label] = json.load(handle)
        paths[label] = path

    analysis = analyze_prompt_robustness_aggregate(
        analyses,
        analysis_paths=paths,
        comparison_axis=args.comparison_axis,
    )
    write_prompt_robustness_aggregate_json(analysis, args.output_json)
    write_prompt_robustness_aggregate_markdown(analysis, args.output_md)
    axis = args.comparison_axis.replace("_", "-")
    print(f"wrote cross-{axis} prompt robustness to {args.output_md}")


if __name__ == "__main__":
    main()
