from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.prompt_robustness_multi_pair_matrix import (
    analyze_prompt_robustness_multi_pair_matrix,
    write_prompt_robustness_multi_pair_matrix_json,
    write_prompt_robustness_multi_pair_matrix_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prompt robustness over a model x multi-source-pair matrix."
    )
    parser.add_argument(
        "--analysis", action="append", required=True, metavar="MODEL:PAIR=PATH"
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    analyses: dict[str, dict[str, dict]] = {}
    paths: dict[str, dict[str, Path]] = {}
    for spec in args.analysis:
        if "=" not in spec:
            raise ValueError(f"analysis spec must be MODEL:PAIR=PATH: {spec}")
        key, raw_path = spec.split("=", 1)
        if ":" not in key:
            raise ValueError(f"analysis key must be MODEL:PAIR: {key}")
        model, source_pair = (part.strip() for part in key.split(":", 1))
        path = Path(raw_path).expanduser()
        if not model or not source_pair:
            raise ValueError(f"model and source-pair labels must be non-empty: {key}")
        if source_pair in analyses.setdefault(model, {}):
            raise ValueError(f"duplicate analysis key: {model}:{source_pair}")
        with path.open("r", encoding="utf-8") as handle:
            analyses[model][source_pair] = json.load(handle)
        paths.setdefault(model, {})[source_pair] = path

    analysis = analyze_prompt_robustness_multi_pair_matrix(
        analyses,
        analysis_paths=paths,
    )
    write_prompt_robustness_multi_pair_matrix_json(analysis, args.output_json)
    write_prompt_robustness_multi_pair_matrix_markdown(analysis, args.output_md)
    print(f"wrote model x multi-source-pair prompt matrix to {args.output_md}")


if __name__ == "__main__":
    main()
