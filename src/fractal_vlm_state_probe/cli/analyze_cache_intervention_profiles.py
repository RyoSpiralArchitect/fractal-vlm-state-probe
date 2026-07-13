from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cache_intervention_profile import (
    compare_cache_intervention_profiles,
    write_profile_comparison_json,
    write_profile_comparison_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare layer profiles from cache intervention analysis JSONs."
    )
    parser.add_argument(
        "--analysis",
        action="append",
        required=True,
        metavar="KEY=PATH",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    analyses = _load_analysis_specs(args.analysis)
    result = compare_cache_intervention_profiles(analyses)
    write_profile_comparison_json(result, args.output_json)
    write_profile_comparison_markdown(result, args.output_md)
    print(f"wrote cache intervention profile comparison to {args.output_json}")
    print(f"wrote cache intervention profile comparison to {args.output_md}")


def _load_analysis_specs(specs: list[str]) -> dict[str, dict]:
    analyses = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"analysis spec must be KEY=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        label = label.strip()
        if not label or label in analyses:
            raise ValueError(f"analysis labels must be non-empty and unique: {label!r}")
        path = Path(raw_path).expanduser()
        with path.open("r", encoding="utf-8") as handle:
            analyses[label] = json.load(handle)
    return analyses


if __name__ == "__main__":
    main()
