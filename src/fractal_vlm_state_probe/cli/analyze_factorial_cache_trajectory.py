from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.factorial_cache_trajectory import (
    analyze_factorial_cache_trajectory,
    write_factorial_cache_trajectory_json,
    write_factorial_cache_trajectory_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze factorial cache argmax trajectories across replay lengths."
    )
    parser.add_argument("--analysis", action="append", required=True, metavar="KEY=PATH")
    parser.add_argument("--phase", default="after")
    parser.add_argument("--probe-id", default="forced_family_choice")
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
    result = analyze_factorial_cache_trajectory(
        analyses,
        analysis_paths=paths,
        phase=args.phase,
        probe_id=args.probe_id,
    )
    write_factorial_cache_trajectory_json(result, args.output_json)
    write_factorial_cache_trajectory_markdown(result, args.output_md)
    print(f"wrote factorial cache trajectory to {args.output_md}")


if __name__ == "__main__":
    main()
