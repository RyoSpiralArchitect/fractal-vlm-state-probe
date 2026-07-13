from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cache_tensor_replication import (
    analyze_cache_tensor_replication,
    write_cache_tensor_replication_json,
    write_cache_tensor_replication_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate full source-cache tensor directions across source pairs."
    )
    parser.add_argument(
        "--analysis", action="append", required=True, metavar="KEY=PATH"
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    analyses = {}
    paths = {}
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

    analysis = analyze_cache_tensor_replication(
        analyses,
        analysis_paths=paths,
    )
    write_cache_tensor_replication_json(analysis, args.output_json)
    write_cache_tensor_replication_markdown(analysis, args.output_md)
    print(f"wrote source-cache tensor replication to {args.output_md}")


if __name__ == "__main__":
    main()
