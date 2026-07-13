from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.cache_prefix_audit import (
    analyze_cache_prefix_audits,
    write_cache_prefix_audit_json,
    write_cache_prefix_audit_markdown,
)
from fractal_vlm_state_probe.compare_runs import load_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MLX-VLM multimodal prompt-cache prefix audits."
    )
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"run spec must be LABEL=PATH: {spec}")
        label, raw_path = spec.split("=", 1)
        if not label or label in runs:
            raise ValueError(f"run labels must be non-empty and unique: {label!r}")
        runs[label] = load_run(Path(raw_path).expanduser())
    analysis = analyze_cache_prefix_audits(runs)
    write_cache_prefix_audit_json(analysis, args.output_json)
    write_cache_prefix_audit_markdown(analysis, args.output_md)
    print(f"wrote cache prefix audit to {args.output_md}")


if __name__ == "__main__":
    main()
