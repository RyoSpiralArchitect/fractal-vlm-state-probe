from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.cross_model_replication import (
    analyze_cross_model_replication,
    write_cross_model_replication_json,
    write_cross_model_replication_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize independent source-pair factorial replication across models."
    )
    parser.add_argument("--family-trajectory", required=True, type=Path)
    parser.add_argument("--frequency-trajectory", required=True, type=Path)
    parser.add_argument("--frame-count", default=1, type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    with args.family_trajectory.open("r", encoding="utf-8") as handle:
        family = json.load(handle)
    with args.frequency_trajectory.open("r", encoding="utf-8") as handle:
        frequency = json.load(handle)
    analysis = analyze_cross_model_replication(
        family,
        frequency,
        family_path=args.family_trajectory,
        frequency_path=args.frequency_trajectory,
        frame_count=args.frame_count,
    )
    write_cross_model_replication_json(analysis, args.output_json)
    write_cross_model_replication_markdown(analysis, args.output_md)
    print(f"wrote cross-model replication summary to {args.output_md}")


if __name__ == "__main__":
    main()
