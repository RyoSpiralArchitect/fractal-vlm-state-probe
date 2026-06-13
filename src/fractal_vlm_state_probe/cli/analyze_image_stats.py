from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.image_stats import (
    analyze_manifest_batch,
    analyze_manifest_image_stats,
    write_image_stats_json,
    write_image_stats_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze low-level image statistics for stimulus manifests.")
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--include-frame-stats",
        action="store_true",
        help="Include per-frame statistics in JSON. Aggregate-only JSON is used by default.",
    )
    args = parser.parse_args()

    if len(args.manifest) == 1:
        analysis = analyze_manifest_image_stats(
            args.manifest[0],
            max_frames=args.max_frames,
            include_frame_stats=args.include_frame_stats,
        )
    else:
        analysis = analyze_manifest_batch(
            args.manifest,
            max_frames=args.max_frames,
            include_frame_stats=args.include_frame_stats,
        )
    write_image_stats_json(analysis, args.output_json)
    write_image_stats_markdown(analysis, args.output_md)
    print(f"wrote image statistics to {args.output_md}")


if __name__ == "__main__":
    main()
