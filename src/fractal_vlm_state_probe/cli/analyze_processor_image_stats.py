from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.processor_image_stats import (
    analyze_manifest_processor_image_stats,
    analyze_processor_manifest_batch,
    load_hf_processor,
    write_processor_image_stats_json,
    write_processor_image_stats_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze stimulus image statistics after model processor pixel_values conversion."
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--model", required=True, help="HF model id or local path used to load AutoProcessor.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--include-frame-stats",
        action="store_true",
        help="Include per-frame processor statistics in JSON. Aggregate-only JSON is used by default.",
    )
    args = parser.parse_args()

    processor = load_hf_processor(args.model, trust_remote_code=args.trust_remote_code)
    if len(args.manifest) == 1:
        analysis = analyze_manifest_processor_image_stats(
            args.manifest[0],
            processor=processor,
            patch_size=args.patch_size,
            max_frames=args.max_frames,
            include_frame_stats=args.include_frame_stats,
        )
    else:
        analysis = analyze_processor_manifest_batch(
            args.manifest,
            processor=processor,
            patch_size=args.patch_size,
            max_frames=args.max_frames,
            include_frame_stats=args.include_frame_stats,
        )
    analysis["processor_model"] = args.model
    write_processor_image_stats_json(analysis, args.output_json)
    write_processor_image_stats_markdown(analysis, args.output_md)
    print(f"wrote processor image statistics to {args.output_md}")


if __name__ == "__main__":
    main()
