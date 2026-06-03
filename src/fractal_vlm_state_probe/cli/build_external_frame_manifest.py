from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.external_frames import (
    build_external_frame_manifest,
    load_condition,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a manifest for an external frame sequence."
    )
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--frame-glob", default="*")
    args = parser.parse_args()

    condition = load_condition(args.condition)
    manifest = build_external_frame_manifest(
        frames_dir=args.frames_dir,
        output_path=args.output,
        condition=condition,
        fps=args.fps,
        frame_glob=args.frame_glob,
    )
    print(f"wrote manifest for {len(manifest['frames'])} frames to {args.output}")


if __name__ == "__main__":
    main()

