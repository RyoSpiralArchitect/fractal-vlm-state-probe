from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.control_stimulus import render_blank_stimulus


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic blank control frames.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--total-frames", type=int, default=50)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--rgb", type=_parse_rgb, default=(0, 0, 0))
    parser.add_argument("--condition-id", default="blank_visual_null")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = render_blank_stimulus(
        args.output,
        width=args.width,
        height=args.height,
        total_frames=args.total_frames,
        fps=args.fps,
        rgb=args.rgb,
        condition_id=args.condition_id,
        overwrite=args.overwrite,
    )
    print(f"wrote {len(manifest['frames'])} blank frames and manifest to {args.output / 'manifest.json'}")


def _parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rgb must contain integers") from exc
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("--rgb must be three integers in 0..255")
    return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    main()
