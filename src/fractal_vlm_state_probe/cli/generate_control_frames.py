from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.control_stimulus import (
    GENERATED_CONTROL_KINDS,
    TRANSFORM_CONTROL_KINDS,
    GeneratedControlSpec,
    render_generated_control_stimulus,
    render_manifest_transform,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic visual control stimuli.")
    parser.add_argument("--kind", required=True, choices=GENERATED_CONTROL_KINDS + TRANSFORM_CONTROL_KINDS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--total-frames", type=int, default=50)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rgb", type=_parse_rgb, default=(0, 0, 0))
    parser.add_argument("--cell-size", type=int, default=24)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--dot-density", type=float, default=0.035)
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--source-frame-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.kind in GENERATED_CONTROL_KINDS:
        manifest = render_generated_control_stimulus(
            GeneratedControlSpec(
                kind=args.kind,
                width=args.width,
                height=args.height,
                total_frames=args.total_frames,
                fps=args.fps,
                seed=args.seed,
                rgb=args.rgb,
                cell_size=args.cell_size,
                sites=args.sites,
                dot_density=args.dot_density,
                motion_speed=args.motion_speed,
                condition_id=args.condition_id,
            ),
            args.output,
            overwrite=args.overwrite,
        )
    else:
        if args.source_manifest is None:
            parser.error("--source-manifest is required for transform controls")
        manifest = render_manifest_transform(
            args.output,
            source_manifest_path=args.source_manifest,
            transform_kind=args.kind,
            seed=args.seed,
            source_frame_index=args.source_frame_index,
            max_frames=args.max_frames,
            condition_id=args.condition_id,
            overwrite=args.overwrite,
        )

    print(f"wrote {len(manifest['frames'])} control frames and manifest to {args.output / 'manifest.json'}")


def _parse_rgb(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--rgb must contain integers") from exc
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("--rgb must be three integers in 0..255")
    return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    main()
