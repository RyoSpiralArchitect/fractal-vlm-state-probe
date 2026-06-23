from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.cross_palette_batch import (
    parse_pair_spec,
    prepare_cross_palette_factorial_batch,
)
from fractal_vlm_state_probe.processor_image_stats import load_hf_processor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare replicated 2x2 cross-palette factorial manifests and image-stat contrasts."
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="PAIR_ID=M_MANIFEST,J_MANIFEST",
        help="Repeat for each Mandelbrot/Julia source pair.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--processor-model",
        default=None,
        help="Optional HF model id/local path for processor-space image statistics.",
    )
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    processor = None
    if args.processor_model:
        processor = load_hf_processor(args.processor_model, trust_remote_code=args.trust_remote_code)

    summary = prepare_cross_palette_factorial_batch(
        pair_specs=[parse_pair_spec(raw) for raw in args.pair],
        output_root=args.output_root,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
        processor=processor,
        processor_model=args.processor_model,
        patch_size=args.patch_size,
    )
    print(f"wrote cross-palette factorial batch to {args.output_root / 'cross_palette_factorial_batch_summary.md'}")
    print(f"prepared {summary['pair_count']} pair(s)")


if __name__ == "__main__":
    main()
