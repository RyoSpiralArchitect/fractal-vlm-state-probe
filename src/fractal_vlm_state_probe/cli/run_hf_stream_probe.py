from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.hf_stream import HFStreamRunConfig, run_hf_stream_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Hugging Face VLM stream probe.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=None, help="HF model id or local path.")
    parser.add_argument("--local-path", default=None, help="Local model path; overrides --model.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--probe-max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--delivery-mode",
        choices=["visual_stream", "text_only_stream", "blank_visual_stream", "probe_only"],
        default="visual_stream",
        help="Control whether stream turns deliver real frames, text-only controls, blank images, or no stream turns.",
    )
    parser.add_argument(
        "--blank-rgb",
        type=_parse_rgb,
        default=(0, 0, 0),
        help="RGB triple for --delivery-mode blank_visual_stream, for example 0,0,0.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--probe-cache-policy",
        choices=["isolated", "shared_append", "no_cache"],
        default="isolated",
        help="HF replays full transcript; shared_append deliberately appends probe turns.",
    )
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument(
        "--trace-max-layers",
        type=int,
        default=4,
        help="Maximum layers to summarize. Use -1 for all layers.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-frame-artifacts",
        action="store_true",
        help="Do not copy selected frame images next to the output JSON.",
    )
    args = parser.parse_args()

    model_ref = args.local_path or args.model
    if not model_ref:
        parser.error("one of --model or --local-path is required")

    result = run_hf_stream_probe(
        HFStreamRunConfig(
            manifest_path=args.manifest,
            output_path=args.output,
            model_ref=model_ref,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            max_new_tokens=args.max_new_tokens,
            probe_max_new_tokens=args.probe_max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            delivery_mode=args.delivery_mode,
            blank_rgb=args.blank_rgb,
            dry_run=args.dry_run,
            probe_cache_policy=args.probe_cache_policy,
            trace_every=args.trace_every,
            trace_max_layers=None if args.trace_max_layers < 0 else args.trace_max_layers,
            torch_dtype=args.torch_dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
            include_frame_artifacts=not args.no_frame_artifacts,
        )
    )
    selected = result["stimulus"]["frame_count_selected"]
    print(f"wrote HF stream probe result for {selected} selected frames to {args.output}")


def _parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--blank-rgb must contain integers") from exc
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("--blank-rgb must be three integers in 0..255")
    return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    main()
