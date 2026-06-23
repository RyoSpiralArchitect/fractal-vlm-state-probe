from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.mlx_stream import StreamRunConfig, run_stream_probe
from fractal_vlm_state_probe.prompts import available_probe_presets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fractal frame stream probe.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=64)
    parser.add_argument("--probe-preset", choices=available_probe_presets(), default="default")
    parser.add_argument(
        "--generation-readout-top-k",
        type=int,
        default=10,
        help="Save top-k token logprobs for each generated token when the backend exposes them.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--probe-temperature",
        type=float,
        default=None,
        help="Temperature for before/mid/after probes. Defaults to --temperature.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=None,
        help="Optional base seed reset before probe phases. Uses seed, seed+1, seed+2 for before/mid/after.",
    )
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
        help=(
            "Use isolated branch caches for probes by default. shared_append is "
            "available for deliberate contamination studies."
        ),
    )
    parser.add_argument(
        "--cache-summary-every",
        type=int,
        default=10,
        help="Capture stream cache summaries every N frames plus first/mid/last. 0 disables.",
    )
    parser.add_argument(
        "--cache-summary-max-layers",
        type=int,
        default=4,
        help="Maximum layers to summarize per capture. Use -1 for all layers.",
    )
    parser.add_argument(
        "--no-prompt-cache-state",
        action="store_true",
        help="Disable MLX-VLM PromptCacheState even if it is available.",
    )
    parser.add_argument(
        "--no-frame-artifacts",
        action="store_true",
        help="Do not copy selected frame images next to the output JSON.",
    )
    args = parser.parse_args()

    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=args.manifest,
            output_path=args.output,
            model_id=args.model,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            max_tokens=args.max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            probe_preset=args.probe_preset,
            generation_readout_top_k=args.generation_readout_top_k,
            temperature=args.temperature,
            probe_temperature=args.probe_temperature,
            seed=args.seed,
            probe_seed=args.probe_seed,
            delivery_mode=args.delivery_mode,
            blank_rgb=args.blank_rgb,
            dry_run=args.dry_run,
            use_prompt_cache_state=not args.no_prompt_cache_state,
            probe_cache_policy=args.probe_cache_policy,
            cache_summary_every=args.cache_summary_every,
            cache_summary_max_layers=None
            if args.cache_summary_max_layers < 0
            else args.cache_summary_max_layers,
            include_frame_artifacts=not args.no_frame_artifacts,
        )
    )
    selected = result["stimulus"]["frame_count_selected"]
    print(f"wrote stream probe result for {selected} selected frames to {args.output}")


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
