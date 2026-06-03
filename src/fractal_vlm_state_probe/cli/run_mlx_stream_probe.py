from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.mlx_stream import StreamRunConfig, run_stream_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fractal frame stream probe.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
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
            temperature=args.temperature,
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


if __name__ == "__main__":
    main()
