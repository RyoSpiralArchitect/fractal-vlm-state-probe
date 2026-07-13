from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.cache_tensor_artifact import (
    parse_cache_tensor_capture_spec,
)
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.prompts import available_probe_presets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one ordered multi-image cumulative replay and probe its cache."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=64)
    parser.add_argument(
        "--probe-preset", choices=available_probe_presets(), default="forced_choice"
    )
    parser.add_argument(
        "--after-probe-protocol",
        choices=["direct_multimodal_replay", "legacy_cache_branch"],
        default="direct_multimodal_replay",
        help="Use a valid fresh multimodal read by default; legacy cache branching is audit-only.",
    )
    parser.add_argument(
        "--capture-direct-probe-cache",
        action="store_true",
        help="Also summarize the fresh direct-probe cache; off by default because source ACK cache is primary.",
    )
    parser.add_argument("--generation-readout-top-k", type=int, default=10)
    parser.add_argument(
        "--save-full-vocab-first-step",
        action="store_true",
        help="Save compressed full-vocabulary first-step logprobs for each probe.",
    )
    parser.add_argument(
        "--source-cache-only",
        action="store_true",
        help="Run only the fresh multimodal ACK forward and omit before/after probes.",
    )
    parser.add_argument(
        "--capture-cache-tensor",
        action="append",
        default=[],
        metavar="LAYER[:keys|values]",
        help="Save one trimmed source-cache tensor sidecar. Repeat for more layers.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--probe-temperature", type=float, default=0.0)
    parser.add_argument("--cache-summary-max-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-frame-artifacts", action="store_true")
    args = parser.parse_args()
    if args.source_cache_only and args.save_full_vocab_first_step:
        parser.error(
            "--source-cache-only cannot be combined with --save-full-vocab-first-step"
        )
    if args.source_cache_only and args.capture_direct_probe_cache:
        parser.error(
            "--source-cache-only cannot be combined with --capture-direct-probe-cache"
        )

    result = run_cumulative_replay_probe(
        CumulativeReplayRunConfig(
            manifest_path=args.manifest,
            output_path=args.output,
            model_id=args.model,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            max_tokens=args.max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            probe_preset=args.probe_preset,
            after_probe_protocol=args.after_probe_protocol,
            capture_direct_probe_cache=args.capture_direct_probe_cache,
            generation_readout_top_k=args.generation_readout_top_k,
            save_full_vocab_first_step=args.save_full_vocab_first_step,
            source_cache_only=args.source_cache_only,
            cache_tensor_captures=tuple(
                parse_cache_tensor_capture_spec(raw)
                for raw in args.capture_cache_tensor
            ),
            temperature=args.temperature,
            probe_temperature=args.probe_temperature,
            cache_summary_max_layers=None
            if args.cache_summary_max_layers < 0
            else args.cache_summary_max_layers,
            seed=args.seed,
            probe_seed=args.probe_seed,
            dry_run=args.dry_run,
            include_frame_artifacts=not args.no_frame_artifacts,
        )
    )
    print(f"wrote cumulative replay probe to {args.output}")
    print(f"frames replayed: {result['stimulus']['frame_count_selected']}")


if __name__ == "__main__":
    main()
