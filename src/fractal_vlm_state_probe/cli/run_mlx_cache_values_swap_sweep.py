from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.mlx_cache_swap import (
    CREATIVE_IMPRESSION_PROMPT,
    ValuesSwapRunConfig,
)
from fractal_vlm_state_probe.mlx_cache_swap_sweep import (
    ValuesSwapSweepConfig,
    run_values_swap_probe_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a matched MLX values-swap sweep over probe seeds and cache layers."
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--donor-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--source-label", default="source")
    parser.add_argument("--donor-label", default="donor")
    parser.add_argument("--layer-index", action="append", dest="layer_indices", type=int)
    parser.add_argument("--probe-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--probe-phase", choices=["mid", "after"], default="mid")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--probe-id", default="creative_visual_impression")
    parser.add_argument("--probe-prompt", default=CREATIVE_IMPRESSION_PROMPT)
    parser.add_argument("--cache-summary-every", type=int, default=10)
    parser.add_argument("--cache-summary-max-layers", type=int, default=4)
    parser.add_argument("--generation-readout-top-k", type=int, default=50)
    parser.add_argument("--no-frame-artifacts", action="store_true")
    parser.add_argument("--no-reciprocal", action="store_true")
    parser.add_argument("--no-self-sham", action="store_true")
    args = parser.parse_args()

    layer_indices = tuple(args.layer_indices or [23])
    probe_seeds = tuple(args.probe_seeds)
    base = ValuesSwapRunConfig(
        source_manifest_path=args.source_manifest,
        donor_manifest_path=args.donor_manifest,
        output_path=args.output,
        model_id=args.model,
        source_label=args.source_label,
        donor_label=args.donor_label,
        layer_index=layer_indices[0],
        probe_phase=args.probe_phase,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        max_tokens=args.max_tokens,
        probe_max_tokens=args.probe_max_tokens,
        temperature=args.temperature,
        probe_temperature=args.probe_temperature,
        seed=args.seed,
        probe_seed=probe_seeds[0],
        probe_id=args.probe_id,
        probe_prompt=args.probe_prompt,
        cache_summary_every=args.cache_summary_every,
        cache_summary_max_layers=None
        if args.cache_summary_max_layers < 0
        else args.cache_summary_max_layers,
        generation_readout_top_k=args.generation_readout_top_k,
        include_frame_artifacts=not args.no_frame_artifacts,
        include_reciprocal=not args.no_reciprocal,
    )
    result = run_values_swap_probe_sweep(
        ValuesSwapSweepConfig(
            run=base,
            layer_indices=layer_indices,
            probe_seeds=probe_seeds,
            include_self_sham=not args.no_self_sham,
        )
    )
    print(f"wrote values-swap sweep to {args.output}")
    print(f"trials: {result['trial_count']}")


if __name__ == "__main__":
    main()
