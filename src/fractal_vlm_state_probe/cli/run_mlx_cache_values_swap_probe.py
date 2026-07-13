from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.mlx_cache_swap import (
    CREATIVE_IMPRESSION_PROMPT,
    ValuesSwapRunConfig,
    run_values_swap_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a pinpoint MLX PromptCacheState values-swap creative probe."
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--donor-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--source-label", default="source")
    parser.add_argument("--donor-label", default="donor")
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--probe-phase", choices=["mid", "after"], default="mid")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--probe-id", default="creative_visual_impression")
    parser.add_argument("--probe-prompt", default=CREATIVE_IMPRESSION_PROMPT)
    parser.add_argument("--cache-summary-every", type=int, default=10)
    parser.add_argument("--cache-summary-max-layers", type=int, default=4)
    parser.add_argument("--generation-readout-top-k", type=int, default=20)
    parser.add_argument("--no-frame-artifacts", action="store_true")
    parser.add_argument("--no-reciprocal", action="store_true")
    args = parser.parse_args()

    result = run_values_swap_probe(
        ValuesSwapRunConfig(
            source_manifest_path=args.source_manifest,
            donor_manifest_path=args.donor_manifest,
            output_path=args.output,
            model_id=args.model,
            source_label=args.source_label,
            donor_label=args.donor_label,
            layer_index=args.layer_index,
            probe_phase=args.probe_phase,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            max_tokens=args.max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            temperature=args.temperature,
            probe_temperature=args.probe_temperature,
            seed=args.seed,
            probe_seed=args.probe_seed,
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
    )
    print(f"wrote values-swap probe result to {args.output}")
    for label, record in result["probes"].items():
        text = " ".join((record.get("assistant_text") or "").split())
        print(f"{label}: {text[:160]}")


if __name__ == "__main__":
    main()
