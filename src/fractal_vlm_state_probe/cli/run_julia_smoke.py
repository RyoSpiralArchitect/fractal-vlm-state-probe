from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.mlx_stream import StreamRunConfig, run_stream_probe
from fractal_vlm_state_probe.stimulus import load_spec, render_stimulus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Julia smoke stimulus and optionally run MLX-VLM."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/julia_smoke.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/smoke/julia_b"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-mlx", action="store_true")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=48)
    parser.add_argument("--cache-summary-max-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    stimulus_dir = args.output_root / "stimulus"
    spec = load_spec(args.config)
    manifest = render_stimulus(spec, stimulus_dir, overwrite=args.overwrite)
    manifest_path = stimulus_dir / "manifest.json"
    print(f"wrote Julia stimulus with {len(manifest['frames'])} frames to {manifest_path}")

    if args.run_mlx:
        output_path = args.output_root / "julia_b_mlx.json"
        result = run_stream_probe(
            StreamRunConfig(
                manifest_path=manifest_path,
                output_path=output_path,
                model_id=args.model,
                max_frames=args.max_frames,
                max_tokens=args.max_tokens,
                probe_max_tokens=args.probe_max_tokens,
                cache_summary_max_layers=args.cache_summary_max_layers,
                seed=args.seed,
            )
        )
        print(
            "wrote Julia MLX probe for "
            f"{result['stimulus']['frame_count_selected']} frames to {output_path}"
        )


if __name__ == "__main__":
    main()
