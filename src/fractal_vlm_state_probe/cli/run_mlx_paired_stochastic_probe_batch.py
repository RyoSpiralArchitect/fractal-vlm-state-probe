from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from fractal_vlm_state_probe.compare_runs import (
    compare_runs,
    load_run,
    write_comparison_json,
    write_comparison_markdown,
)
from fractal_vlm_state_probe.control_stimulus import render_blank_stimulus
from fractal_vlm_state_probe.mlx_stream import StreamRunConfig, run_stream_probe
from fractal_vlm_state_probe.paired_stochastic import (
    analyze_paired_stochastic_batch,
    write_paired_stochastic_json,
    write_paired_stochastic_markdown,
)
from fractal_vlm_state_probe.stimulus import load_spec, render_stimulus, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired stochastic-probe MLX batch.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mandelbrot-config", type=Path, default=Path("configs/mandelbrot_smoke.json"))
    parser.add_argument("--julia-config", type=Path, default=Path("configs/julia_smoke.json"))
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--probe-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--stream-seed", type=int, default=20260604)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--cache-summary-every", type=int, default=10)
    parser.add_argument("--cache-summary-max-layers", type=int, default=4)
    parser.add_argument("--probe-cache-policy", choices=["isolated", "shared_append", "no_cache"], default="isolated")
    parser.add_argument("--blank-rgb", type=_parse_rgb, default=(0, 0, 0))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = _prepare_manifests(args)
    batch_records = []
    for probe_seed in args.probe_seeds:
        seed_dir = args.output_root / f"probe_seed_{probe_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run_paths = {
            "null_blank": seed_dir / "null_blank_mlx.json",
            "mandelbrot": seed_dir / "mandelbrot_mlx.json",
            "julia": seed_dir / "julia_mlx.json",
        }
        for condition_key, manifest_path in manifests.items():
            _run_or_reuse(
                output_path=run_paths[condition_key],
                manifest_path=manifest_path,
                args=args,
                probe_seed=probe_seed,
            )
        comparisons = _write_seed_comparisons(seed_dir=seed_dir, run_paths=run_paths)
        batch_records.append(
            {
                "probe_seed": probe_seed,
                "stream_seed": args.stream_seed,
                "runs": {key: str(path) for key, path in run_paths.items()},
                "comparisons": comparisons,
            }
        )

    summary = {
        "schema_version": 1,
        "batch_kind": "mlx_paired_stochastic_probe_batch",
        "model": args.model,
        "stream_seed": args.stream_seed,
        "probe_seeds": args.probe_seeds,
        "frames": args.frames,
        "stream_temperature": args.temperature,
        "probe_temperature": args.probe_temperature,
        "probe_seed_policy": "same base probe seed is applied to all conditions; MLX run resets before/mid/after to seed, seed+1, seed+2",
        "conditions": {key: str(path) for key, path in manifests.items()},
        "records": batch_records,
    }
    write_json(args.output_root / "paired_stochastic_batch_summary.json", summary)
    (args.output_root / "paired_stochastic_batch_summary.md").write_text(
        _format_batch_summary(summary),
        encoding="utf-8",
    )

    analysis = analyze_paired_stochastic_batch(batch_records)
    write_paired_stochastic_json(analysis, args.output_root / "paired_stochastic_analysis.json")
    write_paired_stochastic_markdown(analysis, args.output_root / "paired_stochastic_analysis.md")
    print(f"wrote paired stochastic analysis to {args.output_root / 'paired_stochastic_analysis.md'}")


def _prepare_manifests(args: argparse.Namespace) -> dict[str, Path]:
    stimuli_root = args.output_root / "stimuli"
    mandelbrot_spec = replace(load_spec(args.mandelbrot_config), total_frames=args.frames)
    julia_spec = replace(load_spec(args.julia_config), total_frames=args.frames)
    mandelbrot_dir = stimuli_root / "mandelbrot"
    julia_dir = stimuli_root / "julia"
    null_dir = stimuli_root / "null_blank"

    if args.overwrite or not (mandelbrot_dir / "manifest.json").exists():
        render_stimulus(mandelbrot_spec, mandelbrot_dir, overwrite=args.overwrite)
    if args.overwrite or not (julia_dir / "manifest.json").exists():
        render_stimulus(julia_spec, julia_dir, overwrite=args.overwrite)
    if args.overwrite or not (null_dir / "manifest.json").exists():
        render_blank_stimulus(
            null_dir,
            width=mandelbrot_spec.width,
            height=mandelbrot_spec.height,
            total_frames=args.frames,
            fps=mandelbrot_spec.fps,
            rgb=args.blank_rgb,
            overwrite=args.overwrite,
        )
    return {
        "null_blank": null_dir / "manifest.json",
        "mandelbrot": mandelbrot_dir / "manifest.json",
        "julia": julia_dir / "manifest.json",
    }


def _run_or_reuse(
    *,
    output_path: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    probe_seed: int,
) -> None:
    if output_path.exists() and not args.overwrite:
        print(f"reusing {output_path}")
        return
    print(f"running probe_seed={probe_seed} manifest={manifest_path} -> {output_path}")
    run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id=args.model,
            max_frames=args.frames,
            max_tokens=args.max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            temperature=args.temperature,
            probe_temperature=args.probe_temperature,
            dry_run=args.dry_run,
            probe_cache_policy=args.probe_cache_policy,
            cache_summary_every=args.cache_summary_every,
            cache_summary_max_layers=args.cache_summary_max_layers,
            seed=args.stream_seed,
            probe_seed=probe_seed,
            delivery_mode="visual_stream",
        )
    )


def _write_seed_comparisons(
    *,
    seed_dir: Path,
    run_paths: dict[str, Path],
) -> dict[str, dict[str, str]]:
    comparisons = {
        "null_vs_mandelbrot": (run_paths["null_blank"], run_paths["mandelbrot"]),
        "null_vs_julia": (run_paths["null_blank"], run_paths["julia"]),
        "mandelbrot_vs_julia": (run_paths["mandelbrot"], run_paths["julia"]),
    }
    records = {}
    comparison_dir = seed_dir / "comparisons"
    for comparison_id, (left, right) in comparisons.items():
        comparison = compare_runs(load_run(left), load_run(right))
        md_path = comparison_dir / f"{comparison_id}.md"
        json_path = comparison_dir / f"{comparison_id}.json"
        write_comparison_markdown(comparison, md_path)
        write_comparison_json(comparison, json_path)
        records[comparison_id] = {"markdown": str(md_path), "json": str(json_path)}
    return records


def _format_batch_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# MLX Paired Stochastic Probe Batch",
        "",
        f"- Model: `{summary['model']}`",
        f"- Stream seed: `{summary['stream_seed']}`",
        f"- Probe seeds: `{', '.join(str(seed) for seed in summary['probe_seeds'])}`",
        f"- Frames per run: `{summary['frames']}`",
        f"- Stream temperature: `{summary['stream_temperature']}`",
        f"- Probe temperature: `{summary['probe_temperature']}`",
        "",
        "## Conditions",
        "",
    ]
    for condition_key, manifest_path in summary["conditions"].items():
        lines.append(f"- `{condition_key}`: `{manifest_path}`")
    lines.extend(["", "## Records", ""])
    for record in summary["records"]:
        lines.append(f"### Probe seed {record['probe_seed']}")
        lines.append("")
        for run_id, run_path in record["runs"].items():
            lines.append(f"- run `{run_id}`: `{run_path}`")
        for comparison_id, paths in record["comparisons"].items():
            lines.append(f"- comparison `{comparison_id}`: `{paths['markdown']}`")
        lines.append("")
    return "\n".join(lines)


def _parse_rgb(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--blank-rgb must contain integers") from exc
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("--blank-rgb must be three integers in 0..255")
    return parts[0], parts[1], parts[2]


if __name__ == "__main__":
    main()
