from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

from fractal_vlm_state_probe.compare_runs import (
    compare_runs,
    load_run,
    write_comparison_json,
    write_comparison_markdown,
)
from fractal_vlm_state_probe.control_stimulus import (
    GENERATED_CONTROL_KINDS,
    GeneratedControlSpec,
    render_blank_stimulus,
    render_generated_control_stimulus,
)
from fractal_vlm_state_probe.mlx_stream import StreamRunConfig, run_stream_probe
from fractal_vlm_state_probe.paired_stochastic import (
    analyze_paired_stochastic_batch,
    write_paired_stochastic_json,
    write_paired_stochastic_markdown,
)
from fractal_vlm_state_probe.stimulus import load_spec, render_stimulus, write_json

FRACTAL_CONDITIONS = ("mandelbrot", "julia")
NON_FRACTAL_GENERATED_CONDITIONS = tuple(kind for kind in GENERATED_CONTROL_KINDS if kind != "blank")
PATTERN_CONDITION_CHOICES = ("null_blank",) + FRACTAL_CONDITIONS + NON_FRACTAL_GENERATED_CONDITIONS
DEFAULT_PATTERN_CONDITIONS = (
    "null_blank",
    "mandelbrot",
    "julia",
    "checkerboard",
    "voronoi",
    "quasicrystal",
    "white_noise",
    "blue_noise",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired stochastic-probe MLX batch over pattern controls.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--conditions", nargs="+", choices=PATTERN_CONDITION_CHOICES, default=list(DEFAULT_PATTERN_CONDITIONS))
    parser.add_argument("--mandelbrot-config", type=Path, default=Path("configs/mandelbrot_smoke.json"))
    parser.add_argument("--julia-config", type=Path, default=Path("configs/julia_smoke.json"))
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument("--probe-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--stream-seed", type=int, default=20260604)
    parser.add_argument("--stimulus-seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--cache-summary-every", type=int, default=10)
    parser.add_argument("--cache-summary-max-layers", type=int, default=4)
    parser.add_argument("--probe-cache-policy", choices=["isolated", "shared_append", "no_cache"], default="isolated")
    parser.add_argument("--blank-rgb", type=_parse_rgb, default=(0, 0, 0))
    parser.add_argument("--cell-size", type=int, default=24)
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--dot-density", type=float, default=0.035)
    parser.add_argument("--motion-speed", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = prepare_pattern_manifests(
        output_root=args.output_root,
        conditions=args.conditions,
        mandelbrot_config=args.mandelbrot_config,
        julia_config=args.julia_config,
        frames=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stimulus_seed=args.stimulus_seed,
        blank_rgb=args.blank_rgb,
        cell_size=args.cell_size,
        sites=args.sites,
        dot_density=args.dot_density,
        motion_speed=args.motion_speed,
        overwrite=args.overwrite,
    )

    batch_records = []
    for probe_seed in args.probe_seeds:
        seed_dir = args.output_root / f"probe_seed_{probe_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run_paths = {condition_key: seed_dir / f"{condition_key}_mlx.json" for condition_key in manifests}
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
                "stimulus_seed": args.stimulus_seed,
                "runs": {key: str(path) for key, path in run_paths.items()},
                "comparisons": comparisons,
            }
        )

    summary = {
        "schema_version": 1,
        "batch_kind": "mlx_pattern_paired_stochastic_probe_batch",
        "model": args.model,
        "stream_seed": args.stream_seed,
        "stimulus_seed": args.stimulus_seed,
        "probe_seeds": args.probe_seeds,
        "frames": args.frames,
        "stream_temperature": args.temperature,
        "probe_temperature": args.probe_temperature,
        "probe_seed_policy": "same base probe seed is applied to all conditions; MLX run resets before/mid/after to seed, seed+1, seed+2",
        "conditions": {key: str(path) for key, path in manifests.items()},
        "records": batch_records,
    }
    write_json(args.output_root / "pattern_batch_summary.json", summary)
    (args.output_root / "pattern_batch_summary.md").write_text(
        _format_pattern_batch_summary(summary),
        encoding="utf-8",
    )

    analysis = analyze_paired_stochastic_batch(batch_records)
    write_paired_stochastic_json(analysis, args.output_root / "paired_stochastic_analysis.json")
    write_paired_stochastic_markdown(analysis, args.output_root / "paired_stochastic_analysis.md")
    print(f"wrote pattern paired stochastic analysis to {args.output_root / 'paired_stochastic_analysis.md'}")


def prepare_pattern_manifests(
    *,
    output_root: Path,
    conditions: list[str],
    mandelbrot_config: Path,
    julia_config: Path,
    frames: int,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    stimulus_seed: int = 7,
    blank_rgb: tuple[int, int, int] = (0, 0, 0),
    cell_size: int = 24,
    sites: int = 32,
    dot_density: float = 0.035,
    motion_speed: float = 1.0,
    overwrite: bool = False,
) -> dict[str, Path]:
    condition_keys = _unique_conditions(conditions)
    stimuli_root = output_root / "stimuli"
    mandelbrot_spec = replace(load_spec(mandelbrot_config), total_frames=frames)
    julia_spec = replace(load_spec(julia_config), total_frames=frames)

    target_width = width or mandelbrot_spec.width
    target_height = height or mandelbrot_spec.height
    target_fps = fps or mandelbrot_spec.fps
    mandelbrot_spec = replace(mandelbrot_spec, width=target_width, height=target_height, fps=target_fps)
    julia_spec = replace(julia_spec, width=target_width, height=target_height, fps=target_fps)

    manifests: dict[str, Path] = {}
    for condition_key in condition_keys:
        condition_dir = stimuli_root / condition_key
        manifest_path = condition_dir / "manifest.json"
        if not overwrite and manifest_path.exists():
            manifests[condition_key] = manifest_path
            continue

        if condition_key == "null_blank":
            render_blank_stimulus(
                condition_dir,
                width=target_width,
                height=target_height,
                total_frames=frames,
                fps=target_fps,
                rgb=blank_rgb,
                overwrite=overwrite,
            )
        elif condition_key == "mandelbrot":
            render_stimulus(mandelbrot_spec, condition_dir, overwrite=overwrite)
        elif condition_key == "julia":
            render_stimulus(julia_spec, condition_dir, overwrite=overwrite)
        elif condition_key in NON_FRACTAL_GENERATED_CONDITIONS:
            render_generated_control_stimulus(
                GeneratedControlSpec(
                    kind=condition_key,  # type: ignore[arg-type]
                    width=target_width,
                    height=target_height,
                    total_frames=frames,
                    fps=target_fps,
                    seed=stimulus_seed,
                    cell_size=cell_size,
                    sites=sites,
                    dot_density=dot_density,
                    motion_speed=motion_speed,
                ),
                condition_dir,
                overwrite=overwrite,
            )
        else:
            raise ValueError(f"unsupported pattern condition: {condition_key}")
        manifests[condition_key] = manifest_path
    return manifests


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
    records = {}
    comparison_dir = seed_dir / "comparisons"
    for left_key, right_key in combinations(run_paths, 2):
        comparison_id = f"{left_key}_vs_{right_key}"
        comparison = compare_runs(load_run(run_paths[left_key]), load_run(run_paths[right_key]))
        md_path = comparison_dir / f"{comparison_id}.md"
        json_path = comparison_dir / f"{comparison_id}.json"
        write_comparison_markdown(comparison, md_path)
        write_comparison_json(comparison, json_path)
        records[comparison_id] = {"markdown": str(md_path), "json": str(json_path)}
    return records


def _format_pattern_batch_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# MLX Pattern Paired Stochastic Probe Batch",
        "",
        f"- Model: `{summary['model']}`",
        f"- Stream seed: `{summary['stream_seed']}`",
        f"- Stimulus seed: `{summary['stimulus_seed']}`",
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


def _unique_conditions(conditions: list[str]) -> list[str]:
    seen = set()
    output = []
    for condition in conditions:
        if condition in seen:
            raise ValueError(f"duplicate pattern condition: {condition}")
        if condition not in PATTERN_CONDITION_CHOICES:
            raise ValueError(f"unsupported pattern condition: {condition}")
        seen.add(condition)
        output.append(condition)
    return output


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
