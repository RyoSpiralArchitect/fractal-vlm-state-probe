from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path
from typing import Any

from fractal_vlm_state_probe.compare_runs import (
    compare_runs,
    load_run,
    write_comparison_json,
    write_comparison_markdown,
)
from fractal_vlm_state_probe.mlx_cumulative_replay import (
    CumulativeReplayRunConfig,
    run_cumulative_replay_probe,
)
from fractal_vlm_state_probe.mlx_stream import (
    StreamRunConfig,
    _load_mlx_runtime,
    run_stream_probe,
)
from fractal_vlm_state_probe.paired_stochastic import (
    analyze_paired_stochastic_batch,
    write_paired_stochastic_json,
    write_paired_stochastic_markdown,
)
from fractal_vlm_state_probe.prompts import available_probe_presets
from fractal_vlm_state_probe.stimulus import write_json

_MANIFEST_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired stochastic-probe MLX batch over arbitrary manifests.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        metavar="KEY=PATH",
        help="Manifest to include in the batch. Repeat for each condition.",
    )
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    parser.add_argument(
        "--context-protocol",
        choices=["incremental_cache", "cumulative_replay"],
        default="incremental_cache",
        help="Incremental image turns or one ordered multi-image replay turn.",
    )
    parser.add_argument("--probe-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--stream-seed", type=int, default=20260604)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--probe-max-tokens", type=int, default=80)
    parser.add_argument("--probe-preset", choices=available_probe_presets(), default="default")
    parser.add_argument(
        "--generation-readout-top-k",
        type=int,
        default=10,
        help="Save top-k token logprobs for each generated token when the backend exposes them.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--probe-temperature", type=float, default=0.7)
    parser.add_argument("--cache-summary-every", type=int, default=10)
    parser.add_argument(
        "--cache-summary-max-layers",
        type=int,
        default=4,
        help="Maximum layers to summarize per capture. Use -1 for all layers.",
    )
    parser.add_argument("--probe-cache-policy", choices=["isolated", "shared_append", "no_cache"], default="isolated")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-frame-artifacts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifests = parse_manifest_specs(args.manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)

    batch_records = []
    runtime_holder: dict[str, Any] = {}
    for probe_seed in args.probe_seeds:
        seed_dir = args.output_root / f"probe_seed_{probe_seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run_paths = {key: seed_dir / f"{key}_mlx.json" for key in manifests}
        for key, manifest_path in manifests.items():
            _run_or_reuse(
                output_path=run_paths[key],
                manifest_path=manifest_path,
                args=args,
                probe_seed=probe_seed,
                runtime_holder=runtime_holder,
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
        "batch_kind": "mlx_manifest_paired_stochastic_probe_batch",
        "model": args.model,
        "context_protocol": args.context_protocol,
        "stream_seed": args.stream_seed,
        "probe_seeds": args.probe_seeds,
        "max_frames": args.max_frames,
        "frame_stride": args.frame_stride,
        "stream_temperature": args.temperature,
        "probe_temperature": args.probe_temperature,
        "probe_preset": args.probe_preset,
        "generation_readout_top_k": args.generation_readout_top_k,
        "cache_summary_every": args.cache_summary_every,
        "cache_summary_max_layers": args.cache_summary_max_layers,
        "probe_seed_policy": "same base probe seed is applied to all conditions; MLX phases use seed, seed+1, seed+2",
        "conditions": {key: str(path) for key, path in manifests.items()},
        "records": batch_records,
    }
    write_json(args.output_root / "manifest_batch_summary.json", summary)
    (args.output_root / "manifest_batch_summary.md").write_text(
        _format_manifest_batch_summary(summary),
        encoding="utf-8",
    )

    analysis = analyze_paired_stochastic_batch(batch_records)
    write_paired_stochastic_json(analysis, args.output_root / "paired_stochastic_analysis.json")
    write_paired_stochastic_markdown(analysis, args.output_root / "paired_stochastic_analysis.md")
    print(f"wrote manifest paired stochastic analysis to {args.output_root / 'paired_stochastic_analysis.md'}")


def parse_manifest_specs(raw_specs: list[str]) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"manifest spec must be KEY=PATH: {raw}")
        key, raw_path = raw.split("=", 1)
        key = key.strip()
        if not key or not _MANIFEST_KEY_PATTERN.match(key):
            raise ValueError(f"manifest key must contain only letters, digits, dot, underscore, or hyphen: {key}")
        if key in manifests:
            raise ValueError(f"duplicate manifest key: {key}")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"manifest does not exist: {path}")
        manifests[key] = path
    if len(manifests) < 2:
        raise ValueError("at least two manifests are required")
    return manifests


def _run_or_reuse(
    *,
    output_path: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    probe_seed: int,
    runtime_holder: dict[str, Any],
) -> None:
    if output_path.exists() and not args.overwrite:
        print(f"reusing {output_path}")
        return
    print(f"running probe_seed={probe_seed} manifest={manifest_path} -> {output_path}")
    if not args.dry_run and "mlx" not in runtime_holder:
        runtime_holder["mlx"] = _load_mlx_runtime(args.model)
    cache_summary_max_layers = (
        None if args.cache_summary_max_layers < 0 else args.cache_summary_max_layers
    )
    if args.context_protocol == "cumulative_replay":
        run_cumulative_replay_probe(
            CumulativeReplayRunConfig(
                manifest_path=manifest_path,
                output_path=output_path,
                model_id=args.model,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
                max_tokens=args.max_tokens,
                probe_max_tokens=args.probe_max_tokens,
                probe_preset=args.probe_preset,
                generation_readout_top_k=args.generation_readout_top_k,
                temperature=args.temperature,
                probe_temperature=args.probe_temperature,
                dry_run=args.dry_run,
                cache_summary_max_layers=cache_summary_max_layers,
                seed=args.stream_seed,
                probe_seed=probe_seed,
                include_frame_artifacts=not args.no_frame_artifacts,
            ),
            mlx_runtime=runtime_holder.get("mlx"),
        )
        return
    run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id=args.model,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            max_tokens=args.max_tokens,
            probe_max_tokens=args.probe_max_tokens,
            probe_preset=args.probe_preset,
            generation_readout_top_k=args.generation_readout_top_k,
            temperature=args.temperature,
            probe_temperature=args.probe_temperature,
            dry_run=args.dry_run,
            probe_cache_policy=args.probe_cache_policy,
            cache_summary_every=args.cache_summary_every,
            cache_summary_max_layers=cache_summary_max_layers,
            seed=args.stream_seed,
            probe_seed=probe_seed,
            delivery_mode="visual_stream",
            include_frame_artifacts=not args.no_frame_artifacts,
        ),
        mlx_runtime=runtime_holder.get("mlx"),
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


def _format_manifest_batch_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# MLX Manifest Paired Stochastic Probe Batch",
        "",
        f"- Model: `{summary['model']}`",
        f"- Context protocol: `{summary['context_protocol']}`",
        f"- Stream seed: `{summary['stream_seed']}`",
        f"- Probe seeds: `{', '.join(str(seed) for seed in summary['probe_seeds'])}`",
        f"- Max frames per run: `{summary['max_frames']}`",
        f"- Frame stride: `{summary['frame_stride']}`",
        f"- Stream temperature: `{summary['stream_temperature']}`",
        f"- Probe temperature: `{summary['probe_temperature']}`",
        f"- Probe preset: `{summary['probe_preset']}`",
        f"- Generation readout top-k: `{summary['generation_readout_top_k']}`",
        f"- Cache summary every: `{summary['cache_summary_every']}`",
        f"- Cache summary max layers: `{summary['cache_summary_max_layers']}`",
        "",
        "## Conditions",
        "",
    ]
    for key, manifest_path in summary["conditions"].items():
        lines.append(f"- `{key}`: `{manifest_path}`")
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


if __name__ == "__main__":
    main()
