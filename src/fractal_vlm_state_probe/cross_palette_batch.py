from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .control_stimulus import render_cross_palette_manifest_transform
from .factorial_image_contrast import analyze_factorial_image_contrast
from .image_stats import analyze_manifest_batch
from .processor_image_stats import analyze_processor_manifest_batch
from .stimulus import write_json


@dataclass(frozen=True)
class CrossPalettePairSpec:
    pair_id: str
    mandelbrot_manifest: Path
    julia_manifest: Path
    mandelbrot_alias: str | None = None
    julia_alias: str | None = None


def prepare_cross_palette_factorial_batch(
    *,
    pair_specs: list[CrossPalettePairSpec],
    output_root: Path,
    max_frames: int | None = None,
    overwrite: bool = False,
    processor: Any | None = None,
    processor_model: str | None = None,
    patch_size: int | None = 14,
) -> dict[str, Any]:
    if not pair_specs:
        raise ValueError("at least one pair spec is required")
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for pair in pair_specs:
        records.append(
            _prepare_pair(
                pair=pair,
                output_root=output_root,
                max_frames=max_frames,
                overwrite=overwrite,
                processor=processor,
                processor_model=processor_model,
                patch_size=patch_size,
            )
        )

    summary = {
        "schema_version": 1,
        "batch_kind": "cross_palette_factorial_replication_batch",
        "output_root": str(output_root),
        "max_frames": max_frames,
        "processor_model": processor_model,
        "patch_size": patch_size,
        "pair_count": len(records),
        "records": records,
    }
    write_json(output_root / "cross_palette_factorial_batch_summary.json", summary)
    (output_root / "cross_palette_factorial_batch_summary.md").write_text(
        format_cross_palette_factorial_batch_markdown(summary),
        encoding="utf-8",
    )
    return summary


def parse_pair_spec(raw: str) -> CrossPalettePairSpec:
    if "=" not in raw:
        raise ValueError(f"pair spec must be PAIR_ID=M_MANIFEST,J_MANIFEST: {raw}")
    pair_id, raw_paths = raw.split("=", 1)
    pair_id = pair_id.strip()
    if not pair_id:
        raise ValueError("pair id must not be empty")
    parts = [part.strip() for part in raw_paths.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"pair spec must contain exactly two manifest paths: {raw}")
    return CrossPalettePairSpec(
        pair_id=pair_id,
        mandelbrot_manifest=Path(parts[0]).expanduser(),
        julia_manifest=Path(parts[1]).expanduser(),
    )


def format_cross_palette_factorial_batch_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-Palette Factorial Replication Batch",
        "",
        f"- Pair count: `{summary['pair_count']}`",
        f"- Max frames: `{summary.get('max_frames')}`",
        f"- Processor model: `{summary.get('processor_model')}`",
        f"- Patch size: `{summary.get('patch_size')}`",
        "",
        "## Pairs",
        "",
        "| Pair | MM | JJ | MJ | JM | Raw contrast | Processor contrast |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in summary["records"]:
        manifests = record["manifests"]
        analyses = record["analyses"]
        lines.append(
            "| "
            f"`{record['pair_id']}` | "
            f"`{manifests['mm']['condition_id']}` | "
            f"`{manifests['jj']['condition_id']}` | "
            f"`{manifests['mj']['condition_id']}` | "
            f"`{manifests['jm']['condition_id']}` | "
            f"`{analyses['raw_factorial_image_contrast_md']}` | "
            f"`{analyses.get('processor_factorial_image_contrast_md')}` |"
        )

    lines.extend(["", "## Suggested MLX Commands", ""])
    for record in summary["records"]:
        lines.append(f"### {record['pair_id']}")
        lines.append("")
        lines.append("```bash")
        lines.append(record["suggested_commands"]["mlx_manifest_probe_batch"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _prepare_pair(
    *,
    pair: CrossPalettePairSpec,
    output_root: Path,
    max_frames: int | None,
    overwrite: bool,
    processor: Any | None,
    processor_model: str | None,
    patch_size: int | None,
) -> dict[str, Any]:
    if not pair.mandelbrot_manifest.exists():
        raise FileNotFoundError(f"mandelbrot manifest does not exist: {pair.mandelbrot_manifest}")
    if not pair.julia_manifest.exists():
        raise FileNotFoundError(f"julia manifest does not exist: {pair.julia_manifest}")

    pair_dir = output_root / pair.pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)
    mm_manifest = pair.mandelbrot_manifest
    jj_manifest = pair.julia_manifest
    mm_condition_id = _condition_id(mm_manifest)
    jj_condition_id = _condition_id(jj_manifest)
    mj_condition_id = pair.mandelbrot_alias or f"{mm_condition_id}_spatial_{jj_condition_id}_palette"
    jm_condition_id = pair.julia_alias or f"{jj_condition_id}_spatial_{mm_condition_id}_palette"

    mj_manifest = pair_dir / "mj_mandelbrot_spatial_julia_palette" / "manifest.json"
    jm_manifest = pair_dir / "jm_julia_spatial_mandelbrot_palette" / "manifest.json"
    render_cross_palette_manifest_transform(
        mj_manifest.parent,
        source_manifest_path=mm_manifest,
        palette_manifest_path=jj_manifest,
        max_frames=max_frames,
        condition_id=mj_condition_id,
        overwrite=overwrite,
    )
    render_cross_palette_manifest_transform(
        jm_manifest.parent,
        source_manifest_path=jj_manifest,
        palette_manifest_path=mm_manifest,
        max_frames=max_frames,
        condition_id=jm_condition_id,
        overwrite=overwrite,
    )

    manifests = {
        "mm": mm_manifest,
        "jj": jj_manifest,
        "mj": mj_manifest,
        "jm": jm_manifest,
    }
    raw_stats = analyze_manifest_batch(list(manifests.values()), max_frames=max_frames, include_frame_stats=True)
    raw_stats_json = pair_dir / "raw_image_stats.json"
    write_json(raw_stats_json, raw_stats)
    raw_contrast = analyze_factorial_image_contrast(
        stats=raw_stats,
        mm_condition_id=mm_condition_id,
        jj_condition_id=jj_condition_id,
        mj_condition_id=mj_condition_id,
        jm_condition_id=jm_condition_id,
    )
    raw_contrast_json = pair_dir / "raw_factorial_image_contrast.json"
    raw_contrast_md = pair_dir / "raw_factorial_image_contrast.md"
    write_json(raw_contrast_json, raw_contrast)
    raw_contrast_md.write_text(_format_factorial_image_contrast_inline(raw_contrast), encoding="utf-8")

    analyses: dict[str, Any] = {
        "raw_image_stats_json": str(raw_stats_json),
        "raw_factorial_image_contrast_json": str(raw_contrast_json),
        "raw_factorial_image_contrast_md": str(raw_contrast_md),
    }
    if processor is not None:
        processor_stats = analyze_processor_manifest_batch(
            list(manifests.values()),
            processor=processor,
            patch_size=patch_size,
            max_frames=max_frames,
            include_frame_stats=True,
        )
        processor_stats["processor_model"] = processor_model
        processor_stats_json = pair_dir / "processor_image_stats.json"
        write_json(processor_stats_json, processor_stats)
        processor_contrast = analyze_factorial_image_contrast(
            stats=processor_stats,
            mm_condition_id=mm_condition_id,
            jj_condition_id=jj_condition_id,
            mj_condition_id=mj_condition_id,
            jm_condition_id=jm_condition_id,
        )
        processor_contrast_json = pair_dir / "processor_factorial_image_contrast.json"
        processor_contrast_md = pair_dir / "processor_factorial_image_contrast.md"
        write_json(processor_contrast_json, processor_contrast)
        processor_contrast_md.write_text(
            _format_factorial_image_contrast_inline(processor_contrast),
            encoding="utf-8",
        )
        analyses.update(
            {
                "processor_image_stats_json": str(processor_stats_json),
                "processor_factorial_image_contrast_json": str(processor_contrast_json),
                "processor_factorial_image_contrast_md": str(processor_contrast_md),
            }
        )

    manifest_records = {
        cell: {
            "path": str(path),
            "condition_id": _condition_id(path),
        }
        for cell, path in manifests.items()
    }
    return {
        "pair_id": pair.pair_id,
        "manifests": manifest_records,
        "analyses": analyses,
        "suggested_commands": {
            "mlx_manifest_probe_batch": _manifest_probe_command(
                pair_id=pair.pair_id,
                output_root=output_root,
                manifests=manifests,
                max_frames=max_frames,
            ),
            "factorial_cache_contrast": _factorial_cache_command(
                pair_id=pair.pair_id,
                output_root=output_root,
            ),
        },
    }


def _format_factorial_image_contrast_inline(analysis: dict[str, Any]) -> str:
    from .factorial_image_contrast import format_factorial_image_contrast_markdown

    return format_factorial_image_contrast_markdown(analysis)


def _condition_id(manifest_path: Path) -> str:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    condition = manifest.get("stimulus_condition") or {}
    condition_id = condition.get("condition_id")
    if not condition_id:
        raise ValueError(f"manifest missing stimulus_condition.condition_id: {manifest_path}")
    return str(condition_id)


def _manifest_probe_command(
    *,
    pair_id: str,
    output_root: Path,
    manifests: dict[str, Path],
    max_frames: int | None,
) -> str:
    batch_root = output_root / pair_id / "manifest_probe_seed_0"
    lines = [
        "python3 scripts/run_mlx_manifest_probe_batch.py \\",
        f"  --output-root {batch_root} \\",
        f"  --manifest mm={manifests['mm']} \\",
        f"  --manifest jj={manifests['jj']} \\",
        f"  --manifest mj={manifests['mj']} \\",
        f"  --manifest jm={manifests['jm']} \\",
        "  --probe-seeds 0 \\",
    ]
    if max_frames is not None:
        lines.append(f"  --max-frames {max_frames} \\")
    lines.extend(
        [
            "  --model HuggingFaceTB/SmolVLM2-2.2B-Instruct \\",
            "  --context-protocol cumulative_replay \\",
            "  --after-probe-protocol direct_multimodal_replay \\",
            "  --save-full-vocab-first-step \\",
            "  --generation-readout-top-k 20 \\",
            "  --temperature 0 \\",
            "  --probe-temperature 0 \\",
            "  --probe-max-tokens 2 \\",
            "  --cache-summary-max-layers -1 \\",
            "  --probe-preset forced_choice \\",
            "  --no-frame-artifacts",
        ]
    )
    return "\n".join(lines)


def _factorial_cache_command(*, pair_id: str, output_root: Path) -> str:
    pair_root = output_root / pair_id
    root = pair_root / "manifest_probe_seed_0" / "probe_seed_0"
    return "\n".join(
        [
            "python3 scripts/analyze_factorial_cache_contrast.py \\",
            f"  --mm {root}/mm_mlx.json \\",
            f"  --jj {root}/jj_mlx.json \\",
            f"  --mj {root}/mj_mlx.json \\",
            f"  --jm {root}/jm_mlx.json \\",
            f"  --output-json {pair_root}/factorial_cache_contrast.json \\",
            f"  --output-md {pair_root}/factorial_cache_contrast.md",
        ]
    )
