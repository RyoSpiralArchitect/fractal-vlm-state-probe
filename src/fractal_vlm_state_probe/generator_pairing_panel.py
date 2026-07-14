from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from .control_stimulus import (
    GENERATED_CONTROL_KINDS,
    GeneratedControlKind,
    GeneratedControlSpec,
    render_generated_control_stimulus,
)
from .cross_palette_batch import (
    CrossPalettePairSpec,
    prepare_cross_palette_factorial_batch,
)
from .stimulus import write_json


MARGINAL_AUDIT_FIELDS = {
    "luminance_mean",
    "luminance_std",
    "luminance_entropy_bits",
    "colorfulness",
}
_GENERATED_SPEC_FIELDS = {
    "width",
    "height",
    "total_frames",
    "fps",
    "seed",
    "rgb",
    "cell_size",
    "sites",
    "dot_density",
    "motion_speed",
}


def prepare_generator_pairing_panel(
    *,
    config: dict[str, Any],
    output_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize a hierarchical generator-pairing panel and its 2x2 cells."""
    _validate_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    defaults = dict(config.get("source_defaults") or {})
    pair_records = []
    pair_specs = []

    for pair in config["pairs"]:
        pair_id = str(pair["pair_id"])
        source_a = _materialize_source(
            pair["source_a"],
            defaults=defaults,
            output_root=output_root,
            pair_id=pair_id,
            role="a",
            overwrite=overwrite,
        )
        source_b = _materialize_source(
            pair["source_b"],
            defaults=defaults,
            output_root=output_root,
            pair_id=pair_id,
            role="b",
            overwrite=overwrite,
        )
        pair_specs.append(
            CrossPalettePairSpec(
                pair_id=pair_id,
                mandelbrot_manifest=Path(source_a["manifest_path"]),
                julia_manifest=Path(source_b["manifest_path"]),
            )
        )
        pair_records.append(
            {
                "pair_id": pair_id,
                "broad_class": str(pair["broad_class"]),
                "pairing_family": str(pair["pairing_family"]),
                "replicate": int(pair["replicate"]),
                "source_a": source_a,
                "source_b": source_b,
            }
        )

    factorial_root = output_root / "factorials"
    factorial_summary = prepare_cross_palette_factorial_batch(
        pair_specs=pair_specs,
        output_root=factorial_root,
        max_frames=int(config.get("max_frames", defaults.get("total_frames", 1))),
        overwrite=overwrite,
    )
    factorial_by_pair = {
        record["pair_id"]: record for record in factorial_summary["records"]
    }
    for record in pair_records:
        record["factorial"] = factorial_by_pair[record["pair_id"]]

    source_hashes = [
        source["first_frame_sha256"]
        for record in pair_records
        for source in (record["source_a"], record["source_b"])
    ]
    broad_counts = Counter(record["broad_class"] for record in pair_records)
    pairing_counts = Counter(record["pairing_family"] for record in pair_records)
    summary = {
        "schema_version": 1,
        "analysis_kind": "generator_pairing_factorial_panel",
        "panel_id": str(config["panel_id"]),
        "output_root": str(output_root),
        "source_pair_is_atomic": True,
        "source_pair_count": len(pair_records),
        "source_count": len(source_hashes),
        "unique_source_first_frame_hash_count": len(set(source_hashes)),
        "broad_class_counts": dict(sorted(broad_counts.items())),
        "pairing_family_counts": dict(sorted(pairing_counts.items())),
        "broad_class_by_pair": {
            record["pair_id"]: record["broad_class"] for record in pair_records
        },
        "pairing_family_by_pair": {
            record["pair_id"]: record["pairing_family"] for record in pair_records
        },
        "records": pair_records,
        "raw_marginal_audit": _raw_marginal_audit(pair_records),
        "factorial_batch_summary_json": str(
            factorial_root / "cross_palette_factorial_batch_summary.json"
        ),
        "interpretation_notes": [
            "Broad classes contain multiple generator-pairing families; replicate seeds are nested inside pairing family.",
            "Each complete MM/JJ/MJ/JM source-pair factorial is one atomic directional unit.",
            "Simultaneously exchanging source A and source B leaves the interaction contrast invariant and is not an independent experiment.",
            "Raw marginal audit fields check only palette-preserved summaries; spatial and frequency structure remain intentionally free to differ.",
        ],
    }
    write_json(output_root / "generator_pairing_panel_summary.json", summary)
    (output_root / "generator_pairing_panel_summary.md").write_text(
        format_generator_pairing_panel_markdown(summary),
        encoding="utf-8",
    )
    return summary


def load_generator_pairing_panel_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("generator-pairing panel config must be a JSON object")
    return config


def format_generator_pairing_panel_markdown(summary: dict[str, Any]) -> str:
    audit = summary["raw_marginal_audit"]
    lines = [
        "# Generator-Pairing Factorial Panel",
        "",
        f"- Panel: `{summary['panel_id']}`",
        f"- Source-pair units: `{summary['source_pair_count']}`",
        f"- Unique first-frame source hashes: "
        f"`{summary['unique_source_first_frame_hash_count']}/{summary['source_count']}`",
        f"- Max audited raw marginal leak: "
        f"`{audit['max_abs_spatial_or_interaction_effect']:.8g}`",
        "",
        "## Hierarchy",
        "",
        "| Pair | Broad class | Pairing family | Replicate | Source A | Source B |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for record in summary["records"]:
        lines.append(
            f"| `{record['pair_id']}` | `{record['broad_class']}` | "
            f"`{record['pairing_family']}` | {record['replicate']} | "
            f"`{record['source_a']['condition_id']}` | "
            f"`{record['source_b']['condition_id']}` |"
        )
    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in summary["interpretation_notes"])
    lines.append("")
    return "\n".join(lines)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("generator-pairing panel config schema_version must be 1")
    if not str(config.get("panel_id") or "").strip():
        raise ValueError("generator-pairing panel config requires panel_id")
    pairs = config.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("generator-pairing panel config requires non-empty pairs")
    defaults = config.get("source_defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("source_defaults must be an object")
    unknown_defaults = sorted(set(defaults) - _GENERATED_SPEC_FIELDS)
    if unknown_defaults:
        raise ValueError(f"unsupported source defaults: {unknown_defaults}")

    pair_ids = set()
    replicate_keys = set()
    family_to_broad = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("every pair entry must be an object")
        pair_id = str(pair.get("pair_id") or "").strip()
        broad = str(pair.get("broad_class") or "").strip()
        family = str(pair.get("pairing_family") or "").strip()
        replicate = pair.get("replicate")
        if not pair_id or not broad or not family:
            raise ValueError(
                "every pair requires pair_id, broad_class, and pairing_family"
            )
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        if not isinstance(replicate, int) or replicate < 1:
            raise ValueError(f"replicate must be a positive integer: {pair_id}")
        replicate_key = (family, replicate)
        if replicate_key in replicate_keys:
            raise ValueError(
                f"duplicate replicate {replicate} for pairing family {family}"
            )
        replicate_keys.add(replicate_key)
        prior_broad = family_to_broad.setdefault(family, broad)
        if prior_broad != broad:
            raise ValueError(
                f"pairing family {family} occurs in multiple broad classes"
            )
        for role in ("source_a", "source_b"):
            _validate_source(pair.get(role), pair_id=pair_id, role=role)


def _validate_source(source: Any, *, pair_id: str, role: str) -> None:
    if not isinstance(source, dict):
        raise ValueError(f"{pair_id} {role} must be an object")
    has_manifest = bool(source.get("manifest"))
    has_kind = bool(source.get("kind"))
    if has_manifest == has_kind:
        raise ValueError(f"{pair_id} {role} requires exactly one of manifest or kind")
    if has_kind and source["kind"] not in GENERATED_CONTROL_KINDS:
        raise ValueError(
            f"{pair_id} {role} has unsupported generated kind: {source['kind']}"
        )
    unknown = sorted(
        set(source) - _GENERATED_SPEC_FIELDS - {"manifest", "kind", "condition_id"}
    )
    if unknown:
        raise ValueError(f"{pair_id} {role} has unsupported fields: {unknown}")


def _materialize_source(
    source: dict[str, Any],
    *,
    defaults: dict[str, Any],
    output_root: Path,
    pair_id: str,
    role: str,
    overwrite: bool,
) -> dict[str, Any]:
    if source.get("manifest"):
        manifest_path = Path(str(source["manifest"])).expanduser()
        if not manifest_path.exists():
            raise FileNotFoundError(f"source manifest does not exist: {manifest_path}")
        return _source_record(manifest_path, generated=False)

    values = dict(defaults)
    values.update(
        {key: value for key, value in source.items() if key in _GENERATED_SPEC_FIELDS}
    )
    required = {"width", "height", "total_frames", "fps", "seed"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{pair_id} source {role} missing generated fields: {missing}")
    kind = str(source["kind"])
    seed = int(values["seed"])
    condition_id = str(
        source.get("condition_id") or f"{pair_id}_{role}_{kind}_seed{seed}"
    )
    source_dir = output_root / "sources" / condition_id
    spec = GeneratedControlSpec(
        kind=cast(GeneratedControlKind, kind),
        width=int(values["width"]),
        height=int(values["height"]),
        total_frames=int(values["total_frames"]),
        fps=float(values["fps"]),
        seed=seed,
        rgb=tuple(int(value) for value in values.get("rgb", (0, 0, 0))),
        cell_size=int(values.get("cell_size", 24)),
        sites=int(values.get("sites", 32)),
        dot_density=float(values.get("dot_density", 0.035)),
        motion_speed=float(values.get("motion_speed", 1.0)),
        condition_id=condition_id,
    )
    render_generated_control_stimulus(spec, source_dir, overwrite=overwrite)
    return _source_record(source_dir / "manifest.json", generated=True)


def _source_record(manifest_path: Path, *, generated: bool) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    frames = manifest.get("frames") or []
    if not frames:
        raise ValueError(f"source manifest has no frames: {manifest_path}")
    condition = manifest.get("stimulus_condition") or {}
    condition_id = condition.get("condition_id")
    if not condition_id:
        raise ValueError(f"source manifest has no condition id: {manifest_path}")
    return {
        "manifest_path": str(manifest_path),
        "condition_id": str(condition_id),
        "kind": (manifest.get("stimulus_config") or {}).get("kind"),
        "generated_by_panel": generated,
        "frame_count": len(frames),
        "first_frame_sha256": str(frames[0]["sha256"]),
    }


def _raw_marginal_audit(pair_records: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    field_records = 0
    for pair in pair_records:
        path = Path(pair["factorial"]["analyses"]["raw_factorial_image_contrast_json"])
        with path.open("r", encoding="utf-8") as handle:
            analysis = json.load(handle)
        for record in analysis["records"]:
            if record["field"] not in MARGINAL_AUDIT_FIELDS:
                continue
            field_records += 1
            values.extend(
                [
                    float(record["abs_spatial_main_effect"]),
                    float(record["abs_interaction_effect"]),
                ]
            )
    return {
        "fields": sorted(MARGINAL_AUDIT_FIELDS),
        "field_record_count": field_records,
        "effect_value_count": len(values),
        "max_abs_spatial_or_interaction_effect": max(values),
    }
