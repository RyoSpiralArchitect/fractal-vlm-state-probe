from __future__ import annotations

from pathlib import Path
from typing import Any

from .stimulus import write_json


def analyze_cache_prefix_audits(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    records = []
    for label, run in runs.items():
        records.extend(_run_audit_records(label, run))
    available = [record for record in records if record["available"]]
    safe = [
        record
        for record in available
        if record.get("reuse_safe_under_token_prefix_contract") is True
    ]
    return {
        "schema_version": 1,
        "analysis_kind": "mlx_vlm_multimodal_cache_prefix_audit",
        "run_count": len(runs),
        "record_count": len(records),
        "available_record_count": len(available),
        "safe_record_count": len(safe),
        "unsafe_record_count": len(available) - len(safe),
        "runs": {
            label: {
                "model_id": run.get("model_id"),
                "run_kind": run.get("run_kind"),
                "context_policy": run.get("context_policy"),
            }
            for label, run in runs.items()
        },
        "records": records,
        "interpretation_notes": [
            "A multimodal cache branch is not treated as valid when the reconstructed prompt does not retain the full cached token prefix.",
            "MLX-VLM 0.4.4 trims caches by token-prefix length, so token/cache sequence-length mismatch is also unsafe.",
            "This audit does not invalidate fresh single-turn multimodal cache summaries or direct multimodal readouts.",
        ],
    }


def write_cache_prefix_audit_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_cache_prefix_audit_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_cache_prefix_audit_markdown(analysis), encoding="utf-8")


def format_cache_prefix_audit_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Multimodal Cache Prefix Audit",
        "",
        f"- Runs: `{analysis['run_count']}`",
        f"- Available audits: `{analysis['available_record_count']}`",
        f"- Safe: `{analysis['safe_record_count']}`",
        f"- Unsafe: `{analysis['unsafe_record_count']}`",
        "",
        "| Run | Location | Source tokens | Formatted tokens | Common prefix | Prefix fraction | Cache lengths | Token/cache aligned | Safe |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for record in analysis["records"]:
        if not record["available"]:
            continue
        cache_lengths = ", ".join(
            str(value) for value in record["cache_sequence_lengths"]
        )
        lines.append(
            f"| `{record['run_label']}` | `{record['location']}` | "
            f"{record['source_token_count']} | {record['formatted_prompt_token_count']} | "
            f"{record['common_prefix_token_count']} | {record['common_prefix_fraction']:.6f} | "
            f"{cache_lengths or 'n/a'} | `{record['token_cache_length_aligned']}` | "
            f"`{record['reuse_safe_under_token_prefix_contract']}` |"
        )
    lines.extend(["", "## Interpretation Notes", ""])
    lines.extend(f"- {note}" for note in analysis.get("interpretation_notes", []))
    lines.append("")
    return "\n".join(lines)


def _run_audit_records(label: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for index, event in enumerate(run.get("stream_events") or []):
        records.append(
            _audit_record(
                label,
                f"stream_event:{index}:frame={event.get('frame_index')}",
                event.get("cache_prefix_audit"),
            )
        )
    for phase, probes in (run.get("probes") or {}).items():
        for probe in probes or []:
            records.append(
                _audit_record(
                    label,
                    f"probe:{phase}:{probe.get('probe_id')}",
                    probe.get("cache_prefix_audit"),
                )
            )
    return records


def _audit_record(
    label: str,
    location: str,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    value = dict(audit or {"available": False, "reason": "audit not recorded"})
    return {
        "run_label": label,
        "location": location,
        **value,
    }
