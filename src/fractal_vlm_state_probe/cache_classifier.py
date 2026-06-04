from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .stimulus import write_json

CACHE_STAT_FIELDS = ("mean", "variance", "std", "abs_mean", "l2_norm")


@dataclass(frozen=True)
class CacheFeatureRecord:
    path: str
    label: str
    seed: int | None
    run_id: str
    features: dict[str, float]


def load_run_records(paths: list[Path]) -> list[CacheFeatureRecord]:
    records = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        records.append(run_to_feature_record(data, path))
    return records


def run_to_feature_record(data: dict[str, Any], path: Path) -> CacheFeatureRecord:
    stimulus = data.get("stimulus") or {}
    condition = stimulus.get("condition") or {}
    seed = (data.get("reproducibility") or {}).get("seed")
    run_id = path.name.removesuffix("_mlx.json").removesuffix(".json")
    label = condition.get("condition_id") or run_id
    features: dict[str, float] = {}

    for event in data.get("stream_events") or []:
        summary = event.get("cache_summary")
        if summary:
            _add_summary_features(
                features,
                summary,
                prefix=f"stream.frame_{int(event.get('frame_index', -1)):06d}",
            )

    probes = data.get("probes") or {}
    for phase in ("mid", "after"):
        for record in probes.get(phase, []) or []:
            summary = record.get("source_cache_summary_before_probe")
            if summary:
                _add_summary_features(
                    features,
                    summary,
                    prefix=f"probe_source.{phase}.{record.get('probe_id', 'probe')}",
                )

    return CacheFeatureRecord(
        path=str(path),
        label=str(label),
        seed=int(seed) if seed is not None else None,
        run_id=run_id,
        features=features,
    )


def analyze_cache_feature_records(records: list[CacheFeatureRecord]) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("at least two records are required")
    feature_names, matrix = _feature_matrix(records)
    labels = [record.label for record in records]
    seeds = [record.seed for record in records]

    duplicate_groups = _duplicate_feature_groups(records, feature_names)
    loo = _nearest_centroid_leave_one_out(matrix, labels)
    seed_eval = _nearest_centroid_leave_one_seed_out(matrix, labels, seeds)
    centroid_distances = _centroid_distances(matrix, labels)

    return {
        "schema_version": 1,
        "analysis_kind": "cache_summary_condition_classifier",
        "record_count": len(records),
        "feature_count": len(feature_names),
        "labels": dict(sorted(Counter(labels).items())),
        "seeds": sorted({seed for seed in seeds if seed is not None}),
        "records": [
            {
                "path": record.path,
                "label": record.label,
                "seed": record.seed,
                "run_id": record.run_id,
                "feature_count": len(record.features),
                "feature_hash": _feature_hash(record, feature_names),
            }
            for record in records
        ],
        "duplicate_feature_groups": duplicate_groups,
        "nearest_centroid_leave_one_out": loo,
        "nearest_centroid_leave_one_seed_out": seed_eval,
        "centroid_distances": centroid_distances,
        "interpretation_notes": [
            "This classifier uses saved cache-summary statistics, not raw hidden states or raw KV tensors.",
            "Perfect classification on deterministic duplicate seeds is reproducibility evidence, not independent stochastic replication.",
            "A positive result means condition information is present in the measured summary features.",
        ],
    }


def write_classifier_json(analysis: dict[str, Any], output_path: Path) -> None:
    write_json(output_path, analysis)


def write_classifier_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_classifier_markdown(analysis), encoding="utf-8")


def format_classifier_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Cache Summary Condition Classifier",
        "",
        f"- Records: `{analysis['record_count']}`",
        f"- Features: `{analysis['feature_count']}`",
        f"- Labels: `{_format_counts(analysis['labels'])}`",
        f"- Seeds: `{', '.join(str(seed) for seed in analysis['seeds'])}`",
        "",
        "## Classifier",
        "",
        _format_eval("Leave-one-out", analysis["nearest_centroid_leave_one_out"]),
        "",
        _format_eval("Leave-one-seed-out", analysis["nearest_centroid_leave_one_seed_out"]),
        "",
        "## Centroid Distances",
        "",
        "| Left | Right | Standardized Euclidean distance |",
        "| --- | --- | ---: |",
    ]
    for item in analysis["centroid_distances"]:
        lines.append(f"| `{item['left']}` | `{item['right']}` | `{item['distance']:.6g}` |")

    lines.extend(["", "## Duplicate Feature Groups", ""])
    if not analysis["duplicate_feature_groups"]:
        lines.append("No exact duplicate feature groups were found.")
    for group in analysis["duplicate_feature_groups"]:
        lines.append(
            "- "
            f"`{group['label']}` hash `{group['feature_hash']}` "
            f"seeds=`{', '.join(str(seed) for seed in group['seeds'])}` "
            f"count=`{group['count']}`"
        )

    lines.extend(["", "## Interpretation Notes", ""])
    for note in analysis["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def discover_run_paths(batch_root: Path) -> list[Path]:
    return sorted(path for path in batch_root.glob("seed_*/*_mlx.json") if path.is_file())


def _add_summary_features(features: dict[str, float], summary: dict[str, Any], *, prefix: str) -> None:
    for layer in summary.get("layers") or []:
        layer_index = layer.get("layer_index")
        for tensor_name in ("keys", "values"):
            tensor = layer.get(tensor_name)
            if not tensor:
                continue
            for field in CACHE_STAT_FIELDS:
                value = _as_float(tensor.get(field))
                if value is not None:
                    features[f"{prefix}.layer_{layer_index}.{tensor_name}.{field}"] = value


def _feature_matrix(records: list[CacheFeatureRecord]) -> tuple[list[str], np.ndarray]:
    feature_names = sorted({name for record in records for name in record.features})
    if not feature_names:
        raise ValueError("no cache-summary features found")
    matrix = np.array(
        [[record.features.get(name, 0.0) for name in feature_names] for record in records],
        dtype=np.float64,
    )
    return feature_names, matrix


def _nearest_centroid_leave_one_out(matrix: np.ndarray, labels: list[str]) -> dict[str, Any]:
    predictions = []
    for index in range(len(labels)):
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[index] = False
        prediction = _predict_nearest_centroid(
            train_matrix=matrix[train_mask],
            train_labels=[label for idx, label in enumerate(labels) if train_mask[idx]],
            test_row=matrix[index],
        )
        predictions.append({"index": index, "actual": labels[index], "predicted": prediction})
    return _classification_report(predictions)


def _nearest_centroid_leave_one_seed_out(
    matrix: np.ndarray,
    labels: list[str],
    seeds: list[int | None],
) -> dict[str, Any]:
    seed_values = sorted({seed for seed in seeds if seed is not None})
    if len(seed_values) < 2:
        return {"available": False, "reason": "fewer than two non-null seeds"}
    predictions = []
    for seed in seed_values:
        test_indices = [index for index, value in enumerate(seeds) if value == seed]
        train_indices = [index for index, value in enumerate(seeds) if value != seed]
        train_labels = [labels[index] for index in train_indices]
        if len(set(train_labels)) < 2:
            continue
        train_matrix = matrix[train_indices]
        for index in test_indices:
            prediction = _predict_nearest_centroid(
                train_matrix=train_matrix,
                train_labels=train_labels,
                test_row=matrix[index],
            )
            predictions.append(
                {
                    "index": index,
                    "seed": seed,
                    "actual": labels[index],
                    "predicted": prediction,
                }
            )
    report = _classification_report(predictions)
    report["available"] = bool(predictions)
    return report


def _predict_nearest_centroid(
    *,
    train_matrix: np.ndarray,
    train_labels: list[str],
    test_row: np.ndarray,
) -> str:
    standardized_train, standardized_test = _standardize_train_test(train_matrix, test_row)
    centroids = {}
    for label in sorted(set(train_labels)):
        label_rows = standardized_train[[index for index, value in enumerate(train_labels) if value == label]]
        centroids[label] = label_rows.mean(axis=0)
    distances = {
        label: float(np.linalg.norm(standardized_test - centroid))
        for label, centroid in centroids.items()
    }
    return min(distances, key=distances.get)


def _centroid_distances(matrix: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    standardized = _standardize(matrix)
    centroids = {}
    for label in sorted(set(labels)):
        rows = standardized[[index for index, value in enumerate(labels) if value == label]]
        centroids[label] = rows.mean(axis=0)

    records = []
    sorted_labels = sorted(centroids)
    for left_index, left in enumerate(sorted_labels):
        for right in sorted_labels[left_index + 1 :]:
            records.append(
                {
                    "left": left,
                    "right": right,
                    "distance": float(np.linalg.norm(centroids[left] - centroids[right])),
                }
            )
    return records


def _classification_report(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {"available": False, "accuracy": None, "predictions": [], "confusion": {}}
    correct = sum(1 for item in predictions if item["actual"] == item["predicted"])
    labels = sorted({item["actual"] for item in predictions} | {item["predicted"] for item in predictions})
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for item in predictions:
        confusion[item["actual"]][item["predicted"]] += 1
    return {
        "available": True,
        "accuracy": correct / len(predictions),
        "correct": correct,
        "total": len(predictions),
        "predictions": predictions,
        "confusion": confusion,
    }


def _duplicate_feature_groups(
    records: list[CacheFeatureRecord],
    feature_names: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[CacheFeatureRecord]] = defaultdict(list)
    for record in records:
        groups[(record.label, _feature_hash(record, feature_names))].append(record)
    return [
        {
            "label": label,
            "feature_hash": feature_hash,
            "count": len(items),
            "seeds": [item.seed for item in items],
            "paths": [item.path for item in items],
        }
        for (label, feature_hash), items in sorted(groups.items())
        if len(items) > 1
    ]


def _feature_hash(record: CacheFeatureRecord, feature_names: list[str]) -> str:
    payload = [(name, round(record.features.get(name, 0.0), 12)) for name in feature_names]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _standardize(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-12] = 1.0
    return (matrix - mean) / std


def _standardize_train_test(train_matrix: np.ndarray, test_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_matrix.mean(axis=0)
    std = train_matrix.std(axis=0)
    std[std < 1e-12] = 1.0
    return (train_matrix - mean) / std, (test_row - mean) / std


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))


def _format_eval(name: str, report: dict[str, Any]) -> str:
    if not report.get("available"):
        return f"### {name}\n\nNot available: {report.get('reason', 'no predictions')}"
    lines = [
        f"### {name}",
        "",
        f"- Accuracy: `{report['accuracy']:.3f}` ({report['correct']}/{report['total']})",
        "",
        "| Actual | " + " | ".join(f"`{label}`" for label in sorted(report["confusion"])) + " |",
        "| --- | " + " | ".join("---:" for _ in sorted(report["confusion"])) + " |",
    ]
    labels = sorted(report["confusion"])
    for actual in labels:
        row = report["confusion"][actual]
        lines.append("| `" + actual + "` | " + " | ".join(str(row[label]) for label in labels) + " |")
    return "\n".join(lines)
