from __future__ import annotations

from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np

from .cache_tensor_artifact import load_cache_tensor_artifact
from .cache_tensor_factorial import cache_tensor_regions
from .probe_readout import CELL_KEYS


def analyze_pairing_seed_holdout(
    analyses: dict[str, dict[str, Any]],
    *,
    analysis_paths: dict[str, Path],
    metadata: dict[str, dict[str, Any]],
    image_bands: int = 4,
) -> dict[str, Any]:
    if set(analyses) != set(metadata) or set(analyses) != set(analysis_paths):
        raise ValueError("analyses, paths and hierarchy metadata must align")
    if image_bands < 1:
        raise ValueError("image_bands must be positive")
    targets = {
        (a["model_id"], a["layer_index"], a["tensor"]) for a in analyses.values()
    }
    if len(targets) != 1:
        raise ValueError("holdout analysis requires one model/layer/tensor target")
    validate_pairing_holdout_metadata(metadata)
    vectors = {}
    reference_regions = None
    reference_shape = None
    for label, analysis in analyses.items():
        vector, regions = _load_interaction(analysis)
        if reference_shape is None:
            reference_shape, reference_regions = vector.shape, regions
        if vector.shape != reference_shape or regions != reference_regions:
            raise ValueError(f"tensor shape or token regions differ for {label}")
        vectors[label] = vector
    assert reference_shape is not None and reference_regions is not None
    if len(reference_shape) != 4 or reference_shape[0] != 1:
        raise ValueError("head views require [1, KV heads, sequence, dimension]")

    views = []
    for region in ("image_tokens", "post_image"):
        positions = reference_regions[region]
        if not positions:
            raise ValueError(f"holdout analysis requires identified {region} positions")
        region_vectors = {k: np.take(v, positions, axis=-2) for k, v in vectors.items()}
        views.append(_view(region, region_vectors, metadata, primary=True))
        for head in range(reference_shape[1]):
            selected = {k: v[:, head : head + 1] for k, v in region_vectors.items()}
            views.append(
                _view(
                    f"{region}/head_{head}", selected, metadata, parent=region_vectors
                )
            )
        if region == "image_tokens":
            if len(positions) < image_bands:
                raise ValueError(
                    "image region has fewer positions than requested bands"
                )
            for band, indices in enumerate(
                np.array_split(np.arange(len(positions)), image_bands)
            ):
                selected = {
                    k: np.take(v, indices, axis=-2) for k, v in region_vectors.items()
                }
                record = _view(
                    f"image_tokens/band_{band}",
                    selected,
                    metadata,
                    parent=region_vectors,
                )
                record["sequence_positions"] = [
                    positions[int(index)] for index in indices
                ]
                views.append(record)

    model, layer, tensor = next(iter(targets))
    return {
        "schema_version": 1,
        "analysis_kind": "source_cache_pairing_seed_holdout",
        "model_id": model,
        "layer_index": layer,
        "tensor": tensor,
        "tensor_shape": list(reference_shape),
        "token_region_position_counts": {
            name: len(positions) for name, positions in reference_regions.items()
        },
        "within_image_span_non_image_positions": sorted(
            set(reference_regions["non_image_tokens"])
            - set(reference_regions["pre_image"])
            - set(reference_regions["post_image"])
        ),
        "source_pair_count": len(analyses),
        "pairing_family_count": len({v["pairing_family"] for v in metadata.values()}),
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "reference_policy": "normalize the sum of independently unit-normalized reference vectors",
        "analysis_paths": {k: str(v) for k, v in analysis_paths.items()},
        "hierarchy_metadata": metadata,
        "views": views,
        "interpretation_notes": [
            "Test seeds never contribute to reference directions or view selection.",
            "The exact correspondence test moves complete two-seed test-family blocks within broad class.",
            "Image bands describe token order; a two-dimensional patch map is not assumed.",
            "Head and band p-values are exploratory and unadjusted.",
        ],
    }


def evaluate_pairing_holdout(
    vectors: dict[str, np.ndarray], metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(vectors) != set(metadata):
        raise ValueError("vectors and hierarchy metadata must align")
    validate_pairing_holdout_metadata(metadata)
    shapes = {v.shape for v in vectors.values()}
    if len(shapes) != 1:
        raise ValueError("holdout vectors must have the same shape")
    units = {}
    for label, vector in vectors.items():
        flat = np.asarray(vector, dtype=np.float64).reshape(-1)
        if not np.isfinite(flat).all():
            raise ValueError(f"non-finite interaction vector: {label}")
        norm = float(np.linalg.norm(flat))
        if norm == 0:
            return {"available": False, "reason": f"zero interaction vector: {label}"}
        units[label] = flat / norm
    families = sorted({v["pairing_family"] for v in metadata.values()})
    family_broad = {
        f: next(v["broad_class"] for v in metadata.values() if v["pairing_family"] == f)
        for f in families
    }
    references = {}
    for family in families:
        labels = [
            k
            for k, v in metadata.items()
            if v["pairing_family"] == family and v["replicate"] in (1, 2)
        ]
        reference = units[labels[0]] + units[labels[1]]
        norm = float(np.linalg.norm(reference))
        if norm <= 1e-12:
            return {
                "available": False,
                "reason": f"cancelling reference direction: {family}",
            }
        references[family] = reference / norm

    rows = []
    family_scores = []
    for family in families:
        test_labels = sorted(
            k
            for k, v in metadata.items()
            if v["pairing_family"] == family and v["replicate"] in (3, 4)
        )
        scores = []
        alternatives = [
            f
            for f in families
            if f != family and family_broad[f] == family_broad[family]
        ]
        for label in test_labels:
            score = {f: float(np.dot(units[label], references[f])) for f in families}
            winners = [
                f for f in families if abs(score[f] - max(score.values())) <= 1e-12
            ]
            rows.append(
                {
                    "label": label,
                    "pairing_family": family,
                    "broad_class": family_broad[family],
                    "replicate": metadata[label]["replicate"],
                    "reference_cosines": score,
                    "own_family_cosine": score[family],
                    "same_broad_other_family_cosine": float(
                        np.mean([score[f] for f in alternatives])
                    ),
                    "margin": score[family]
                    - float(np.mean([score[f] for f in alternatives])),
                    "retrieved_family": winners[0] if len(winners) == 1 else None,
                    "retrieval_correct": len(winners) == 1 and winners[0] == family,
                }
            )
            scores.append([score[f] for f in families])
        family_scores.append(np.mean(scores, axis=0))
    matrix = np.asarray(family_scores)
    indices_by_broad = [
        [i for i, f in enumerate(families) if family_broad[f] == broad]
        for broad in sorted(set(family_broad.values()))
    ]
    family_margins = {}
    for i, family in enumerate(families):
        other = [
            j
            for j in range(len(families))
            if j != i and family_broad[families[j]] == family_broad[family]
        ]
        family_margins[family] = float(matrix[i, i] - np.mean(matrix[i, other]))
    observed = float(np.mean(list(family_margins.values())))
    null = []
    for assignments in product(
        *(permutations(indices) for indices in indices_by_broad)
    ):
        margins = []
        for indices, assigned in zip(indices_by_broad, assignments):
            for row, column in zip(indices, assigned):
                other = [j for j in indices if j != column]
                margins.append(matrix[row, column] - float(np.mean(matrix[row, other])))
        null.append(float(np.mean(margins)))
    extreme = sum(value >= observed - 1e-12 for value in null)
    return {
        "available": True,
        "test_seed_count": len(rows),
        "reference_labels": sorted(
            k for k, v in metadata.items() if v["replicate"] in (1, 2)
        ),
        "test_labels": sorted(
            k for k, v in metadata.items() if v["replicate"] in (3, 4)
        ),
        "own_family_cosine_mean": float(
            np.mean([r["own_family_cosine"] for r in rows])
        ),
        "other_family_cosine_mean": float(
            np.mean([r["same_broad_other_family_cosine"] for r in rows])
        ),
        "mean_family_margin": observed,
        "family_margins": family_margins,
        "positive_family_margin_count": sum(v > 0 for v in family_margins.values()),
        "retrieval_correct_count": sum(r["retrieval_correct"] for r in rows),
        "retrieval_tied_count": sum(r["retrieved_family"] is None for r in rows),
        "exact_block_test": {
            "assignment_count": len(null),
            "extreme_count": extreme,
            "p_greater": extreme / len(null),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "permutation_unit": "complete two-test-seed pairing-family block",
            "multiplicity_adjusted": False,
        },
        "test_seed_scores": rows,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in p_values):
        raise ValueError("p-values must be finite and between zero and one")
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    result = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * p_values[index]))
        result[index] = previous
    return result


def validate_pairing_holdout_metadata(metadata: dict[str, dict[str, Any]]) -> None:
    if not metadata:
        raise ValueError("holdout metadata is empty")
    families = {}
    for record in metadata.values():
        family = record["pairing_family"]
        families.setdefault(family, []).append(record)
    for family, records in families.items():
        if sorted(r["replicate"] for r in records) != [1, 2, 3, 4]:
            raise ValueError(f"family {family} requires unique replicates 1, 2, 3, 4")
        if len({r["broad_class"] for r in records}) != 1:
            raise ValueError(f"family {family} crosses broad classes")
    counts = {}
    for records in families.values():
        broad = records[0]["broad_class"]
        counts[broad] = counts.get(broad, 0) + 1
    if len(counts) != 2 or any(n < 2 or n > 4 for n in counts.values()):
        raise ValueError(
            "exact holdout test supports two broad classes with 2-4 families each"
        )


def _load_interaction(
    analysis: dict[str, Any],
) -> tuple[np.ndarray, dict[str, list[int]]]:
    if analysis.get("analysis_kind") != "source_cache_tensor_factorial_contrast":
        raise ValueError("expected a source-cache tensor factorial analysis")
    arrays = {}
    regions = None
    for cell in CELL_KEYS:
        record = analysis["cells"][cell]
        arrays[cell] = load_cache_tensor_artifact(
            Path(record["source_path"]), record["cache_tensor_artifact"]
        ).astype(np.float64)
        current = cache_tensor_regions(
            record["cache_token_layout"], sequence_length=arrays[cell].shape[-2]
        )
        if regions is not None and current != regions:
            raise ValueError("factorial cell token partitions differ")
        regions = current
    if len({v.shape for v in arrays.values()}) != 1:
        raise ValueError("factorial cell tensor shapes differ")
    assert regions is not None
    return arrays["jj"] - arrays["jm"] - arrays["mj"] + arrays["mm"], regions


def _view(
    name: str,
    vectors: dict[str, np.ndarray],
    metadata: dict[str, dict[str, Any]],
    *,
    primary: bool = False,
    parent: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    result = {
        "view": name,
        "primary": primary,
        **evaluate_pairing_holdout(vectors, metadata),
        "interaction_l2_norm_by_pair": {
            label: float(np.linalg.norm(vector.reshape(-1)))
            for label, vector in vectors.items()
        },
    }
    if parent is not None:
        result["interaction_energy_fraction_by_pair"] = {
            label: float(np.sum(vectors[label] ** 2) / np.sum(parent[label] ** 2))
            if np.any(parent[label])
            else None
            for label in vectors
        }
    return result


def format_pairing_seed_holdout(analysis: dict[str, Any]) -> str:
    lines = [
        "# Pairing Direction: Frozen-Reference Seed Validation",
        "",
        f"Model: `{analysis['model_id']}`, layer {analysis['layer_index']} `{analysis['tensor']}`.",
        "Reference replicates: 1, 2. Test replicates: 3, 4.",
        "",
        "| View | Own cosine | Other same-class cosine | Margin | Positive families | Retrieval | Exact p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view in analysis["views"]:
        if not view["available"]:
            lines.append(
                f"| {view['view']} | unavailable: {view['reason']} | | | | | |"
            )
            continue
        lines.append(
            f"| {view['view']} | {view['own_family_cosine_mean']:.5f} | "
            f"{view['other_family_cosine_mean']:.5f} | {view['mean_family_margin']:.5f} | "
            f"{view['positive_family_margin_count']}/{analysis['pairing_family_count']} | "
            f"{view['retrieval_correct_count']}/{view['test_seed_count']} | "
            f"{view['exact_block_test']['p_greater']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Head/band p-values are exploratory and unadjusted. Primary Holm correction is in the study summary.",
            "",
        ]
    )
    return "\n".join(lines)
