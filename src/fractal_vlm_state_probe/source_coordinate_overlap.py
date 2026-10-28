from __future__ import annotations

from itertools import product

import numpy as np

from .cache_direction_holdout import validate_pairing_holdout_metadata
from .permutation_maps import MAPS, NEW_MAPS

COMPONENTS = (
    "shared_fixed_content",
    "shared_moved_content",
    "noncoincident_content",
    "padding",
)
ATOL = 1e-12


def validate_config(config):
    expected = {
        "schema_version": 1,
        "study_id": "source_coordinate_overlap_v1",
        "maps": list(MAPS),
        "view": "raw_rgb",
        "components": list(COMPONENTS),
        "source_shape": [240, 320],
        "canvas_shape": [320, 320, 3],
        "content_rows": [40, 280],
        "reference_replicates": [1, 2],
        "test_replicates": [3, 4],
        "normalization": "unchanged_full_vector_norms_before_partition",
        "aggregate_map_scope": "three_new_maps_only",
        "aggregate_regimes": ["same_map", "cross_map"],
        "expected_map_pairs": 16,
        "expected_raw_vectors": 128,
        "planned_dot_products": 2048,
        "planned_additive_terms": 8192,
        "reconstruction_atol": ATOL,
        "ratio_denominator_floor": ATOL,
        "new_statistical_tests": 0,
        "new_processor_calls": 0,
        "new_model_forwards": 0,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("source-coordinate decomposition differs from registration")


def partition_maps(left, right, *, shape=(240, 320), padding_rows=40):
    n = int(np.prod(shape))
    for mapping in (left, right):
        if (
            mapping.shape != (n,)
            or not np.issubdtype(mapping.dtype, np.integer)
            or not np.array_equal(np.sort(mapping), np.arange(n))
        ):
            raise ValueError("source-coordinate map must be an integer bijection")
    if padding_rows < 0:
        raise ValueError("padding cannot be negative")
    categories = np.full((shape[0] + 2 * padding_rows, shape[1]), 3, dtype=np.uint8)
    coincident = left == right
    fixed = left == np.arange(n)
    categories[padding_rows : padding_rows + shape[0]] = np.where(
        coincident, np.where(fixed, 0, 1), 2
    ).reshape(shape)
    return categories


def partition_receipt(categories):
    counts = {name: int(np.sum(categories == i)) for i, name in enumerate(COMPONENTS)}
    content = categories.size - counts["padding"]
    return {
        "pixel_counts": counts,
        "canvas_pixels": int(categories.size),
        "content_pixels": content,
        "coincident_content_fraction": (counts[COMPONENTS[0]] + counts[COMPONENTS[1]])
        / content,
        "content_fractions": {k: counts[k] / content for k in COMPONENTS[:3]},
    }


def full_unit(vector):
    array = np.asarray(vector, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("normalization requires a finite nonempty vector")
    norm = float(np.linalg.norm(array))
    return (array / norm, norm) if norm else (None, 0.0)


def pairing_directions(reference, test, metadata):
    validate_pairing_holdout_metadata(metadata)
    if set(reference) != set(metadata) or set(test) != set(metadata):
        raise ValueError("reference/test metadata coverage differs")
    families = sorted({m["pairing_family"] for m in metadata.values()})
    refs, tests, norms = (
        {},
        {},
        {"reference_inputs": {}, "reference_sums": {}, "tests": {}},
    )
    for family in families:
        labels = sorted(
            k
            for k, m in metadata.items()
            if m["pairing_family"] == family and m["replicate"] in (1, 2)
        )
        units = []
        for label in labels:
            unit, norm = full_unit(reference[label])
            if unit is None:
                return None, None, f"zero reference input: {label}"
            units.append(unit)
            norms["reference_inputs"][label] = norm
        refs[family], norm = full_unit(units[0] + units[1])
        if norm <= ATOL:
            return None, None, f"cancelling reference: {family}"
        norms["reference_sums"][family] = norm
    for label, meta in metadata.items():
        if meta["replicate"] not in (3, 4):
            continue
        tests[label], norm = full_unit(test[label])
        if tests[label] is None:
            return None, None, f"zero test input: {label}"
        norms["tests"][label] = norm
    return refs, tests, norms


def linear_summary(scores, metadata):
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    if set(scores) != set(labels) or any(
        set(s) != set(families) for s in scores.values()
    ):
        raise ValueError("linear score coverage differs")
    broad = {
        f: next(m["broad_class"] for m in metadata.values() if m["pairing_family"] == f)
        for f in families
    }
    rows = []
    for label in labels:
        values, meta = scores[label], metadata[label]
        if not np.isfinite(list(values.values())).all():
            raise ValueError("nonfinite additive score")
        family = meta["pairing_family"]
        other = float(
            np.mean(
                [
                    v
                    for f, v in values.items()
                    if f != family and broad[f] == meta["broad_class"]
                ]
            )
        )
        rows.append(
            {
                "label": label,
                **meta,
                "own_family_cosine_contribution": values[family],
                "other_family_cosine_contribution": other,
                "margin": values[family] - other,
            }
        )
    margins = {
        f: float(np.mean([r["margin"] for r in rows if r["pairing_family"] == f]))
        for f in families
    }
    return {
        "test_seed_count": len(rows),
        "test_seed_margins": rows,
        "family_margins": margins,
        "mean_family_margin": float(np.mean(list(margins.values()))),
        "positive_family_margin_count": sum(v > 0 for v in margins.values()),
        "negative_family_margin_count": sum(v < 0 for v in margins.values()),
    }


def signed_ratio(value, total):
    return value / total if abs(total) > ATOL else None


def summarize_parts(full_scores, component_scores, metadata):
    full = linear_summary(full_scores, metadata)
    components = {c: linear_summary(s, metadata) for c, s in component_scores.items()}
    for item in components.values():
        item["mean_margin_fraction_of_full"] = signed_ratio(
            item["mean_family_margin"], full["mean_family_margin"]
        )
        item["family_margin_fractions_of_full"] = {
            f: signed_ratio(v, full["family_margins"][f])
            for f, v in item["family_margins"].items()
        }
    coincident = linear_summary(
        {
            label: {
                family: sum(component_scores[c][label][family] for c in COMPONENTS[:2])
                for family in values
            }
            for label, values in full_scores.items()
        },
        metadata,
    )
    coincident["mean_margin_fraction_of_full"] = signed_ratio(
        coincident["mean_family_margin"], full["mean_family_margin"]
    )
    max_error = max(
        abs(sum(item["family_margins"][f] for item in components.values()) - value)
        for f, value in full["family_margins"].items()
    )
    seed_error = max(
        abs(
            sum(item["test_seed_margins"][i]["margin"] for item in components.values())
            - row["margin"]
        )
        for i, row in enumerate(full["test_seed_margins"])
    )
    if (
        max(max_error, seed_error) > ATOL
        or abs(
            sum(c["mean_family_margin"] for c in components.values())
            - full["mean_family_margin"]
        )
        > ATOL
    ):
        raise ValueError("additive margins do not reconstruct full margins")
    return {
        "full": full,
        "components": components,
        "coincident_content": coincident,
        "family_margin_reconstruction_max_abs_error": max_error,
        "test_seed_margin_reconstruction_max_abs_error": seed_error,
    }


def decompose_pair(reference, test, metadata, categories):
    refs, tests, norms = pairing_directions(reference, test, metadata)
    if refs is None:
        return {"available": False, "reason": norms}
    if any(
        a.shape != (*categories.shape, 3) for a in [*refs.values(), *tests.values()]
    ):
        raise ValueError("directions and coordinate partition shapes differ")
    full_scores, parts, dots = {}, {c: {} for c in COMPONENTS}, []
    masks = {c: categories == i for i, c in enumerate(COMPONENTS)}
    for label, vector in tests.items():
        full_scores[label] = {}
        for c in COMPONENTS:
            parts[c][label] = {}
        for family, ref in refs.items():
            # Normalize once on the full arrays; these are signed score terms, not masked cosines.
            products = vector * ref
            values = {c: float(np.sum(products[masks[c]])) for c in COMPONENTS}
            full = float(np.dot(vector.ravel(), ref.ravel()))
            error = abs(sum(values.values()) - full)
            if error > ATOL:
                raise ValueError(
                    "additive dot product does not reconstruct the full cosine"
                )
            full_scores[label][family] = full
            for c in COMPONENTS:
                parts[c][label][family] = values[c]
            dots.append(
                {
                    "test_label": label,
                    **metadata[label],
                    "reference_family": family,
                    "full_cosine": full,
                    "components": values,
                    "reconstruction_abs_error": error,
                }
            )
    return {
        "available": True,
        "normalization": norms,
        "full_scores": full_scores,
        "component_scores": parts,
        "dots": dots,
        **summarize_parts(full_scores, parts, metadata),
    }


def analyze(vectors, maps, metadata, baseline):
    if set(vectors) != set(MAPS) or set(maps) != set(MAPS):
        raise ValueError("all four maps are required")
    previous = [r for r in baseline["map_matrix"] if r["view"] == "raw_rgb"]
    if len(previous) != 16 or {
        (r["reference_map"], r["test_map"]) for r in previous
    } != set(product(MAPS, repeat=2)):
        raise ValueError("baseline raw matrix coverage differs")
    aggregates_old = [r for r in baseline["primary_tests"] if r["view"] == "raw_rgb"]
    if len(aggregates_old) != 2 or {r["regime"] for r in aggregates_old} != {
        "same_map",
        "cross_map",
    }:
        raise ValueError("baseline raw aggregate coverage differs")
    matrix, dots, panels = [], [], {}
    score_error, margin_error = 0.0, 0.0
    for left, right in product(MAPS, repeat=2):
        categories = partition_maps(maps[left], maps[right])
        result = decompose_pair(vectors[left], vectors[right], metadata, categories)
        old = next(
            r for r in previous if (r["reference_map"], r["test_map"]) == (left, right)
        )
        if result["available"] != old["available"]:
            raise ValueError("decomposition availability differs from baseline")
        row = {
            "reference_map": left,
            "test_map": right,
            "partition": partition_receipt(categories),
        }
        panels[(left, right)] = result
        if result["available"]:
            expected = {
                s["label"]: s["reference_cosines"] for s in old["test_seed_scores"]
            }
            score_error = max(
                score_error,
                max(
                    abs(value - expected[label][f])
                    for label, values in result["full_scores"].items()
                    for f, value in values.items()
                ),
            )
            margin_error = max(
                margin_error,
                abs(result["full"]["mean_family_margin"] - old["mean_family_margin"]),
            )
            dots.extend(
                {"reference_map": left, "test_map": right, **d} for d in result["dots"]
            )
        matrix.append(
            {
                **row,
                **{
                    k: v
                    for k, v in result.items()
                    if k not in ("full_scores", "component_scores", "dots")
                },
            }
        )
    aggregates = []
    for regime in ("same_map", "cross_map"):
        pairs = [
            (a, b)
            for a in NEW_MAPS
            for b in NEW_MAPS
            if (a == b) == (regime == "same_map")
        ]
        selected = [panels[p] for p in pairs]
        row = {
            "regime": regime,
            "ordered_map_pairs": [list(p) for p in pairs],
            "available": all(r["available"] for r in selected),
        }
        old = next(r for r in aggregates_old if r["regime"] == regime)
        if row["available"] != old["available"]:
            raise ValueError("aggregate availability differs from baseline")
        if row["available"]:

            def average(field, component=None):
                values = [
                    r[field] if component is None else r[field][component]
                    for r in selected
                ]
                return {
                    label: {
                        f: float(np.mean([s[label][f] for s in values])) for f in scores
                    }
                    for label, scores in values[0].items()
                }

            scores = average("full_scores")
            parts = {c: average("component_scores", c) for c in COMPONENTS}
            row.update(summarize_parts(scores, parts, metadata))
            expected = {
                s["label"]: s["reference_cosines"] for s in old["test_seed_scores"]
            }
            score_error = max(
                score_error,
                max(
                    abs(value - expected[label][f])
                    for label, values in scores.items()
                    for f, value in values.items()
                ),
            )
            margin_error = max(
                margin_error,
                abs(row["full"]["mean_family_margin"] - old["mean_family_margin"]),
            )
        else:
            row["reason"] = "at least one required map pair is unavailable"
        aggregates.append(row)
    if score_error > ATOL or margin_error > ATOL:
        raise ValueError("full scores do not reproduce the frozen baseline")
    return {
        "matrix": matrix,
        "aggregates": aggregates,
        "dot_products": dots,
        "baseline_score_max_abs_error": score_error,
        "baseline_mean_margin_max_abs_error": margin_error,
        "planned_dot_products": 2048,
        "available_dot_products": len(dots),
        "new_statistical_tests": 0,
    }
