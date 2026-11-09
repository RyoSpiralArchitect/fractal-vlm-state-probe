"""Full-vector crossed reference scores; no acquisition or alignment fitting."""

import hashlib
import math

import numpy as np

from .stat_matched_source import CELLS, FLOORS, checked_norm, family_reference


def direction(cells):
    if set(cells) != set(CELLS):
        raise ValueError("complete factorial quartet required")
    values = {k: np.asarray(cells[k], dtype=np.float64).ravel(order="C") for k in CELLS}
    if (
        not values["mm"].size
        or len({v.shape for v in values.values()}) != 1
        or any(not np.isfinite(v).all() for v in values.values())
    ):
        raise ValueError("finite equal-sized cells required")
    z = np.ascontiguousarray(
        (values["jj"] - values["mj"]) - (values["jm"] - values["mm"]), dtype="<f8"
    )
    norm = checked_norm(z)
    scale = sum(checked_norm(values[k]) for k in CELLS)
    if not math.isfinite(scale):
        raise ValueError("cell scale overflow")
    return (z / norm if norm else None), {
        "dimension": z.size,
        "norm": norm,
        "cell_scale": scale,
        "vector_sha256": hashlib.sha256(z.tobytes()).hexdigest(),
    }


def average(values):
    return None if any(v is None for v in values) else float(np.mean(values))


def endpoints(margins, tests, families):
    rows = []
    for i, pair in enumerate(tests):
        fixed = [margins[0][b][i] for b in (1, 2, 3)]
        matched = [margins[b][b][i] for b in (1, 2, 3)]
        anchor = margins[0][0][i]
        f, m = average(fixed), average(matched)
        diff = average(
            [
                None if x is None or y is None else x - y
                for x, y in zip(matched, fixed, strict=True)
            ]
        )
        f = None if f is None or anchor is None else f - anchor
        m = None if m is None or anchor is None else m - anchor
        residual = None if any(v is None for v in (f, m, diff)) else m - f - diff
        if residual is not None and abs(residual) > 1e-12:
            raise ValueError("endpoint closure failed")
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "family": pair["pairing_family"],
                "fixed_change": f,
                "matched_change": m,
                "reference_difference": diff,
                "closure_residual": residual,
            }
        )
    family_rows = [
        {
            "family": family,
            **{
                k: average([r[k] for r in rows if r["family"] == family])
                for k in ("fixed_change", "matched_change", "reference_difference")
            },
        }
        for family in families
    ]
    overall = {
        k: average([r[k] for r in family_rows])
        for k in ("fixed_change", "matched_change", "reference_difference")
    }
    for row in [*family_rows, overall]:
        values = [
            row[k] for k in ("fixed_change", "matched_change", "reference_difference")
        ]
        if (
            all(v is not None for v in values)
            and abs(values[1] - values[0] - values[2]) > 1e-12
        ):
            raise ValueError("aggregate endpoint closure failed")
    return rows, family_rows, overall


def profiles(load_cells, spec, view):
    """Load a quartet once; share floor-independent arithmetic, never select floors."""
    pairs = spec["sources"]["pairs"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    groups = {p["pairing_family"]: p["broad_class"] for p in pairs}
    tests = [p for p in pairs if p["split"] == "test"]
    if len(families) != 8 or len(tests) != 16:
        raise ValueError("registered family/test denominator required")
    refs, ref_rows, records = {}, [], []
    for a in range(4):
        for family in families:
            donors = [
                p
                for p in pairs
                if p["split"] == "reference" and p["pairing_family"] == family
            ]
            if len(donors) != 2:
                raise ValueError("two reference replicates required")
            units, metadata = [], []
            for p in donors:
                unit, row = direction(load_cells(p["pair_id"], a, view))
                units.append(unit)
                metadata.append(row)
                records.append(
                    {"pair_id": p["pair_id"], "split": "reference", "shift": a, **row}
                )
            ref, reason = family_reference(units)
            refs[a, family] = ref
            ref_rows.append(
                {
                    "shift": a,
                    "family": family,
                    "unit_sum_norm": checked_norm(units[0] + units[1])
                    if all(u is not None for u in units)
                    else None,
                    "reason": reason,
                    "replicates": metadata,
                }
            )
    scores = np.full((4, 4, 16, 8), np.nan)
    test_meta = {}
    for b in range(4):
        for i, p in enumerate(tests):
            unit, row = direction(load_cells(p["pair_id"], b, view))
            test_meta[b, i] = row
            records.append(
                {"pair_id": p["pair_id"], "split": "test", "shift": b, **row}
            )
            if unit is not None:
                for a in range(4):
                    for j, family in enumerate(families):
                        ref = refs[a, family]
                        if ref is not None:
                            scores[a, b, i, j] = float(np.dot(unit, ref))
    for floor in FLOORS:
        masked = scores.copy()
        references = []
        for row in ref_rows:
            available = row["reason"] is None and all(
                r["norm"] > floor * r["cell_scale"] for r in row["replicates"]
            )
            j = families.index(row["family"])
            if not available:
                masked[row["shift"], :, :, j] = np.nan
            references.append(
                {k: v for k, v in row.items() if k != "replicates"}
                | {
                    "available": available,
                    "reason": row["reason"]
                    if row["reason"]
                    else (None if available else "below_amplitude_guard"),
                }
            )
        for (b, i), row in test_meta.items():
            if row["norm"] <= floor * row["cell_scale"]:
                masked[:, b, i, :] = np.nan
        own = np.full((4, 4, 16), np.nan)
        other = np.full_like(own, np.nan)
        for i, p in enumerate(tests):
            j = families.index(p["pairing_family"])
            peers = [
                k
                for k, f in enumerate(families)
                if k != j and groups[f] == p["broad_class"]
            ]
            if len(peers) != 3:
                raise ValueError("three same-class competitors required")
            own[:, :, i] = masked[:, :, i, j]
            other[:, :, i] = masked[:, :, i, peers].mean(axis=-1)
        margin = own - other

        def nullable(x):
            return np.where(np.isfinite(x), x, None).tolist()

        pair_rows, family_rows, overall = endpoints(nullable(margin), tests, families)
        interactions = []
        for row in records:
            available = row["norm"] > floor * row["cell_scale"]
            interactions.append(
                {
                    **row,
                    "threshold": floor * row["cell_scale"],
                    "available": available,
                    "reason": None
                    if available
                    else (
                        "exact_zero" if row["norm"] == 0 else "below_amplitude_guard"
                    ),
                }
            )
        yield {
            "view": view,
            "relative_floor": floor,
            "families": families,
            "test_pair_ids": [p["pair_id"] for p in tests],
            "scores": nullable(masked),
            "own": nullable(own),
            "other": nullable(other),
            "margins": nullable(margin),
            "pairs": pair_rows,
            "family_results": family_rows,
            "overall": overall,
            "references": references,
            "interactions": interactions,
            "available_test_pairs": {
                k: sum(r[k] is not None for r in pair_rows) for k in overall
            },
            "p_values": None,
            "confidence_intervals": None,
            "equivalence_tested": False,
        }


def compare(actual, expected, *, path="root"):
    """Strict missingness/schema comparison with registered absolute numeric tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"schema differs at {path}")
        return max(
            [0.0]
            + [compare(actual[k], v, path=f"{path}/{k}") for k, v in expected.items()]
        )
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"shape differs at {path}")
        return max(
            [0.0]
            + [
                compare(a, b, path=f"{path}/{i}")
                for i, (a, b) in enumerate(zip(actual, expected, strict=True))
            ]
        )
    if expected is None or isinstance(expected, (bool, str, int)):
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"value/missingness differs at {path}")
        return 0.0
    if isinstance(actual, bool) or not isinstance(actual, (float, int)):
        raise ValueError(f"numeric type differs at {path}")
    error = abs(actual - expected)
    if not math.isfinite(actual) or not math.isfinite(expected) or error > 1e-12:
        raise ValueError(f"numeric value differs at {path}: {error}")
    return error


def primary_parity(profile, original):
    if profile["view"] not in ("encoder", "projector"):
        raise ValueError("original row parity is a learned-site check")
    for key in ("families", "test_pair_ids"):
        compare(profile[key], original[key])
    compare(profile["scores"][0], original["scores"])
    compare(profile["margins"][0], original["margins"])
    compare(profile["overall"]["fixed_change"], original["paired_margin_change"])
    compare(
        profile["available_test_pairs"]["fixed_change"],
        original["available_test_pairs"],
    )
