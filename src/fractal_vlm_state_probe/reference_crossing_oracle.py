"""Independent blocked scalar reductions and native-array decoding for crossings."""

import hashlib
import math

import numpy as np

from .stat_matched_source_storage import inside

BLOCK = 32768


def load(root, receipt, view):
    path = inside(root, receipt["path"])
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    a = np.load(path, allow_pickle=False)
    if (
        digest.hexdigest() != receipt["sha256"]
        or list(a.shape) != receipt["shape"]
        or str(a.dtype) != receipt["dtype"]
        or hashlib.sha256(a.tobytes(order="C")).hexdigest() != receipt["word_sha256"]
    ):
        raise ValueError("oracle native file/word/layout mismatch")
    if view in ("encoder", "projector"):
        a = (a.astype("<u4") * np.uint32(65536)).view("<f4")
        if hashlib.sha256(a.tobytes()).hexdigest() != receipt["expanded_word_sha256"]:
            raise ValueError("oracle expanded native words mismatch")
    return a.astype(np.float64).ravel(order="C")


def dot(a, b):
    return math.fsum(
        float(np.sum(a[i : i + BLOCK] * b[i : i + BLOCK], dtype=np.float64))
        for i in range(0, len(a), BLOCK)
    )


def norm(a):
    # Match the registered unscaled norm receipts exactly; score reductions below
    # are independently blocked rather than using the production BLAS dot.
    v = float(np.linalg.norm(a))
    if not math.isfinite(v) or (v == 0 and np.any(a)):
        raise ValueError("oracle norm overflow/underflow")
    return v


def direction(cells):
    if set(cells) != {"mm", "mj", "jm", "jj"}:
        raise ValueError("oracle needs quartet")
    x = {k: np.array(v, dtype=np.float64).flatten(order="C") for k, v in cells.items()}
    if (
        len({a.shape for a in x.values()}) != 1
        or not x["mm"].size
        or any(not np.isfinite(a).all() for a in x.values())
    ):
        raise ValueError("oracle finite equal-sized cells required")
    z = np.subtract(np.subtract(x["jj"], x["mj"]), np.subtract(x["jm"], x["mm"]))
    n = norm(z)
    scale = sum(norm(x[k]) for k in ("mm", "mj", "jm", "jj"))
    if not math.isfinite(scale):
        raise ValueError("oracle cell scale overflow")
    return (z / n if n else None), {
        "dimension": len(z),
        "norm": n,
        "cell_scale": scale,
        "vector_sha256": hashlib.sha256(z.astype("<f8").tobytes()).hexdigest(),
    }


def mean(values):
    return None if any(x is None for x in values) else math.fsum(values) / len(values)


def profiles(load_cells, spec, view):
    pairs = spec["sources"]["pairs"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    tests = [p for p in pairs if p["split"] == "test"]
    groups = {p["pairing_family"]: p["broad_class"] for p in pairs}
    refs, ref_metadata, records = {}, {}, []
    for a in range(4):
        for family in families:
            donors = [
                p
                for p in pairs
                if p["split"] == "reference" and p["pairing_family"] == family
            ]
            if len(donors) != 2:
                raise ValueError("oracle needs two references")
            units, meta = [], []
            for p in donors:
                u, row = direction(load_cells(p["pair_id"], a, view))
                units.append(u)
                meta.append(row)
                records.append(
                    {"pair_id": p["pair_id"], "split": "reference", "shift": a, **row}
                )
            total = None if any(u is None for u in units) else units[0] + units[1]
            n = None if total is None else norm(total)
            ref = total / n if n is not None and n > 1e-12 else None
            reason = (
                "unavailable_reference_replicate"
                if total is None
                else ("cancelling_unit_references" if ref is None else None)
            )
            refs[a, family] = ref
            ref_metadata[a, family] = (meta, n, reason)
    base, test_meta = {}, {}
    for b in range(4):
        for i, p in enumerate(tests):
            u, row = direction(load_cells(p["pair_id"], b, view))
            test_meta[b, i] = row
            records.append(
                {"pair_id": p["pair_id"], "split": "test", "shift": b, **row}
            )
            for a in range(4):
                base[a, b, i] = [
                    None if u is None or refs[a, f] is None else dot(u, refs[a, f])
                    for f in families
                ]
    for floor in (1e-8, 1e-6, 1e-4):
        refs_ok, reference_rows = {}, []
        for (a, family), (meta, n, reason) in ref_metadata.items():
            ok = reason is None and all(
                r["norm"] > floor * r["cell_scale"] for r in meta
            )
            refs_ok[a, family] = ok
            reference_rows.append(
                {
                    "shift": a,
                    "family": family,
                    "unit_sum_norm": n,
                    "reason": reason or (None if ok else "below_amplitude_guard"),
                    "available": ok,
                }
            )
        scores, own, other, margins = [], [], [], []
        for a in range(4):
            ss, oo, cc, mm = [], [], [], []
            for b in range(4):
                s, o, c, m = [], [], [], []
                for i, p in enumerate(tests):
                    r = test_meta[b, i]
                    valid = r["norm"] > floor * r["cell_scale"]
                    row = [
                        v if valid and refs_ok[a, f] else None
                        for f, v in zip(families, base[a, b, i], strict=True)
                    ]
                    j = families.index(p["pairing_family"])
                    competitors = [
                        row[k]
                        for k, f in enumerate(families)
                        if k != j and groups[f] == p["broad_class"]
                    ]
                    co = mean(competitors)
                    s.append(row)
                    o.append(row[j])
                    c.append(co)
                    m.append(None if row[j] is None or co is None else row[j] - co)
                ss.append(s)
                oo.append(o)
                cc.append(c)
                mm.append(m)
            scores.append(ss)
            own.append(oo)
            other.append(cc)
            margins.append(mm)
        pair_rows = []
        for i, p in enumerate(tests):
            fixed = mean([margins[0][b][i] for b in (1, 2, 3)])
            matched = mean([margins[b][b][i] for b in (1, 2, 3)])
            differences = [
                None
                if margins[b][b][i] is None or margins[0][b][i] is None
                else margins[b][b][i] - margins[0][b][i]
                for b in (1, 2, 3)
            ]
            anchor = margins[0][0][i]
            f = None if fixed is None or anchor is None else fixed - anchor
            m = None if matched is None or anchor is None else matched - anchor
            d = mean(differences)
            residual = None if any(v is None for v in (f, m, d)) else m - f - d
            if residual is not None and abs(residual) > 1e-12:
                raise ValueError("oracle endpoint closure failed")
            pair_rows.append(
                {
                    "pair_id": p["pair_id"],
                    "family": p["pairing_family"],
                    "fixed_change": f,
                    "matched_change": m,
                    "reference_difference": d,
                    "closure_residual": residual,
                }
            )
        keys = ("fixed_change", "matched_change", "reference_difference")
        family_rows = [
            {
                "family": family,
                **{
                    k: mean([r[k] for r in pair_rows if r["family"] == family])
                    for k in keys
                },
            }
            for family in families
        ]
        overall = {k: mean([r[k] for r in family_rows]) for k in keys}
        for row in [*family_rows, overall]:
            values = [row[k] for k in keys]
            if (
                all(v is not None for v in values)
                and abs(values[1] - values[0] - values[2]) > 1e-12
            ):
                raise ValueError("oracle aggregate closure failed")
        interactions = []
        for r in records:
            ok = r["norm"] > floor * r["cell_scale"]
            interactions.append(
                {
                    **r,
                    "threshold": floor * r["cell_scale"],
                    "available": ok,
                    "reason": None
                    if ok
                    else ("exact_zero" if r["norm"] == 0 else "below_amplitude_guard"),
                }
            )
        yield {
            "view": view,
            "relative_floor": floor,
            "families": families,
            "test_pair_ids": [p["pair_id"] for p in tests],
            "scores": scores,
            "own": own,
            "other": other,
            "margins": margins,
            "pairs": pair_rows,
            "family_results": family_rows,
            "overall": overall,
            "references": reference_rows,
            "interactions": interactions,
            "available_test_pairs": {
                k: sum(r[k] is not None for r in pair_rows) for k in keys
            },
            "p_values": None,
            "confidence_intervals": None,
            "equivalence_tested": False,
        }
