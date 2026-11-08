"""Separate exact-rational arithmetic over the saved floating-point scores."""

from fractions import Fraction


def decompose(profile, spec):
    metadata = {p["pair_id"]: p for p in spec["sources"]["pairs"]}
    family_class = {p["pairing_family"]: p["broad_class"] for p in metadata.values()}
    names = profile["families"]

    def avg(items):
        return (
            None
            if any(v is None for v in items)
            else sum(items, Fraction()) / len(items)
        )

    def difference(items):
        return None if any(v is None for v in items) else avg(items[1:]) - items[0]

    def floating(value):
        return None if value is None else float(value)

    def totals(group, original):
        values = {
            k: avg([r[k] for r in group])
            for k in ("delta_own", "delta_other", "other_contribution", "margin_change")
        }
        return {
            **{k: floating(v) for k, v in values.items()},
            "source_margin_change": original,
            "closure_residual": None if values["margin_change"] is None else 0.0,
        }

    exact, pairs, exact_shifts = [], [], []
    for i, pair_id in enumerate(profile["test_pair_ids"]):
        pair = metadata[pair_id]
        own_index = names.index(pair["pairing_family"])
        rivals = [
            j
            for j in range(8)
            if names[j] != pair["pairing_family"]
            and family_class[names[j]] == pair["broad_class"]
        ]
        shifts = []
        for s in range(4):
            row = [None if v is None else Fraction(v) for v in profile["scores"][s][i]]
            own, other = row[own_index], avg([row[j] for j in rivals])
            shifts.append(
                {
                    "shift": s,
                    "own": own,
                    "other": other,
                    "margin": None if own is None or other is None else own - other,
                }
            )
        exact_shifts.append(shifts)
        do = difference([r["own"] for r in shifts])
        dc = difference([r["other"] for r in shifts])
        item = {
            "delta_own": do,
            "delta_other": dc,
            "other_contribution": None if dc is None else -dc,
            "margin_change": difference([r["margin"] for r in shifts]),
        }
        exact.append(item)
        original = profile["margins"]
        old = (
            None
            if any(original[s][i] is None for s in range(4))
            else floating(
                avg([Fraction(original[s][i]) for s in (1, 2, 3)])
                - Fraction(original[0][i])
            )
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "family": pair["pairing_family"],
                "broad_class": pair["broad_class"],
                "shifts": [
                    {k: (v if k == "shift" else floating(v)) for k, v in r.items()}
                    for r in shifts
                ],
                **totals([item], old),
            }
        )
    families, family_exact = [], []
    for i, name in enumerate(names):
        group = [
            r
            for r, key in zip(exact, profile["test_pair_ids"], strict=True)
            if metadata[key]["pairing_family"] == name
        ]
        family_exact.append({k: avg([r[k] for r in group]) for k in group[0]})
        families.append(
            {
                "family": name,
                "fixed_pairs": 2,
                **totals(group, profile["family_deltas"][i]),
            }
        )
    overall = totals(family_exact, profile["paired_margin_change"])
    return {
        "view": profile["view"],
        "relative_floor": profile["relative_floor"],
        "primary_learned_profile": profile["primary_learned_profile"],
        "available": overall["margin_change"] is not None,
        "fixed_test_pairs": 16,
        "available_test_pairs": sum(r["margin_change"] is not None for r in exact),
        "fixed_families": 8,
        "shifts": [
            {
                "shift": s,
                **{
                    k: floating(avg([r[s][k] for r in exact_shifts]))
                    for k in ("own", "other", "margin")
                },
            }
            for s in range(4)
        ],
        "pairs": pairs,
        "families": families,
        "overall": overall,
    }
