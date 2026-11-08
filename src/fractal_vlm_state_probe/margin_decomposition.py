"""Fixed own/other accounting over saved correspondence scores only."""

import math

TOLERANCE = 1e-12
COMPONENTS = ("delta_own", "delta_other", "other_contribution", "margin_change")


def mean(values):
    if not values:
        raise ValueError("empty fixed denominator")
    return None if any(v is None for v in values) else math.fsum(values) / len(values)


def close(actual, expected):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError("missingness differs")
    elif (
        isinstance(actual, bool)
        or isinstance(expected, bool)
        or not isinstance(actual, (int, float))
        or not isinstance(expected, (int, float))
        or not math.isfinite(actual)
        or not math.isfinite(expected)
        or abs(actual - expected) > TOLERANCE
    ):
        raise ValueError("saved margin or decomposition differs")


def score(value):
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or abs(value) > 1 + TOLERANCE
    ):
        raise ValueError("finite cosine or explicit null required")
    return float(value)


def aggregate(rows, source_D):
    result = {key: mean([r[key] for r in rows]) for key in COMPONENTS}
    close(result["margin_change"], source_D)
    closure = (
        None
        if result["margin_change"] is None
        else result["delta_own"]
        + result["other_contribution"]
        - result["margin_change"]
    )
    close(closure, None if source_D is None else 0.0)
    return {**result, "source_margin_change": source_D, "closure_residual": closure}


def decompose(profile, spec):
    pairs = spec["sources"]["pairs"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    tests = [p for p in pairs if p["split"] == "test"]
    classes = {p["pairing_family"]: p["broad_class"] for p in pairs}
    if (
        len(tests) != 16
        or len(families) != 8
        or profile["families"] != families
        or profile["test_pair_ids"] != [p["pair_id"] for p in tests]
    ):
        raise ValueError("fixed test/family identities or order differ")
    raw = profile["scores"]
    margins = profile["margins"]
    if (
        len(raw) != 4
        or any(len(r) != 16 or any(len(v) != 8 for v in r) for r in raw)
        or len(margins) != 4
        or any(len(r) != 16 for r in margins)
    ):
        raise ValueError("fixed score/margin shape differs")
    values = [[[score(v) for v in row] for row in shift] for shift in raw]
    pair_rows, shift_rows = [], []
    for i, pair in enumerate(tests):
        own = families.index(pair["pairing_family"])
        others = [
            j
            for j, f in enumerate(families)
            if j != own and classes[f] == pair["broad_class"]
        ]
        if len(others) != 3:
            raise ValueError("exactly three same-class competitors required")
        shifts = []
        for s in range(4):
            o, c = values[s][i][own], mean([values[s][i][j] for j in others])
            m = None if o is None or c is None else o - c
            close(m, margins[s][i])
            shifts.append({"shift": s, "own": o, "other": c, "margin": m})
        delta_o = (
            None
            if any(r["own"] is None for r in shifts)
            else mean([r["own"] for r in shifts[1:]]) - shifts[0]["own"]
        )
        delta_c = (
            None
            if any(r["other"] is None for r in shifts)
            else mean([r["other"] for r in shifts[1:]]) - shifts[0]["other"]
        )
        # Keep the margin unavailable if either required contributor is missing.
        d = None if delta_o is None or delta_c is None else delta_o - delta_c
        original = (
            None
            if any(margins[s][i] is None for s in range(4))
            else mean([margins[s][i] for s in (1, 2, 3)]) - margins[0][i]
        )
        row = {
            "delta_own": delta_o,
            "delta_other": delta_c,
            "other_contribution": None if delta_c is None else -delta_c,
            "margin_change": d,
        }
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "family": pair["pairing_family"],
                "broad_class": pair["broad_class"],
                "shifts": shifts,
                **aggregate([row], original),
            }
        )
    if len(profile["family_deltas"]) != 8:
        raise ValueError("fixed family denominator differs")
    family_rows = []
    for i, family in enumerate(families):
        group = [r for r in pair_rows if r["family"] == family]
        if len(group) != 2:
            raise ValueError("two test pairs per family required")
        family_rows.append(
            {
                "family": family,
                "fixed_pairs": 2,
                **aggregate(group, profile["family_deltas"][i]),
            }
        )
    for s in range(4):
        shift_rows.append(
            {
                "shift": s,
                **{
                    k: mean([r["shifts"][s][k] for r in pair_rows])
                    for k in ("own", "other", "margin")
                },
            }
        )
    overall = aggregate(family_rows, profile["paired_margin_change"])
    available = overall["margin_change"] is not None
    available_pairs = sum(r["margin_change"] is not None for r in pair_rows)
    if (
        profile["available"] is not available
        or profile["available_test_pairs"] != available_pairs
        or profile["fixed_test_pairs"] != 16
    ):
        raise ValueError("source availability/denominators differ")
    return {
        "view": profile["view"],
        "relative_floor": profile["relative_floor"],
        "primary_learned_profile": profile["primary_learned_profile"],
        "available": available,
        "fixed_test_pairs": 16,
        "available_test_pairs": available_pairs,
        "fixed_families": 8,
        "shifts": shift_rows,
        "pairs": pair_rows,
        "families": family_rows,
        "overall": overall,
    }


def profile_identities(spec):
    return [
        (v, f)
        for v in spec["analysis"]["views"]
        for f in ((1e-8, 1e-6, 1e-4) if v in ("encoder", "projector") else (None,))
    ]


def validate_profiles(profiles, spec):
    if [(p["view"], p["relative_floor"]) for p in profiles] != profile_identities(spec):
        raise ValueError("all eighteen original views/floors required in order")
    for p in profiles:
        if p["primary_learned_profile"] is not (
            p["view"] in ("encoder", "projector") and p["relative_floor"] == 1e-6
        ):
            raise ValueError("primary profile flag differs")


def compare(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("oracle object coverage differs")
        return max((compare(actual[k], v) for k, v in expected.items()), default=0.0)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("oracle row coverage differs")
        return max(
            (compare(a, b) for a, b in zip(actual, expected, strict=True)), default=0.0
        )
    if isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError("oracle numeric type differs")
        close(actual, expected)
        return abs(actual - expected)
    if actual != expected or type(actual) is not type(expected):
        raise ValueError("oracle label/missingness differs")
    return 0.0
