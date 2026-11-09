"""Replay Note 0093 public scalar aggregates without project imports or inference.

Decimal arithmetic starts at the published JSON score/norm/spectral values.
The 4e-15 comparison tolerance accommodates binary64 intermediate rounding; it
is an audit tolerance, not an input-matching gate or scientific effect threshold.
Native vectors, input words, FFTs and acquisition chronology are not verified.
Run: python scripts/verify_stat_matched_source_publication.py [--repo PATH]
Only JSON is written to stdout. No files, models or processors are created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from decimal import Decimal, DecimalException, localcontext
from pathlib import Path

D = Decimal
TOLERANCE = D("4e-15")
RESULTS = Path("examples/research_notes/0093_stat_matched_source_results")
PANEL = Path("configs/stat_matched_source_panel_v1.json")
FLOORS = (D("1e-8"), D("1e-6"), D("1e-4"))
LEARNED = ("encoder", "projector")


class AuditFailure(ValueError):
    """An explicit failure, retained when Python assertions are disabled."""


def require(condition, label):
    if not condition:
        raise AuditFailure(label)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise AuditFailure(f"non-finite JSON number: {value}")


def decode(text):
    return json.loads(text, parse_float=D, parse_constant=reject_constant,
                      object_pairs_hook=object_pairs)


def number(value):
    require(type(value) in (int, D), "expected a numeric scalar")
    value = D(value)
    require(value.is_finite(), "non-finite numeric scalar")
    return value


def average(values):
    values = list(values)
    return None if not values or None in values else sum(values, D(0)) / len(values)


def unique(rows, key, label):
    result = {key(row): row for row in rows}
    require(len(result) == len(rows), f"duplicate {label}")
    return result


def shape(values, dimensions):
    return isinstance(values, list) and len(values) == dimensions[0] and (
        len(dimensions) == 1 or all(shape(v, dimensions[1:]) for v in values))


class Comparison:
    def __init__(self):
        self.counts = Counter()
        self.errors = defaultdict(lambda: D(0))

    def check(self, label, actual, recorded):
        self.counts[label] += 1
        if actual is None or recorded is None:
            require(actual is recorded, f"{label}: missingness differs")
            return
        error = abs(number(actual) - number(recorded))
        self.errors[label] = max(self.errors[label], error)
        require(error <= TOLERANCE, f"{label}: rounding residual {error}")


def check_geometry(profile, pair_map, test_ids, families, check):
    require(all(type(r["shift"]) is int for r in profile["interactions"]),
            "interaction shifts must be integers")
    interactions = unique(profile["interactions"],
                          lambda r: (r["pair_id"], r["shift"]), "interaction ID")
    require(set(interactions) == {(p, s) for p in pair_map for s in range(4)},
            "interaction ID coverage differs")
    for row in interactions.values():
        norm, scale = number(row["norm"]), number(row["cell_scale"])
        require(norm >= 0 and scale > 0, "invalid norm or cell scale")
        if profile["view"] == "raw_rgb/global_moments":
            require(norm == 0, "raw global-moment interaction must remain exact zero")
        require(type(row["dimension"]) is int and row["dimension"] > 0,
                "invalid interaction dimension")
        if profile["relative_floor"] is None:
            envelope = D("4e-10") * D(row["dimension"]).sqrt()
            check.check("baseline_error_envelope", envelope, row["baseline_error_envelope"])
            threshold = 100 * envelope
        else:
            require(row["baseline_error_envelope"] is None, "learned baseline envelope")
            threshold = profile["relative_floor"] * scale
        check.check("threshold", threshold, row["threshold"])
        require(row["available"] is (norm > threshold), "amplitude availability differs")
        if norm == 0:
            require(row["reason"] == "exact_zero", "zero interaction reason differs")
        elif row["available"]:
            require(row["reason"] is None, "available interaction has failure reason")
    refs = unique(profile["references"], lambda r: r["family"], "reference family")
    require(set(refs) == set(families), "reference family coverage differs")
    for family, ref in refs.items():
        require(type(ref["available"]) is bool, "reference availability must be boolean")
        if ref["available"]:
            require(ref["reason"] is None, "available reference has failure reason")
        members = [p for p, meta in pair_map.items()
                   if meta["split"] == "reference" and meta["pairing_family"] == family]
        if not all(interactions[p, 0]["available"] for p in members):
            require(not ref["available"] and ref["reason"] == "unavailable_reference_replicate",
                    "reference missingness differs")
        # Available replicates alone cannot establish absence of vector cancellation.
    for s, matrix in enumerate(profile["scores"]):
        for i, pid in enumerate(test_ids):
            for j, family in enumerate(families):
                available = interactions[pid, s]["available"] and refs[family]["available"]
                require((matrix[i][j] is not None) == available, "score missingness differs")
    diagnostics = unique(profile["diagnostics"],
                         lambda r: (r["pair_id"], r["shift"]), "diagnostic ID")
    require(all(type(r["shift"]) is int for r in diagnostics.values()),
            "diagnostic shifts must be integers")
    require(set(diagnostics) == {(p, s) for p in test_ids for s in (1, 2, 3)},
            "diagnostic ID coverage differs")
    ratios, cosines = [], []
    for (pid, s), row in diagnostics.items():
        z0, zs = interactions[pid, 0], interactions[pid, s]
        ratio = zs["norm"] / z0["norm"] if z0["available"] and zs["available"] else None
        check.check("norm_ratio", ratio, row["norm_ratio"])
        if ratio is None:
            require(row["direction_cosine"] is row["relative_change"] is None,
                    "unavailable direction diagnostics were imputed")
            continue
        cosine = number(row["direction_cosine"])
        require(abs(cosine) <= 1 + TOLERANCE, "diagnostic cosine outside bounds")
        require(0 <= number(row["relative_change"]) <= 1 + TOLERANCE,
                "relative change outside bounds")
        if profile["view"] in LEARNED:
            relative = max(D(0), 1 + ratio * ratio - 2 * ratio * cosine).sqrt() / (1 + ratio)
            check.check("relative_change", relative, row["relative_change"])
        ratios.append(ratio)
        cosines.append(cosine)
    return {
        "mean_direction_cosine": average(cosines),
        "norm_ratio_min": min(ratios) if ratios else None,
        "norm_ratio_max": max(ratios) if ratios else None,
        "minimum_relative_amplitude": min(r["norm"] / r["cell_scale"]
                                          for r in interactions.values()),
    }


def check_profile(profile, pair_map, test_ids, families, classes, check):
    require(profile["test_pair_ids"] == test_ids and profile["families"] == families,
            "profile pair/family order differs from registration")
    require(shape(profile["scores"], (4, 16, 8))
            and shape(profile["score_matrices"], (4, 8, 8))
            and shape(profile["margins"], (4, 16))
            and shape(profile["family_deltas"], (8,)), "profile array shape differs")
    require(profile["fixed_test_pairs"] == 16, "fixed pair denominator differs")
    require(profile["p_values"] is profile["confidence_intervals"] is None
            and profile["equivalence_tested"] is False, "inference scope differs")
    primary = profile["view"] in LEARNED and profile["relative_floor"] == D("1e-6")
    require(profile["primary_learned_profile"] is primary, "primary flag differs")
    for matrix in profile["scores"]:
        for row in matrix:
            require(all(v is None or abs(number(v)) <= 1 + TOLERANCE for v in row),
                    "score cosine outside bounds")
    indices = {f: [i for i, pid in enumerate(test_ids)
                   if pair_map[pid]["pairing_family"] == f] for f in families}
    others = {f: [j for j, g in enumerate(families) if g != f and classes[g] == classes[f]]
              for f in families}
    margins = []
    for s, scores in enumerate(profile["scores"]):
        row = []
        for i, pid in enumerate(test_ids):
            family = pair_map[pid]["pairing_family"]
            own = scores[i][families.index(family)]
            other = average(scores[i][j] for j in others[family])
            value = None if own is None or other is None else own - other
            check.check("pair_margin", value, profile["margins"][s][i])
            row.append(value)
        margins.append(row)
        for f, family in enumerate(families):
            for target in range(8):
                value = average(scores[i][target] for i in indices[family])
                check.check("family_score_matrix", value, profile["score_matrices"][s][f][target])
    deltas = [None if any(row[i] is None for row in margins)
              else average(margins[s][i] for s in (1, 2, 3)) - margins[0][i]
              for i in range(16)]
    family_deltas = [average(deltas[i] for i in indices[f]) for f in families]
    for actual, stored in zip(family_deltas, profile["family_deltas"]):
        check.check("family_delta", actual, stored)
    endpoint = average(family_deltas)
    check.check("endpoint", endpoint, profile["paired_margin_change"])
    available = sum(v is not None for v in deltas)
    require(available == profile["available_test_pairs"]
            and profile["available"] is (available == 16), "endpoint availability differs")
    require(profile["reason"] == (None if available == 16
            else "required_margin_unavailable_no_filtering"), "endpoint reason differs")
    geometry = check_geometry(profile, pair_map, test_ids, families, check)
    return {
        "view": profile["view"], "relative_floor": profile["relative_floor"],
        "available_test_pairs": available, "paired_margin_change": endpoint,
        "mean_identity_margin": average(margins[0]),
        "mean_nonidentity_margin": average(v for row in margins[1:] for v in row),
        "pair_deltas": dict(zip(test_ids, deltas)),
        "family_deltas": dict(zip(families, family_deltas)),
        "negative_family_deltas": sum(v is not None and v < 0 for v in family_deltas),
        "positive_family_deltas": sum(v is not None and v > 0 for v in family_deltas),
        **geometry,
    }


def spectral_summaries(spectra, pair_map, cells):
    rows = unique(spectra, lambda r: r["identity"], "spectral ID")
    expected = {f"{p}/s{s}/{c}" for p in pair_map for s in range(4) for c in cells}
    require(set(rows) == expected and len(rows) == 512, "spectral ID coverage differs")
    metrics = {
        "raw_rgb": ("spectral_centroid", "high_frequency_energy_ratio"),
        "processor_pixel_values": ("spectral_centroid", "high_frequency_energy_ratio",
            "spectral_centroid_cycles_per_patch", "energy_ratio_above_half_cycle_per_patch",
            "energy_ratio_above_one_cycle_per_patch"),
    }
    for row in rows.values():
        require(set(row["spectra"]) == set(metrics), "spectral domains differ")
        for domain, fields in metrics.items():
            require(set(row["spectra"][domain]) == set(fields), "spectral fields differ")
            for field, value in row["spectra"][domain].items():
                if "per_patch" in field:
                    require(value is None, "patch-relative diagnostic must remain unavailable")
                else:
                    require(0 <= number(value) <= 1 + TOLERANCE,
                            "normalized spectral descriptor outside [0, 1]")
    result = []
    for domain, fields in metrics.items():
        for field in fields:
            values = [r["spectra"][domain][field] for r in rows.values()]
            for split in ("reference", "test"):
                ids = [p for p, meta in pair_map.items() if meta["split"] == split]
                delta, identity = [], []
                for pid in ids:
                    for cell in cells:
                        before = rows[f"{pid}/s0/{cell}"]["spectra"][domain][field]
                        identity.append(before)
                        for s in (1, 2, 3):
                            after = rows[f"{pid}/s{s}/{cell}"]["spectra"][domain][field]
                            delta.append(None if before is None or after is None else after - before)
                finite = [v for v in delta if v is not None]
                result.append({"domain": domain, "metric": field, "split": split,
                    "null_records_of_512": values.count(None), "paired_comparisons": len(delta),
                    "available_comparisons": len(finite), "mean_identity": average(identity),
                    "mean_paired_change": average(delta),
                    "maximum_absolute_change": max(map(abs, finite)) if finite else None,
                    "changes_above_1e_minus_12": sum(abs(v) > D("1e-12") for v in finite)})
    return result


def audit(repo):
    names = ("profiles.jsonl", "spectra.jsonl", "summary.json", "verification_receipts.json")
    raw = {name: (repo / RESULTS / name).read_bytes() for name in names}
    config_bytes = (repo / PANEL).read_bytes()
    config = decode(config_bytes)
    summary = decode(raw["summary.json"])
    decode(raw["verification_receipts.json"])
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    require(set(summary["output_sha256"]) == {"profiles.jsonl", "spectra.jsonl"},
            "output hash inventory differs")
    for name, digest in summary["output_sha256"].items():
        require(hashes[name] == digest, f"public output hash mismatch: {name}")
    profiles = [decode(line) for line in raw["profiles.jsonl"].splitlines()]
    spectra = [decode(line) for line in raw["spectra.jsonl"].splitlines()]
    pair_map = unique(config["sources"]["pairs"], lambda p: p["pair_id"], "registered pair ID")
    test_ids = [p for p, meta in pair_map.items() if meta["split"] == "test"]
    families = list(dict.fromkeys(p["pairing_family"] for p in pair_map.values()))
    classes = {p["pairing_family"]: p["broad_class"] for p in pair_map.values()}
    require(len(pair_map) == 32 and len(test_ids) == 16 and len(families) == 8,
            "registered pair/family counts differ")
    for family in families:
        members = [m for m in pair_map.values() if m["pairing_family"] == family]
        require({(m["split"], m["replicate"]) for m in members}
                == {("reference", 5), ("reference", 6), ("test", 7), ("test", 8)},
                "registered split/replicates differ")
        require(len({m["broad_class"] for m in members}) == 1, "inconsistent family class")
        require(sum(classes[g] == classes[family] for g in families) == 4,
                "within-class competitor count differs")
    require(config["low_norm"]["learned_sensitivity_relative_floors"] == list(FLOORS),
            "registered floors differ")
    views = config["analysis"]["views"]
    require(len(views) == len(set(views)) == 14 and all(v in views for v in LEARNED),
            "registered views differ")
    profile_map = unique(profiles, lambda p: (p["view"], p["relative_floor"]), "profile")
    expected = {(v, f) for v in views for f in (FLOORS if v in LEARNED else (None,))}
    require(set(profile_map) == expected and len(profiles) == 18, "profile inventory differs")
    require(summary["profiles"] == 18
            and summary["counts"]["primary_test_pairs"] == 16
            and summary["counts"]["known_pairing_families"] == 8,
            "summary profile/pair/family counts differ")
    require(summary["learned_floors"] == list(FLOORS)
            and summary["all_learned_floors_same_primary_results"] is True,
            "summary floor declarations differ")
    require(summary["p_values"] is summary["confidence_intervals"] is None
            and summary["equivalence_tested"] is False,
            "summary inference scope differs")
    check = Comparison()
    results = [check_profile(p, pair_map, test_ids, families, classes, check) for p in profiles]
    for site in LEARNED:
        base = profile_map[site, FLOORS[0]]
        for floor in FLOORS[1:]:
            other = profile_map[site, floor]
            ignored = {"relative_floor", "primary_learned_profile", "interactions"}
            require({k: v for k, v in base.items() if k not in ignored}
                    == {k: v for k, v in other.items() if k not in ignored}, "floor profiles differ")
            require([{k: v for k, v in r.items() if k != "threshold"} for r in base["interactions"]]
                    == [{k: v for k, v in r.items() if k != "threshold"} for r in other["interactions"]],
                    "floor interactions differ")
    targets = unique(summary["primary_endpoints"], lambda r: r["view"], "summary endpoint")
    require(set(targets) == set(LEARNED), "summary primary inventory differs")
    for row in results:
        if row["relative_floor"] == D("1e-6"):
            target = targets[row["view"]]
            require(target["relative_floor"] == row["relative_floor"],
                    "summary primary floor differs")
            require(target["available"] is (row["available_test_pairs"] == 16),
                    "summary primary availability differs")
            for key in ("paired_margin_change", "mean_identity_margin", "mean_nonidentity_margin",
                        "mean_direction_cosine", "norm_ratio_min", "norm_ratio_max",
                        "available_test_pairs", "negative_family_deltas", "positive_family_deltas"):
                check.check("summary_" + key, row[key], target[key])
    baselines = [r for r in results if r["relative_floor"] is None]
    available = [r for r in baselines if r["paired_margin_change"] is not None]
    require(len(available) == summary["available_baseline_profiles"] == 11,
            "baseline availability count differs")
    require([r["view"] for r in baselines if r["paired_margin_change"] is None]
            == ["raw_rgb/global_moments"], "unavailable baseline differs")
    require(summary["unavailable_profiles"] == [{
        "view": "raw_rgb/global_moments", "available_test_pairs": 0,
        "reason": "required_margin_unavailable_no_filtering"}],
        "summary missingness declaration differs")
    check.check("maximum_baseline_D", max(abs(r["paired_margin_change"]) for r in available),
                summary["maximum_absolute_baseline_D"])
    cells = config["construction"]["cells"]
    require(cells == ["mm", "mj", "jm", "jj"], "factorial cell order differs")
    return {
        "status": "VERIFIED_PUBLIC_SCORE_AGGREGATES", "decimal_precision": 70,
        "rounding_comparison_tolerance": TOLERANCE,
        "input_sha256": {**hashes, str(PANEL): hashlib.sha256(config_bytes).hexdigest()},
        "comparison_counts": dict(check.counts), "maximum_rounding_residuals": dict(check.errors),
        "profiles": results, "spectra": spectral_summaries(spectra, pair_map, cells),
        "floor_profiles_identical_except_threshold_and_flags": True,
        "new_model_contexts": 0, "new_processor_calls": 0,
        "scope": "Published scores, norms and spectra are inputs; native arrays, reference cancellation, "
                 "input-word matching, FFT values and execution chronology require retained local artifacts.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        with localcontext() as ctx:
            ctx.prec = 70
            result = audit(args.repo.resolve())
    except (OSError, ValueError, TypeError, KeyError, IndexError, DecimalException) as exc:
        print(json.dumps({"status": "FAILED_PUBLIC_SCORE_AUDIT", "error": str(exc)}))
        return 1
    print(json.dumps(result, default=str, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
