from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.cli.run_source_coordinate_overlap import load_sources
from fractal_vlm_state_probe.padding_policy import _checked_artifact
from fractal_vlm_state_probe.permutation_maps import MAPS, NEW_MAPS
from fractal_vlm_state_probe.source_coordinate_overlap import (
    COMPONENTS,
    validate_config,
)
from fractal_vlm_state_probe.stimulus import write_json


def oracle_partition(left, right, *, shape=(240, 320), padding_rows=40):
    size = shape[0] * shape[1]
    if left.shape != (size,) or right.shape != (size,):
        raise ValueError("oracle map shape differs")
    if sorted(left.tolist()) != list(range(size)) or sorted(right.tolist()) != list(
        range(size)
    ):
        raise ValueError("oracle requires coordinate permutations")
    codes = []
    for site, (a, b) in enumerate(zip(left, right)):
        codes.append(2 if a != b else (0 if a == site else 1))
    result = np.full((shape[0] + 2 * padding_rows, shape[1]), 3, dtype=np.int64)
    result[padding_rows : padding_rows + shape[0]] = np.asarray(codes).reshape(shape)
    return result


def oracle_directions(reference, test, metadata):
    def unit(array):
        norm = float(np.sqrt(np.sum(array * array)))
        return (array / norm, norm) if norm else (None, 0)

    refs, tests, norms = (
        {},
        {},
        {"reference_inputs": {}, "reference_sums": {}, "tests": {}},
    )
    for family in sorted({m["pairing_family"] for m in metadata.values()}):
        selected = sorted(
            label
            for label, m in metadata.items()
            if m["pairing_family"] == family and m["replicate"] in (1, 2)
        )
        values = []
        for label in selected:
            value, norm = unit(reference[label])
            if value is None:
                return None
            values.append(value)
            norms["reference_inputs"][label] = norm
        ref, norm = unit(values[0] + values[1])
        if norm <= 1e-12:
            return None
        refs[family] = ref
        norms["reference_sums"][family] = norm
    for label, m in metadata.items():
        if m["replicate"] in (3, 4):
            tests[label], norm = unit(test[label])
            if tests[label] is None:
                return None
            norms["tests"][label] = norm
    return refs, tests, norms


def verify_linear(saved, scores, metadata):
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    if (
        set(scores) != set(labels)
        or saved["test_seed_count"] != len(labels)
        or len(saved["test_seed_margins"]) != len(labels)
        or {r["label"] for r in saved["test_seed_margins"]} != set(labels)
        or set(saved["family_margins"]) != set(families)
    ):
        raise ValueError("linear-summary denominator differs")
    lookup = {r["label"]: r for r in saved["test_seed_margins"]}
    margins = {f: [] for f in families}
    for label in labels:
        meta = metadata[label]
        if any(lookup[label][k] != v for k, v in meta.items()) or set(
            scores[label]
        ) != set(families):
            raise ValueError("linear-summary metadata differs")
        others = [
            f
            for f in families
            if f != meta["pairing_family"]
            and next(
                m["broad_class"] for m in metadata.values() if m["pairing_family"] == f
            )
            == meta["broad_class"]
        ]
        own = scores[label][meta["pairing_family"]]
        other = sum(scores[label][f] for f in others) / len(others)
        margin = own - other
        for key, value in (
            ("own_family_cosine_contribution", own),
            ("other_family_cosine_contribution", other),
            ("margin", margin),
        ):
            if (
                not np.isfinite(lookup[label][key])
                or abs(lookup[label][key] - value) > 1e-12
            ):
                raise ValueError("linear-summary test margin differs")
        margins[meta["pairing_family"]].append(margin)
    expected = {f: sum(v) / len(v) for f, v in margins.items()}
    if (
        any(
            not np.isfinite(saved["family_margins"][f])
            or abs(v - saved["family_margins"][f]) > 1e-12
            for f, v in expected.items()
        )
        or not np.isfinite(saved["mean_family_margin"])
        or abs(sum(expected.values()) / len(expected) - saved["mean_family_margin"])
        > 1e-12
        or saved["positive_family_margin_count"]
        != sum(v > 0 for v in saved["family_margins"].values())
        or saved["negative_family_margin_count"]
        != sum(v < 0 for v in saved["family_margins"].values())
    ):
        raise ValueError("linear-summary family margin differs")
    if "retrieval_correct_count" in saved or "exact_block_test" in saved:
        raise ValueError(
            "component summaries cannot acquire retrieval or statistical tests"
        )


def check_ratio(saved, numerator, denominator):
    expected = numerator / denominator if abs(denominator) > 1e-12 else None
    if (
        (expected is None) != (saved is None)
        or expected is not None
        and (
            not np.isfinite(saved)
            or not np.isclose(saved, expected, rtol=1e-12, atol=1e-12)
        )
    ):
        raise ValueError("signed component ratio differs")


def verify_summaries(row, full, parts, metadata):
    if set(row["components"]) != set(COMPONENTS):
        raise ValueError("component coverage differs")
    verify_linear(row["full"], full, metadata)
    for c in COMPONENTS:
        saved = row["components"][c]
        verify_linear(saved, parts[c], metadata)
        check_ratio(
            saved["mean_margin_fraction_of_full"],
            saved["mean_family_margin"],
            row["full"]["mean_family_margin"],
        )
        if set(saved["family_margin_fractions_of_full"]) != set(
            row["full"]["family_margins"]
        ):
            raise ValueError("component family ratio coverage differs")
        for f, ratio in saved["family_margin_fractions_of_full"].items():
            check_ratio(
                ratio, saved["family_margins"][f], row["full"]["family_margins"][f]
            )
    combined = {
        label: {
            f: parts[COMPONENTS[0]][label][f] + parts[COMPONENTS[1]][label][f]
            for f in values
        }
        for label, values in full.items()
    }
    verify_linear(row["coincident_content"], combined, metadata)
    check_ratio(
        row["coincident_content"]["mean_margin_fraction_of_full"],
        row["coincident_content"]["mean_family_margin"],
        row["full"]["mean_family_margin"],
    )
    for key in (
        "family_margin_reconstruction_max_abs_error",
        "test_seed_margin_reconstruction_max_abs_error",
    ):
        if not np.isfinite(row[key]) or not 0 <= row[key] <= 1e-12:
            raise ValueError("margin reconstruction receipt differs")


def verify_decomposition(result, dots, vectors, maps, metadata, baseline):
    matrix = result["matrix"]
    if len(matrix) != 16 or {
        (r["reference_map"], r["test_map"]) for r in matrix
    } != set(product(MAPS, repeat=2)):
        raise ValueError("decomposition matrix coverage differs")
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    available_pairs = {
        (r["reference_map"], r["test_map"]) for r in matrix if r["available"]
    }
    expected_dots = {
        (a, b, label, f)
        for a, b in available_pairs
        for label in labels
        for f in families
    }
    lookup = {
        (r["reference_map"], r["test_map"], r["test_label"], r["reference_family"]): r
        for r in dots
    }
    if (
        len(lookup) != len(dots)
        or set(lookup) != expected_dots
        or len(dots) != result["available_dot_products"]
    ):
        raise ValueError("decomposition score ledger coverage differs")
    panels, maximum_error, compared = {}, 0.0, 0
    for row in matrix:
        a, b = row["reference_map"], row["test_map"]
        codes = oracle_partition(maps[a], maps[b])
        counts = np.bincount(codes.ravel(), minlength=4)
        content = int(counts[:3].sum())
        receipt = row["partition"]
        if (
            receipt["pixel_counts"] != dict(zip(COMPONENTS, counts.tolist()))
            or receipt["canvas_pixels"] != codes.size
            or receipt["content_pixels"] != content
            or abs(
                receipt["coincident_content_fraction"]
                - (counts[0] + counts[1]) / content
            )
            > 1e-15
            or any(
                abs(receipt["content_fractions"][c] - counts[i] / content) > 1e-15
                for i, c in enumerate(COMPONENTS[:3])
            )
        ):
            raise ValueError("coordinate partition differs from oracle")
        old = next(
            r
            for r in baseline["map_matrix"]
            if r["view"] == "raw_rgb" and r["reference_map"] == a and r["test_map"] == b
        )
        directions = oracle_directions(vectors[a], vectors[b], metadata)
        if (
            row["available"] != (directions is not None)
            or row["available"] != old["available"]
        ):
            raise ValueError("decomposition availability differs")
        if directions is None:
            panels[(a, b)] = None
            continue
        refs, tests, norms = directions
        if set(row["normalization"]) != set(norms):
            raise ValueError("full-normalization receipt coverage differs")
        for group, values in norms.items():
            if set(row["normalization"][group]) != set(values) or any(
                not np.isclose(
                    row["normalization"][group][label], norm, rtol=1e-12, atol=1e-12
                )
                for label, norm in values.items()
            ):
                raise ValueError("full-normalization receipt differs")
        full, parts = {}, {c: {} for c in COMPONENTS}
        old_scores = {
            s["label"]: s["reference_cosines"] for s in old["test_seed_scores"]
        }
        for label in labels:
            full[label] = {}
            for c in COMPONENTS:
                parts[c][label] = {}
            for family in families:
                saved = lookup[(a, b, label, family)]
                if any(saved[k] != v for k, v in metadata[label].items()) or set(
                    saved["components"]
                ) != set(COMPONENTS):
                    raise ValueError("score ledger metadata differs")
                pixels = np.einsum("hwc,hwc->hw", tests[label], refs[family])
                values = np.bincount(codes.ravel(), weights=pixels.ravel(), minlength=4)
                total = float(np.sum(pixels))
                for c, value in zip(COMPONENTS, values):
                    error = abs(float(value) - saved["components"][c])
                    if not np.isfinite(saved["components"][c]) or error > 1e-12:
                        raise ValueError("independent additive score differs")
                    maximum_error = max(maximum_error, error)
                    parts[c][label][family] = float(value)
                    compared += 1
                if (
                    not np.isfinite(saved["full_cosine"])
                    or abs(total - saved["full_cosine"]) > 1e-12
                    or abs(total - old_scores[label][family]) > 1e-12
                    or abs(sum(saved["components"].values()) - saved["full_cosine"])
                    > 1e-12
                    or not 0 <= saved["reconstruction_abs_error"] <= 1e-12
                ):
                    raise ValueError("full score reconstruction differs")
                full[label][family] = total
        verify_summaries(row, full, parts, metadata)
        panels[(a, b)] = (full, parts)
    if len(result["aggregates"]) != 2 or {
        r["regime"] for r in result["aggregates"]
    } != {"same_map", "cross_map"}:
        raise ValueError("aggregate regime coverage differs")
    for row in result["aggregates"]:
        pairs = [
            (a, b)
            for a in NEW_MAPS
            for b in NEW_MAPS
            if (a == b) == (row["regime"] == "same_map")
        ]
        if row["ordered_map_pairs"] != [list(p) for p in pairs] or row[
            "available"
        ] != all(panels[p] is not None for p in pairs):
            raise ValueError("aggregate map weights or availability differ")
        if not row["available"]:
            continue
        full = {
            label: {
                f: sum(panels[p][0][label][f] for p in pairs) / len(pairs)
                for f in families
            }
            for label in labels
        }
        parts = {
            c: {
                label: {
                    f: sum(panels[p][1][c][label][f] for p in pairs) / len(pairs)
                    for f in families
                }
                for label in labels
            }
            for c in COMPONENTS
        }
        verify_summaries(row, full, parts, metadata)
        old = next(
            r
            for r in baseline["primary_tests"]
            if r["view"] == "raw_rgb" and r["regime"] == row["regime"]
        )
        if abs(
            row["full"]["mean_family_margin"] - old["mean_family_margin"]
        ) > 1e-12 or any(
            abs(v - old["family_margins"][f]) > 1e-12
            for f, v in row["full"]["family_margins"].items()
        ):
            raise ValueError("aggregate baseline margin differs")
    if result["new_statistical_tests"] != 0 or result["planned_dot_products"] != 2048:
        raise ValueError("descriptive analysis contract differs")
    return {
        "ordered_map_partitions": 16,
        "dot_products_recomputed": len(dots),
        "additive_terms_recomputed": compared,
        "aggregate_regimes_recomputed": 2,
        "maximum_component_oracle_abs_error": maximum_error,
    }


def publish(root, output, *, figures=True):
    result, frozen = (
        _read(root / "study_summary.json"),
        _read(root / "frozen_specification.json"),
    )
    validate_config(frozen["config"])
    if (root / "execution_failure.json").exists():
        raise ValueError("failed execution requires a separate audit")
    for name, field in (
        ("frozen_specification.json", "frozen_specification_sha256"),
        ("source_vectors.json", "source_vectors_sha256"),
        ("dot_products.jsonl", "dot_products_sha256"),
    ):
        if _sha(root / name) != result[field]:
            raise ValueError("overlap measurement receipt changed")
    repo = Path(__file__).resolve().parents[3]
    for name, digest in frozen["code_sha256"].items():
        if _sha(repo / name) != digest:
            raise ValueError(f"registered code changed: {name}")
    for path, digest in frozen["source_sha256"].items():
        if _sha(Path(path)) != digest:
            raise ValueError("baseline source changed")
    baseline, vectors, maps, metadata, receipts = load_sources(
        Path(frozen["baseline_root"]), Path(frozen["baseline_publication"])
    )
    if (
        receipts != _read(root / "source_vectors.json")
        or metadata != frozen["hierarchy_metadata"]
    ):
        raise ValueError("source vector/hierarchy receipt differs")
    original_path = Path(baseline["frozen_specification"]["input_summary"])
    if (
        _sha(original_path)
        != baseline["frozen_specification"]["source_sha256"][str(original_path)]
    ):
        raise ValueError("raw input receipt changed")
    original = _read(original_path)
    errors = []
    for record in original["records"]:
        for key in MAPS:
            cells = {
                c: _checked_artifact(a)[0].astype(np.int16)
                for c, a in record["maps"][key]["artifacts"].items()
            }
            if set(cells) != {"mm", "jj", "mj", "jm"}:
                raise ValueError("raw factorial cell coverage differs")
            expected = (cells["jj"] + cells["mm"] - cells["jm"] - cells["mj"]).astype(
                np.float64
            ) / 255
            error = float(np.max(np.abs(expected - vectors[key][record["pair_id"]])))
            if error > 1e-12:
                raise ValueError("raw interaction differs from integer cell oracle")
            errors.append(error)
    if len(errors) != 128:
        raise ValueError("raw interaction reconstruction denominator differs")
    dots = [
        json.loads(line)
        for line in (root / "dot_products.jsonl").read_text().splitlines()
    ]
    verification = verify_decomposition(result, dots, vectors, maps, metadata, baseline)
    verification.update(
        raw_cells_reloaded=512,
        raw_interactions_recomputed=128,
        maximum_raw_oracle_abs_error=max(errors),
    )
    if any(
        result[k] != 0
        for k in ("new_input_vectors", "new_processor_calls", "new_model_forwards")
    ):
        raise ValueError("analysis added unregistered computation")
    if output.exists():
        raise FileExistsError("use a fresh overlap publication directory")
    output.mkdir(parents=True)
    (output / "dot_products.jsonl").write_bytes((root / "dot_products.jsonl").read_bytes())
    write_json(
        output / "source_vectors.json",
        {
            "schema_version": 1,
            "analysis_kind": "raw_source_coordinate_overlap_source_vectors",
            "vectors": receipts,
        },
    )
    summary = {
        **result,
        "analysis_kind": "raw_source_coordinate_overlap_publication",
        "frozen_specification": frozen,
        "source_summary_sha256": _sha(root / "study_summary.json"),
        "exporter_sha256": _sha(Path(__file__)),
        "independent_verification": verification,
        "ledgers": {
            "dot_products.jsonl": {
                "rows": len(dots),
                "sha256": _sha(output / "dot_products.jsonl"),
            },
            "source_vectors.json": {
                "rows": len(receipts),
                "sha256": _sha(output / "source_vectors.json"),
            },
        },
        "claim_boundaries": [
            "This is signed full-normalization score accounting, not a masked cosine or a pixel/model intervention.",
            "Coordinate coincidence is not equality of RGB values, source images or necessarily unmoved sites; padding is separate.",
            "Component ratios may be negative or exceed one; they are not energy, semantic retention or causal fractions.",
            "All views are descriptive and dependent; no component p-value, component retrieval or new image/model observation is added.",
            "Raw-coordinate accounting does not identify processor or cache mechanisms or predict a renormalized ablation.",
        ],
    }
    if figures:
        plot_results(summary, output)
        summary["figures"] = {
            n: _sha(output / n)
            for n in (
                "cross_map_accounting.png",
                "cross_map_accounting.pdf",
                "family_contributions.png",
                "family_contributions.pdf",
            )
        }
    write_json(output / "summary.json", summary)
    return summary


def plot_results(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cross = next(r for r in summary["aggregates"] if r["regime"] == "cross_map")
    if not cross["available"]:
        raise ValueError("figures require available cross-map accounting")
    names = ["Shared fixed", "Shared moved", "Noncoincident", "Padding", "Full margin"]
    values = [cross["components"][c]["mean_family_margin"] for c in COMPONENTS] + [
        cross["full"]["mean_family_margin"]
    ]
    colors = ["#2376a8", "#459f81", "#c75d37", "#8b8794", "#333333"]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(range(5), values, color=colors, height=0.58)
    ax.set(
        yticks=range(5),
        yticklabels=names,
        xlabel="Signed mean-margin contribution (full norms unchanged)",
    )
    ax.invert_yaxis()
    ax.axvline(0, color="0.4", linewidth=0.8)
    span = max(abs(v) for v in values) or 1
    ax.set_xlim(min(0, min(values)) - span * 0.08, max(values) + span * 0.35)
    for i, value in enumerate(values):
        ax.text(
            max(value, 0) + span * 0.02,
            i,
            f"{value:.6f}" if abs(value) >= 1e-6 else f"{value:.2e}",
            va="center",
            fontsize=10,
        )
    ax.set_title("Raw cross-map correspondence: additive accounting")
    fig.subplots_adjust(left=0.21, right=0.97, top=0.89, bottom=0.19)
    fig.text(
        0.5,
        0.04,
        "Three new maps; six ordered cross-map pairs. Descriptive decomposition, not a causal ablation.",
        ha="center",
        fontsize=9,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"cross_map_accounting.{extension}", dpi=160)
    plt.close(fig)
    families = list(cross["full"]["family_margins"])
    labels = {
        "geometry_checkerboard_hex_tiling": "Checkerboard / hex",
        "geometry_square_triangle_tiling": "Square / triangle",
        "geometry_voronoi_hex_tiling": "Voronoi / hex",
        "geometry_voronoi_quasicrystal": "Voronoi / quasicrystal",
        "stochastic_blue_random_dots": "Blue noise / dots",
        "stochastic_sparse_dense_dots": "Sparse / dense dots",
        "stochastic_white_blue_noise": "White / blue noise",
        "stochastic_white_random_dots": "White noise / dots",
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, saved, color, label in (
        (-0.18, cross["coincident_content"], colors[0], "Coincident content"),
        (
            0.18,
            cross["components"]["noncoincident_content"],
            colors[2],
            "Noncoincident content",
        ),
    ):
        ax.barh(
            np.arange(8) + offset,
            [saved["family_margins"][f] for f in families],
            height=0.32,
            color=color,
            label=label,
        )
    ax.scatter(
        [cross["full"]["family_margins"][f] for f in families],
        range(8),
        marker="D",
        color="#333333",
        s=24,
        label="Full margin",
        zorder=3,
    )
    ax.set(
        yticks=range(8),
        yticklabels=[labels[f] for f in families],
        xlabel="Signed family-margin contribution",
    )
    ax.axvline(0, color="0.4", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title("All eight families retained, including negative margins")
    ax.legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.90, bottom=0.17)
    fig.text(
        0.5,
        0.04,
        "Full = coincident + noncoincident + padding. Padding is retained in the numerical ledger.",
        ha="center",
        fontsize=9,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output / f"family_contributions.{extension}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Independently verify raw source-coordinate overlap accounting."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = publish(args.run_root.resolve(), args.output_root.resolve())
    print(json.dumps(summary["independent_verification"], indent=2), flush=True)


if __name__ == "__main__":
    main()
