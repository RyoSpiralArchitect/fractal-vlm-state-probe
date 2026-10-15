from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_tensor_artifact import load_cache_tensor_artifact
from fractal_vlm_state_probe.cache_tensor_factorial import cache_tensor_regions
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.cli.summarize_graded_cache_study import (
    TARGETS,
    summarize_decompositions,
    verify_decomposition,
    verify_transfer,
)
from fractal_vlm_state_probe.cli.summarize_padding_cache_study import (
    _array,
    verify_exact_test,
)
from fractal_vlm_state_probe.map_seed_cache import REGIONS
from fractal_vlm_state_probe.padding_cache import changed_input_keys
from fractal_vlm_state_probe.permutation_maps import ANCHOR, MAPS, NEW_MAPS
from fractal_vlm_state_probe.stimulus import write_json


def verify_aggregate(row, matrix_rows, metadata):
    same = row["regime"] == "same_map"
    if row["regime"] not in ("same_map", "cross_map"):
        raise ValueError("invalid primary regime")
    pairs = [(a, b) for a in NEW_MAPS for b in NEW_MAPS if (a == b) == same]
    selected = [r for r in matrix_rows if (r["reference_map"], r["test_map"]) in pairs]
    if len(selected) != len(pairs) or {
        (r["reference_map"], r["test_map"]) for r in selected
    } != set(pairs):
        raise ValueError("aggregate matrix coverage differs")
    if any(not r["available"] for r in selected):
        if row["available"]:
            raise ValueError("aggregate incorrectly drops an unavailable pair")
        return 0
    if (
        not row["available"]
        or row["map_pair_count"] != len(pairs)
        or row["ordered_map_pairs"] != [list(p) for p in sorted(pairs)]
    ):
        raise ValueError("aggregate map recipe differs")
    labels = sorted(k for k, m in metadata.items() if m["replicate"] in (3, 4))
    families = sorted({m["pairing_family"] for m in metadata.values()})
    if len(row["test_seed_scores"]) != 16 or row["test_seed_count"] != 16:
        raise ValueError("map dependence inflated the test-seed denominator")
    saved = {s["label"]: s for s in row["test_seed_scores"]}
    if set(saved) != set(labels):
        raise ValueError("aggregate test labels differ")
    correct, tied, checked = 0, 0, 0
    for label in labels:
        source = [
            next(s for s in r["test_seed_scores"] if s["label"] == label)
            for r in selected
        ]
        values = {
            f: sum(s["reference_cosines"][f] for s in source) / len(source)
            for f in families
        }
        for f, value in values.items():
            if abs(value - saved[label]["reference_cosines"][f]) > 1e-12:
                raise ValueError("aggregate cosine weights differ")
            checked += 1
        winners = [
            f for f in families if abs(values[f] - max(values.values())) <= 1e-12
        ]
        retrieved = winners[0] if len(winners) == 1 else None
        expected_correct = retrieved == metadata[label]["pairing_family"]
        if (
            saved[label]["retrieved_family"] != retrieved
            or saved[label]["retrieval_correct"] != expected_correct
        ):
            raise ValueError("aggregate retrieval differs")
        correct += expected_correct
        tied += retrieved is None
    if correct != row["retrieval_correct_count"] or tied != row["retrieval_tied_count"]:
        raise ValueError("aggregate retrieval count differs")
    verify_exact_test(row, metadata)
    return checked


def verify_factorial(analysis, cells):
    contrasts = {
        "spatial_contrast": cells["jj"] + cells["jm"] - cells["mm"] - cells["mj"],
        "palette_contrast": cells["jj"] + cells["mj"] - cells["mm"] - cells["jm"],
        "interaction_contrast": cells["jj"] + cells["mm"] - cells["jm"] - cells["mj"],
    }
    regions = {
        "all_effective": list(range(397)),
        "pre_image": list(range(42)),
        "image_tokens": list(range(42, 298)),
        "post_image": list(range(298, 397)),
        "non_image_tokens": [*range(42), *range(298, 397)],
    }
    if {r["region"] for r in analysis["regions"]} != set(regions):
        raise ValueError("factorial region coverage differs")
    for row in analysis["regions"]:
        positions = regions[row["region"]]
        arrays = {k: np.take(v, positions, axis=-2) for k, v in contrasts.items()}
        energy = {k: float(np.sum(v * v)) for k, v in arrays.items()}
        total = sum(energy.values())
        for key, value in energy.items():
            share = value / total if total else None
            actual = row["balanced_contrast_energy"]["energy_shares"][key]
            if (share is None) != (actual is None) or (
                share is not None and abs(share - actual) > 1e-12
            ):
                raise ValueError("factorial balanced energy share differs")
        effects = {
            "spatial_main_effect": arrays["spatial_contrast"] / 2,
            "palette_main_effect": arrays["palette_contrast"] / 2,
            "interaction": arrays["interaction_contrast"],
        }
        for name, vector in effects.items():
            norm = float(np.sqrt(np.sum(vector * vector)))
            actual = row["effects"][name]
            index = np.unravel_index(np.argmax(np.abs(vector)), vector.shape)
            if (
                not np.isclose(norm, actual["l2_norm"], rtol=1e-12, atol=1e-12)
                or actual["argmax_sequence_position"] != positions[index[-2]]
            ):
                raise ValueError("factorial norm or position differs")
    v = contrasts["interaction_contrast"]
    total = float(np.sum(v * v))
    fraction = float(np.sum(v[:, :, 42:298] ** 2) / total) if total else None
    actual = analysis["interaction_partition"]["image_energy_fraction"]
    if (fraction is None) != (actual is None) or (
        fraction is not None and abs(fraction - actual) > 1e-12
    ):
        raise ValueError("interaction partition differs")
    return contrasts["interaction_contrast"]


def publish(root, output, *, figures=True):
    result, frozen = (
        _read(root / "study_summary.json"),
        _read(root / "frozen_specification.json"),
    )
    metadata = frozen["hierarchy_metadata"]
    for name, field in (
        ("frozen_specification.json", "frozen_specification_sha256"),
        ("calibration.json", "calibration_summary_sha256"),
        ("capture_registry.json", "capture_registry_sha256"),
        ("frozen_panel_references.json", "frozen_reference_registry_sha256"),
        ("panel_anchor_rechecks.json", "panel_anchor_rechecks_sha256"),
    ):
        if _sha(root / name) != result[field]:
            raise ValueError(f"measurement receipt changed: {name}")
    repo = Path(__file__).resolve().parents[3]
    for path, digest in frozen["code_sha256"].items():
        if _sha(repo / path) != digest:
            raise ValueError(f"frozen implementation changed: {path}")
    for key in ("historical_run_sha256", "historical_audit_sha256", "manifest_sha256"):
        for path, digest in frozen[key].items():
            if _sha(Path(path)) != digest:
                raise ValueError(f"source changed: {path}")
    if (root / "execution_failures.json").exists() and _read(
        root / "execution_failures.json"
    ):
        raise ValueError("execution failures require a separate audit")
    registry, refs = (
        _read(root / "capture_registry.json"),
        _read(root / "frozen_panel_references.json"),
    )
    expected_paths = {
        f"calibration/{label}_{cell}.json"
        for label in metadata
        for cell in ("mm", "jj", "mj", "jm")
    }
    expected_paths |= {
        f"panel/{'reference' if m['replicate'] in (1, 2) else 'test'}/{key}/{label}_{cell}.json"
        for label, m in metadata.items()
        for key in MAPS
        for cell in ("mm", "jj", "mj", "jm")
    }
    if (
        len(registry) != 640
        or set(registry) != expected_paths
        or len(refs) != 256
        or refs
        != {k: v for k, v in registry.items() if k.startswith("panel/reference/")}
    ):
        raise ValueError("capture/reference denominator differs")
    calibration, rechecks = (
        _read(root / "calibration.json"),
        _read(root / "panel_anchor_rechecks.json"),
    )
    if calibration["status"] != "PASS":
        raise ValueError("anchor calibration incomplete")
    checks = []
    for name, rows in (
        ("calibration", calibration["checks"]),
        ("panel_anchor", rechecks),
    ):
        if len(rows) != 384 or {
            (r["pair_id"], r["cell"], r["target"]) for r in rows
        } != set(product(metadata, ("mm", "jj", "mj", "jm"), TARGETS)):
            raise ValueError("anchor comparison denominator differs")
        for row in rows:
            before, after = (
                _array(Path(row["before_path"]), row["target"]),
                _array(Path(row["after_path"]), row["target"]),
            )
            if (
                before.shape != after.shape
                or before.dtype != after.dtype
                or before.tobytes() != after.tobytes()
                or not row["bitwise_equal"]
                or row["max_abs_difference"] != 0
            ):
                raise ValueError("anchor equality does not reproduce")
            checks.append({"comparison": name, **row})
    captures, actual_inputs, responses = [], {}, Counter()
    for relative, receipt in registry.items():
        path = root / relative
        if (
            _sha(path) != receipt["run_sha256"]
            or _sha(Path(receipt["input_audit_path"])) != receipt["input_audit_sha256"]
        ):
            raise ValueError("capture receipt changed")
        run, audit = _read(path), _read(Path(receipt["input_audit_path"]))
        event = run["stream_events"][0]
        if (
            len(receipt["tensor_artifacts"]) != 3
            or event["cache_tensor_artifacts"] != receipt["tensor_artifacts"]
            or audit["native"] != audit["effective"]
            or audit["override_image_sizes"] is not None
            or audit["context"]["frozen_specification_sha256"]
            != result["frozen_specification_sha256"]
        ):
            raise ValueError("capture target/input contract differs")
        for artifact in receipt["tensor_artifacts"]:
            array = load_cache_tensor_artifact(path, artifact)
            if (
                list(array.shape) != [1, 2, 397, 64]
                or str(array.dtype) != "float32"
                or artifact["source_dtype"] != "mlx.core.bfloat16"
                or not np.isfinite(array).all()
            ):
                raise ValueError("tensor shape/dtype/finiteness differs")
        regions = cache_tensor_regions(event["cache_token_layout"], sequence_length=397)
        counts = [len(regions[r]) for r in ("pre_image", "image_tokens", "post_image")]
        if counts != [42, 256, 99] or [
            s["token_id"] for s in event["generation"]["steps"]
        ] != [785, 2168, 2168]:
            raise ValueError("source trace or layout differs")
        responses[event["assistant_text"]] += 1
        actual_inputs[relative] = audit["effective"]
        captures.append(
            {
                "run_path": str(path),
                **receipt,
                "prepared_inputs": audit,
                "source_response": event["assistant_text"],
                "token_region_counts": dict(
                    zip(("pre_image", "image_tokens", "post_image"), counts)
                ),
            }
        )
    if dict(responses) != {"The image": 640}:
        raise ValueError("source response changed")
    input_path = Path(frozen["input_summary_path"])
    if _sha(input_path) != frozen["input_summary_sha256"]:
        raise ValueError("input study changed")
    inputs = {r["pair_id"]: r for r in _read(input_path)["records"]}
    for label, meta in metadata.items():
        phase = "reference" if meta["replicate"] in (1, 2) else "test"
        for cell in ("mm", "jj", "mj", "jm"):
            old_path = (
                Path(frozen["historical_root"])
                / "panel"
                / phase
                / "selected_06_of_08"
                / f"{label}_{cell}.input_audit.json"
            )
            old = _read(old_path)["effective"]
            if actual_inputs[f"calibration/{label}_{cell}.json"] != old:
                raise ValueError("anchor calibration payload differs")
            for key in MAPS:
                native = actual_inputs[f"panel/{phase}/{key}/{label}_{cell}.json"]
                if (
                    changed_input_keys(old, native)
                    != ([] if key == ANCHOR else ["pixel_values"])
                    or native["pixel_values"]["sha256"]
                    != inputs[label]["maps"][key]["cells"][cell]["pixel_values_sha256"]
                ):
                    raise ValueError("actual map pixels or metadata isolation differs")
    analyses, vectors = {}, {}
    if set(result["factorial_paths"]) != {
        f"{key}/{t}" for key in MAPS for t in TARGETS
    }:
        raise ValueError("factorial target coverage differs")
    for key, paths in result["factorial_paths"].items():
        map_key, target = key.split("/")
        if set(paths) != set(metadata):
            raise ValueError("factorial block coverage differs")
        vectors[(map_key, target)], analyses[(map_key, target)] = {}, {}
        for label, raw in paths.items():
            analysis = _read(Path(raw))
            phase = "reference" if metadata[label]["replicate"] in (1, 2) else "test"
            cells = {}
            if set(analysis["cells"]) != {"mm", "jj", "mj", "jm"}:
                raise ValueError("factorial cell coverage differs")
            for cell, receipt in analysis["cells"].items():
                expected = root / "panel" / phase / map_key / f"{label}_{cell}.json"
                if (
                    Path(receipt["source_path"]).resolve() != expected.resolve()
                    or receipt["cache_tensor_artifact"]
                    not in registry[str(expected.relative_to(root))]["tensor_artifacts"]
                ):
                    raise ValueError("factorial source differs from capture")
                cells[cell] = load_cache_tensor_artifact(
                    expected, receipt["cache_tensor_artifact"]
                ).astype(np.float64)
            vectors[(map_key, target)][label] = verify_factorial(analysis, cells)
            analyses[(map_key, target)][label] = analysis

    def regional(key, target, view):
        start, stop = (42, 298) if view == "image_tokens" else (298, 397)
        return {
            label: v[:, :, start:stop] for label, v in vectors[(key, target)].items()
        }

    matrix = result["map_matrix"]
    expected_views = set(product(MAPS, MAPS, TARGETS, REGIONS))
    if (
        len(matrix) != 96
        or {(r["reference_map"], r["test_map"], r["target"], r["view"]) for r in matrix}
        != expected_views
    ):
        raise ValueError("map matrix coverage differs")
    cosines = 0
    for row in matrix:
        cosines += verify_transfer(
            row,
            metadata,
            regional(row["reference_map"], row["target"], row["view"]),
            regional(row["test_map"], row["target"], row["view"]),
        )
    primary = result["primary_tests"]
    if len(primary) != 12 or {
        (r["regime"], r["target"], r["view"]) for r in primary
    } != set(product(("same_map", "cross_map"), TARGETS, REGIONS)):
        raise ValueError("primary family coverage differs")
    aggregate_cosines = 0
    for row in primary:
        selected = [
            r
            for r in matrix
            if (r["target"], r["view"]) == (row["target"], row["view"])
        ]
        aggregate_cosines += verify_aggregate(row, selected, metadata)
    order = sorted(
        range(12),
        key=lambda i: (
            primary[i]["exact_block_test"]["p_greater"]
            if primary[i]["available"]
            else 1
        ),
    )
    previous = 0
    for rank, index in enumerate(order):
        row = primary[index]
        p = row["exact_block_test"]["p_greater"] if row["available"] else 1
        previous = max(previous, min(1, (12 - rank) * p))
        if (
            row["holm_family_size"] != 12
            or abs(row["holm_p_greater"] - previous) > 1e-14
        ):
            raise ValueError("joint Holm differs")
    decompositions = result["directional_decompositions"]
    expected_decompositions = {
        (a, b, t, v, label)
        for a, b in combinations(MAPS, 2)
        for t in TARGETS
        for v in REGIONS
        for label in metadata
    }
    if (
        len(decompositions) != 1152
        or {
            (
                r["reference_condition"],
                r["compared_condition"],
                r["target"],
                r["view"],
                r["pair_id"],
            )
            for r in decompositions
        }
        != expected_decompositions
    ):
        raise ValueError("directional decomposition coverage differs")
    for row in decompositions:
        label = row["pair_id"]
        if any(row[k] != v for k, v in metadata[label].items()) or row["phase"] != (
            "reference" if metadata[label]["replicate"] in (1, 2) else "test"
        ):
            raise ValueError("decomposition metadata differs")
        verify_decomposition(
            row,
            regional(row["reference_condition"], row["target"], row["view"])[label],
            regional(row["compared_condition"], row["target"], row["view"])[label],
        )
    if len(result["same_cross_gaps"]) != 6:
        raise ValueError("regime gap coverage differs")
    for gap in result["same_cross_gaps"]:
        rows = {
            r["regime"]: r
            for r in primary
            if (r["target"], r["view"]) == (gap["target"], gap["view"])
        }
        same, cross = rows["same_map"], rows["cross_map"]
        if same["available"] and cross["available"]:
            if abs(
                gap["same_minus_cross_mean_margin"]
                - same["mean_family_margin"]
                + cross["mean_family_margin"]
            ) > 1e-12 or any(
                abs(value - same["family_margins"][f] + cross["family_margins"][f])
                > 1e-12
                for f, value in gap["family_margin_differences"].items()
            ):
                raise ValueError("same/cross descriptive gap differs")
        elif gap["same_minus_cross_mean_margin"] is not None:
            raise ValueError("unavailable gap must remain null")
    historical = _read(Path(frozen["historical_root"]) / "study_summary.json")
    for row in matrix:
        if row["reference_map"] != ANCHOR or row["test_map"] != ANCHOR:
            continue
        old = next(
            r
            for r in historical["primary_tests"]
            if r["condition"] == "selected_06_of_08"
            and (r["target"], r["view"]) == (row["target"], row["view"])
        )
        for field in (
            "test_seed_scores",
            "mean_family_margin",
            "family_margins",
            "retrieval_correct_count",
        ):
            if row[field] != old[field]:
                raise ValueError("historical anchor correspondence does not reproduce")
    integrity = {}
    for key in MAPS:
        for phase, seeds in (("reference", (1, 2)), ("test", (3, 4))):
            selected = [
                a
                for (m, _), blocks in analyses.items()
                if m == key
                for label, a in blocks.items()
                if metadata[label]["replicate"] in seeds
            ]
            regional_rows = [{r["region"]: r for r in a["regions"]} for a in selected]
            if len(selected) != 48:
                raise ValueError("factorial phase denominator differs")
            integrity[f"{key}/{phase}"] = {
                "factorials": 48,
                "pre_image_all_effects_zero": sum(
                    all(e["l2_norm"] == 0 for e in r["pre_image"]["effects"].values())
                    for r in regional_rows
                ),
                "image_interaction_argmax": sum(
                    r["all_effective"]["effects"]["interaction"][
                        "argmax_sequence_position"
                    ]
                    in range(42, 298)
                    for r in regional_rows
                ),
                "image_interaction_energy_above_0_9": sum(
                    a["interaction_partition"]["image_energy_fraction"] > 0.9
                    for a in selected
                ),
                "dominance_by_region": {
                    region: dict(
                        Counter(
                            max(
                                r[region]["balanced_contrast_energy"]["energy_shares"],
                                key=r[region]["balanced_contrast_energy"][
                                    "energy_shares"
                                ].get,
                            )
                            for r in regional_rows
                        )
                    )
                    for region in ("image_tokens", "all_effective")
                },
            }
    if output.exists():
        raise FileExistsError("use a fresh publication root")
    output.mkdir(parents=True)
    ledgers = {
        "captures.jsonl": captures,
        "anchor_checks.jsonl": checks,
        "directional_decompositions.jsonl": decompositions,
    }
    for name, rows in ledgers.items():
        with (output / name).open("w") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
    summary = {k: v for k, v in result.items() if k != "directional_decompositions"}
    summary.update(
        analysis_kind="fixed_fraction_map_seed_cache_publication",
        registered_design_commit=_read(root / "registration.json")["commit"],
        source_summary_sha256=_sha(root / "study_summary.json"),
        exporter_sha256=_sha(Path(__file__)),
        verification_helper_sha256={
            name: _sha(Path(__file__).with_name(name))
            for name in (
                "summarize_graded_cache_study.py",
                "summarize_padding_cache_study.py",
            )
        },
        frozen_specification=frozen,
        model_snapshot=_read(root / "model_snapshot.json"),
        runtime_identity=_read(root / "runtime_identity.json"),
        calibration_status="PASS",
        actual_source_response_counts=dict(responses),
        factorial_integrity=integrity,
        directional_summary=summarize_decompositions(decompositions),
        ledgers={
            name: {"rows": len(rows), "sha256": _sha(output / name)}
            for name, rows in ledgers.items()
        },
        independent_verification={
            "tensor_sidecars_checked": 1920,
            "anchor_tensor_equalities_recomputed": 768,
            "matrix_cosines_recomputed_from_tensors": cosines,
            "aggregate_cosines_recomputed": aggregate_cosines,
            "exact_synchronized_assignment_tests_recomputed": 12,
            "joint_holm_family_size": 12,
            "directional_decompositions_recomputed": 1152,
            "factorial_localization_and_energy_checks": 384,
            "historical_anchor_matrix_views_reproduced": 6,
        },
        execution_failures=[],
        claim_boundaries=[
            "Twelve primary tests aggregate only three new maps; the historical anchor is descriptive and calibration-only for primary inference.",
            "The 16 averaged test-seed scores are dependent repeated-image observations across maps, not 48 or 96 new independent images.",
            "Same-minus-cross gaps, all 96 matrix views and 1152 directional records are descriptive; significance labels do not test regime differences.",
            "Input marginals/padding are fixed but processor spectra can drift between maps; selection sets and destinations both change.",
            "Reused input-selected source images, only three new maps, and repeated anchor cells limit generalization.",
            "No universal map law, geometry-only cause, semantic mechanism, persistent state or cache-to-readout mediation is established.",
        ],
    )
    if figures:
        plot_results(summary, output)
        summary["figures"] = {
            name: _sha(output / name)
            for name in (
                "map_matrix.png",
                "map_matrix.pdf",
                "same_cross_primary.png",
                "same_cross_primary.pdf",
            )
        }
    write_json(output / "summary.json", summary)
    return summary


def plot_results(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D

    labels = ("L1 keys", "L12 keys", "L23 values")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    values = [r["mean_family_margin"] for r in summary["map_matrix"] if r["available"]]
    scale = max([abs(v) for v in values] + [1e-6])
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    for i, view in enumerate(REGIONS):
        for j, (target, label) in enumerate(zip(TARGETS, labels)):
            ax = axes[i, j]
            matrix = np.full((4, 4), np.nan)
            for row in summary["map_matrix"]:
                if row["target"] == target and row["view"] == view and row["available"]:
                    matrix[
                        MAPS.index(row["test_map"]), MAPS.index(row["reference_map"])
                    ] = row["mean_family_margin"]
            shown = ax.imshow(matrix, cmap="RdBu", norm=norm)
            for y in range(4):
                for x in range(4):
                    v = matrix[y, x]
                    ax.text(
                        x,
                        y,
                        f"{v:.3f}" if np.isfinite(v) else "NA",
                        ha="center",
                        va="center",
                        color="white" if abs(v) > scale * 0.6 else "black",
                        fontsize=9,
                    )
            ax.set(
                xticks=range(4),
                xticklabels=("H", "16", "17", "18"),
                yticks=range(4),
                yticklabels=("H", "16", "17", "18"),
                title=f"{label}: {view.replace('_', ' ')}",
                xlabel="Reference map",
                ylabel="Test map",
            )
    fig.subplots_adjust(
        left=0.07, right=0.87, bottom=0.11, top=0.89, hspace=0.38, wspace=0.35
    )
    fig.colorbar(
        shown, cax=fig.add_axes([0.90, 0.18, 0.016, 0.64]), label="Mean family margin"
    )
    fig.suptitle(
        "Same and different maps at fixed 3/4 selected-site fraction", fontsize=17
    )
    fig.text(
        0.5,
        0.025,
        "H: historical anchor; 16/17/18: new map seeds 20261016/17/18. All matrix cells are descriptive, not independent tests.",
        ha="center",
        fontsize=9,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"map_matrix.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"same_map": "#176b58", "cross_map": "#a44763"}
    for ax, view in zip(axes, REGIONS):
        ax.axhline(0, color="#888888", linewidth=0.8)
        for index, target in enumerate(TARGETS):
            for regime, offset in (("same_map", -0.10), ("cross_map", 0.10)):
                row = next(
                    r
                    for r in summary["primary_tests"]
                    if (r["regime"], r["target"], r["view"]) == (regime, target, view)
                )
                if row["available"]:
                    value, color = row["mean_family_margin"], colors[regime]
                    ax.plot(
                        index + offset,
                        value,
                        "o",
                        color=color,
                        markersize=9,
                        markerfacecolor=color
                        if row["holm_p_greater"] < 0.05
                        else "white",
                    )
        ax.set(
            xticks=range(3),
            xticklabels=labels,
            title=view.replace("_", " "),
            ylabel="Mean family margin",
            xlim=(-0.45, 2.45),
        )
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Primary correspondence across three new maps", fontsize=17)
    fig.legend(
        handles=[
            Line2D(
                [], [], marker="o", color=c, linestyle="none", label=k.replace("_", " ")
            )
            for k, c in colors.items()
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Filled: joint Holm < 0.05 across 12 tests. Same/cross gaps are descriptive; no between-regime test is implied.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.24, wspace=0.28)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"same_cross_primary.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Verify and publish synchronized same-map and cross-map cache tests."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = publish(args.run_root, args.output_root)
    print(json.dumps(result["independent_verification"], indent=2))


if __name__ == "__main__":
    main()
