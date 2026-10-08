from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_tensor_artifact import load_cache_tensor_artifact
from fractal_vlm_state_probe.cache_tensor_factorial import cache_tensor_regions
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.cli.summarize_padding_cache_study import (
    _array,
    verify_exact_test,
    verify_primary_vectors,
)
from fractal_vlm_state_probe.graded_cache import LEVELS, REGIONS
from fractal_vlm_state_probe.padding_cache import changed_input_keys
from fractal_vlm_state_probe.stimulus import write_json

TARGETS = ("layer_001_keys", "layer_012_keys", "layer_023_values")


def verify_decomposition(row, reference, compared):
    a, b = reference.astype(np.float64).ravel(), compared.astype(np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid decomposition vectors")
    aa, bb = float(np.sum(a * a)), float(np.sum(b * b))
    if aa == 0:
        if row["available"]:
            raise ValueError("zero-reference decomposition must be unavailable")
        return
    if not row["available"]:
        raise ValueError("nonzero-reference decomposition incorrectly unavailable")
    alpha = float(np.sum(a * b) / aa)
    perpendicular = b - alpha * a
    delta = b - a
    energy = float(np.sum(delta * delta))
    perpendicular_energy = float(np.sum(perpendicular * perpendicular))
    parallel_energy = (alpha - 1) ** 2 * aa
    expected = {
        "cosine": float(np.sum(a * b) / np.sqrt(aa * bb)) if bb else None,
        "norm_ratio": float(np.sqrt(bb / aa)),
        "reference_rms": float(np.sqrt(aa / a.size)),
        "compared_rms": float(np.sqrt(bb / b.size)),
        "projection_coefficient": alpha,
        "delta_rms": float(np.sqrt(energy / a.size)),
        "parallel_delta_rms": float(np.sqrt(parallel_energy / a.size)),
        "perpendicular_delta_rms": float(np.sqrt(perpendicular_energy / a.size)),
        "perpendicular_delta_energy_fraction": perpendicular_energy / energy
        if energy
        else None,
        "relative_delta_l2": float(np.sqrt(energy / aa)),
    }
    for field, value in expected.items():
        if (value is None and row[field] is not None) or (
            value is not None
            and (
                row[field] is None
                or not np.isclose(value, row[field], rtol=1e-12, atol=1e-12)
            )
        ):
            raise ValueError(f"decomposition differs: {field}")
    if row["relative_pythagorean_residual"] > 1e-12 or (
        energy and abs(energy - parallel_energy - perpendicular_energy) / energy > 1e-12
    ):
        raise ValueError("directional decomposition does not reconstruct delta energy")


def verify_transfer(row, metadata, endpoint, compared):
    if "exact_block_test" in row:
        raise ValueError("endpoint transfer was registered as descriptive only")
    mixed = {
        k: endpoint[k] if metadata[k]["replicate"] in (1, 2) else compared[k]
        for k in metadata
    }
    if any(not np.any(vector) for vector in mixed.values()):
        if row["available"]:
            raise ValueError("zero-vector transfer must be unavailable")
        return 0
    families = sorted({m["pairing_family"] for m in metadata.values()})
    for family in families:
        labels = [
            k
            for k, m in metadata.items()
            if m["pairing_family"] == family and m["replicate"] in (1, 2)
        ]
        a, b = (endpoint[k].astype(np.float64).ravel() for k in labels)
        summed = a / np.sqrt(np.sum(a * a)) + b / np.sqrt(np.sum(b * b))
        if np.sqrt(np.sum(summed * summed)) <= 1e-12:
            if row["available"]:
                raise ValueError("cancelling-reference transfer must be unavailable")
            return 0
    if not row["available"]:
        raise ValueError("nondegenerate transfer incorrectly unavailable")
    reference_labels = sorted(
        k for k, m in metadata.items() if m["replicate"] in (1, 2)
    )
    test_labels = sorted(set(metadata) - set(reference_labels))
    if (
        row["reference_labels"] != reference_labels
        or row["test_labels"] != test_labels
        or len(row["test_seed_scores"]) != 16
        or {s["label"] for s in row["test_seed_scores"]} != set(test_labels)
    ):
        raise ValueError("transfer reference/test labels differ")

    def unit(array):
        flat = array.astype(np.float64).ravel()
        norm = float(np.sqrt(np.sum(flat * flat)))
        if norm == 0:
            raise ValueError("available transfer contains a zero vector")
        return flat / norm

    refs = {}
    for family in families:
        selected = [
            k for k in reference_labels if metadata[k]["pairing_family"] == family
        ]
        refs[family] = unit(unit(endpoint[selected[0]]) + unit(endpoint[selected[1]]))
    checked, correct, tied, margins = 0, 0, 0, {f: [] for f in families}
    for score in row["test_seed_scores"]:
        label = score["label"]
        meta = metadata[label]
        family = meta["pairing_family"]
        if score["pairing_family"] != family or score["replicate"] != meta["replicate"]:
            raise ValueError("transfer test metadata differs")
        vector = unit(compared[label])
        values = {f: float(np.sum(vector * ref)) for f, ref in refs.items()}
        for f, value in values.items():
            if abs(value - score["reference_cosines"][f]) > 1e-12:
                raise ValueError("transfer cosine differs from tensors")
            checked += 1
        alternatives = [
            f
            for f in families
            if f != family
            and next(
                m["broad_class"] for m in metadata.values() if m["pairing_family"] == f
            )
            == meta["broad_class"]
        ]
        margin = values[family] - float(np.mean([values[f] for f in alternatives]))
        if abs(margin - score["margin"]) > 1e-12:
            raise ValueError("transfer margin differs")
        margins[family].append(margin)
        winners = [
            f for f, v in values.items() if abs(v - max(values.values())) <= 1e-12
        ]
        retrieved = winners[0] if len(winners) == 1 else None
        if score["retrieved_family"] != retrieved or score["retrieval_correct"] != (
            retrieved == family
        ):
            raise ValueError("transfer retrieval or tie differs")
        correct += retrieved == family
        tied += retrieved is None
    expected = {f: float(np.mean(v)) for f, v in margins.items()}
    if (
        any(abs(expected[f] - row["family_margins"][f]) > 1e-12 for f in families)
        or abs(np.mean(list(expected.values())) - row["mean_family_margin"]) > 1e-12
        or correct != row["retrieval_correct_count"]
        or tied != row["retrieval_tied_count"]
        or sum(v > 0 for v in expected.values()) != row["positive_family_margin_count"]
    ):
        raise ValueError("transfer aggregate differs")
    return checked


def summarize_decompositions(rows):
    output = []
    keys = sorted(
        {
            (r["compared_condition"], r["reference_condition"], r["target"], r["view"])
            for r in rows
        }
    )
    for compared, reference, target, view in keys:
        block = [
            r
            for r in rows
            if (
                r["compared_condition"],
                r["reference_condition"],
                r["target"],
                r["view"],
            )
            == (compared, reference, target, view)
        ]
        for phase in ("all", "reference", "test"):
            chosen = [r for r in block if phase == "all" or r["phase"] == phase]
            metrics = {}
            for field in (
                "cosine",
                "norm_ratio",
                "projection_coefficient",
                "relative_delta_l2",
                "perpendicular_delta_energy_fraction",
                "reference_rms",
                "compared_rms",
            ):
                values = [
                    r[field] for r in chosen if r["available"] and r[field] is not None
                ]
                metrics[field] = {
                    "available": len(values),
                    "total": len(chosen),
                    "min": min(values) if values else None,
                    "median": float(np.median(values)) if values else None,
                    "max": max(values) if values else None,
                }
            output.append(
                {
                    "compared_condition": compared,
                    "reference_condition": reference,
                    "target": target,
                    "view": view,
                    "phase": phase,
                    "metrics": metrics,
                }
            )
    return output


def publish(root, output, *, figures=True):
    result = _read(root / "study_summary.json")
    frozen = _read(root / "frozen_specification.json")
    metadata = frozen["hierarchy_metadata"]
    for file, field in (
        ("frozen_specification.json", "frozen_specification_sha256"),
        ("calibration.json", "calibration_summary_sha256"),
        ("capture_registry.json", "capture_registry_sha256"),
        ("frozen_panel_references.json", "frozen_reference_registry_sha256"),
        ("panel_endpoint_rechecks.json", "panel_endpoint_rechecks_sha256"),
    ):
        if _sha(root / file) != result[field]:
            raise ValueError(f"measurement receipt changed: {file}")
    repo = Path(__file__).resolve().parents[3]
    for path, digest in frozen["code_sha256"].items():
        if _sha(repo / path) != digest:
            raise ValueError(f"measurement implementation changed: {path}")
    registry = _read(root / "capture_registry.json")
    refs = _read(root / "frozen_panel_references.json")
    if (
        len(registry) != 512
        or len(refs) != 192
        or refs
        != {k: v for k, v in registry.items() if k.startswith("panel/reference/")}
    ):
        raise ValueError("capture or frozen-reference denominator differs")
    calibration = _read(root / "calibration.json")
    endpoint = _read(root / "panel_endpoint_rechecks.json")
    if (
        calibration["status"] != "PASS"
        or len(calibration["checks"]) != 384
        or len(endpoint) != 384
    ):
        raise ValueError("endpoint calibration is incomplete")
    checks = []
    for name, rows in (
        ("calibration", calibration["checks"]),
        ("panel_endpoint", endpoint),
    ):
        if len({(r["pair_id"], r["cell"], r["target"]) for r in rows}) != 384:
            raise ValueError("duplicate endpoint comparison")
        for row in rows:
            before, after = (
                _array(Path(row["before_path"]), row["target"]),
                _array(Path(row["after_path"]), row["target"]),
            )
            if (
                before.dtype != after.dtype
                or before.shape != after.shape
                or before.tobytes() != after.tobytes()
                or not row["bitwise_equal"]
                or row["max_abs_difference"] != 0
            ):
                raise ValueError("recorded endpoint equality does not reproduce")
            checks.append({"comparison": name, **row})
    capture_rows, responses, partitions, actual_inputs = [], Counter(), Counter(), {}
    for raw, receipt in registry.items():
        path = root / raw
        if (
            _sha(path) != receipt["run_sha256"]
            or _sha(Path(receipt["input_audit_path"])) != receipt["input_audit_sha256"]
        ):
            raise ValueError("source run or input audit changed")
        run, audit = _read(path), _read(Path(receipt["input_audit_path"]))
        event = run["stream_events"][0]
        if (
            len(receipt["tensor_artifacts"]) != 3
            or event["cache_tensor_artifacts"] != receipt["tensor_artifacts"]
            or audit["native"] != audit["effective"]
            or audit["override_image_sizes"] is not None
        ):
            raise ValueError("capture target/input contract differs")
        if (
            audit["context"]["frozen_specification_sha256"]
            != result["frozen_specification_sha256"]
        ):
            raise ValueError("input audit belongs to a different registration")
        for artifact in receipt["tensor_artifacts"]:
            array = load_cache_tensor_artifact(path, artifact)
            if (
                list(array.shape) != [1, 2, 397, 64]
                or str(array.dtype) != "float32"
                or artifact["source_dtype"] != "mlx.core.bfloat16"
                or not np.isfinite(array).all()
            ):
                raise ValueError("selected tensor shape/dtype/finiteness differs")
        regions = cache_tensor_regions(event["cache_token_layout"], sequence_length=397)
        counts = tuple(
            len(regions[r]) for r in ("pre_image", "image_tokens", "post_image")
        )
        if counts != (42, 256, 99) or [
            s["token_id"] for s in event["generation"]["steps"]
        ] != [785, 2168, 2168]:
            raise ValueError("source trace or cache layout differs")
        responses[event["assistant_text"]] += 1
        partitions[str(counts)] += 1
        actual_inputs[raw] = audit["effective"]
        capture_rows.append(
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
    if dict(responses) != {"The image": 512}:
        raise ValueError("source response changed")
    study_path = Path(frozen["input_summary_path"])
    if _sha(study_path) != frozen["input_summary_sha256"]:
        raise ValueError("input study changed")
    input_records = {r["pair_id"]: r for r in _read(study_path)["records"]}
    for label, meta in metadata.items():
        phase = "reference" if meta["replicate"] in (1, 2) else "test"
        for cell in ("mm", "jj", "mj", "jm"):
            historical_path = (
                Path(frozen["historical_root"])
                / "panel"
                / phase
                / "palette_mean"
                / f"{label}_{cell}.json"
            )
            old = _read(historical_path.with_suffix(".input_audit.json"))["effective"]
            if old != actual_inputs[f"calibration/{label}_{cell}.json"]:
                raise ValueError("calibration prepared inputs differ from endpoint")
            for level in LEVELS:
                native = actual_inputs[f"panel/{phase}/{level}/{label}_{cell}.json"]
                if (
                    changed_input_keys(old, native)
                    != ([] if level == LEVELS[-1] else ["pixel_values"])
                    or native["pixel_values"]["sha256"]
                    != input_records[label]["levels"][level]["cells"][cell][
                        "pixel_values_sha256"
                    ]
                ):
                    raise ValueError("graded actual-input isolation does not reproduce")
    primary = result["primary_tests"]
    if (
        len(primary) != 18
        or len(result["exploratory_views"]) != 72
        or len(result["endpoint_reference_transfer"]) != 12
    ):
        raise ValueError("view denominator differs")
    primary_cosines = verify_primary_vectors(result, metadata)
    for row in primary + result["exploratory_views"]:
        verify_exact_test(row, metadata)
    order = sorted(
        range(18),
        key=lambda i: (
            primary[i]["exact_block_test"]["p_greater"]
            if primary[i]["available"]
            else 1
        ),
    )
    previous = 0
    for rank, index in enumerate(order):
        p = (
            primary[index]["exact_block_test"]["p_greater"]
            if primary[index]["available"]
            else 1
        )
        previous = max(previous, min(1, (18 - rank) * p))
        if (
            primary[index]["holm_family_size"] != 18
            or abs(primary[index]["holm_p_greater"] - previous) > 1e-14
        ):
            raise ValueError("joint Holm correction differs")
    vectors, analyses = {}, {}
    for key, paths in result["factorial_paths"].items():
        level, target = key.rsplit("/", 1)
        vectors[(level, target)], analyses[(level, target)] = {}, {}
        for label, raw in paths.items():
            analysis = _read(Path(raw))
            phase = "reference" if metadata[label]["replicate"] in (1, 2) else "test"
            cells = {}
            for cell, receipt in analysis["cells"].items():
                expected_path = root / "panel" / phase / level / f"{label}_{cell}.json"
                if (
                    Path(receipt["source_path"]).resolve() != expected_path.resolve()
                    or receipt["cache_tensor_artifact"]
                    not in registry[str(expected_path.relative_to(root))][
                        "tensor_artifacts"
                    ]
                ):
                    raise ValueError(
                        "factorial source does not identify a captured panel cell"
                    )
                cells[cell] = load_cache_tensor_artifact(
                    expected_path, receipt["cache_tensor_artifact"]
                ).astype(np.float64)
            vectors[(level, target)][label] = (
                cells["jj"] - cells["jm"] - cells["mj"] + cells["mm"]
            )
            analyses[(level, target)][label] = analysis

    def region_vectors(level, target, region):
        start, stop = (42, 298) if region == "image_tokens" else (298, 397)
        return {k: v[:, :, start:stop] for k, v in vectors[(level, target)].items()}

    transferred = 0
    for row in result["endpoint_reference_transfer"]:
        if (
            row["reference_condition"] != LEVELS[-1]
            or row["test_condition"] not in LEVELS[:-1]
        ):
            raise ValueError("transfer reference level differs")
        transferred += verify_transfer(
            row,
            metadata,
            region_vectors(LEVELS[-1], row["target"], row["view"]),
            region_vectors(row["test_condition"], row["target"], row["view"]),
        )
        native = next(
            r
            for r in primary
            if (r["condition"], r["target"], r["view"])
            == (row["test_condition"], row["target"], row["view"])
        )
        if (
            row["available"]
            and native["available"]
            and (
                abs(
                    row["endpoint_minus_native_reference_margin"]
                    - (row["mean_family_margin"] - native["mean_family_margin"])
                )
                > 1e-12
            )
        ):
            raise ValueError("transfer/native reference contrast differs")
    decompositions = result["directional_decompositions"]
    if (
        len(decompositions) != 576
        or len(
            {
                (
                    r["compared_condition"],
                    r["reference_condition"],
                    r["target"],
                    r["view"],
                    r["pair_id"],
                )
                for r in decompositions
            }
        )
        != 576
    ):
        raise ValueError("decomposition denominator differs")
    for row in decompositions:
        label = row["pair_id"]
        if row["phase"] != (
            "reference" if metadata[label]["replicate"] in (1, 2) else "test"
        ) or any(row[k] != v for k, v in metadata[label].items()):
            raise ValueError("decomposition metadata or phase differs")
        verify_decomposition(
            row,
            region_vectors(row["reference_condition"], row["target"], row["view"])[
                label
            ],
            region_vectors(row["compared_condition"], row["target"], row["view"])[
                label
            ],
        )
    for row in result["paired_margin_differences"]:
        left = next(
            r
            for r in primary
            if (r["condition"], r["target"], r["view"])
            == (row["compared_condition"], row["target"], row["view"])
        )
        right = next(
            r
            for r in primary
            if (r["condition"], r["target"], r["view"])
            == (row["reference_condition"], row["target"], row["view"])
        )
        if row["available"] and (
            abs(
                row["compared_minus_reference_margin"]
                - left["mean_family_margin"]
                + right["mean_family_margin"]
            )
            > 1e-12
            or any(
                abs(value - left["family_margins"][f] + right["family_margins"][f])
                > 1e-12
                for f, value in row["family_margin_differences"].items()
            )
        ):
            raise ValueError("paired margin difference does not reproduce")
    historical_result = _read(Path(frozen["historical_root"]) / "study_summary.json")
    for row in primary:
        if row["condition"] != LEVELS[-1]:
            continue
        old = next(
            r
            for r in historical_result["primary_tests"]
            if r["condition"] == "shared_pixel_permutation/palette_mean"
            and (r["target"], r["view"]) == (row["target"], row["view"])
        )
        for field in (
            "mean_family_margin",
            "family_margins",
            "test_seed_scores",
            "exact_block_test",
            "retrieval_correct_count",
        ):
            if row[field] != old[field]:
                raise ValueError("full endpoint raw correspondence result differs")
    integrity = {}
    for level in LEVELS:
        for phase, seeds in (("reference", (1, 2)), ("test", (3, 4))):
            selected = [
                a
                for (analysis_level, _), blocks in analyses.items()
                if analysis_level == level
                for label, a in blocks.items()
                if metadata[label]["replicate"] in seeds
            ]
            regional = [{r["region"]: r for r in a["regions"]} for a in selected]
            if len(selected) != 48:
                raise ValueError("factorial phase denominator differs")
            integrity[f"{level}/{phase}"] = {
                "factorials": 48,
                "pre_image_all_effects_zero": sum(
                    all(e["l2_norm"] == 0 for e in r["pre_image"]["effects"].values())
                    for r in regional
                ),
                "image_interaction_argmax": sum(
                    r["all_effective"]["effects"]["interaction"][
                        "argmax_sequence_position"
                    ]
                    in range(42, 298)
                    for r in regional
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
                            for r in regional
                        )
                    )
                    for region in ("image_tokens", "all_effective")
                },
            }
    if output.exists():
        raise FileExistsError("use a fresh publication root")
    output.mkdir(parents=True)
    ledgers = {
        "captures.jsonl": capture_rows,
        "endpoint_checks.jsonl": checks,
        "directional_decompositions.jsonl": decompositions,
    }
    for name, rows in ledgers.items():
        with (output / name).open("w") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
    summary = {
        **result,
        "analysis_kind": "graded_cache_correspondence_and_direction_snapshot",
        "date": frozen["config"]["registered_date"],
        "registered_design_commit": _read(root / "registration.json")["commit"],
        "source_summary_sha256": _sha(root / "study_summary.json"),
        "exporter_sha256": _sha(Path(__file__)),
        "verification_helper_sha256": _sha(
            Path(__file__).with_name("summarize_padding_cache_study.py")
        ),
        "frozen_specification": frozen,
        "model_snapshot": _read(root / "model_snapshot.json"),
        "runtime_identity": _read(root / "runtime_identity.json"),
        "calibration_status": "PASS",
        "actual_source_response_counts": dict(responses),
        "cache_partition_counts": dict(partitions),
        "factorial_integrity": integrity,
        "directional_summary": summarize_decompositions(decompositions),
        "ledgers": {
            name: {"rows": len(rows), "sha256": _sha(output / name)}
            for name, rows in ledgers.items()
        },
        "independent_verification": {
            "tensor_sidecars_checked": 1536,
            "endpoint_tensor_equalities_recomputed": 768,
            "primary_cosines_recomputed_from_tensors": primary_cosines,
            "transfer_cosines_recomputed_from_tensors": transferred,
            "exact_assignment_tests_recomputed": 90,
            "joint_holm_family_size": 18,
            "directional_decompositions_recomputed": 576,
            "full_endpoint_raw_primary_results_reproduced": 6,
        },
        "execution_failures": _read(root / "execution_failures.json")
        if (root / "execution_failures.json").exists()
        else [],
        "claim_boundaries": [
            "All levels pass within-level input gates; their spectra differ across levels.",
            "The 18 primary tests share one Holm family; 72 head/band views remain exploratory.",
            "Endpoint-reference transfer, vector decompositions and paired margin differences are descriptive, not additional inferential tests.",
            "The same 32 blocks under shared transformations are dependent observations, not 576 independent samples.",
            "Calibration repeats and repeated endpoint panel cells are separately identified; no new independent image cohort is introduced.",
            "Reference directions exclude test-cache vectors, but all images informed input selection and the endpoint was already measured.",
            "No geometry-only cause, semantic mechanism, persistent state or cache-to-readout mediation is established.",
        ],
    }
    if figures:
        plot_results(summary, output)
        summary["figures"] = {
            name: _sha(output / name)
            for name in (
                "graded_correspondence.png",
                "graded_correspondence.pdf",
                "directional_decomposition.png",
                "directional_decomposition.pdf",
            )
        }
    write_json(output / "summary.json", summary)
    return summary


def plot_results(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ("#176b58", "#286d99", "#ad4d62")
    labels = ("L1 keys", "L12 keys", "L23 values")
    x = [0.75, 0.875, 1.0]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, region in zip(axes, REGIONS):
        for target, color, label in zip(TARGETS, colors, labels):
            native = [
                next(
                    r
                    for r in summary["primary_tests"]
                    if (r["condition"], r["target"], r["view"])
                    == (level, target, region)
                )
                for level in LEVELS
            ]
            transferred = [
                next(
                    r
                    for r in summary["endpoint_reference_transfer"]
                    if (r["test_condition"], r["target"], r["view"])
                    == (level, target, region)
                )
                for level in LEVELS[:-1]
            ]
            ax.plot(
                x,
                [r.get("mean_family_margin", np.nan) for r in native],
                color=color,
                label=label,
            )
            ax.plot(
                x,
                [r.get("mean_family_margin", np.nan) for r in transferred]
                + [native[-1].get("mean_family_margin", np.nan)],
                color=color,
                linestyle="--",
                alpha=0.8,
            )
            for xx, row in zip(x, native):
                if not row["available"]:
                    continue
                ax.scatter(
                    xx,
                    row["mean_family_margin"],
                    s=45,
                    facecolors=color if row["holm_p_greater"] < 0.05 else "white",
                    edgecolors=color,
                    zorder=3,
                )
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set(
            title=region.replace("_", " "),
            xlabel="Selected-site fraction",
            ylabel="Equal-family correspondence margin",
            xticks=x,
            xticklabels=["3/4", "7/8", "1"],
        )
        ax.grid(axis="y", alpha=0.15)
    handles = [
        Line2D([0], [0], color=color, label=label)
        for color, label in zip(colors, labels)
    ]
    handles += [
        Line2D([0], [0], color="#555555", label="Native-level references"),
        Line2D(
            [0], [0], color="#555555", linestyle="--", label="Full-endpoint references"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
    )
    passed = sum(
        r["available"] and r["holm_p_greater"] < 0.05 for r in summary["primary_tests"]
    )
    fig.suptitle(
        "Graded cache correspondence and fixed-reference transfer", fontsize=15
    )
    fig.text(
        0.5,
        0.015,
        f"Filled native markers: joint Holm < 0.05 ({passed}/18). Dashed transfer and level differences are descriptive.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.25, top=0.88, wspace=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"graded_correspondence.{suffix}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = ("cosine", "norm_ratio", "perpendicular_delta_energy_fraction")
    titles = (
        "Cosine to full endpoint",
        "Norm ratio to full endpoint",
        "Perpendicular share of delta energy",
    )
    for i, region in enumerate(REGIONS):
        for ax, field, title in zip(axes[i], fields, titles):
            for target, color, label in zip(TARGETS, colors, labels):
                rows = [
                    next(
                        r
                        for r in summary["directional_summary"]
                        if (
                            r["compared_condition"],
                            r["reference_condition"],
                            r["target"],
                            r["view"],
                            r["phase"],
                        )
                        == (level, LEVELS[-1], target, region, "all")
                    )["metrics"][field]
                    for level in LEVELS[:-1]
                ]
                ax.plot(
                    x[:2],
                    [r["median"] if r["median"] is not None else np.nan for r in rows],
                    "o-",
                    color=color,
                    label=label,
                )
                ax.fill_between(
                    x[:2],
                    [r["min"] if r["min"] is not None else np.nan for r in rows],
                    [r["max"] if r["max"] is not None else np.nan for r in rows],
                    color=color,
                    alpha=0.12,
                )
            ax.set(
                title=f"{region.replace('_', ' ')}\n{title}",
                xticks=x[:2],
                xticklabels=["3/4", "7/8"],
                xlabel="Compared selected-site fraction",
            )
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Direction, amplitude and non-parallel displacement", fontsize=15)
    fig.legend(
        handles=handles[:3],
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.03),
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Lines: medians; shading: range across 32 dependent blocks. Zero-delta endpoint shares are undefined and omitted.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(bottom=0.16, top=0.87, hspace=0.48, wspace=0.32)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"directional_decomposition.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Verify and publish graded cache correspondence and direction."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = publish(args.run_root, args.output_root, figures=not args.no_figures)
    print(json.dumps(result["independent_verification"], indent=2))
    print(f"wrote {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
