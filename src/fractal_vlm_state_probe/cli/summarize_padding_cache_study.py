from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import permutations, product
from pathlib import Path

import numpy as np

from fractal_vlm_state_probe.cache_direction_holdout import _load_interaction
from fractal_vlm_state_probe.cache_tensor_artifact import load_cache_tensor_artifact
from fractal_vlm_state_probe.cache_tensor_factorial import cache_tensor_regions
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import _read, _sha
from fractal_vlm_state_probe.padding_cache import tensor_difference
from fractal_vlm_state_probe.stimulus import write_json


def verify_exact_test(row, metadata):
    if not row["available"]:
        return
    scores = row["test_seed_scores"]
    expected_labels = {k for k, v in metadata.items() if v["replicate"] in (3, 4)}
    if {s["label"] for s in scores} != expected_labels or set(
        row["reference_labels"]
    ) != {k for k, v in metadata.items() if v["replicate"] in (1, 2)}:
        raise ValueError("reference/test label sets differ")
    families = sorted({v["pairing_family"] for v in metadata.values()})
    broad = {
        f: next(v["broad_class"] for v in metadata.values() if v["pairing_family"] == f)
        for f in families
    }
    matrix = np.array(
        [
            [
                np.mean(
                    [
                        s["reference_cosines"][reference]
                        for s in scores
                        if s["pairing_family"] == family
                    ]
                )
                for reference in families
            ]
            for family in families
        ]
    )
    groups = [
        [i for i, f in enumerate(families) if broad[f] == category]
        for category in sorted(set(broad.values()))
    ]
    observed = np.mean(
        [
            matrix[i, i] - np.mean([matrix[i, j] for j in group if j != i])
            for group in groups
            for i in group
        ]
    )
    null = []
    for assignment in product(*(permutations(group) for group in groups)):
        null.append(
            np.mean(
                [
                    matrix[i, j] - np.mean([matrix[i, k] for k in group if k != j])
                    for group, assigned in zip(groups, assignment)
                    for i, j in zip(group, assigned)
                ]
            )
        )
    extreme = sum(x >= observed - 1e-12 for x in null)
    if (
        len(scores) != 16
        or len(null) != 576
        or abs(observed - row["mean_family_margin"]) > 1e-12
        or extreme != row["exact_block_test"]["extreme_count"]
        or extreme / len(null) != row["exact_block_test"]["p_greater"]
    ):
        raise ValueError("saved exact correspondence test does not reproduce")
    for score in scores:
        family = score["pairing_family"]
        if (
            family != metadata[score["label"]]["pairing_family"]
            or score["replicate"] != metadata[score["label"]]["replicate"]
        ):
            raise ValueError("test score metadata differs")
        others = [f for f in families if f != family and broad[f] == broad[family]]
        margin = score["reference_cosines"][family] - np.mean(
            [score["reference_cosines"][f] for f in others]
        )
        if abs(margin - score["margin"]) > 1e-12:
            raise ValueError("seed margin differs from its cosine row")
    family_margins = {
        f: float(np.mean([s["margin"] for s in scores if s["pairing_family"] == f]))
        for f in families
    }
    if (
        any(
            abs(m - row["family_margins"][f]) > 1e-12 for f, m in family_margins.items()
        )
        or sum(m > 0 for m in family_margins.values())
        != row["positive_family_margin_count"]
    ):
        raise ValueError("family margin summary differs")


def _array(path, target):
    run = _read(path)
    artifact = next(
        a
        for a in run["stream_events"][0]["cache_tensor_artifacts"]
        if f"layer_{a['layer_index']:03d}_{a['tensor']}" == target
    )
    return load_cache_tensor_artifact(path, artifact)


def verify_primary_vectors(result, metadata):
    checked = 0
    for key, paths in result["factorial_paths"].items():
        condition, target = key.rsplit("/", 1)
        interactions = {}
        for label, path in paths.items():
            analysis = _read(Path(path))
            cells = {
                c: load_cache_tensor_artifact(
                    Path(a["source_path"]), a["cache_tensor_artifact"]
                ).astype(np.float64)
                for c, a in analysis["cells"].items()
            }
            interactions[label] = cells["jj"] - cells["jm"] - cells["mj"] + cells["mm"]
        for region, start, stop in (
            ("image_tokens", 42, 298),
            ("post_image", 298, 397),
        ):
            row = next(
                r
                for r in result["primary_tests"]
                if (r["condition"], r["target"], r["view"])
                == (condition, target, region)
            )
            if not row["available"]:
                continue
            units = {}
            for label, tensor in interactions.items():
                vector = tensor[:, :, start:stop].ravel()
                norm = float(np.sqrt(np.sum(vector * vector)))
                if norm == 0:
                    raise ValueError("available primary view contains a zero vector")
                units[label] = vector / norm
            refs = {}
            for family in sorted({v["pairing_family"] for v in metadata.values()}):
                labels = [
                    k
                    for k, v in metadata.items()
                    if v["pairing_family"] == family and v["replicate"] in (1, 2)
                ]
                combined = units[labels[0]] + units[labels[1]]
                norm = float(np.sqrt(np.sum(combined * combined)))
                if norm <= 1e-12:
                    raise ValueError("available primary view has cancelling references")
                refs[family] = combined / norm
            for score in row["test_seed_scores"]:
                for family, reference in refs.items():
                    cosine = float(np.sum(units[score["label"]] * reference))
                    if abs(cosine - score["reference_cosines"][family]) > 1e-12:
                        raise ValueError(
                            "saved primary cosine differs from captured tensors"
                        )
                    checked += 1
    return checked


def publish(root: Path, output: Path, *, figures: bool = True) -> dict:
    result = _read(root / "study_summary.json")
    frozen = _read(root / "frozen_specification.json")
    calibration = _read(root / "calibration.json")
    registry = _read(root / "capture_registry.json")
    references = _read(root / "frozen_panel_references.json")
    for file, key in (
        ("frozen_specification.json", "frozen_specification_sha256"),
        ("calibration.json", "calibration_summary_sha256"),
        ("capture_registry.json", "capture_registry_sha256"),
        ("frozen_panel_references.json", "frozen_reference_registry_sha256"),
    ):
        if _sha(root / file) != result[key]:
            raise ValueError(f"source receipt changed: {file}")
    repo = Path(__file__).resolve().parents[3]
    for path, digest in frozen["code_sha256"].items():
        if _sha(repo / path) != digest:
            raise ValueError(f"measurement implementation changed: {path}")
    if (
        len(registry) != 520
        or len(references) != 128
        or references
        != {k: v for k, v in registry.items() if k.startswith("panel/reference/")}
    ):
        raise ValueError("capture or frozen-reference denominator differs")
    rows, responses, partitions = [], Counter(), Counter()
    for raw, receipt in registry.items():
        path = root / raw
        if (
            _sha(path) != receipt["run_sha256"]
            or _sha(Path(receipt["input_audit_path"])) != receipt["input_audit_sha256"]
        ):
            raise ValueError("captured run or actual-input audit changed")
        run = _read(path)
        event = run["stream_events"][0]
        if (
            event["cache_tensor_artifacts"] != receipt["tensor_artifacts"]
            or len(receipt["tensor_artifacts"]) != 3
        ):
            raise ValueError("tensor receipt changed")
        for artifact in receipt["tensor_artifacts"]:
            array = load_cache_tensor_artifact(path, artifact)
            if list(array.shape) != [1, 2, 397, 64] or not np.isfinite(array).all():
                raise ValueError("captured tensor shape or finiteness differs")
        regions = cache_tensor_regions(event["cache_token_layout"], sequence_length=397)
        counts = tuple(
            len(regions[r]) for r in ("pre_image", "image_tokens", "post_image")
        )
        partitions[str(counts)] += 1
        responses[event["assistant_text"]] += 1
        if counts != (42, 256, 99) or [
            s["token_id"] for s in event["generation"]["steps"]
        ] != [785, 2168, 2168]:
            raise ValueError("source response trace or token partition differs")
        rows.append(
            {
                "run_path": str(path),
                **receipt,
                "prepared_inputs": _read(Path(receipt["input_audit_path"])),
                "source_response": event["assistant_text"],
                "token_region_counts": dict(
                    zip(("pre_image", "image_tokens", "post_image"), counts)
                ),
            }
        )
    if dict(responses) != {"The image": 520}:
        raise ValueError("source responses differ")
    actual_inputs = {
        str(Path(row["run_path"]).relative_to(root)): row["prepared_inputs"][
            "effective"
        ]
        for row in rows
    }
    for check in calibration["prepared_input_checks"]:
        name = f"{check['pair_id']}_{check['cell']}.json"
        before = actual_inputs[f"calibration/rectangular/{name}"]
        after = actual_inputs[f"calibration/black_square/{name}"]
        if (
            sorted(
                k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
            )
            != ["image_sizes"]
            or before["image_sizes"]["values"] != [[320, 240]]
            or after["image_sizes"]["values"] != [[320, 320]]
        ):
            raise ValueError("calibration prepared-input difference does not reproduce")
    for cell in ("mm", "jj", "mj", "jm"):
        for index, opposite in ((0, "black_square"), (1, "rectangular")):
            crossed = actual_inputs[f"calibration/metadata_cross/{cell}_{index}.json"]
            native = actual_inputs[
                f"calibration/{opposite}/{frozen['config']['metadata_cross_anchor']}_{cell}.json"
            ]
            if crossed != native:
                raise ValueError(
                    "crossed metadata inputs differ from natural counterpart"
                )
    pixel_only_pairs = 0
    for raw, before in actual_inputs.items():
        parts = Path(raw).parts
        if parts[0] != "panel" or parts[2] != "black":
            continue
        after = actual_inputs[str(Path(parts[0], parts[1], "palette_mean", parts[3]))]
        if sorted(
            k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
        ) != ["pixel_values"]:
            raise ValueError("panel padding contrast changes more than pixel_values")
        pixel_only_pairs += 1
    if pixel_only_pairs != 128:
        raise ValueError("paired panel-input denominator differs")
    all_checks, calibration_counts = [], {}
    for name, expected in (
        ("historical_checks", 384),
        ("black_square_checks", 384),
        ("crossed_metadata_checks", 24),
    ):
        checks = calibration[name]
        if len(checks) != expected:
            raise ValueError("calibration denominator differs")
        for check in checks:
            before = _array(Path(check["before_path"]), check["target"])
            after = _array(Path(check["after_path"]), check["target"])
            if (
                before.dtype != after.dtype
                or before.shape != after.shape
                or before.tobytes() != after.tobytes()
                or not check["bitwise_equal"]
                or check["max_abs_difference"] != 0
            ):
                raise ValueError("calibration tensor equality does not reproduce")
            all_checks.append({"comparison": name, **check})
        calibration_counts[name] = {
            "checks": len(checks),
            "bitwise_equal": sum(c["bitwise_equal"] for c in checks),
            "by_target": dict(Counter(c["target"] for c in checks)),
        }
    if (
        calibration["status"] != "PASS"
        or len(calibration["prepared_input_checks"]) != 128
        or any(
            c["changed_keys"] != ["image_sizes"]
            for c in calibration["prepared_input_checks"]
        )
    ):
        raise ValueError("black-sham prepared-input isolation failed")
    metadata = frozen["hierarchy_metadata"]
    primary_cosines_checked = verify_primary_vectors(result, metadata)
    for row in result["primary_tests"] + result["exploratory_views"]:
        verify_exact_test(row, metadata)
    primary = result["primary_tests"]
    if len(primary) != 12 or len(result["exploratory_views"]) != 48:
        raise ValueError("test family denominator differs")
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
        p = (
            primary[index]["exact_block_test"]["p_greater"]
            if primary[index]["available"]
            else 1
        )
        previous = max(previous, min(1, (12 - rank) * p))
        if (
            abs(primary[index]["holm_p_greater"] - previous) > 1e-14
            or primary[index]["holm_family_size"] != 12
        ):
            raise ValueError("joint Holm correction differs")
    integrity = {}
    for condition in frozen["config"]["conditions"]:
        for phase, seeds in (("reference", (1, 2)), ("test", (3, 4))):
            analyses = [
                _read(Path(path))
                for key, paths in result["factorial_paths"].items()
                if key.startswith(condition + "/")
                for label, path in paths.items()
                if metadata[label]["replicate"] in seeds
            ]
            if len(analyses) != 48:
                raise ValueError("factorial denominator differs")
            by_region = [{r["region"]: r for r in a["regions"]} for a in analyses]
            integrity[f"{condition}/{phase}"] = {
                "factorials": len(analyses),
                "pre_image_all_effects_zero": sum(
                    all(e["l2_norm"] == 0 for e in r["pre_image"]["effects"].values())
                    for r in by_region
                ),
                "image_interaction_argmax": sum(
                    r["all_effective"]["effects"]["interaction"][
                        "argmax_sequence_position"
                    ]
                    in range(42, 298)
                    for r in by_region
                ),
                "image_interaction_energy_above_0_9": sum(
                    a["interaction_partition"]["image_energy_fraction"] > 0.9
                    for a in analyses
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
                            for r in by_region
                        )
                    )
                    for region in ("image_tokens", "all_effective")
                },
            }
    vector_pairs = []
    for target in ("layer_001_keys", "layer_012_keys", "layer_023_values"):
        before_paths = result["factorial_paths"][
            f"shared_pixel_permutation/black/{target}"
        ]
        after_paths = result["factorial_paths"][
            f"shared_pixel_permutation/palette_mean/{target}"
        ]
        for label in before_paths:
            before, before_regions = _load_interaction(_read(Path(before_paths[label])))
            after, after_regions = _load_interaction(_read(Path(after_paths[label])))
            if before_regions != after_regions:
                raise ValueError("paired interaction regions differ")
            vector_pairs.append(
                {
                    "pair_id": label,
                    "target": target,
                    **metadata[label],
                    "regions": {
                        region: tensor_difference(
                            np.take(before, positions, axis=-2),
                            np.take(after, positions, axis=-2),
                        )
                        for region, positions in before_regions.items()
                        if region in ("image_tokens", "post_image")
                    },
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    for filename, ledger in (
        ("captures.jsonl", rows),
        ("calibration_checks.jsonl", all_checks),
    ):
        with (output / filename).open("w", encoding="utf-8") as handle:
            for row in ledger:
                handle.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
    summary = {
        **result,
        "analysis_kind": "padding_cache_calibration_and_panel_snapshot",
        "date": frozen["config"]["registered_date"],
        "registered_design_commit": "a5e96e3",
        "source_summary_sha256": _sha(root / "study_summary.json"),
        "exporter_sha256": _sha(Path(__file__)),
        "frozen_specification": frozen,
        "model_snapshot": _read(root / "model_snapshot.json"),
        "runtime_identity": _read(root / "runtime_identity.json"),
        "calibration_status": calibration["status"],
        "calibration_checks": calibration_counts,
        "actual_source_response_counts": dict(responses),
        "cache_partition_counts": dict(partitions),
        "panel_pairs_differing_only_in_pixel_values": pixel_only_pairs,
        "factorial_integrity": integrity,
        "paired_interaction_vectors": vector_pairs,
        "ledgers": {
            name: {"sha256": _sha(output / name), "rows": count}
            for name, count in (
                ("captures.jsonl", 520),
                ("calibration_checks.jsonl", 792),
            )
        },
        "independent_verification": {
            "tensor_sidecars_checked": 1560,
            "calibration_equalities_recomputed": 792,
            "primary_reference_test_cosines_recomputed_from_tensors": primary_cosines_checked,
            "exact_assignment_tests_recomputed": 60,
            "joint_holm_family_size": 12,
        },
        "execution_failures": [
            _read(p) for p in sorted((root / "execution_failures").glob("*.json"))
        ],
        "claim_boundaries": [
            "Calibration equality is bounded to this fixed FastVLM single-image path and three targets.",
            "Only permutation/palette-mean is fully eligible under the input frequency-summary gates; black is an unmatched comparator.",
            "Palette-mean padding changes RGB marginals; this is not a frequency-only intervention or full-spectrum equality.",
            "References exclude test-seed cache vectors, but the existing test images informed input-condition selection in Note 0045.",
            "All twelve primary tests share one Holm family; the 48 head/band tests are exploratory.",
            "Calibration replays are excluded from selected experimental counts; no direct probe, unseen-family test or cache intervention is added.",
        ],
    }
    if figures:
        plot_summary(primary, output)
        summary["figures"] = {
            name: _sha(output / name)
            for name in ("pairing_margins.png", "pairing_margins.pdf")
        }
    write_json(output / "summary.json", summary)
    return summary


def plot_summary(primary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = ("layer_001_keys", "layer_012_keys", "layer_023_values")
    views = ("image_tokens", "post_image")
    labels = [
        f"L{int(t.split('_')[1])} {t.split('_')[2]} / {'image' if v == 'image_tokens' else 'post-image'}"
        for t in targets
        for v in views
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(6)
    for offset, policy, color, label in (
        (-0.17, "black", "#49565C", "Black padding (unmatched comparator)"),
        (
            0.17,
            "palette_mean",
            "#218774",
            "Palette-mean padding (32/32 input accepted)",
        ),
    ):
        rows = [
            next(
                r
                for r in primary
                if r["condition"].endswith("/" + policy)
                and r["target"] == t
                and r["view"] == v
            )
            for t in targets
            for v in views
        ]
        values = [r["mean_family_margin"] if r["available"] else np.nan for r in rows]
        ax.barh(x + offset, values, height=0.30, color=color, label=label)
        for position, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.3f}",
                    (value, position),
                    xytext=(4 if value >= 0 else -4, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=9,
                )
    ax.set_yticks(x, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#798488", lw=0.8)
    ax.set_xlabel("Own-family cosine minus other-family cosine within broad class")
    ax.set_title(
        "Pairing correspondence after fixed pixel permutation", loc="left", pad=18
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=1, frameon=False)
    ax.margins(x=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.01,
        0.05,
        f"{sum(r['available'] and r['holm_p_greater'] < 0.05 for r in primary)}/12 primary tests pass joint Holm at 0.05. Positive bars do not establish corrected correspondence.",
        fontsize=8,
    )
    fig.text(
        0.01,
        0.015,
        "Eight existing families; reference seeds 1/2, test seeds 3/4. Changed marginals; not a frequency-only effect.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"pairing_margins.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Verify and publish the calibrated padding-cache experiment."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = publish(args.run_root, args.output_root, figures=not args.no_figures)
    print(
        f"published {result['panel_source_cells']} panel cells and {result['calibration_cells']} calibration cells",
        flush=True,
    )


if __name__ == "__main__":
    main()
