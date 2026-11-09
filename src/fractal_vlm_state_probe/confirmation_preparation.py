"""Development-only precision planning and ungenerated cohort seed metadata."""

import copy
import hashlib
import math
import shutil
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .stat_matched_source import rng_seeds
from .stat_matched_source_storage import inside, ledger, read, sha, write

PLAN = "examples/research_notes/0099_confirmation_preparation/plan.json"


def allocated_pairs(spec, plan):
    candidate = plan["candidate"]
    families = list(
        dict.fromkeys(p["pairing_family"] for p in spec["sources"]["pairs"])
    )
    pairs = []
    for family in families:
        template = next(
            p for p in spec["sources"]["pairs"] if p["pairing_family"] == family
        )
        for split, replicates in (
            ("reference", candidate["reference_replicates"]),
            ("test", candidate["test_replicates"]),
        ):
            for replicate in replicates:
                p = copy.deepcopy(template)
                p.update(
                    pair_id=f"{family}_confirmation_r{replicate}",
                    split=split,
                    replicate=replicate,
                )
                for role in ("source_a", "source_b"):
                    key = f"{candidate['source_namespace']}|{family}|{replicate}|{role}"
                    digest = hashlib.sha256(key.encode("ascii")).digest()
                    p[role]["seed"] = 1 + int.from_bytes(digest[:8], "big") % (
                        2**31 - 2
                    )
                pairs.append(p)
    if len(pairs) != 80 or len({p["pair_id"] for p in pairs}) != 80:
        raise ValueError("complete 80-pair metadata allocation required")
    return pairs


def seed_audit(pairs, defaults, historical):
    seen, old = {}, {}
    for row in historical:
        for seed in rng_seeds(row["source"], row["defaults"]):
            old.setdefault(seed, []).append(row["identity"])
    for p in pairs:
        for role in ("source_a", "source_b"):
            identity = p["pair_id"] + "/" + role
            for seed in rng_seeds(p[role], defaults):
                if seed in seen or seed in old:
                    raise ValueError(f"source initializer collision: {identity}")
                seen[seed] = identity
    return {
        "metadata_only": True,
        "historical_sources": len(historical),
        "new_sources_allocated": 160,
        "new_unique_initializers": len(seen),
        "collisions": [],
        "decoded_content_overlap_checked": False,
        "historical_inventory_complete_for_all_possible_sources": False,
    }


def residuals(profiles, spec, settings):
    tests = [p for p in spec["sources"]["pairs"] if p["split"] == "test"]
    families = list(dict.fromkeys(p["pairing_family"] for p in tests))
    if len(families) != 8 or len(tests) != 16:
        raise ValueError("complete development denominator required")
    matrix = np.empty((8, 2, 2), dtype=np.float64)
    for j, view in enumerate(settings["views"]):
        matches = [
            p
            for p in profiles
            if p["view"] == view and p["relative_floor"] == settings["relative_floor"]
        ]
        if len(matches) != 1:
            raise ValueError("one fixed development profile required per site")
        p = matches[0]
        if p["test_pair_ids"] != [t["pair_id"] for t in tests]:
            raise ValueError("development pair order differs")
        if [r["pair_id"] for r in p["pairs"]] != p["test_pair_ids"]:
            raise ValueError("development pair endpoint inventory differs")
        for fi, family in enumerate(families):
            rows = [r for r in p["pairs"] if r["family"] == family]
            if len(rows) != 2:
                raise ValueError("two development pairs per fixed family required")
            values = [r[settings["endpoint"]] for r in rows]
            if any(
                v is None or isinstance(v, bool) or not math.isfinite(v) for v in values
            ):
                raise ValueError("complete finite development endpoints required")
            center = math.fsum(values) / 2
            matrix[fi, :, j] = [v - center for v in values]
    return families, matrix


def precision_plan(profiles, spec, settings):
    families, centered = residuals(profiles, spec, settings)
    rng = np.random.Generator(np.random.PCG64(settings["simulation_seed"]))
    critical = NormalDist().inv_cdf(0.9875)
    rows = []
    for n in settings["candidate_test_pairs_per_family"]:
        draw = rng.integers(0, 2, size=(settings["simulation_trials"], 8, n))
        sampled = centered[np.arange(8)[None, :, None], draw]
        variance = np.var(sampled, axis=2, ddof=1)
        half = critical * np.sqrt(variance.sum(axis=1) / n) / 8
        # Independent analytic finite two-point formula for each simulated draw.
        counts = draw.sum(axis=2)
        delta = centered[:, 1] - centered[:, 0]
        independent_variance = (counts * (n - counts) / (n * (n - 1)))[
            :, :, None
        ] * delta[None] ** 2
        independent_half = critical * np.sqrt(independent_variance.sum(axis=1) / n) / 8
        error = float(np.max(np.abs(half - independent_half)))
        if error > 1e-12:
            raise ValueError(
                "precision simulation differs from two-point variance oracle"
            )
        for scale in settings["residual_sd_multipliers"]:
            for j, view in enumerate(settings["views"]):
                values = half[:, j] * scale
                q = np.quantile(values, [0.5, 0.9, 0.95], method="linear")
                rows.append(
                    {
                        "view": view,
                        "test_pairs_per_family": n,
                        "total_test_pairs": n * 8,
                        "sd_multiplier": scale,
                        "half_width_median": float(q[0]),
                        "half_width_p90": float(q[1]),
                        "half_width_p95": float(q[2]),
                        "fraction_within_target": float(
                            np.mean(values <= settings["planning_target_half_width"])
                        ),
                        "maximum_oracle_error_unscaled": error,
                    }
                )
    return {
        "schema_version": 1,
        "analysis_kind": "development_precision_sensitivity",
        "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATION_OR_COVERAGE_VALIDATION",
        "families": families,
        "development_pairs": 16,
        "centered_pair_residuals": centered.tolist(),
        "rows": rows,
        "normal_critical_value": critical,
        "planning_target_half_width": settings["planning_target_half_width"],
        "budget_choice_test_pairs_per_family": settings[
            "budget_choice_test_pairs_per_family"
        ],
        "sample_size_selected_by_observed_effect_or_significance": False,
        "new_map_or_reference_variance_estimated": False,
        "exact_bootstrap_coverage_established": False,
        "rng": "numpy_Generator_PCG64",
        "numpy_version": np.__version__,
    }


def prepare(repo, output):
    repo, output = Path(repo).resolve(), Path(output).absolute()
    plan = read(repo / PLAN)
    if (
        plan["status"] != "DEVELOPMENT_PLANNING_AND_METADATA_ONLY"
        or any(
            plan[k] is not False
            for k in (
                "confirmation_generation_authorized",
                "processor_execution_authorized",
                "model_execution_authorized",
            )
        )
        or any(
            type(v) is not int or v != 0
            for v in plan["new_confirmation_observations"].values()
        )
    ):
        raise ValueError("preparation-only zero-confirmation boundary required")
    if output != repo / plan["preparation_output_root"]:
        raise ValueError("canonical fresh preparation root required")
    inside(repo, plan["preparation_output_root"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        for path, digest in plan["sources"].items():
            if sha(inside(repo, path)) != digest:
                raise ValueError(f"pinned development input differs: {path}")
        spec = read(repo / "configs/stat_matched_source_panel_v1.json")
        pairs = allocated_pairs(spec, plan)
        frozen = read(repo / "runs/stat_matched_source_capture_v1/frozen.json")
        history = list(frozen["source_inventory"]["metadata"])
        history += [
            {
                "source": p[role],
                "defaults": spec["sources"]["defaults"],
                "identity": p["pair_id"] + "/" + role,
            }
            for p in spec["sources"]["pairs"]
            for role in ("source_a", "source_b")
        ]
        audit = seed_audit(pairs, spec["sources"]["defaults"], history)
        precision = precision_plan(
            ledger(
                repo / "examples/research_notes/0098_reference_crossing_results",
                "profiles.jsonl",
            ),
            spec,
            plan["development_precision"],
        )
        write(
            output / "allocated_source_metadata.json",
            {
                "schema_version": 1,
                "analysis_kind": "ungenerated_confirmation_source_metadata",
                "pairs": pairs,
                "defaults": spec["sources"]["defaults"],
                "source_images_generated": 0,
            },
        )
        write(
            output / "seed_audit.json",
            {
                "schema_version": 1,
                "analysis_kind": "confirmation_seed_metadata_audit",
                **audit,
            },
        )
        write(output / "precision.json", precision)
        free = shutil.disk_usage(repo).free
        result = {
            "schema_version": 1,
            "analysis_kind": "confirmation_preparation",
            "status": "PREPARED_NOT_CALIBRATED_NOT_RELEASED",
            "plan_sha256": sha(repo / PLAN),
            "source_images_generated": 0,
            "coordinate_maps_generated": 0,
            "processor_calls": 0,
            "model_contexts": 0,
            "confirmation_scores": 0,
            "allocated_reference_pairs": 16,
            "allocated_test_pairs": 64,
            "cohort_actual_content_sealed": False,
            "source_content_overlap_checked": False,
            "map_candidate_processor_qualified": False,
            "free_bytes_observed": free,
            "future_required_launch_free_bytes": plan["budget_candidates"][
                "required_launch_free_bytes"
            ],
            "future_disk_gate_met_at_preparation": free
            >= plan["budget_candidates"]["required_launch_free_bytes"],
            "execution_authorized": False,
            "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        }
        write(output / "summary.json", result)
        return result
    except Exception as error:
        write(output / "failure.json", {"error": str(error), "retry_authorized": False})
        raise
