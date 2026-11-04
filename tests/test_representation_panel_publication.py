import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fractal_vlm_state_probe.cli.summarize_controlled_overlap import (
    oracle_bootstrap,
    oracle_summary,
    verify_summary,
)
from fractal_vlm_state_probe.controlled_overlap_cache_panel import ordered_jobs
from fractal_vlm_state_probe.representation_panel import (
    ENDPOINTS,
    SHAPES,
    SITES,
    array_path,
    run_path,
    seal,
)


@pytest.fixture(scope="module")
def publication():
    repo = Path(__file__).parents[1]
    root = repo / "examples/research_notes/0070_representation_correspondence"
    content = (root / "summary.json").read_bytes()
    assert (
        hashlib.sha256(content).hexdigest()
        == "40327bc266271e300b436cd8d1125a81222d209fe1bef37c464bbecf93d500a4"
    )
    summary = json.loads(content)
    assert (
        summary["status"] == "COMPLETE"
        and summary["analysis_kind"] == "representation_panel_publication"
    )
    assert {p.name for p in root.iterdir()} == {
        "summary.json",
        *summary["ledgers"],
        *summary["figures"],
    }
    for name, digest in summary["figures"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    ledgers = {}
    for name, receipt in summary["ledgers"].items():
        content = (root / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == receipt["sha256"]
        ledgers[name] = [json.loads(x) for x in content.splitlines()]
        assert len(ledgers[name]) == receipt["rows"]
    return repo, summary, ledgers


def test_published_capture_receipts_sources_and_reference_seal(publication):
    repo, summary, ledgers = publication
    frozen = summary["frozen_specification"]
    assert (
        frozen["registered_design_commit"] == "996656c4263a716af3ac7e48eb84ae3c10c39371"
    )
    assert (
        frozen["release"]["preceding_merge_commit"]
        == "61b0e16a5f93fafd02d243288cb22ed4a2c48529"
    )
    assert len(frozen["code_sha256"]) == 67
    assert all(
        hashlib.sha256((repo / p).read_bytes()).hexdigest() == h
        for p, h in frozen["code_sha256"].items()
    )
    assert (
        frozen["finite_codec_words_checked"] == 65280
        and frozen["historical_codec_arrays_checked"] == 256
    )
    capture, check = summary["capture_summary"], summary["independent_verification"]
    assert (
        capture["calibration_contexts"],
        capture["panel_cells"],
        capture["new_source_contexts"],
    ) == (384, 2048, 2432)
    assert capture["same_model_instance"] and capture["new_direct_probe_cells"] == 0
    assert summary["recalibration_parity"]["representation_array_pairs_exact"] == 256
    assert {k: len(v) for k, v in ledgers.items()} == {
        "cells.jsonl": 2048,
        "cache_parity.jsonl": 2048,
        "vectors.jsonl": 1024,
        "directions.jsonl": 512,
        "scores.jsonl": 56,
    }
    assert [
        check[k]
        for k in (
            "packed_arrays_reexpanded",
            "full_contrast_byte_equalities",
            "score_matrices",
            "direction_records",
        )
    ] == [4096, 1024, 56, 512]
    assert check["model_forwards"] == check["processor_calls"] == 0
    assert (
        check["cache_tensor_pairs"] == 6144
        and check["cache_scalar_pairs"] == 6144 * 50816
    )
    assert check["maximum_score_abs_error"] <= 1e-12
    assert (
        not summary["full_cell_arrays_published"]
        and summary["confirmation_cohort"] == "not_generated_not_sealed"
    )
    jobs = ordered_jobs(frozen["hierarchy_metadata"])
    cells = ledgers["cells.jsonl"]
    gate = seal(cells[:256], jobs)
    assert all(summary["reference_seal"][k] == v for k, v in gate.items())
    sealed_bytes = (
        json.dumps(cells[:256], indent=2, ensure_ascii=False) + "\n"
    ).encode()
    assert (
        hashlib.sha256(sealed_bytes).hexdigest() == summary["reference_seal"]["sha256"]
    )
    for i, (job, row, parity) in enumerate(
        zip(jobs, cells, ledgers["cache_parity.jsonl"], strict=True)
    ):
        assert all(row[k] == v for k, v in job.items())
        assert row["ordinal"] == parity["ordinal"] == 385 + i
        assert all(row[k] == parity[k] for k in ("map", "pair_id", "cell"))
        assert row["path"] == run_path(job) and row["representation"][
            "path"
        ] == array_path(job)
        assert row["observer_restored"] and row["observer_calls"] == 1
        assert row["representation"]["packed_dtype"] == "<u2"
        assert set(parity["targets"]) == {"1:keys", "12:keys", "23:values"}
        for k, t in parity["targets"].items():
            assert t["scalar_count"] == 50816
            assert (
                t["before_array_sha256"]
                == t["after_array_sha256"]
                == row["cache_array_sha256"][k]
            )
        for site, shape in SHAPES.items():
            a = row["representation"]["arrays"][site]
            assert (
                a["shape"] == list(shape)
                and a["dtype"] == "float32"
                and a["source_dtype"] == "mlx.core.bfloat16"
            )
            assert a["scalar_count"] == np.prod(shape) and len(a["array_sha256"]) == 64
    for r in summary["comparison_context"].values():
        p = repo / r["publication"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["sha256"]
        data = json.loads(p.read_text())
        assert (
            r["primary_endpoints"] == data["primary_endpoints"]
            and r["aggregates"] == data["aggregates"]
        )


def test_primary_ranges_and_direction_receipts_replay_without_source_tensors(
    publication,
):
    _, summary, ledgers = publication
    f = summary["frozen_specification"]
    meta = f["hierarchy_metadata"]
    labels = sorted(k for k, v in meta.items() if v["replicate"] in (3, 4))
    families = sorted({v["pairing_family"] for v in meta.values()})
    cube = np.full((2, 4, 2, 3, 16, 8), np.nan)
    scores = ledgers["scores.jsonl"]
    assert {(r["site"], r["map"]) for r in scores} == {
        (s, m) for s in SITES for m in f["map_recipes"]
    }
    for row in scores:
        recipe = f["map_recipes"][row["map"]]
        assert (
            all(row[k] == v for k, v in recipe.items())
            and row["reference_map"] == recipe["reference"]
        )
        if not row["available"]:
            assert "scores" not in row and row["reason"]
            continue
        assert row["labels"] == labels and row["families"] == families
        matrix = np.asarray(row["scores"])
        assert matrix.shape == (16, 8) and np.isfinite(matrix).all()
        verify_summary(row["summary"], matrix, labels, families, meta)
        if recipe["q"] is not None:
            cube[
                SITES.index(row["site"]),
                recipe["mask"],
                recipe["destination"],
                (0, 1, 4).index(recipe["q"]),
            ] = matrix
    for row in summary["aggregates"]:
        v = cube[SITES.index(row["site"]), :, :, (0, 1, 4).index(row["q_sixteenths"])]
        assert row["available"] == bool(np.isfinite(v).all())
        if row["available"]:
            verify_summary(row, v.mean(axis=(0, 1)), labels, families, meta)
    margins = np.full((2, 4, 2, 3, 8, 2), np.nan)
    for idx in np.ndindex(2, 4, 2, 3):
        if np.isfinite(cube[idx]).all():
            values = oracle_summary(cube[idx], labels, families, meta)[0]
            for fi, fam in enumerate(families):
                margins[idx + (fi,)] = [
                    v
                    for k, v in zip(labels, values)
                    if meta[k]["pairing_family"] == fam
                ]
    spec = f["release"]["bootstrap"]
    assert (
        spec["draws"] == 10000
        and spec["seed"] == 2026110301
        and spec["quantiles"] == [0.025 / 4, 1 - 0.025 / 4]
    )
    ranges = oracle_bootstrap(margins, spec)
    assert {(e["site"], e["endpoint"]) for e in summary["primary_endpoints"]} == {
        (s, e) for s in SITES for e in ENDPOINTS
    }
    for row in summary["primary_endpoints"]:
        vi, ei = SITES.index(row["site"]), ENDPOINTS.index(row["endpoint"])
        available = bool(
            np.isfinite(margins[vi, :, :, 0]).all()
            and np.isfinite(margins[vi, :, :, 2]).all()
        )
        assert (
            row["available"] == available
            and row["p_value"] is None
            and not row["equivalence_tested"]
        )
        if not available:
            assert "mean" not in row and "exploratory_range" not in row
            continue
        v = (
            margins[vi, :, :, 0]
            if ei == 0
            else margins[vi, :, :, 2] - margins[vi, :, :, 0]
        )
        np.testing.assert_allclose(row["mean"], v.mean(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            row["mask_values"], v.mean(axis=(1, 2, 3)), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            [row["family_values"][k] for k in families],
            v.mean(axis=(0, 1, 3)),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            row["exploratory_range"], ranges[vi, ei], rtol=0, atol=1e-12
        )
    vectors = {(r["site"], r["map"], r["pair_id"]): r for r in ledgers["vectors.jsonl"]}
    assert len(vectors) == 1024
    for row in ledgers["directions.jsonl"]:
        s, m, d, q, k = [
            row[key]
            for key in ("site", "mask", "destination", "q_sixteenths", "pair_id")
        ]
        a = vectors[(s, f"m{m}_q0_d{d}", k)]["norm"]
        b = vectors[(s, f"m{m}_q{q}_d{d}", k)]["norm"]
        np.testing.assert_allclose(
            [row["zero_norm"], row["other_norm"]], [a, b], rtol=1e-11, atol=1e-12
        )
        if a and b:
            r = b / a
            np.testing.assert_allclose(row["norm_ratio"], r, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(
                row["relative_l2_difference"] ** 2,
                1 + r * r - 2 * r * row["cosine"],
                rtol=1e-10,
                atol=1e-12,
            )
        else:
            assert row["cosine"] is None
