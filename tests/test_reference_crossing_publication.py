"""Portable replay of the registered crossings, with no native-array reads."""

import math
from pathlib import Path

import pytest

from fractal_vlm_state_probe import reference_crossing as core
from fractal_vlm_state_probe.stat_matched_source_storage import ledger, read, sha

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "examples/research_notes/0098_reference_crossing_results"


def test_public_hashes_native_receipts_and_scope():
    summary = read(PUB / "summary.json")
    assert summary["status"] == "VERIFIED_POSTHOC_REFERENCE_CROSSINGS"
    for path, digest in summary["output_sha256"].items():
        assert sha(PUB / path) == digest
    receipts = read(PUB / "verification_receipts.json")
    assert receipts["first_summary"] == receipts["second_summary"]
    assert receipts["first_tree_sha256"] == receipts["second_tree_sha256"]
    assert receipts["first_summary"]["status"] == "VERIFIED_NATIVE_CROSSINGS"
    assert (
        summary["implementation_commit"]
        == receipts["first_summary"]["provenance"]["implementation_commit"]
    )
    for path, digest in receipts["first_summary"]["provenance"][
        "implementation_sha256"
    ].items():
        assert sha(ROOT / path) == digest
    assert (
        summary["model_contexts"]
        == summary["processor_calls"]
        == summary["source_generation"]
        == 0
    )
    assert summary["p_values"] is None and summary["confidence_intervals"] is None
    assert summary["equivalence_tested"] is False
    assert summary["full_native_payloads_published"] is False


@pytest.mark.parametrize("index", range(12))
def test_each_profile_reconstructs_from_all_crossed_scores(index):
    profiles = ledger(PUB, "profiles.jsonl")
    summary = read(PUB / "summary.json")
    plan = read(
        ROOT / "examples/research_notes/0096_reference_crossing_registration/plan.json"
    )
    assert [(p["view"], p["relative_floor"]) for p in profiles] == [
        (v, f) for v in plan["views"] for f in plan["relative_floors"]
    ]
    p = profiles[index]
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    tests = [r for r in spec["sources"]["pairs"] if r["split"] == "test"]
    groups = {r["pairing_family"]: r["broad_class"] for r in spec["sources"]["pairs"]}
    for a in range(4):
        for b in range(4):
            for i, pair in enumerate(tests):
                row = p["scores"][a][b][i]
                assert len(row) == 8
                own = p["families"].index(pair["pairing_family"])
                peers = [
                    row[j]
                    for j, f in enumerate(p["families"])
                    if j != own and groups[f] == pair["broad_class"]
                ]
                other = None if any(v is None for v in peers) else math.fsum(peers) / 3
                margin = None if row[own] is None or other is None else row[own] - other
                for actual, expected in (
                    (p["own"][a][b][i], row[own]),
                    (p["other"][a][b][i], other),
                    (p["margins"][a][b][i], margin),
                ):
                    if expected is None:
                        assert actual is None
                    else:
                        assert actual == pytest.approx(expected, abs=1e-12, rel=0)
    pairs, families, overall = core.endpoints(p["margins"], tests, p["families"])
    core.compare(p["pairs"], pairs)
    core.compare(p["family_results"], families)
    core.compare(p["overall"], overall)
    assert len(pairs) == 16 and len(families) == 8 and len(p["references"]) == 32
    assert len(p["interactions"]) == 128
    if p["relative_floor"] == 1e-6:
        row = next(r for r in summary["primary_profiles"] if r["view"] == p["view"])
        core.compare(row["overall"], overall)
        core.compare(row["family_results"], families)
    if p["view"] in ("encoder", "projector"):
        old = next(
            r
            for r in ledger(
                ROOT / "examples/research_notes/0093_stat_matched_source_results",
                "profiles.jsonl",
            )
            if (r["view"], r["relative_floor"]) == (p["view"], p["relative_floor"])
        )
        core.primary_parity(p, old)


def test_figure_receipt_and_all_family_documentation():
    receipt = read(PUB / "figure_receipt.json")
    assert receipt["source_summary_sha256"] == sha(PUB / "summary.json")
    assert receipt["figure_sha256"] == sha(PUB / "family_reference_conditions.png")
    assert receipt["script_sha256"] == sha(ROOT / "scripts/plot_reference_crossing.py")
    assert receipt["all_families_retained"] == 8
    note = (ROOT / "docs/research_notes/0098_reference_crossing_results.md").read_text()
    for label in (
        "Voronoi / quasicrystal",
        "Checkerboard / hex",
        "Square / triangle",
        "Voronoi / hex",
        "White / blue noise",
        "White noise / random dots",
        "Blue noise / random dots",
        "Sparse / dense dots",
    ):
        assert "| " + label + " |" in note
    assert "not a registered statistical equivalence" in note
