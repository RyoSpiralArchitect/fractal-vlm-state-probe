"""Analytic toy scores only; no new decomposition of the real score table."""

import copy
import json
from pathlib import Path

import pytest

from fractal_vlm_state_probe import margin_decomposition as core
from fractal_vlm_state_probe import margin_decomposition_oracle as oracle
from fractal_vlm_state_probe.cli import run_margin_decomposition as cli
from fractal_vlm_state_probe.stat_matched_source_storage import read, write

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    spec = read(ROOT / "configs/stat_matched_source_panel_v1.json")
    pairs = [p for p in spec["sources"]["pairs"] if p["split"] == "test"]
    names = list(dict.fromkeys(p["pairing_family"] for p in pairs))
    p = {
        "view": "encoder",
        "relative_floor": 1e-6,
        "primary_learned_profile": True,
        "families": names,
        "test_pair_ids": [q["pair_id"] for q in pairs],
        "scores": [],
    }
    for shift in range(4):
        rows = []
        for i, pair in enumerate(pairs):
            own = names.index(pair["pairing_family"])
            rows.append(
                [
                    0.6 + 0.005 * i - 0.1 * shift
                    if j == own
                    else 0.1 + 0.002 * j + 0.05 * shift
                    if j // 4 == own // 4
                    else -0.3 + 0.01 * shift
                    for j in range(8)
                ]
            )
        p["scores"].append(rows)
    return annotate(p, spec), spec


def annotate(p, spec):
    pairs = {q["pair_id"]: q for q in spec["sources"]["pairs"]}
    groups = {q["pairing_family"]: q["broad_class"] for q in pairs.values()}
    p["margins"] = []
    for shift in p["scores"]:
        values = []
        for key, row in zip(p["test_pair_ids"], shift, strict=True):
            q = pairs[key]
            own = p["families"].index(q["pairing_family"])
            others = [
                j
                for j, name in enumerate(p["families"])
                if j != own and groups[name] == q["broad_class"]
            ]
            values.append(
                None
                if any(row[j] is None for j in (own, *others))
                else row[own] - sum(row[j] for j in others) / 3
            )
        p["margins"].append(values)
    d = [
        None
        if any(p["margins"][s][i] is None for s in range(4))
        else sum(p["margins"][s][i] for s in (1, 2, 3)) / 3 - p["margins"][0][i]
        for i in range(16)
    ]
    p["family_deltas"] = [
        None if any(v is None for v in d[i : i + 2]) else sum(d[i : i + 2]) / 2
        for i in range(0, 16, 2)
    ]
    p["available"] = all(v is not None for v in d)
    p["available_test_pairs"] = sum(v is not None for v in d)
    p["fixed_test_pairs"] = 16
    p["paired_margin_change"] = sum(d) / 16 if p["available"] else None
    return p


def test_analytic_own_decline_and_competitor_rise_close_to_margin():
    p, spec = fixture()
    result = core.decompose(p, spec)
    assert result["overall"]["delta_own"] == pytest.approx(-0.2)
    assert result["overall"]["delta_other"] == pytest.approx(0.1)
    assert result["overall"]["other_contribution"] == pytest.approx(-0.1)
    assert result["overall"]["margin_change"] == pytest.approx(-0.3)
    assert core.compare(result, oracle.decompose(p, spec)) < 1e-12
    assert len(result["pairs"]) == 16 and len(result["families"]) == 8


@pytest.mark.parametrize("required", [True, False])
def test_missing_required_reference_propagates_without_filtering(required):
    p, spec = fixture()
    p["scores"][1][0][0 if required else 4] = None
    annotate(p, spec)
    result = core.decompose(p, spec)
    assert result["available_test_pairs"] == (15 if required else 16)
    assert result["available"] is (not required)
    assert (result["overall"]["margin_change"] is None) is required
    assert len(result["pairs"]) == 16 and len(result["families"]) == 8
    assert core.compare(result, oracle.decompose(p, spec)) < 1e-12


def test_all_null_direction_remains_unavailable_not_zero():
    p, spec = fixture()
    p["scores"] = [[[None] * 8 for _ in range(16)] for _ in range(4)]
    annotate(p, spec)
    result = core.decompose(p, spec)
    assert result["overall"]["margin_change"] is None
    assert result["available_test_pairs"] == 0
    assert core.compare(result, oracle.decompose(p, spec)) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, 1.01])
def test_invalid_score_is_rejected(bad):
    p, spec = fixture()
    p["scores"][0][0][0] = bad
    with pytest.raises(ValueError):
        core.decompose(p, spec)


@pytest.mark.parametrize(
    "kind", ["score_shape", "order", "margin", "family", "total", "availability"]
)
def test_source_shape_identity_and_arithmetic_corruption_fail_closed(kind):
    p, spec = fixture()
    if kind == "score_shape":
        p["scores"][0][0].pop()
    elif kind == "order":
        p["test_pair_ids"][:2] = p["test_pair_ids"][:2][::-1]
    elif kind == "margin":
        p["margins"][0][0] += 0.01
    elif kind == "family":
        p["family_deltas"][0] += 0.01
    elif kind == "total":
        p["paired_margin_change"] += 0.01
    else:
        p["available_test_pairs"] = 15
    with pytest.raises(ValueError):
        core.decompose(p, spec)


def test_all_views_and_floors_are_required():
    p, spec = fixture()
    rows = [
        {
            **p,
            "view": v,
            "relative_floor": f,
            "primary_learned_profile": v in ("encoder", "projector") and f == 1e-6,
        }
        for v, f in core.profile_identities(spec)
    ]
    core.validate_profiles(rows, spec)
    with pytest.raises(ValueError):
        core.validate_profiles(rows[:-1], spec)
    with pytest.raises(ValueError):
        core.validate_profiles(rows[::-1], spec)


@pytest.fixture
def software_run(tmp_path, monkeypatch):
    p, spec = fixture()
    plan = read(ROOT / cli.PLAN)
    rows = [
        {
            **copy.deepcopy(p),
            "view": v,
            "relative_floor": f,
            "primary_learned_profile": v in ("encoder", "projector") and f == 1e-6,
        }
        for v, f in core.profile_identities(spec)
    ]
    write(tmp_path / cli.PLAN, plan)
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(
        cli,
        "qualify",
        lambda repo: (plan, rows, spec, {"capture_sha256": "software_fixture"}),
    )
    monkeypatch.setattr(cli, "committed_code", lambda *args: {"software_fixture": True})
    root = tmp_path / plan["output_root"]
    assert cli.run("run", root)["status"] == "DECOMPOSED_UNVERIFIED"
    return tmp_path, plan, root


def test_software_file_run_two_oracle_replays_are_identical(software_run):
    repo, plan, source = software_run
    a, b = [repo / n for n in plan["verification_roots"]]
    first = cli.run("verify", a, capture=source)
    second = cli.run("verify", b, capture=source)
    assert first == second and first["status"] == "VERIFIED_SCORE_DECOMPOSITION"
    assert cli.files(a) == cli.files(b)
    with pytest.raises(FileExistsError):
        cli.run("run", source)


@pytest.mark.parametrize("mutation", ["extra_file", "summary_count", "row_value"])
def test_file_replay_rejects_refreshable_corruption(software_run, mutation):
    repo, plan, root = software_run
    if mutation == "extra_file":
        write(root / "capture.json", {"unexpected": True})
    elif mutation == "summary_count":
        data = read(root / "summary.json")
        data["model_contexts"] = 1
        (root / "summary.json").write_text(json.dumps(data), encoding="utf-8")
    else:
        rows = [
            json.loads(s)
            for s in (root / "decomposition.jsonl").read_text().splitlines()
        ]
        rows[0]["overall"]["delta_own"] += 0.1
        (root / "decomposition.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        summary = read(root / "summary.json")
        summary["output_sha256"]["decomposition.jsonl"] = cli.sha(
            root / "decomposition.jsonl"
        )
        (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    output = repo / plan["verification_roots"][0]
    with pytest.raises(ValueError):
        cli.run("verify", output, capture=root)
    assert read(output / "failure.json")["primary_results_withheld"]


def test_parent_qualification_is_file_only_and_hash_bound():
    plan, rows, _, _ = cli.qualify(ROOT)
    assert len(rows) == plan["profiles"] == 18
    assert (
        plan["model_contexts"]
        == plan["processor_calls"]
        == plan["source_generation"]
        == 0
    )
    assert plan["prior_outcomes_known"]


@pytest.mark.parametrize("kind", ["parent", "implementation"])
def test_qualifier_rejects_omitted_hash_coverage(monkeypatch, kind):
    original = cli.read

    def changed(path):
        data = original(path)
        if path == ROOT / cli.PLAN:
            if kind == "parent":
                data["sources"].pop(next(iter(data["sources"])))
            else:
                data["implementation_files"].pop()
        return data

    monkeypatch.setattr(cli, "read", changed)
    with pytest.raises(ValueError, match="hash coverage"):
        cli.qualify(ROOT)
