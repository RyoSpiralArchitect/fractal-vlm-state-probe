from __future__ import annotations

from fractal_vlm_state_probe.seeding import set_global_seed


def test_set_global_seed_records_seed() -> None:
    record = set_global_seed(42)
    assert record["seed"] == 42
    assert "python.random" in record["applied"]


def test_set_global_seed_allows_none() -> None:
    assert set_global_seed(None) == {"seed": None, "applied": []}

