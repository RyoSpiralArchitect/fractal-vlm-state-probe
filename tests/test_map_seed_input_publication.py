import copy

import pytest

from fractal_vlm_state_probe.cli.summarize_map_seed_input_study import (
    verify_panel_counts,
)
from fractal_vlm_state_probe.permutation_maps import MAPS, summarize_input_maps
from test_permutation_maps import complete_records


def test_publication_recounts_failed_maps_without_rescuing():
    records = complete_records()
    records[0]["maps"][MAPS[-1]]["gate"]["accepted"] = False
    assert not verify_panel_counts(records, summarize_input_maps(records))


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted_blocks", 32),
        ("total_blocks", 31),
        ("rejected_pair_ids", []),
        ("complete_four_seed_families", ["f0"]),
    ],
)
def test_publication_rejects_count_or_failure_suppression(field, value):
    records = complete_records()
    records[0]["maps"][MAPS[-1]]["gate"]["accepted"] = False
    counts = copy.deepcopy(summarize_input_maps(records))
    counts[MAPS[-1]][field] = value
    with pytest.raises(ValueError):
        verify_panel_counts(records, counts)


def test_publication_rejects_duplicate_blocks():
    records = complete_records()
    counts = summarize_input_maps(records)
    records[-1] = copy.deepcopy(records[0])
    with pytest.raises(ValueError):
        verify_panel_counts(records, counts)
