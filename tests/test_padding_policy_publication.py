import json

import pytest

from fractal_vlm_state_probe.cli.summarize_padding_policy_study import publish
from fractal_vlm_state_probe.padding_policy import padding_gate
from fractal_vlm_state_probe.stimulus import sha256_file


@pytest.fixture
def study():
    cells = {
        cell: {
            "processor": {
                "spectral_centroid": 0.1,
                "high_frequency_energy_ratio": 0.02,
                "luminance_std": 0.2,
            },
            "pixel_values_shape": [3, 4, 4],
            "registered_expanded_palette_verified": True,
            "marginal_audit": {
                "content_pixels_unchanged": True,
                "padding_pixels_equal_registered_fill": True,
                "mean_max_abs_error": 0,
                "variance_max_abs_error": 0,
                "joint_rgb_total_variation_from_content": 0.25,
            },
            "black_pad_sham": {
                "pixel_values_bitwise_equal": True,
                "whole_processor_payload_equal": False,
                "changed_non_pixel_payload_keys": ["image_sizes"],
            },
        }
        for cell in ("mm", "jj", "mj", "jm")
    }
    return {
        "date": "fixture",
        "frozen_specification": {"fixture_only": True},
        "frozen_specification_sha256": "fixture",
        "block_count": 1,
        "conditions": {
            "original/black": {
                "accepted_blocks": 1,
                "all_input_blocks_accepted": True,
            }
        },
        "records": [
            {
                "pair_id": "fixture_pair",
                "pairing_family": "fixture_family",
                "broad_class": "fixture",
                "replicate": 1,
                "conditions": {
                    "original/black": {
                        "gate": padding_gate(cells),
                        "cells": cells,
                        "artifacts": {cell: {} for cell in cells},
                    }
                },
            }
        ],
        "claim_boundaries": ["fixture only"],
    }


def test_publication_preserves_ledger_and_metadata_caveat(tmp_path, study):
    source = tmp_path / "study.json"
    source.write_text(json.dumps(study))
    output = tmp_path / "publication"
    summary = publish(source, output, figures=False)
    ledger = [
        json.loads(line) for line in (output / "cells.jsonl").read_text().splitlines()
    ]
    assert len(ledger) == summary["cell_ledger"]["rows"] == 4
    assert summary["cell_ledger"]["sha256"] == sha256_file(output / "cells.jsonl")
    assert summary["checks"]["black_pixel_shams_bitwise_equal"] == 4
    assert summary["checks"]["black_shams_full_processor_payload_equal"] == 0
    assert summary["model_cache_calibration_status"] == "NOT_MEASURED"
    assert summary["new_cache_forwards"] == 0
    assert json.loads((output / "summary.json").read_text()) == summary


@pytest.mark.parametrize("corruption", ["gate", "count"])
def test_publication_rejects_inconsistent_saved_results(tmp_path, study, corruption):
    if corruption == "gate":
        study["records"][0]["conditions"]["original/black"]["gate"]["accepted"] = False
    else:
        study["conditions"]["original/black"]["accepted_blocks"] = 0
    source = tmp_path / "study.json"
    source.write_text(json.dumps(study))
    with pytest.raises(ValueError):
        publish(source, tmp_path / "publication", figures=False)
