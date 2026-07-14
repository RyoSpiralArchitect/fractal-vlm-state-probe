from __future__ import annotations

from pathlib import Path

from fractal_vlm_state_probe.fractals import FractalSpec
from fractal_vlm_state_probe.mlx_stream import (
    StreamRunConfig,
    _collect_stream,
    audit_prompt_cache_prefix,
    _mid_probe_after_position,
    _mlx_tensor_stats,
    _mlx_sequence_position_stats,
    _new_prompt_cache_state,
    _package_version,
    _probe_phase_seeds,
    _should_summarize_cache,
    cache_layout_sequence_positions,
    run_stream_probe,
    summarize_mlx_logprobs,
    summarize_prompt_cache_token_layout,
)
from fractal_vlm_state_probe.stimulus import render_stimulus


def test_mid_probe_after_position_consumes_half_stream() -> None:
    assert _mid_probe_after_position(0) is None
    assert _mid_probe_after_position(1) == 0
    assert _mid_probe_after_position(11) == 5
    assert _mid_probe_after_position(12) == 5


def test_should_summarize_cache_keeps_key_positions() -> None:
    assert _should_summarize_cache(0, frame_count=12, every=10)
    assert _should_summarize_cache(5, frame_count=12, every=10)
    assert _should_summarize_cache(10, frame_count=12, every=10)
    assert _should_summarize_cache(11, frame_count=12, every=10)
    assert not _should_summarize_cache(4, frame_count=12, every=10)
    assert not _should_summarize_cache(0, frame_count=12, every=0)


def test_probe_phase_seeds_are_phase_stable() -> None:
    assert _probe_phase_seeds(None) == {"before": None, "mid": None, "after": None}
    assert _probe_phase_seeds(10) == {"before": 10, "mid": 11, "after": 12}


def test_package_version_reports_installed_mlx_vlm() -> None:
    assert _package_version("mlx-vlm") != "unknown"


def test_new_prompt_cache_state_does_not_reuse_batch_state() -> None:
    class PromptCacheState:
        def __init__(self) -> None:
            self.token_ids = []
            self.cache = []

    prototype = PromptCacheState()
    first = _new_prompt_cache_state(prototype)
    second = _new_prompt_cache_state(prototype)

    assert first is not prototype
    assert second is not prototype
    assert first is not second


def test_summarize_mlx_logprobs_keeps_sorted_top_k() -> None:
    import mlx.core as mx

    class Processor:
        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            return f"T{ids[0]}"

    records = summarize_mlx_logprobs(
        mx.array([-3.0, -0.5, -1.5, -0.25]),
        processor=Processor(),
        top_k=2,
    )

    assert records == [
        {"token_id": 3, "token": "T3", "logprob": -0.25},
        {"token_id": 1, "token": "T1", "logprob": -0.5},
    ]


def test_collect_stream_records_generation_step_top_logprobs() -> None:
    import mlx.core as mx

    class Chunk:
        text = "A"
        token = 3
        logprobs = mx.array([-3.0, -0.5, -1.5, -0.25])
        prompt_tokens = 4
        generation_tokens = 1
        total_tokens = 5

    class Processor:
        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            return f"T{ids[0]}"

    generation = _collect_stream([Chunk()], processor=Processor(), top_k=2)

    assert generation["text"] == "A"
    assert generation["summary"]["score_steps"] == 1
    step = generation["summary"]["steps"][0]
    assert step["token_id"] == 3
    assert step["token"] == "T3"
    assert step["token_logprob"] == -0.25
    assert [item["token_id"] for item in step["top_logprobs"]] == [3, 1]


def test_collect_stream_can_persist_first_step_full_vocab_metadata() -> None:
    import mlx.core as mx

    class Chunk:
        text = "A"
        token = 1
        logprobs = mx.array([-1.5, -0.25, -2.0])

    captured = []

    def writer(logprobs: object) -> dict:
        captured.append(logprobs)
        return {"path": "sidecar.npz", "vocab_size": 3}

    generation = _collect_stream(
        [Chunk()],
        top_k=2,
        first_step_full_vocab_writer=writer,
    )

    assert len(captured) == 1
    assert generation["summary"]["steps"][0]["full_vocab_sidecar"] == {
        "path": "sidecar.npz",
        "vocab_size": 3,
    }


def test_mlx_tensor_stats_promotes_float16_reductions() -> None:
    import math

    import mlx.core as mx

    tensor = mx.array([[[65504.0, -65504.0]]], dtype=mx.float16)
    stats = _mlx_tensor_stats(mx, tensor)

    assert math.isfinite(stats["variance"])
    assert math.isfinite(stats["l2_norm"])
    assert math.isfinite(stats["sequence_position_stats"][0]["variance"])
    assert math.isfinite(stats["sequence_position_stats"][0]["l2_norm"])


def test_mlx_sequence_stats_include_requested_positions() -> None:
    import mlx.core as mx

    tensor = mx.zeros((1, 1, 8, 2))
    records = _mlx_sequence_position_stats(
        mx,
        tensor,
        sequence_positions=[3, 5],
    )

    assert [record["position"] for record in records] == [0, 3, 4, 5, 6, 7]


def test_prompt_cache_token_layout_finds_image_runs_and_focus_positions() -> None:
    class CacheEntry:
        offset = 10

    class State:
        token_ids = [10, 151652, 151655, 151655, 151653, 20, 151652, 151655, 151653, 30]
        cache = [CacheEntry()]

    config = {
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "image_token_id": 151655,
    }

    layout = summarize_prompt_cache_token_layout(State(), config)

    assert layout["image_token_count"] == 3
    assert layout["image_token_runs"] == [
        {"start": 2, "end": 3, "length": 2},
        {"start": 7, "end": 7, "length": 1},
    ]
    roles = {
        record["position"]: record["roles"]
        for record in layout["sequence_position_plan"]
    }
    assert "image_run_0_start" in roles[2]
    assert "image_run_0_end" in roles[3]
    assert "vision_end_token_id" in roles[4]
    cache_layout = layout["cache_position_layout"]
    assert cache_layout["available"] is True
    assert cache_layout["strategy"] == "identity"
    assert cache_layout["image_position_runs"] == layout["image_token_runs"]
    assert cache_layout_sequence_positions(layout) == list(range(10))


def test_prompt_cache_token_layout_maps_granite_vision_expansion() -> None:
    class CacheEntry:
        offset = 3073

    class State:
        token_ids = [10] * 51 + [49155] * 1317 + [20] * 106
        cache = [CacheEntry()]

    layout = summarize_prompt_cache_token_layout(
        State(),
        {"model_type": "granite_vision", "image_token_index": 49155},
    )

    cache_layout = layout["cache_position_layout"]
    assert layout["token_count"] == 1474
    assert cache_layout["available"] is True
    assert cache_layout["strategy"] == "granite_vision_single_image_run_replacement"
    assert cache_layout["cache_sequence_length"] == 3073
    assert cache_layout["image_position_count"] == 2916
    assert cache_layout["image_position_runs"] == [
        {"start": 51, "end": 2966, "length": 2916}
    ]
    assert cache_layout["pre_image_position_count"] == 51
    assert cache_layout["post_image_position_count"] == 106
    assert cache_layout["expansion_delta"] == 1599
    assert cache_layout_sequence_positions(layout) == [0, 51, 1508, 1536, 2966, 3072]


def test_prompt_cache_token_layout_maps_llava_qwen2_placeholder_expansion() -> None:
    class CacheEntry:
        offset = 259

    class State:
        token_ids = [10, -200, 11, 12]
        cache = [CacheEntry()]

    layout = summarize_prompt_cache_token_layout(
        State(),
        {"model_type": "llava_qwen2", "image_token_index": -200},
    )

    cache_layout = layout["cache_position_layout"]
    assert layout["token_count"] == 4
    assert cache_layout["available"] is True
    assert cache_layout["strategy"] == "llava_qwen2_single_image_run_replacement"
    assert cache_layout["cache_sequence_length"] == 259
    assert cache_layout["image_position_count"] == 256
    assert cache_layout["image_position_runs"] == [
        {"start": 1, "end": 256, "length": 256}
    ]
    assert cache_layout["pre_image_position_count"] == 1
    assert cache_layout["post_image_position_count"] == 2
    assert cache_layout["expansion_delta"] == 255


def test_prompt_cache_token_layout_fails_closed_for_unknown_expansion() -> None:
    class CacheEntry:
        offset = 8

    class State:
        token_ids = [10, 99, 99, 20]
        cache = [CacheEntry()]

    layout = summarize_prompt_cache_token_layout(
        State(),
        {"model_type": "unknown_vlm", "image_token_id": 99},
    )

    assert layout["cache_position_layout"] == {
        "available": False,
        "coordinate_space": "effective_cache_sequence",
        "processor_token_count": 4,
        "cache_sequence_lengths": [8],
        "processor_image_position_count": 2,
        "cache_sequence_length": 8,
        "reason": "processor_token_and_cache_lengths_differ",
    }
    assert cache_layout_sequence_positions(layout) == []


def test_cache_prefix_audit_rejects_partial_prefix_and_cache_length_mismatch() -> None:
    import numpy as np

    class Tokenized:
        input_ids = np.array([[1, 2, 9, 4, 5]])

    class Tokenizer:
        pad_token = "<pad>"

        def __call__(self, *args: object, **kwargs: object) -> Tokenized:
            return Tokenized()

    class Processor:
        tokenizer = Tokenizer()
        chat_template = "template"

    class CacheEntry:
        keys = np.zeros((1, 1, 7, 2))

    class State:
        token_ids = [1, 2, 3, 4]
        cache = [CacheEntry()]

    audit = audit_prompt_cache_prefix(
        State(),
        formatted_prompt="formatted",
        processor=Processor(),
        model_config={"model_type": "qwen2_5_vl"},
    )

    assert audit["common_prefix_token_count"] == 2
    assert audit["full_source_prefix_match"] is False
    assert audit["cache_sequence_lengths"] == [7]
    assert audit["token_cache_length_aligned"] is False
    assert audit["reuse_safe_under_token_prefix_contract"] is False


def test_mlx_dry_run_records_text_only_delivery(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "text_only.json"
    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            dry_run=True,
            delivery_mode="text_only_stream",
            probe_temperature=0.7,
            probe_seed=21,
            max_frames=2,
        )
    )
    assert result["stimulus_delivery"]["mode"] == "text_only_stream"
    assert result["context_policy"]["probe_temperature"] == 0.7
    assert result["reproducibility"]["probe_seed"] == 21
    assert result["reproducibility"]["probe_phase_seeds"]["after"] == 23
    assert result["stream_events"][0]["delivery"]["num_images"] == 0
    assert "No image is attached" in result["stream_events"][0]["planned_prompt"]
    assert result["frame_artifacts"] == []


def test_mlx_dry_run_records_blank_visual_delivery(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "blank.json"
    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            dry_run=True,
            delivery_mode="blank_visual_stream",
            max_frames=1,
            blank_rgb=(5, 6, 7),
        )
    )
    assert result["stimulus_delivery"]["mode"] == "blank_visual_stream"
    assert result["stream_events"][0]["delivery"]["num_images"] == 1
    assert result["frame_artifacts"][0]["generated_control"]
    assert (tmp_path / result["frame_artifacts"][0]["path"]).exists()


def test_mlx_dry_run_probe_only_omits_stream_turns(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "probe_only.json"
    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            dry_run=True,
            delivery_mode="probe_only",
            max_frames=2,
        )
    )
    assert result["stimulus_delivery"]["mode"] == "probe_only"
    assert result["stimulus"]["source_frame_count_selected"] == 2
    assert result["stimulus"]["frame_count_selected"] == 0
    assert result["stream_events"] == []


def test_mlx_dry_run_uses_forced_choice_probe_preset(tmp_path: Path) -> None:
    manifest_path = _render_test_manifest(tmp_path / "stimulus")
    output_path = tmp_path / "forced_choice.json"
    result = run_stream_probe(
        StreamRunConfig(
            manifest_path=manifest_path,
            output_path=output_path,
            model_id="example/model",
            dry_run=True,
            delivery_mode="probe_only",
            probe_preset="forced_choice",
        )
    )
    assert result["context_policy"]["probe_preset"] == "forced_choice"
    assert result["context_policy"]["probe_count"] == 2
    assert [probe["probe_id"] for probe in result["probes"]["before"]] == [
        "forced_family_choice",
        "forced_frequency_choice",
    ]


def _render_test_manifest(output_dir: Path) -> Path:
    render_stimulus(
        FractalSpec(
            kind="mandelbrot",
            width=32,
            height=24,
            total_frames=2,
            fps=1.0,
            center_start=(-0.75, 0.1),
            center_end=(-0.74, 0.12),
            scale_start=2.0,
            scale_end=1.0,
            max_iter=24,
        ),
        output_dir,
    )
    return output_dir / "manifest.json"
