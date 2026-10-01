from __future__ import annotations

from types import SimpleNamespace

from fractal_vlm_state_probe.mlx_processor_compat import (
    InternVLProcessorCompat,
    _eos_token_ids,
    _apply_granite_vision_chat_template,
    _internvl_content_text,
    ensure_mlx_chat_template_compat,
    ensure_mlx_processor_compat,
    load_mlx_vlm_with_compat,
)


class _Tokenizer:
    detokenizer = object()
    chat_template = "template"

    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return {"messages": messages, **kwargs}

    def decode(self, *args, **kwargs):
        return "decoded"

    def batch_decode(self, *args, **kwargs):
        return ["decoded"]


def test_internvl_compat_normalizes_multimodal_content_for_string_template() -> None:
    tokenizer = _Tokenizer()
    processor = InternVLProcessorCompat(
        tokenizer=tokenizer, inner=lambda **kwargs: kwargs
    )

    rendered = processor.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "content", "content": "system"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "content", "content": "question"},
                ],
            },
        ],
        add_generation_prompt=True,
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "<image>\n<image>\nquestion"},
    ]
    assert rendered["add_generation_prompt"] is True
    assert not hasattr(processor, "image_processor")


def test_internvl_content_text_accepts_existing_string() -> None:
    assert _internvl_content_text("hello") == "hello"


def test_non_internvl_processor_is_unchanged() -> None:
    processor = object()

    resolved, compatibility = ensure_mlx_processor_compat(
        processor,
        SimpleNamespace(model_type="qwen2_5_vl"),
    )

    assert resolved is processor
    assert compatibility is None


def test_granite_vision_compat_uses_tokenizer_multimodal_template() -> None:
    tokenizer = _Tokenizer()
    processor = SimpleNamespace(tokenizer=tokenizer)

    rendered = _apply_granite_vision_chat_template(
        processor,
        SimpleNamespace(model_type="granite_vision"),
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
        ],
        num_images=2,
    )

    assert tokenizer.messages == [
        {"role": "system", "content": [{"type": "text", "text": "system"}]},
        {"role": "user", "content": [{"type": "text", "text": "earlier"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": "question"},
            ],
        },
    ]
    assert rendered["add_generation_prompt"] is True


def test_non_granite_chat_template_is_unchanged() -> None:
    apply_chat_template = object()

    resolved, compatibility = ensure_mlx_chat_template_compat(
        apply_chat_template,
        SimpleNamespace(model_type="qwen2_5_vl"),
    )

    assert resolved is apply_chat_template
    assert compatibility is None


def test_eos_token_ids_normalizes_scalar_list_and_fallback() -> None:
    assert _eos_token_ids({}, 2) == [2]
    assert _eos_token_ids({"eos_token_id": 7}, 2) == [7]
    assert _eos_token_ids({"eos_token_id": [7, 8]}, 2) == [7, 8]


def test_mlx_load_compat_preserves_the_default_path() -> None:
    model = object()
    processor = object()

    resolved_model, resolved_processor, compatibility = load_mlx_vlm_with_compat(
        "example/model",
        default_load=lambda model_id: (model, processor),
    )

    assert resolved_model is model
    assert resolved_processor is processor
    assert compatibility is None


def test_mlx_load_compat_reraises_unrelated_value_errors() -> None:
    def fail(model_id: str):
        raise ValueError("unrelated load failure")

    try:
        load_mlx_vlm_with_compat("example/model", default_load=fail)
    except ValueError as exc:
        assert str(exc) == "unrelated load failure"
    else:
        raise AssertionError("expected the unrelated load error to propagate")
