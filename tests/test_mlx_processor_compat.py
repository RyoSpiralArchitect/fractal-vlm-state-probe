from __future__ import annotations

from types import SimpleNamespace

from fractal_vlm_state_probe.mlx_processor_compat import (
    InternVLProcessorCompat,
    _apply_granite_vision_chat_template,
    _internvl_content_text,
    ensure_mlx_chat_template_compat,
    ensure_mlx_processor_compat,
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
