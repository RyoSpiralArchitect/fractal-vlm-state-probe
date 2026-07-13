from __future__ import annotations

from typing import Any


class InternVLProcessorCompat:
    """Keep InternVL's custom image expansion off mlx-vlm's generic processor path."""

    image_token = "<image>"

    def __init__(self, *, tokenizer: Any, inner: Any) -> None:
        self.tokenizer = tokenizer
        self._inner = inner
        self.detokenizer = getattr(tokenizer, "detokenizer", None)
        self.chat_template = getattr(tokenizer, "chat_template", None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner(*args, **kwargs)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> Any:
        normalized = [
            {
                **message,
                "content": _internvl_content_text(message.get("content", "")),
            }
            for message in messages
        ]
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def ensure_mlx_processor_compat(
    processor: Any,
    model_config: Any,
) -> tuple[Any, str | None]:
    if _config_value(model_config, "model_type") != "internvl_chat":
        return processor, None

    tokenizer = getattr(processor, "tokenizer", processor)
    inner = _build_internvl_processor(tokenizer, model_config)
    return (
        InternVLProcessorCompat(tokenizer=tokenizer, inner=inner),
        "internvl_chat_custom_image_expansion",
    )


def _build_internvl_processor(tokenizer: Any, model_config: Any) -> Any:
    from mlx_vlm.models.internvl_chat.processor import (
        IMG_CONTEXT_TOKEN,
        InternVLChatProcessor,
        InternVLImageProcessor,
        chat_template,
    )

    vision_config = _config_value(model_config, "vision_config")
    image_size = int(_config_value(vision_config, "image_size") or 448)
    patch_size = int(_config_value(vision_config, "patch_size") or 14)
    downsample_ratio = float(_config_value(model_config, "downsample_ratio") or 0.5)
    image_processor = InternVLImageProcessor(
        size=image_size,
        dynamic_min_num=1,
        dynamic_max_num=12,
        dynamic_use_thumbnail=True,
    )

    # ProcessorMixin rejects this custom image processor in recent Transformers.
    # Its inference methods only require these explicit attributes.
    inner = object.__new__(InternVLChatProcessor)
    inner.image_processor = image_processor
    inner.tokenizer = tokenizer
    inner.chat_template = chat_template
    inner.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio**2))
    inner.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return inner


def _internvl_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if not isinstance(content, list):
        return str(content) if content is not None else ""

    image_count = 0
    text_parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type in {"image", "image_url", "input_image"}:
                image_count += 1
                continue
            text = item.get("text") or item.get("content")
            if text:
                text_parts.append(str(text))
        elif item is not None:
            text_parts.append(str(item))
    image_prefix = "<image>\n" * image_count
    return image_prefix + "".join(text_parts)


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)
