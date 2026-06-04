from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ConditionFamily = Literal["fractal", "geometric", "natural", "control"]
TemporalPolicy = Literal["ordered", "shuffled", "reversed", "static_repeat", "text_only"]
SemanticLoad = Literal["none", "low", "medium", "high"]
SourceKind = Literal["generated", "external_frames", "external_video", "text_only"]


@dataclass(frozen=True)
class StimulusCondition:
    condition_id: str
    condition_family: ConditionFamily
    temporal_policy: TemporalPolicy
    semantic_load: SemanticLoad
    deterministic: bool
    source_kind: SourceKind
    comparison_role: str = "experimental"
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StimulusCondition":
        required = {
            "condition_id",
            "condition_family",
            "temporal_policy",
            "semantic_load",
            "deterministic",
            "source_kind",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"missing stimulus condition keys: {', '.join(missing)}")

        condition = cls(
            condition_id=str(data["condition_id"]),
            condition_family=data["condition_family"],
            temporal_policy=data["temporal_policy"],
            semantic_load=data["semantic_load"],
            deterministic=bool(data["deterministic"]),
            source_kind=data["source_kind"],
            comparison_role=str(data.get("comparison_role", "experimental")),
            description=str(data.get("description", "")),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if not self.condition_id:
            raise ValueError("condition_id must not be empty")
        if self.condition_family not in ("fractal", "geometric", "natural", "control"):
            raise ValueError(f"unsupported condition_family: {self.condition_family}")
        if self.temporal_policy not in (
            "ordered",
            "shuffled",
            "reversed",
            "static_repeat",
            "text_only",
        ):
            raise ValueError(f"unsupported temporal_policy: {self.temporal_policy}")
        if self.semantic_load not in ("none", "low", "medium", "high"):
            raise ValueError(f"unsupported semantic_load: {self.semantic_load}")
        if self.source_kind not in (
            "generated",
            "external_frames",
            "external_video",
            "text_only",
        ):
            raise ValueError(f"unsupported source_kind: {self.source_kind}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "condition_family": self.condition_family,
            "temporal_policy": self.temporal_policy,
            "semantic_load": self.semantic_load,
            "deterministic": self.deterministic,
            "source_kind": self.source_kind,
            "comparison_role": self.comparison_role,
            "description": self.description,
        }


def default_generated_fractal_condition(kind: str) -> StimulusCondition:
    return StimulusCondition(
        condition_id=f"{kind}_generated_ordered",
        condition_family="fractal",
        temporal_policy="ordered",
        semantic_load="low",
        deterministic=True,
        source_kind="generated",
        comparison_role="experimental",
        description=f"Generated {kind} frame stream with deterministic parameters.",
    )


def condition_from_config(
    config: dict[str, Any],
    *,
    fallback_kind: str = "mandelbrot",
) -> StimulusCondition:
    raw = config.get("stimulus_condition")
    if raw is None:
        return default_generated_fractal_condition(fallback_kind)
    if not isinstance(raw, dict):
        raise ValueError("stimulus_condition must be an object")
    return StimulusCondition.from_dict(raw)
