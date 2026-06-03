from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .frame_artifacts import prepare_frame_artifacts
from .stimulus import sha256_file

StimulusDeliveryMode = Literal[
    "visual_stream",
    "text_only_stream",
    "blank_visual_stream",
    "probe_only",
]


@dataclass(frozen=True)
class FrameDelivery:
    frame_index: int
    mode: StimulusDeliveryMode
    image_path: Path | None
    frame_artifact: dict[str, Any] | None
    num_images: int
    input_kind: str

    def to_event_record(self, output_base: Path) -> dict[str, Any]:
        image_path = None
        if self.image_path is not None:
            try:
                image_path = str(self.image_path.relative_to(output_base))
            except ValueError:
                image_path = str(self.image_path)
        return {
            "delivery_mode": self.mode,
            "input_kind": self.input_kind,
            "num_images": self.num_images,
            "delivered_image_path": image_path,
        }


def prepare_frame_deliveries(
    *,
    output_path: Path,
    manifest_base: Path,
    frames: list[dict[str, Any]],
    mode: StimulusDeliveryMode,
    include_frame_artifacts: bool,
    blank_rgb: tuple[int, int, int] = (0, 0, 0),
) -> dict[int, FrameDelivery]:
    if mode == "probe_only":
        return {}
    if mode == "visual_stream":
        return _visual_deliveries(
            output_path=output_path,
            manifest_base=manifest_base,
            frames=frames,
            include_frame_artifacts=include_frame_artifacts,
        )
    if mode == "text_only_stream":
        return {
            int(frame["index"]): FrameDelivery(
                frame_index=int(frame["index"]),
                mode=mode,
                image_path=None,
                frame_artifact=None,
                num_images=0,
                input_kind="text_only_timecode",
            )
            for frame in frames
        }
    if mode == "blank_visual_stream":
        return _blank_visual_deliveries(
            output_path=output_path,
            frames=frames,
            include_frame_artifacts=include_frame_artifacts,
            blank_rgb=blank_rgb,
        )
    raise ValueError(f"unsupported delivery mode: {mode}")


def stimulus_delivery_record(
    *,
    mode: StimulusDeliveryMode,
    include_frame_artifacts: bool,
    blank_rgb: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "include_frame_artifacts": include_frame_artifacts,
        "blank_rgb": list(blank_rgb) if mode == "blank_visual_stream" else None,
        "image_policy": _image_policy(mode),
    }


def delivery_prompt_prefix(frame: dict[str, Any], mode: StimulusDeliveryMode, timecode: str) -> str:
    if mode == "text_only_stream":
        return (
            f"This is text-only control step {frame['index']:06d} at {timecode} "
            f"({frame['t_seconds']:.3f} seconds). No image is attached."
        )
    if mode == "blank_visual_stream":
        return (
            f"This is blank-visual control frame {frame['index']:06d} at {timecode} "
            f"({frame['t_seconds']:.3f} seconds). A generated blank image is attached."
        )
    return (
        f"This is frame {frame['index']:06d} at {timecode} "
        f"({frame['t_seconds']:.3f} seconds) in the deterministic visual stream."
    )


def frame_artifact_list(deliveries: dict[int, FrameDelivery]) -> list[dict[str, Any]]:
    return [
        delivery.frame_artifact
        for delivery in deliveries.values()
        if delivery.frame_artifact is not None
    ]


def _visual_deliveries(
    *,
    output_path: Path,
    manifest_base: Path,
    frames: list[dict[str, Any]],
    include_frame_artifacts: bool,
) -> dict[int, FrameDelivery]:
    artifacts = prepare_frame_artifacts(
        output_path=output_path,
        manifest_base=manifest_base,
        frames=frames,
        enabled=include_frame_artifacts,
    )
    deliveries = {}
    for frame in frames:
        frame_index = int(frame["index"])
        artifact = artifacts.get(frame_index)
        if artifact is not None:
            artifact["delivery_mode"] = "visual_stream"
            artifact["generated_control"] = False
        deliveries[frame_index] = FrameDelivery(
            frame_index=frame_index,
            mode="visual_stream",
            image_path=manifest_base / frame["path"],
            frame_artifact=artifact,
            num_images=1,
            input_kind="source_frame",
        )
    return deliveries


def _blank_visual_deliveries(
    *,
    output_path: Path,
    frames: list[dict[str, Any]],
    include_frame_artifacts: bool,
    blank_rgb: tuple[int, int, int],
) -> dict[int, FrameDelivery]:
    artifact_dir = output_path.with_suffix(output_path.suffix + ".frames")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    deliveries = {}
    for frame in frames:
        frame_index = int(frame["index"])
        width = int(frame.get("width") or 224)
        height = int(frame.get("height") or 224)
        dest_path = artifact_dir / f"blank_frame_{frame_index:06d}.png"
        Image.new("RGB", (width, height), color=blank_rgb).save(dest_path)
        artifact = None
        if include_frame_artifacts:
            artifact = {
                "path": str(dest_path.relative_to(output_path.parent)),
                "source_path": frame["path"],
                "sha256": sha256_file(dest_path),
                "width": width,
                "height": height,
                "delivery_mode": "blank_visual_stream",
                "generated_control": True,
                "blank_rgb": list(blank_rgb),
            }
        deliveries[frame_index] = FrameDelivery(
            frame_index=frame_index,
            mode="blank_visual_stream",
            image_path=dest_path,
            frame_artifact=artifact,
            num_images=1,
            input_kind="generated_blank_frame",
        )
    return deliveries


def _image_policy(mode: StimulusDeliveryMode) -> str:
    if mode == "visual_stream":
        return "deliver manifest frame images"
    if mode == "text_only_stream":
        return "deliver frame index and timecode text without images"
    if mode == "blank_visual_stream":
        return "deliver generated blank images at manifest frame dimensions"
    if mode == "probe_only":
        return "deliver no stream turns; run clean before/after probes only"
    raise ValueError(f"unsupported delivery mode: {mode}")
