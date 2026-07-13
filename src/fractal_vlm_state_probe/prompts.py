from __future__ import annotations

from copy import deepcopy
from typing import Any


SYSTEM_PROMPT = (
    "You are participating in a measurement run. Describe and respond only to "
    "the requested task. Do not infer consciousness, subjective experience, or "
    "altered mental states from the visual stream."
)

SYNC_PROMPT = (
    "Focus on this deterministic evolving visual pattern as context for this "
    "run. Return ACK only."
)

DEFAULT_PROBES = [
    {
        "id": "blue_silence_poem",
        "prompt": "Write a single, abstract, creative poem titled 'Blue Silence'.",
    },
]

FORCED_CHOICE_PROBES = [
    {
        "id": "forced_family_choice",
        "prompt": (
            "Forced choice. Based only on the current context, answer exactly one "
            "letter: A for Mandelbrot-like zoom structure, B for Julia-like "
            "filament structure, or C for unclear/neither. Output only A, B, or C."
        ),
        "probe_family": "family",
        "prompt_variant": "baseline",
        "candidate_labels": ["A", "B", "C"],
        "candidate_order": ["A", "B", "C"],
        "candidate_semantics": {
            "A": "mandelbrot",
            "B": "julia",
            "C": "unclear",
        },
    },
    {
        "id": "forced_frequency_choice",
        "prompt": (
            "Forced choice. Based only on the current context, answer exactly one "
            "letter: L for coarse low-frequency layout, H for high-frequency edge "
            "texture, or C for unclear/mixed. Output only L, H, or C."
        ),
        "probe_family": "frequency",
        "prompt_variant": "baseline",
        "candidate_labels": ["L", "H", "C"],
        "candidate_order": ["L", "H", "C"],
        "candidate_semantics": {
            "L": "low_frequency",
            "H": "high_frequency",
            "C": "unclear",
        },
    },
]

FORCED_CHOICE_PARAPHRASE_PROBES = [
    {
        "id": "forced_family_choice_paraphrase",
        "prompt": (
            "Choose the closest description of the attached visual context. "
            "Return one character only: A for a connected bulb-and-cusp "
            "Mandelbrot zoom, B for a Julia-style filament or island field, or "
            "C when neither description is reliable."
        ),
        "probe_family": "family",
        "prompt_variant": "paraphrase",
        "candidate_labels": ["A", "B", "C"],
        "candidate_order": ["A", "B", "C"],
        "candidate_semantics": {
            "A": "mandelbrot",
            "B": "julia",
            "C": "unclear",
        },
    },
    {
        "id": "forced_frequency_choice_paraphrase",
        "prompt": (
            "Judge the dominant spatial scale in the attached visual context. "
            "Return one character only: L when broad regions and coarse "
            "boundaries dominate, H when fine edges and repeated small details "
            "dominate, or C when the balance is unclear."
        ),
        "probe_family": "frequency",
        "prompt_variant": "paraphrase",
        "candidate_labels": ["L", "H", "C"],
        "candidate_order": ["L", "H", "C"],
        "candidate_semantics": {
            "L": "low_frequency",
            "H": "high_frequency",
            "C": "unclear",
        },
    },
]

FORCED_CHOICE_REVERSED_ORDER_PROBES = [
    {
        "id": "forced_family_choice_reversed_order",
        "prompt": (
            "Forced choice. Based only on the current context, answer exactly "
            "one letter: C for unclear/neither, B for Julia-like filament "
            "structure, or A for Mandelbrot-like zoom structure. Output only C, "
            "B, or A."
        ),
        "probe_family": "family",
        "prompt_variant": "reversed_order",
        "candidate_labels": ["A", "B", "C"],
        "candidate_order": ["C", "B", "A"],
        "candidate_semantics": {
            "A": "mandelbrot",
            "B": "julia",
            "C": "unclear",
        },
    },
    {
        "id": "forced_frequency_choice_reversed_order",
        "prompt": (
            "Forced choice. Based only on the current context, answer exactly "
            "one letter: C for unclear/mixed, H for high-frequency edge texture, "
            "or L for coarse low-frequency layout. Output only C, H, or L."
        ),
        "probe_family": "frequency",
        "prompt_variant": "reversed_order",
        "candidate_labels": ["L", "H", "C"],
        "candidate_order": ["C", "H", "L"],
        "candidate_semantics": {
            "L": "low_frequency",
            "H": "high_frequency",
            "C": "unclear",
        },
    },
]

FORCED_CHOICE_ROTATED_LABEL_PROBES = [
    {
        "id": "forced_family_choice_rotated_labels",
        "prompt": (
            "Forced choice. Based only on the current context, answer exactly "
            "one letter: A for unclear/neither, B for Mandelbrot-like zoom "
            "structure, or C for Julia-like filament structure. Output only A, "
            "B, or C."
        ),
        "probe_family": "family",
        "prompt_variant": "rotated_labels",
        "candidate_labels": ["A", "B", "C"],
        "candidate_order": ["A", "B", "C"],
        "candidate_semantics": {
            "A": "unclear",
            "B": "mandelbrot",
            "C": "julia",
        },
    },
    {
        "id": "forced_frequency_choice_rotated_labels",
        "prompt": (
            "Forced choice. Based only on the current context, answer exactly "
            "one letter: L for unclear/mixed, H for coarse low-frequency layout, "
            "or C for high-frequency edge texture. Output only L, H, or C."
        ),
        "probe_family": "frequency",
        "prompt_variant": "rotated_labels",
        "candidate_labels": ["L", "H", "C"],
        "candidate_order": ["L", "H", "C"],
        "candidate_semantics": {
            "L": "unclear",
            "H": "low_frequency",
            "C": "high_frequency",
        },
    },
]

FORCED_CHOICE_ROBUSTNESS_PROBES = [
    *FORCED_CHOICE_PROBES,
    *FORCED_CHOICE_PARAPHRASE_PROBES,
    *FORCED_CHOICE_REVERSED_ORDER_PROBES,
    *FORCED_CHOICE_ROTATED_LABEL_PROBES,
]

CREATIVE_REFLECTION_PROMPT = (
    "Describe the visual impression this evolving pattern leaves behind. "
    "Write a short abstract reflection in 3-5 sentences."
)

CREATIVE_REFLECTION_PROBES = [
    {
        "id": "creative_visual_impression",
        "prompt": CREATIVE_REFLECTION_PROMPT,
    },
]

PROBE_PRESETS = {
    "default": DEFAULT_PROBES,
    "creative_reflection": CREATIVE_REFLECTION_PROBES,
    "forced_choice": FORCED_CHOICE_PROBES,
    "forced_choice_paraphrase": FORCED_CHOICE_PARAPHRASE_PROBES,
    "forced_choice_reversed_order": FORCED_CHOICE_REVERSED_ORDER_PROBES,
    "forced_choice_rotated_labels": FORCED_CHOICE_ROTATED_LABEL_PROBES,
    "forced_choice_robustness": FORCED_CHOICE_ROBUSTNESS_PROBES,
}


def available_probe_presets() -> list[str]:
    return sorted(PROBE_PRESETS)


def resolve_probe_preset(name: str) -> list[dict[str, Any]]:
    try:
        probes = PROBE_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown probe preset: {name}") from exc
    return deepcopy(probes)


def probe_metadata(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in probe.items()
        if key not in {"id", "prompt"}
    }
