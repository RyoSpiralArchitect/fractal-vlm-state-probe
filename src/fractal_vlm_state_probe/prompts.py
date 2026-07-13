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
    },
    {
        "id": "forced_frequency_choice",
        "prompt": (
            "Forced choice. Based only on the current context, answer exactly one "
            "letter: L for coarse low-frequency layout, H for high-frequency edge "
            "texture, or C for unclear/mixed. Output only L, H, or C."
        ),
    },
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
}


def available_probe_presets() -> list[str]:
    return sorted(PROBE_PRESETS)


def resolve_probe_preset(name: str) -> list[dict[str, str]]:
    try:
        probes = PROBE_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown probe preset: {name}") from exc
    return [dict(probe) for probe in probes]
