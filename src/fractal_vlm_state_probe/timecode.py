from __future__ import annotations


def format_timecode(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    minutes, rem_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"

