"""Token and cache coordinates for a single post-image source-prompt edit."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .padding_cache import input_fingerprint

STAGES = ("ack_before", "yes", "ack_after")
ROLES = (
    "pre_image",
    "image",
    "post_image_before_edit",
    "edit",
    "prompt_tail",
    "suffix",
)


def source_prompt(original, config, stage):
    old = config["base_tail"]
    if stage not in STAGES or original.count(old) != 1 or not original.endswith(old):
        raise ValueError("unexpected source prompt or stage")
    if stage == "yes":
        return original.removesuffix(old) + config["edited_tail"]
    return original


def check_inputs(audit, original, edit, stage):
    actual = audit["effective"]
    if audit["override_image_sizes"] is not None or audit["native"] != actual:
        raise ValueError("prepared inputs overridden")
    expected = dict(original)
    old = original["input_ids"]
    ids = np.array(old["values"], dtype=old["dtype"])
    index = edit["processor_index"]
    if (
        stage not in STAGES
        or ids.shape != (1, 140)
        or ids[0, index] != edit["old_id"]
        or edit["old_id"] == edit["new_id"]
    ):
        raise ValueError("historical prompt token contract differs")
    if stage == "yes":
        ids[0, index] = edit["new_id"]
    expected.update(input_fingerprint({"input_ids": ids}))
    if actual != expected:
        raise ValueError(
            "actual prepared inputs do not match the registered token edit"
        )
    return ids[0].tolist()


def token_roles(prepared, cached, event, edit):
    layout, gen = event["cache_token_layout"], event["generation"]
    kv = layout["cache_position_layout"]
    if cached[: len(prepared)] != prepared:
        raise ValueError("tracked cache IDs do not start with actual prepared IDs")
    suffix = cached[len(prepared) :]
    steps = [s["token_id"] for s in gen["steps"]]
    if (
        not 1 <= len(suffix) <= 2
        or gen["generation_tokens"] != len(suffix)
        or steps not in (suffix, suffix + suffix[-1:])
        or gen["prompt_tokens"] != len(prepared)
        or gen["total_tokens"] != len(cached)
    ):
        raise ValueError(
            "realized suffix and generation notifications are inconsistent"
        )
    markers = [i for i, t in enumerate(cached) if t == -200]
    if (
        markers != [42]
        or layout["marker_token_ids"] != {"image_token_index": -200}
        or layout["marker_positions"] != {"image_token_index": markers}
        or layout["token_count"] != len(cached)
        or not kv["available"]
        or not kv["image_partition_available"]
        or kv["strategy"] != "llava_qwen2_single_image_run_replacement"
        or kv["image_position_runs"] != [{"start": 42, "end": 297, "length": 256}]
        or kv["processor_token_count"] != len(cached)
        or kv["cache_sequence_lengths"] != [len(cached) + 255]
        or kv["cache_sequence_length"] != len(cached) + 255
    ):
        raise ValueError("processor/cache image mapping or length differs")
    edit_at = edit["processor_index"]
    if not markers[0] < edit_at < len(prepared) - 1:
        raise ValueError("edit must be in the post-image prompt")
    edges = [
        0,
        markers[0],
        markers[0] + 256,
        edit_at + 255,
        edit_at + 256,
        len(prepared) + 255,
        len(cached) + 255,
    ]
    return {
        "prepared_ids": prepared,
        "cache_state_ids": cached,
        "suffix_ids": suffix,
        "stream_notification_ids": steps,
        "assistant_text": event["assistant_text"],
        "spans": {role: [a, b] for role, a, b in zip(ROLES, edges, edges[1:])},
    }


def region_rows(a, b, left, right):
    for arr, roles in ((a, left), (b, right)):
        if (
            arr.dtype != np.float32
            or arr.shape != (1, 2, roles["spans"]["suffix"][1], 64)
            or not np.isfinite(arr).all()
        ):
            raise ValueError("invalid target for the realized layout")
    for name in ROLES:
        al, ar = left["spans"][name]
        bl, br = right["spans"][name]
        row = {"role": name, "before_span": [al, ar], "edited_span": [bl, br]}
        if name == "suffix" and left["suffix_ids"] != right["suffix_ids"]:
            yield {
                **row,
                "available": False,
                "reason": "realized_suffix_ids_differ",
                "changed_positions": None,
                "changed_scalars": None,
                "rms": None,
                "max_abs": None,
            }
            continue
        if (al, ar) != (bl, br) or al >= ar:
            raise ValueError("unmatched or empty registered region")
        x, y = a[:, :, al:ar, :], b[:, :, bl:br, :]
        different = x.view(np.uint32) != y.view(np.uint32)
        delta = y.astype(np.float64) - x
        yield {
            **row,
            "available": True,
            "reason": None,
            "changed_positions": (
                al + np.flatnonzero(different.any(axis=(0, 1, 3)))
            ).tolist(),
            "changed_scalars": int(different.sum()),
            "rms": float(np.sqrt(np.mean(delta**2))),
            "max_abs": float(np.abs(delta).max()),
        }


def aggregate(regions, pairs, targets):
    stats = {}
    for target in targets:
        per_role = {}
        for role in ROLES:
            group = [r for r in regions if r["target"] == target and r["role"] == role]
            valid = [r for r in group if r["available"]]
            per_role[role] = {
                "planned": len(group),
                "available": len(valid),
                "unavailable": len(group) - len(valid),
                "bitwise_exact": sum(not r["changed_positions"] for r in valid),
                "median_cell_rms": float(np.median([r["rms"] for r in valid]))
                if valid
                else None,
                "maximum_abs": max((r["max_abs"] for r in valid), default=None),
            }
        firsts = Counter(
            str(p["first_changed_comparable_position"][target]) for p in pairs
        )
        stats[target] = {
            "regions": per_role,
            "first_changed_position_counts": dict(firsts),
        }
    return stats
