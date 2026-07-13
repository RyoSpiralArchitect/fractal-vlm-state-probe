from __future__ import annotations

import hashlib
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .cache_tensor_factorial import (
    balanced_factorial_contrast_vectors,
    factorial_effect_vectors,
)
from .probe_readout import CELL_KEYS
from .stimulus import write_json

FORCED_CHOICE_LABELS = {
    "forced_family_choice": ("A", "B", "C"),
    "forced_frequency_choice": ("L", "H", "C"),
}

FORCED_CHOICE_SEMANTICS = {
    "forced_family_choice": {
        "A": "mandelbrot",
        "B": "julia",
        "C": "unclear",
    },
    "forced_frequency_choice": {
        "L": "low_frequency",
        "H": "high_frequency",
        "C": "unclear",
    },
}

PROBE_METADATA_KEYS = (
    "probe_family",
    "prompt_variant",
    "candidate_labels",
    "candidate_order",
    "candidate_semantics",
)


def full_vocab_sidecar_path(
    run_output_path: Path,
    *,
    phase: str,
    probe_id: str,
) -> Path:
    sidecar_dir = run_output_path.parent / f"{run_output_path.stem}_full_vocab"
    return sidecar_dir / f"{phase}__{probe_id}__step_000.npz"


def write_full_vocab_logprob_sidecar(
    logprobs: Any,
    *,
    path: Path,
    relative_to: Path,
) -> dict[str, Any]:
    values = _to_float32_numpy(logprobs).reshape(-1)
    if values.size == 0:
        raise ValueError("full-vocabulary logprob array is empty")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("full-vocabulary logprobs contain NaN or positive infinity")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logprobs=values)
    digest = _sha256(path)
    logsumexp = _logsumexp(values.astype(np.float64))
    return {
        "path": Path(os.path.relpath(path, relative_to)).as_posix(),
        "sha256": digest,
        "format": "numpy_npz",
        "array_key": "logprobs",
        "value_kind": "natural_log_probability",
        "dtype": str(values.dtype),
        "shape": list(values.shape),
        "vocab_size": int(values.size),
        "logsumexp": logsumexp,
        "probability_sum": float(math.exp(logsumexp)),
        "negative_infinity_count": int(np.isneginf(values).sum()),
    }


def _to_float32_numpy(values: Any) -> np.ndarray:
    try:
        return np.asarray(values, dtype=np.float32)
    except (RuntimeError, TypeError, ValueError):
        pass
    if hasattr(values, "astype"):
        try:
            import mlx.core as mx

            promoted = values.astype(mx.float32)
            mx.eval(promoted)
            return np.asarray(promoted, dtype=np.float32)
        except Exception:
            pass
    if hasattr(values, "tolist"):
        return np.asarray(values.tolist(), dtype=np.float32)
    raise TypeError(f"cannot convert {type(values).__name__} logprobs to float32")


def load_full_vocab_logprobs(
    run_path: Path,
    metadata: dict[str, Any],
    *,
    verify_hash: bool = True,
) -> np.ndarray:
    sidecar_path = (run_path.parent / str(metadata["path"])).resolve()
    if verify_hash and _sha256(sidecar_path) != metadata.get("sha256"):
        raise ValueError(f"full-vocabulary sidecar hash mismatch: {sidecar_path}")
    with np.load(sidecar_path, allow_pickle=False) as archive:
        values = np.asarray(
            archive[str(metadata.get("array_key", "logprobs"))], dtype=np.float64
        )
    values = values.reshape(-1)
    expected_vocab = metadata.get("vocab_size")
    if expected_vocab is not None and values.size != int(expected_vocab):
        raise ValueError(
            f"full-vocabulary sidecar size mismatch: {values.size} != {expected_vocab}"
        )
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError(f"invalid full-vocabulary logprobs: {sidecar_path}")
    return values


def analyze_full_vocab_readout_contrast(
    *,
    runs: dict[str, dict[str, Any]],
    run_paths: dict[str, Path],
    max_token_effects: int = 20,
) -> dict[str, Any]:
    if set(runs) != set(CELL_KEYS) or set(run_paths) != set(CELL_KEYS):
        raise ValueError(
            f"runs and run_paths must contain exactly: {', '.join(CELL_KEYS)}"
        )

    records = []
    for phase, probe_id in _phase_probe_keys(runs.values()):
        record = _analyze_phase_probe(
            runs=runs,
            run_paths=run_paths,
            phase=phase,
            probe_id=probe_id,
            max_token_effects=max_token_effects,
        )
        records.append(record)

    available = [record for record in records if record["available"]]
    return {
        "schema_version": 1,
        "analysis_kind": "full_vocab_first_token_probe_readout_contrast",
        "cells": {key: _run_condition(runs[key]) for key in CELL_KEYS},
        "cell_semantics": {
            "mm": "Mandelbrot spatial rank x Mandelbrot palette",
            "jj": "Julia spatial rank x Julia palette",
            "mj": "Mandelbrot spatial rank x Julia palette",
            "jm": "Julia spatial rank x Mandelbrot palette",
        },
        "contrast_formulas": {
            "spatial_main_effect": "((jm - mm) + (jj - mj)) / 2",
            "palette_main_effect": "((mj - mm) + (jj - jm)) / 2",
            "interaction_effect": "jj - jm - mj + mm",
        },
        "balanced_contrast_formulas": {
            "spatial_contrast": "jj + jm - mm - mj",
            "palette_contrast": "jj + mj - mm - jm",
            "interaction_contrast": "jj + mm - jm - mj",
        },
        "record_count": len(records),
        "available_record_count": len(available),
        "records": records,
        "interpretation_notes": [
            "Each record uses the complete saved first-step vocabulary distribution.",
            "Pairwise distances within one four-cell factorial are descriptive, not independent samples.",
            "Probability-space interaction is bounded and primary; tail-sensitive logprob interaction is diagnostic.",
            "Balanced probability contrasts use the same +/-1 cell-coefficient scale before their L2 energy shares are compared.",
            "An interaction energy share of 1/3 is the exchangeable isotropic reference across the three balanced factorial axes, not a fitted null distribution.",
            "Logprob interaction excludes tokens that are non-finite in any cell and reports the excluded count.",
            "A sidecar hash and vocabulary-size check is performed before analysis.",
        ],
    }


def write_full_vocab_readout_json(analysis: dict[str, Any], path: Path) -> None:
    write_json(path, analysis)


def write_full_vocab_readout_markdown(analysis: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_full_vocab_readout_markdown(analysis), encoding="utf-8")


def format_full_vocab_readout_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Full-Vocabulary First-Token Readout Contrast",
        "",
        "## Records",
        "",
        "| Phase | Probe | Variant | Available | Vocab | Max pair JS | Interaction L1 | Interaction max | Token | Weighted log interaction RMS |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for record in analysis["records"]:
        if not record["available"]:
            lines.append(
                f"| `{record['phase']}` | `{record['probe_id']}` | "
                f"`{record.get('prompt_variant') or 'n/a'}` | `false` | n/a | "
                "n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        interaction = record["probability_contrasts"]["interaction"]
        pairwise = record["pairwise_distances"]
        max_js = max((item["jensen_shannon"] for item in pairwise), default=0.0)
        token = interaction["argmax_token"] or str(interaction["argmax_token_id"])
        lines.append(
            f"| `{record['phase']}` | `{record['probe_id']}` | "
            f"`{record.get('prompt_variant') or 'n/a'}` | `true` | "
            f"{record['vocab_size']} | {max_js:.8g} | {interaction['l1_norm']:.8g} | "
            f"{interaction['max_abs']:.8g} | `{token}` | "
            f"{_format_metric(record['logprob_interaction']['reference_weighted_rms'])} |"
        )

    lines.extend(
        [
            "",
            "## Balanced Contrast Calibration",
            "",
            "All three probability contrasts use the same `+1/+1/-1/-1` coefficient scale.",
            "The exchangeable isotropic reference share is `0.33333333`.",
            "",
            "| Phase | Probe | Spatial energy share | Palette energy share | Interaction energy share | Interaction minus 1/3 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in analysis["records"]:
        if not record["available"]:
            lines.append(
                f"| `{record['phase']}` | `{record['probe_id']}` | n/a | n/a | n/a | n/a |"
            )
            continue
        calibration = record["balanced_probability_contrast_energy"]
        shares = calibration["energy_shares"]
        lines.append(
            f"| `{record['phase']}` | `{record['probe_id']}` | "
            f"{_format_metric(shares['spatial_contrast'])} | "
            f"{_format_metric(shares['palette_contrast'])} | "
            f"{_format_metric(shares['interaction_contrast'])} | "
            f"{_format_metric(calibration['interaction_share_minus_exchangeable_reference'])} |"
        )

    lines.extend(["", "## Pairwise Distribution Distances", ""])
    for record in analysis["records"]:
        if not record["available"]:
            continue
        lines.extend(
            [
                f"### `{record['phase']}` / `{record['probe_id']}`",
                "",
                "| Pair | JS | TV | Hellinger | Symmetric KL |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in record["pairwise_distances"]:
            lines.append(
                f"| `{item['left']}` / `{item['right']}` | "
                f"{item['jensen_shannon']:.8g} | {item['total_variation']:.8g} | "
                f"{item['hellinger']:.8g} | {_format_metric(item['symmetric_kl'])} |"
            )
        candidates = record.get("forced_choice_candidates")
        if candidates and candidates.get("available"):
            lines.extend(
                [
                    "",
                    "Forced-choice candidate probabilities:",
                    "",
                    "| Cell | Candidate mass | Label conditionals | Semantic conditionals | Generated token | Generated semantic |",
                    "| --- | ---: | --- | --- | --- | --- |",
                ]
            )
            for cell in CELL_KEYS:
                cell_record = candidates["cells"][cell]
                conditional = ", ".join(
                    f"{label}={value:.6f}"
                    for label, value in cell_record["conditional_probabilities"].items()
                )
                semantic_conditional = ", ".join(
                    f"{label}={value:.6f}"
                    for label, value in cell_record[
                        "semantic_conditional_probabilities"
                    ].items()
                )
                lines.append(
                    f"| `{cell}` | {cell_record['candidate_probability_mass']:.8g} | "
                    f"{conditional} | {semantic_conditional} | "
                    f"`{record['generated_tokens'][cell]}` | "
                    f"`{record['generated_semantics'][cell]}` |"
                )
        lines.extend(["", "Top probability-space interaction tokens:", ""])
        for item in record["top_probability_interaction_tokens"][:10]:
            token = item["token"] or str(item["token_id"])
            lines.append(
                f"- `{token}`: interaction `{item['interaction_effect']:.8g}`, "
                f"mean probability `{item['mean_probability']:.8g}`"
            )
        lines.append("")

    lines.extend(["## Interpretation Notes", ""])
    for note in analysis.get("interpretation_notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _analyze_phase_probe(
    *,
    runs: dict[str, dict[str, Any]],
    run_paths: dict[str, Path],
    phase: str,
    probe_id: str,
    max_token_effects: int,
) -> dict[str, Any]:
    probe_records = {
        cell: _probe_record(run, phase=phase, probe_id=probe_id)
        for cell, run in runs.items()
    }
    metadata = _shared_probe_metadata(
        probe_records,
        phase=phase,
        probe_id=probe_id,
    )
    steps = {
        cell: _first_generation_step(record) for cell, record in probe_records.items()
    }
    missing = [
        cell
        for cell, step in steps.items()
        if not step or not step.get("full_vocab_sidecar")
    ]
    base = {
        "phase": phase,
        "probe_id": probe_id,
        **metadata,
        "available": not missing,
        "missing_cells": missing,
    }
    if missing:
        return base

    logprobs = {
        cell: _normalize_logprobs(
            load_full_vocab_logprobs(run_paths[cell], steps[cell]["full_vocab_sidecar"])
        )
        for cell in CELL_KEYS
    }
    vocab_sizes = {values.size for values in logprobs.values()}
    if len(vocab_sizes) != 1:
        raise ValueError(
            f"vocabulary-size mismatch for {phase}/{probe_id}: {vocab_sizes}"
        )
    probabilities = {cell: np.exp(values) for cell, values in logprobs.items()}
    token_labels = _token_label_map(steps.values())

    pairwise = []
    for left, right in combinations(CELL_KEYS, 2):
        pairwise.append(
            {
                "left": left,
                "right": right,
                **_distribution_distances(logprobs[left], logprobs[right]),
            }
        )

    probability_effects = factorial_effect_vectors(probabilities)
    balanced_probability_contrasts = balanced_factorial_contrast_vectors(probabilities)
    probability_interaction = probability_effects["interaction"]
    reference = sum(probabilities.values()) / len(probabilities)
    logprob_interaction, logprob_interaction_summary = _logprob_interaction(
        logprobs,
        reference,
    )
    max_probability_interaction = float(np.max(np.abs(probability_interaction)))
    top_indices = (
        np.argsort(np.abs(probability_interaction))[-max_token_effects:][::-1]
        if max_probability_interaction > 0.0
        else np.asarray([], dtype=int)
    )
    top_tokens = [
        {
            "token_id": int(token_id),
            "token": token_labels.get(int(token_id), ""),
            "mm": float(probabilities["mm"][token_id]),
            "jj": float(probabilities["jj"][token_id]),
            "mj": float(probabilities["mj"][token_id]),
            "jm": float(probabilities["jm"][token_id]),
            "mean_probability": float(reference[token_id]),
            "interaction_effect": float(probability_interaction[token_id]),
            "abs_interaction_effect": float(abs(probability_interaction[token_id])),
            "logprob_interaction_effect": _finite_float_or_none(
                logprob_interaction[token_id]
            ),
        }
        for token_id in top_indices
    ]

    return {
        **base,
        "vocab_size": vocab_sizes.pop(),
        "sidecars": {cell: steps[cell]["full_vocab_sidecar"] for cell in CELL_KEYS},
        "all_sidecars_byte_identical": len(
            {steps[cell]["full_vocab_sidecar"]["sha256"] for cell in CELL_KEYS}
        )
        == 1,
        "generated_tokens": {
            cell: str(steps[cell].get("token", "")) for cell in CELL_KEYS
        },
        "generated_semantics": {
            cell: _generated_semantic(
                steps[cell],
                metadata.get("candidate_semantics"),
            )
            for cell in CELL_KEYS
        },
        "pairwise_distances": pairwise,
        "probability_contrasts": {
            name: _vector_summary(values, token_labels)
            for name, values in probability_effects.items()
        },
        "balanced_probability_contrasts": {
            name: _vector_summary(values, token_labels)
            for name, values in balanced_probability_contrasts.items()
        },
        "balanced_probability_contrast_energy": _balanced_contrast_energy(
            balanced_probability_contrasts
        ),
        "logprob_interaction": logprob_interaction_summary,
        "forced_choice_candidates": _forced_choice_candidate_summary(
            probe_id=probe_id,
            probabilities=probabilities,
            token_labels=token_labels,
            candidate_labels=metadata.get("candidate_labels"),
            candidate_semantics=metadata.get("candidate_semantics"),
        ),
        "top_probability_interaction_tokens": top_tokens,
    }


def _distribution_distances(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_p = np.exp(left)
    right_p = np.exp(right)
    midpoint_log = np.logaddexp(left, right) - math.log(2.0)
    kl_left_right = _relative_entropy(left_p, left, right)
    kl_right_left = _relative_entropy(right_p, right, left)
    js = 0.5 * (
        _relative_entropy(left_p, left, midpoint_log)
        + _relative_entropy(right_p, right, midpoint_log)
    )
    symmetric_kl = (kl_left_right + kl_right_left) / 2.0
    return {
        "kl_left_right": _nonnegative_finite_or_none(kl_left_right),
        "kl_left_right_is_infinite": math.isinf(kl_left_right),
        "kl_right_left": _nonnegative_finite_or_none(kl_right_left),
        "kl_right_left_is_infinite": math.isinf(kl_right_left),
        "symmetric_kl": _nonnegative_finite_or_none(symmetric_kl),
        "symmetric_kl_is_infinite": math.isinf(symmetric_kl),
        "jensen_shannon": max(0.0, js),
        "total_variation": float(0.5 * np.sum(np.abs(left_p - right_p))),
        "hellinger": float(
            np.sqrt(0.5 * np.sum(np.square(np.sqrt(left_p) - np.sqrt(right_p))))
        ),
    }


def _relative_entropy(
    probabilities: np.ndarray,
    log_probabilities: np.ndarray,
    reference_log_probabilities: np.ndarray,
) -> float:
    support = probabilities > 0.0
    if not np.any(support):
        return 0.0
    deltas = log_probabilities[support] - reference_log_probabilities[support]
    if np.isposinf(deltas).any():
        return math.inf
    return float(np.sum(probabilities[support] * deltas))


def _logprob_interaction(
    logprobs: dict[str, np.ndarray],
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    stacked = np.stack([logprobs[cell] for cell in CELL_KEYS])
    finite_mask = np.all(np.isfinite(stacked), axis=0)
    values = np.full(reference.shape, np.nan, dtype=np.float64)
    values[finite_mask] = (
        logprobs["jj"][finite_mask]
        - logprobs["jm"][finite_mask]
        - logprobs["mj"][finite_mask]
        + logprobs["mm"][finite_mask]
    )
    finite_values = values[finite_mask]
    reference_mass = float(np.sum(reference[finite_mask]))
    if finite_values.size == 0:
        summary = {
            "rms": None,
            "reference_weighted_rms": None,
            "max_abs": None,
        }
    else:
        weighted_square_sum = float(
            np.sum(reference[finite_mask] * np.square(finite_values))
        )
        summary = {
            "rms": float(np.sqrt(np.mean(np.square(finite_values)))),
            "reference_weighted_rms": (
                float(np.sqrt(weighted_square_sum / reference_mass))
                if reference_mass > 0.0
                else None
            ),
            "max_abs": float(np.max(np.abs(finite_values))),
        }
    return values, {
        **summary,
        "finite_token_count": int(np.count_nonzero(finite_mask)),
        "excluded_nonfinite_token_count": int(
            finite_mask.size - np.count_nonzero(finite_mask)
        ),
        "finite_reference_probability_mass": reference_mass,
    }


def _nonnegative_finite_or_none(value: float) -> float | None:
    return max(0.0, value) if math.isfinite(value) else None


def _finite_float_or_none(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _format_metric(value: Any) -> str:
    return f"{float(value):.8g}" if isinstance(value, (int, float)) else "n/a"


def _vector_summary(values: np.ndarray, token_labels: dict[int, str]) -> dict[str, Any]:
    token_id = int(np.argmax(np.abs(values)))
    return {
        "l1_norm": float(np.sum(np.abs(values))),
        "l2_norm": float(np.linalg.norm(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max_abs": float(abs(values[token_id])),
        "argmax_token_id": token_id,
        "argmax_token": token_labels.get(token_id, ""),
        "argmax_signed_effect": float(values[token_id]),
        "sum": float(np.sum(values)),
    }


def _balanced_contrast_energy(
    contrasts: dict[str, np.ndarray],
) -> dict[str, Any]:
    energies = {
        name: float(np.dot(values, values)) for name, values in contrasts.items()
    }
    total_energy = sum(energies.values())
    shares = {
        name: energy / total_energy if total_energy > 0.0 else None
        for name, energy in energies.items()
    }
    interaction_share = shares["interaction_contrast"]
    reference = 1.0 / 3.0
    return {
        "coefficient_scale": "+1/+1/-1/-1 for every contrast",
        "l2_squared": energies,
        "total_l2_squared": total_energy,
        "energy_shares": shares,
        "exchangeable_isotropic_reference_share": reference,
        "interaction_share_minus_exchangeable_reference": (
            None if interaction_share is None else interaction_share - reference
        ),
    }


def _normalize_logprobs(values: np.ndarray) -> np.ndarray:
    return values - _logsumexp(values)


def _logsumexp(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("logprob array has no finite values")
    maximum = float(np.max(finite))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def _phase_probe_keys(runs: Any) -> list[tuple[str, str]]:
    keys = set()
    for run in runs:
        for phase, records in (run.get("probes") or {}).items():
            for record in records or []:
                probe_id = record.get("probe_id")
                if probe_id:
                    keys.add((phase, probe_id))
    phase_order = {"before": 0, "mid": 1, "after": 2}
    return sorted(keys, key=lambda item: (phase_order.get(item[0], 99), item[1]))


def _probe_record(
    run: dict[str, Any],
    *,
    phase: str,
    probe_id: str,
) -> dict[str, Any] | None:
    for record in (run.get("probes") or {}).get(phase, []) or []:
        if record.get("probe_id") == probe_id:
            return record
    return None


def _first_generation_step(record: dict[str, Any] | None) -> dict[str, Any] | None:
    steps = ((record or {}).get("generation") or {}).get("steps") or []
    return steps[0] if steps else None


def _shared_probe_metadata(
    records: dict[str, dict[str, Any] | None],
    *,
    phase: str,
    probe_id: str,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in PROBE_METADATA_KEYS:
        values = [
            record.get(key)
            for record in records.values()
            if record is not None and key in record
        ]
        if not values:
            continue
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"probe metadata mismatch for {phase}/{probe_id}: {key}")
        metadata[key] = values[0]

    for key, value in _default_candidate_metadata(probe_id).items():
        metadata.setdefault(key, value)
    return metadata


def _token_label_map(steps: Any) -> dict[int, str]:
    labels: dict[int, str] = {}
    for step in steps:
        if not step:
            continue
        for item in step.get("top_logprobs") or []:
            token_id = item.get("token_id")
            token = item.get("token")
            if token_id is not None and token:
                labels[int(token_id)] = str(token)
    return labels


def _forced_choice_candidate_summary(
    *,
    probe_id: str,
    probabilities: dict[str, np.ndarray],
    token_labels: dict[int, str],
    candidate_labels: Any = None,
    candidate_semantics: Any = None,
) -> dict[str, Any] | None:
    labels = tuple(candidate_labels or FORCED_CHOICE_LABELS.get(probe_id) or ())
    if not labels:
        return None
    semantics = dict(candidate_semantics or {})
    token_ids = {
        label: sorted(
            token_id
            for token_id, token in token_labels.items()
            if token.strip() == label
        )
        for label in labels
    }
    missing = [label for label, ids in token_ids.items() if not ids]
    if missing:
        return {
            "available": False,
            "labels": list(labels),
            "candidate_semantics": semantics,
            "token_ids": token_ids,
            "missing_labels": missing,
            "cells": {},
        }

    cells = {}
    for cell in CELL_KEYS:
        raw = {
            label: float(np.sum(probabilities[cell][ids]))
            for label, ids in token_ids.items()
        }
        mass = sum(raw.values())
        conditional = {
            label: value / mass if mass else 0.0 for label, value in raw.items()
        }
        cells[cell] = {
            "candidate_probability_mass": mass,
            "probabilities": raw,
            "conditional_probabilities": conditional,
            "semantic_conditional_probabilities": {
                semantics.get(label, label): value
                for label, value in conditional.items()
            },
        }
    return {
        "available": True,
        "labels": list(labels),
        "candidate_semantics": semantics,
        "token_ids": token_ids,
        "missing_labels": [],
        "cells": cells,
    }


def _default_candidate_metadata(probe_id: str) -> dict[str, Any]:
    if probe_id.startswith("forced_family_choice"):
        return {
            "probe_family": "family",
            "candidate_labels": ["A", "B", "C"],
            "candidate_semantics": dict(
                FORCED_CHOICE_SEMANTICS["forced_family_choice"]
            ),
        }
    if probe_id.startswith("forced_frequency_choice"):
        return {
            "probe_family": "frequency",
            "candidate_labels": ["L", "H", "C"],
            "candidate_semantics": dict(
                FORCED_CHOICE_SEMANTICS["forced_frequency_choice"]
            ),
        }
    return {}


def _generated_semantic(
    step: dict[str, Any],
    candidate_semantics: Any,
) -> str | None:
    semantics = dict(candidate_semantics or {})
    return semantics.get(str(step.get("token", "")).strip())


def _run_condition(run: dict[str, Any]) -> str | None:
    return ((run.get("stimulus") or {}).get("condition") or {}).get("condition_id")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
