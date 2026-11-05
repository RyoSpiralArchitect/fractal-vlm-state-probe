"""Verify all full-vocabulary equality/normalization receipts using only NumPy and files."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_bytes())


def file_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def word_sha(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def rows(p, sha):
    if file_sha(p) != sha:
        raise ValueError(f"ledger changed: {p}")
    return [json.loads(line) for line in p.read_bytes().splitlines()]


def vector(root, name, kind, sha):
    with np.load(root / name, allow_pickle=False) as f:
        if f.files != [kind]:
            raise ValueError("unexpected vocabulary array key")
        x = f[kind]
    if (
        x.dtype != np.float32
        or x.shape != (151936,)
        or word_sha(x) != sha
        or np.any(x.view(np.uint32) & 65535)
    ):
        raise ValueError("whole native vocabulary shape/dtype/hash differs")
    return x


def audit(capture, verified, destination):
    s = read(verified / "summary.json")
    seal = read(capture / "capture.json")
    if (
        (capture / "failure.json").exists()
        or s["status"] != "VERIFIED"
        or seal["status"] != "COMPLETE"
        or s["capture_sha256"] != file_sha(capture / "capture.json")
    ):
        raise ValueError("complete unchanged readout calibration required")
    arms = rows(verified / "arms.jsonl", s["ledgers"]["arms.jsonl"]["sha256"])
    comparisons = rows(
        verified / "comparisons.jsonl", s["ledgers"]["comparisons.jsonl"]["sha256"]
    )
    if len(arms) != 288 or len(comparisons) != 96:
        raise ValueError("incomplete readout denominator")
    root = Path(__file__).resolve().parents[1]
    hist = root / "examples/research_notes/0072_prompt_only_source_calibration"
    original = rows(
        hist / "contexts.jsonl",
        read(hist / "summary.json")["ledgers"]["contexts.jsonl"]["sha256"],
    )
    exports = []
    counts = {
        "score_coordinate_pairs": 0,
        "raw_logit_coordinate_pairs": 0,
        "prefix_cache_tensor_pairs": 0,
    }
    for i, anchor in enumerate(original):
        group = arms[3 * i : 3 * i + 3]
        roles = ["native_stream_only", "native_decoder_observed", "prefix_only"]
        if [r["arm"] for r in group] != roles:
            raise ValueError("wrong readout arm order")
        vectors = {}
        for arm, r in zip(roles, group, strict=True):
            if any(r[k] != anchor[k] for k in ("ordinal", "pair_id", "cell", "stage")):
                raise ValueError("anchor identity changed")
            d = capture / "contexts" / f"{i + 1:03d}" / arm
            v = r["vocabulary"]
            if file_sha(d / "logprobs.npz") != v["sidecar"]["sha256"]:
                raise ValueError("vocabulary sidecar file changed")
            scores = vector(d, "logprobs.npz", "logprobs", v["scores_array_sha256"])
            if (
                np.isnan(scores).any()
                or np.isposinf(scores).any()
                or not np.isfinite(scores).any()
            ):
                raise ValueError("invalid native support")
            vectors[arm] = [scores]
            if arm != "native_stream_only":
                logits = vector(d, "logits.npz", "logits", v["logits_array_sha256"])
                if (
                    not np.isfinite(logits).all()
                    or file_sha(d / "logits.npz") != r["logits_file_sha256"]
                ):
                    raise ValueError("raw logits changed")
                vectors[arm].append(logits)
            if arm == "prefix_only":
                actual = read(d / "prefix.json")
                if (
                    actual["actual_input_ids"] != anchor["prepared_ids"]
                    or actual["suffix_ids"] != []
                    or actual["decoder_calls"] != 1
                    or actual["effective_length"] != 395
                ):
                    raise ValueError(
                        "direct readout consumed a suffix or different prefix"
                    )
                (direct_trace,) = read(d / "checkpoints.json")
                observed = (
                    capture / "contexts" / f"{i + 1:03d}" / "native_decoder_observed"
                )
                native_trace = read(observed / "checkpoints.json")[0]
                if (
                    direct_trace != native_trace
                    or direct_trace["input_ids"] != anchor["prepared_ids"]
                ):
                    raise ValueError("native/prefix actual decoder receipts differ")
                direct = np.load(d / "kv_0.npy", allow_pickle=False)
                native = np.load(observed / "kv_0.npy", allow_pickle=False)
                if (
                    direct.shape != (24, 2, 1, 2, 395, 64)
                    or direct.dtype != np.dtype("<u2")
                    or direct.tobytes() != native.tobytes()
                    or word_sha(direct) != direct_trace["sha256"]
                ):
                    raise ValueError("direct native prefix-word mismatch")
                for layer in range(24):
                    for k, kind in enumerate(("keys", "values")):
                        if (
                            word_sha(direct[layer, k])
                            != direct_trace["tensor_sha256"][f"{layer}:{kind}"]
                        ):
                            raise ValueError("one prefix layer/KV digest changed")
                counts["prefix_cache_tensor_pairs"] += 48
                exports.append(
                    {
                        "ordinal": anchor["ordinal"],
                        "pair_id": anchor["pair_id"],
                        "cell": anchor["cell"],
                        "stage": anchor["stage"],
                        **direct_trace,
                    }
                )
        a, b, c = (vectors[k][0] for k in roles)
        if a.tobytes() != b.tobytes() or a.tobytes() != c.tobytes():
            raise ValueError("whole first-step native-score words differ")
        if vectors[roles[1]][1].tobytes() != vectors[roles[2]][1].tobytes():
            raise ValueError("whole first-step raw-logit words differ")
        counts["score_coordinate_pairs"] += 2 * a.size
        counts["raw_logit_coordinate_pairs"] += a.size
        r = comparisons[i]
        if r["score_array_sha256"] != word_sha(a) or r[
            "logits_array_sha256"
        ] != word_sha(vectors[roles[1]][1]):
            raise ValueError("published whole-vocabulary receipt differs")
        native = a.astype(np.float64)
        raw = vectors[roles[1]][1].astype(np.float64)
        m, n = float(native.max()), float(raw.max())
        weights, raww = np.exp(native - m), np.exp(raw - n)
        z = m + math.log(math.fsum(weights))
        p, q = weights / math.fsum(weights), raww / math.fsum(raww)
        values = {
            "native_logsumexp64": z,
            "native_score_probability_sum": math.exp(z),
            "renormalized_probability_sum": math.fsum(p),
            "native_vs_raw_logits_softmax_tv": 0.5 * math.fsum(abs(p - q)),
        }
        for name, value in values.items():
            if not math.isclose(value, r[name], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"independent numerical scalar check differs: {name}")
        if (
            r["natural_first_token_id"] != anchor["suffix_ids"][0]
            or r["argmax_ids"] != np.flatnonzero(a == a.max()).tolist()
            or r["natural_first_token_id"] not in r["argmax_ids"]
        ):
            raise ValueError("native token/full vocabulary alignment differs")
    with (destination / "prefix_boundaries.jsonl").open("x") as f:
        for row in exports:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    result = {
        "schema_version": 1,
        "analysis_kind": "first_step_readout_independent_file_audit",
        "status": "PASS",
        "verified_summary_sha256": file_sha(verified / "summary.json"),
        "rows": 96,
        **counts,
        "prefix_boundaries_sha256": file_sha(destination / "prefix_boundaries.jsonl"),
        "actual_prefix_decoder_calls": 96,
        "prefix_suffix_calls": 0,
        "model_calls": 0,
        "processor_calls": 0,
        "interprompt_effects_computed": 0,
    }
    with (destination / "independent_file_check.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "verified", "destination"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    result = audit(args.capture, args.verified, args.destination)
    print(result["status"], args.destination)
