"""Canonical capture storage, safe paths and append-only operation receipts."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )


def ledger(root, name):
    path = root / name
    return (
        [json.loads(line) for line in path.read_bytes().splitlines()]
        if path.exists()
        else []
    )


def append(root, name, row):
    with (root / name).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        handle.flush()


def inside(root, relative):
    root = root.resolve()
    p = root / relative
    if (
        Path(relative).is_absolute()
        or not p.resolve().is_relative_to(root.resolve())
        or any(x.is_symlink() for x in (p, *p.parents) if x.is_relative_to(root))
    ):
        raise ValueError("unsafe capture artifact path")
    return p


def inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks are not capture artifacts")
        if path.is_file() and str(path.relative_to(root)) not in (
            "capture.json",
            "failure.json",
        ):
            result[str(path.relative_to(root))] = sha(path)
    return result


def save_array(root, relative, values, *, reserve=0):
    p = inside(root, relative)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    if shutil.disk_usage(root).free < reserve + values.nbytes + 4096:
        raise ValueError("saving array would breach disk reserve")
    np.save(p, values, allow_pickle=False)
    return {
        "path": relative,
        "sha256": sha(p),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "word_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


def load_array(root, receipt):
    p = inside(root, receipt["path"])
    if sha(p) != receipt["sha256"]:
        raise ValueError("array file hash differs")
    a = np.load(p, allow_pickle=False)
    if (
        not isinstance(a, np.ndarray)
        or list(a.shape) != receipt["shape"]
        or str(a.dtype) != receipt["dtype"]
        or hashlib.sha256(a.tobytes(order="C")).hexdigest() != receipt["word_sha256"]
    ):
        raise ValueError("array words or layout differ")
    return a


class Recorder:
    def __init__(self, root, reserve):
        self.root, self.reserve = root, reserve
        self.counts = {
            k: {"attempted": 0, "returned": 0} for k in ("processor", "model", "source")
        }
        self.chain = "0" * 64
        self.sequence = 0

    def event(self, kind, **payload):
        self.sequence += 1
        row = {
            "sequence": self.sequence,
            "previous": self.chain,
            "kind": kind,
            **payload,
        }
        self.chain = hashlib.sha256(
            json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        append(self.root, "operations.jsonl", {**row, "digest": self.chain})

    def call(self, kind, identity, function):
        headroom = 32 * 2**20 if self.reserve else 0
        if shutil.disk_usage(self.root).free < self.reserve + headroom:
            raise ValueError("registered disk reserve breached")
        self.counts[kind]["attempted"] += 1
        self.event("begin", operation=kind, identity=identity)
        value = function()
        self.counts[kind]["returned"] += 1
        self.event("return", operation=kind, identity=identity)
        return value
