from __future__ import annotations

import random
from typing import Any


def set_global_seed(seed: int | None, *, include_mlx: bool = False, include_torch: bool = False) -> dict[str, Any]:
    if seed is None:
        return {"seed": None, "applied": []}

    applied: list[str] = []
    random.seed(seed)
    applied.append("python.random")

    try:
        import numpy as np

        np.random.seed(seed % (2**32))
        applied.append("numpy.random")
    except Exception as exc:
        applied.append(f"numpy.random unavailable: {type(exc).__name__}")

    if include_mlx:
        try:
            import mlx.core as mx

            mx.random.seed(seed)
            applied.append("mlx.random")
        except Exception as exc:
            applied.append(f"mlx.random unavailable: {type(exc).__name__}")

    if include_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            applied.append("torch")
        except Exception as exc:
            applied.append(f"torch unavailable: {type(exc).__name__}")

    return {"seed": seed, "applied": applied}

