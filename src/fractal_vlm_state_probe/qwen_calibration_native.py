"""Lazy native adapter. Construction is only permitted after CLI admission."""

import importlib
import json
from contextlib import contextmanager, nullcontext

import numpy as np

from .qwen_representation_observation import observe_qwen_visual


@contextmanager
def decoder_records(target, emit):
    cls = type(target)
    original = cls.__dict__.get("__call__")
    if original is None or getattr(original, "_qwen_calibration_observer", False):
        raise ValueError("explicit, unwrapped decoder call required")
    records, attempts = [], []

    def observed(instance, *args, **kwargs):
        if instance is not target:
            return original(instance, *args, **kwargs)
        if len(attempts) >= 2 or len(args) != 1:
            raise ValueError("registered two-call native decoder path differs")
        attempts.append(args[0])
        emit("decoder_begin")
        result = original(instance, *args, **kwargs)
        records.append((args[0], result))
        emit("decoder_return")
        return result

    observed._qwen_calibration_observer = True
    cls.__call__ = observed
    try:
        yield records
        if len(records) != 2:
            raise ValueError("native first yield must make two decoder calls")
    finally:
        cls.__call__ = original


def first_yield(generator):
    try:
        return next(generator)
    finally:
        generator.close()


class NativeBackend:
    def __init__(self, snapshot, source_prompt, plan):
        from .mlx_stream import _load_mlx_runtime
        from .prompts import SYSTEM_PROMPT

        self.runtime = _load_mlx_runtime(str(snapshot))
        self.model = self.runtime["model"]
        self.processor_object = self.runtime["processor"]
        self.generator = importlib.import_module("mlx_vlm.generate")
        self.mx = importlib.import_module("mlx.core")
        self.plan = plan
        self.prompt = self.runtime["apply_chat_template"](
            self.processor_object, self.runtime["model_config"],
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": source_prompt}], num_images=1)

    def _prepare(self, raw):
        from PIL import Image

        return self.generator.prepare_inputs(
            self.processor_object, images=[Image.fromarray(raw)], audio=None,
            prompts=self.prompt,
            image_token_index=getattr(self.model.config, "image_token_index", None),
            resize_shape=None, add_special_tokens=True)

    @staticmethod
    def _float(value):
        from .cache_tensor_artifact import _to_float32_numpy

        return _to_float32_numpy(value)

    def processor(self, raw):
        prepared = self._prepare(raw)
        return {"pixels": self._float(prepared["pixel_values"]),
                "grid": np.asarray(prepared["image_grid_thw"])}

    def context(self, raw, stage, emit):
        emit("prepare_begin")
        prepared = dict(self._prepare(raw))
        emit("prepare_return")
        ids = prepared.pop("input_ids")
        pixels = prepared.pop("pixel_values")
        mask = prepared.pop("attention_mask")
        grid = prepared["image_grid_thw"]
        # Bound inputs before any native forward, not just after acquisition.
        c = self.plan["limits"]
        from .qwen_calibration_contract import window_order

        window_order(np.asarray(grid), self.plan)
        if (ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= c["max_prefix_sequence"]
                or pixels.shape != (int(np.prod(np.asarray(grid))),
                    c["pixel_channels"] * c["temporal_patch_size"] * c["patch_size"]**2)
                or mask.shape != ids.shape or str(pixels.dtype) not in c["source_dtype_allowlist"]
                or not np.isfinite(self._float(pixels)).all()
                or np.asarray(ids).dtype.kind not in "iu"
                or np.any(np.asarray(ids) < 0) or np.any(np.asarray(ids) >= c["vocab"])
                or not np.isin(np.asarray(mask), [0, 1]).all()):
            raise ValueError("prepared inputs exceed registered native limits")
        extra = {}
        for key, value in prepared.items():
            if key != "image_grid_thw":
                if value is not None and not isinstance(value, (str, bool, int, float)):
                    raise ValueError("unregistered nonpixel processor field")
                extra[key] = value
        if len(json.dumps(extra, allow_nan=False).encode()) > self.plan["storage"]["metadata_and_headers_per_context_allowance_bytes"]:
            raise ValueError("nonpixel metadata exceeds registered cap")
        cache = self.generator.cache.make_prompt_cache(self.model.language_model)
        if len(cache) != c["language_layers"]:
            raise ValueError("native layer count differs before forward")
        tower = self.model.vision_tower
        old_tower, old_merger = type(tower).__call__, type(tower.merger).__call__
        old_decoder = type(self.model.language_model).__call__
        visual = observe_qwen_visual(tower) if stage == "on" else nullcontext(None)
        with visual as sites, decoder_records(self.model.language_model, emit) as calls:
            sampled, scores = first_yield(self.generator.generate_step(
                ids, self.model, pixels, mask, max_tokens=1, temperature=0,
                prompt_cache=cache, prefill_step_size=512, **prepared))
        if len(cache) != c["language_layers"]:
            raise ValueError("native layer count differs")
        native = {"ids": ids, "mask": mask, "pixels": pixels, "grid": grid,
                  "scores": scores}
        for i, (_, output) in enumerate(calls):
            native[f"logits_{i}"] = output.logits[0, -1, :]
        offsets = []
        for i, entry in enumerate(cache):
            offsets.append(int(entry.offset))
            if int(entry.offset) != ids.shape[1] + 1:
                raise ValueError("native cache is not prefix plus one")
            keys, values = entry.state
            native[f"cache_{i:02d}_keys"] = keys
            native[f"cache_{i:02d}_values"] = values
        if sites is not None:
            native.update(site_premerger=sites["merger"][0][0],
                          site_auxiliary=sites["merger"][0][1],
                          site_final=sites["vision"][0][2])
        source_dtypes = {k: str(a.dtype) for k, a in native.items()}
        arrays = {k: np.asarray(a) if k in ("ids", "mask", "grid") else self._float(a)
                  for k, a in native.items()}
        return {"arrays": arrays, "source_dtypes": source_dtypes, "metadata": {
            "sampled_id": int(sampled), "decoder_inputs": [np.asarray(a).ravel().tolist() for a, _ in calls],
            "cache_offsets": offsets, "extra_inputs": extra,
            "observer_calls": {"vision": int(stage == "on"), "merger": int(stage == "on")},
            "observer_restored": (type(tower).__call__ is old_tower and type(tower.merger).__call__ is old_merger
                                  and type(self.model.language_model).__call__ is old_decoder)}}
