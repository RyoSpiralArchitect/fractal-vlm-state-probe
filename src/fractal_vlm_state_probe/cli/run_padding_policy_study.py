from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

from fractal_vlm_state_probe.padding_policy import run_padding_policy_study
from fractal_vlm_state_probe.stimulus import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three padding policies with explicit RGB marginal changes."
    )
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--published-source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fastvlm-model-processor-snapshot", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.source_receipt.read_text())
    published = json.loads(args.published_source.read_text())
    if sha256_file(args.source_receipt) != published["input_receipt_sha256"]:
        raise ValueError("source receipt differs from published snapshot")
    expected = source["processor_provenance"]
    snapshot = args.fastvlm_model_processor_snapshot
    if (
        sha256_file(snapshot / "processing_fastvlm.py")
        != expected["implementation_sha256"]
    ):
        raise ValueError(
            "custom processor source differs from qualified implementation"
        )
    import torch
    from mlx_vlm.utils import load_processor

    torch.set_num_threads(1)
    processor = load_processor(snapshot, add_detokenizer=False, trust_remote_code=True)
    cls = type(processor.image_processor)
    provenance = {
        "snapshot_revision": snapshot.name,
        "preprocessor_config_sha256": sha256_file(
            snapshot / "preprocessor_config.json"
        ),
        "implementation": f"{cls.__module__}.{cls.__name__}",
        "implementation_sha256": sha256_file(Path(inspect.getfile(cls))),
        "processor_mode": "model_loading_path",
    }
    if provenance != expected:
        raise ValueError("processor provenance differs")
    normalization = json.loads((snapshot / "preprocessor_config.json").read_text())
    if (
        not normalization["do_normalize"]
        or not normalization["do_rescale"]
        or normalization["rescale_factor"] != 1 / 255
    ):
        raise ValueError("unsupported pixel normalization")
    runtime = {
        p: importlib.metadata.version(p)
        for p in ("numpy", "Pillow", "torch", "torchvision", "transformers", "mlx-vlm")
    }
    runtime["python"] = platform.python_version()
    summary = run_padding_policy_study(
        source_receipt_path=args.source_receipt,
        published_source_path=args.published_source,
        config=json.loads(args.config.read_text()),
        processor=processor,
        provenance=provenance,
        normalization=normalization,
        runtime=runtime,
        output_root=args.output_root,
    )
    for key, value in summary["conditions"].items():
        print(
            f"{key}: {value['accepted_blocks']}/{value['total_blocks']} accepted",
            flush=True,
        )
    print(
        f"wrote padding comparison to {args.output_root / 'padding_policy_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
