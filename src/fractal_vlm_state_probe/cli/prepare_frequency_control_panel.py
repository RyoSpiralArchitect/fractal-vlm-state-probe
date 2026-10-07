from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from fractal_vlm_state_probe.cli.analyze_pairing_input_holdout import (
    _ModelProcessorPixelView,
)
from fractal_vlm_state_probe.cli.run_pairing_seed_validation import (
    _read,
    _validate_panel_split,
)
from fractal_vlm_state_probe.frequency_control import prepare_frequency_controls
from fractal_vlm_state_probe.stimulus import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare input-only, all-four-cell frequency controls."
    )
    parser.add_argument("--reference-panel", required=True, type=Path)
    parser.add_argument("--test-panel", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fastvlm-model-processor-snapshot", required=True, type=Path)
    parser.add_argument("--qualified-processor-provenance", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    _validate_panel_split(_read(args.reference_panel), _read(args.test_panel))
    snapshot = args.fastvlm_model_processor_snapshot
    config = _read(args.config)
    if _read(snapshot / "config.json")["model_type"] != "llava_qwen2":
        raise ValueError(
            "this stage is registered for the qualified FastVLM processor only"
        )
    import torch
    from mlx_vlm.utils import load_processor

    qualified = _read(args.qualified_processor_provenance)["processor_provenance"]
    if (
        sha256_file(snapshot / "processing_fastvlm.py")
        != qualified["implementation_sha256"]
    ):
        raise ValueError("saved custom processor source differs from qualified code")
    torch.set_num_threads(1)
    model_processor = load_processor(
        snapshot, add_detokenizer=False, trust_remote_code=True
    )
    cls = type(model_processor.image_processor)
    provenance = {
        "snapshot_revision": snapshot.name,
        "preprocessor_config_sha256": sha256_file(
            snapshot / "preprocessor_config.json"
        ),
        "implementation": f"{cls.__module__}.{cls.__name__}",
        "implementation_sha256": sha256_file(Path(inspect.getfile(cls))),
        "processor_mode": "model_loading_path",
    }
    if provenance != qualified:
        raise ValueError("processor differs from the qualified actual-input baseline")
    normalization = _read(snapshot / "preprocessor_config.json")
    if (
        not normalization["do_normalize"]
        or not normalization["do_rescale"]
        or normalization["rescale_factor"] != 1 / 255
    ):
        raise ValueError(
            "unsupported normalization for processor-space RGB reconstruction"
        )
    summary = prepare_frequency_controls(
        reference_panel=args.reference_panel,
        test_panel=args.test_panel,
        config=config,
        processor=_ModelProcessorPixelView(model_processor),
        provenance=provenance,
        normalization=normalization,
        output_root=args.output_root,
    )
    for arm, status in summary["arms"].items():
        print(
            f"{arm}: {status['accepted_blocks']}/{summary['block_count']} accepted blocks",
            flush=True,
        )
    print(
        f"wrote input acceptance to {args.output_root / 'input_acceptance.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
