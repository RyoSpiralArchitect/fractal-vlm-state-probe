from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.generator_pairing_panel import (
    load_generator_pairing_panel_config,
    prepare_generator_pairing_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a hierarchical generator-pairing panel and its "
            "cross-palette factorials."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = prepare_generator_pairing_panel(
        config=load_generator_pairing_panel_config(args.config),
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(
        f"wrote {summary['source_pair_count']} generator-pairing factorials "
        f"to {args.output_root}"
    )


if __name__ == "__main__":
    main()
