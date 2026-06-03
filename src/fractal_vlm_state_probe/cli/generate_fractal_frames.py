from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.stimulus import load_spec, render_stimulus


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic fractal frames.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spec = load_spec(args.config)
    manifest = render_stimulus(spec, args.output, overwrite=args.overwrite)
    print(
        f"wrote {len(manifest['frames'])} frames and manifest to "
        f"{args.output / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()

