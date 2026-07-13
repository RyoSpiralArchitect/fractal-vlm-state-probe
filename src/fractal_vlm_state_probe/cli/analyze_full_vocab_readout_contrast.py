from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.compare_runs import load_run
from fractal_vlm_state_probe.full_vocab_readout import (
    analyze_full_vocab_readout_contrast,
    write_full_vocab_readout_json,
    write_full_vocab_readout_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze full-vocabulary first-token distributions over a 2x2 factorial."
    )
    parser.add_argument("--mm", required=True, type=Path)
    parser.add_argument("--jj", required=True, type=Path)
    parser.add_argument("--mj", required=True, type=Path)
    parser.add_argument("--jm", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--max-token-effects", type=int, default=20)
    args = parser.parse_args()

    run_paths = {key: getattr(args, key) for key in ("mm", "jj", "mj", "jm")}
    analysis = analyze_full_vocab_readout_contrast(
        runs={key: load_run(path) for key, path in run_paths.items()},
        run_paths=run_paths,
        max_token_effects=args.max_token_effects,
    )
    write_full_vocab_readout_json(analysis, args.output_json)
    write_full_vocab_readout_markdown(analysis, args.output_md)
    print(f"wrote full-vocabulary readout contrast to {args.output_md}")


if __name__ == "__main__":
    main()
