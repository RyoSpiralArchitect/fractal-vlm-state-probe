from __future__ import annotations

import argparse
from pathlib import Path

from fractal_vlm_state_probe.run_summary import load_run, summarize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact run JSON summary.")
    parser.add_argument("run_json", type=Path)
    args = parser.parse_args()
    print(summarize_run(load_run(args.run_json)))


if __name__ == "__main__":
    main()

