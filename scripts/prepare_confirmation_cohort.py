"""Allocate metadata and simulate development precision, never generate images."""

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.confirmation_preparation import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(Path(__file__).resolve().parents[1], args.output), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
