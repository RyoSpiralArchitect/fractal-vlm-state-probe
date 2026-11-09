"""Prepare pinned metadata only; there is intentionally no score/run mode."""

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.reference_crossing_registration import prepare

REPO = Path(__file__).resolve().parents[3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(REPO, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
