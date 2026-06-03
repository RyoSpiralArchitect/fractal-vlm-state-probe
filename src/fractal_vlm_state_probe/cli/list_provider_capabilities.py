from __future__ import annotations

import argparse
import json

from fractal_vlm_state_probe.providers import CAPABILITY_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="List provider capability scaffolds.")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()
    payload = {
        adapter_id: capabilities.to_dict()
        for adapter_id, capabilities in sorted(CAPABILITY_REGISTRY.items())
    }
    print(json.dumps(payload, indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()

