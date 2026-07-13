from __future__ import annotations

import argparse
import json
from pathlib import Path

from fractal_vlm_state_probe.prompt_robustness import (
    analyze_prompt_robustness,
    write_prompt_robustness_json,
    write_prompt_robustness_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit forced-choice factorial readouts across prompt variants."
    )
    parser.add_argument("--full-vocab-analysis", required=True, type=Path)
    parser.add_argument("--phase", default="after")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    with args.full_vocab_analysis.open("r", encoding="utf-8") as handle:
        full_vocab_analysis = json.load(handle)
    analysis = analyze_prompt_robustness(
        full_vocab_analysis,
        source_path=args.full_vocab_analysis,
        phase=args.phase,
    )
    write_prompt_robustness_json(analysis, args.output_json)
    write_prompt_robustness_markdown(analysis, args.output_md)
    print(f"wrote prompt robustness audit to {args.output_md}")


if __name__ == "__main__":
    main()
