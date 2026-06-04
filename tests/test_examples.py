from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_example_json_files_are_valid() -> None:
    paths = sorted((REPO_ROOT / "examples").glob("**/*.json"))
    assert paths
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        assert isinstance(data, dict)
        assert data.get("schema_version") == 1
        assert data.get("comparison_kind") or data.get("analysis_kind") or data.get("batch_kind")


def test_documentation_example_links_resolve() -> None:
    docs = [REPO_ROOT / "README.md", REPO_ROOT / "examples" / "README.md"]
    docs.extend(sorted((REPO_ROOT / "docs" / "research_notes").glob("*.md")))
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+examples/[^)]+)\)", text):
            target_path = (path.parent / target).resolve()
            assert target_path.exists(), f"{path} links to missing example {target}"
