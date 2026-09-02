"""Audit the supported source tree and archive boundary.

This is intentionally a small versioned audit script rather than a test under
the ignored ``tests/`` directory. It checks repository architecture without
executing experiment builders or mutating outputs.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "agrinet"


def python_files(path: Path) -> list[Path]:
    return sorted(path.rglob("*.py")) if path.exists() else []


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def main() -> int:
    failures: list[str] = []
    active = python_files(SRC)
    for path in active:
        bad = sorted(
            name
            for name in imports(path)
            if name == "archive.source" or name.startswith("archive.source.")
            or name == "agrinet.rag.distill" or name.startswith("agrinet.rag.distill.")
            or name == "agrinet.data.m1_direct" or name.startswith("agrinet.data.m1_direct.")
        )
        failures.extend(f"{path.relative_to(ROOT)} imports {name}" for name in bad)

    legacy_current = python_files(SRC / "rag" / "distill")
    if legacy_current:
        failures.append("active src/agrinet/rag/distill still contains Python files")

    required = [
        SRC / "research" / "m1",
        SRC / "research" / "hcv",
        SRC / "vlm" / "evaluation",
        ROOT / "archive" / "source" / "rag_distill" / "README.md",
        ROOT / "docs" / "source-map.md",
    ]
    failures.extend(f"missing required boundary path: {path.relative_to(ROOT)}" for path in required if not path.exists())

    if failures:
        print("source boundary audit: FAIL")
        print("\n".join(failures))
        return 1
    print(f"source boundary audit: OK ({len(active)} active Python files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
