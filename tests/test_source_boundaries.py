"""Regression checks for supported versus archived source boundaries."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _python_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_active_source_does_not_import_archived_or_removed_distillation_code() -> None:
    forbidden = ("archive.source", "agrinet.rag.distill", "agrinet.data.m1_direct")
    offenders = {
        str(path.relative_to(ROOT)): marker
        for path in _python_sources(ROOT / "src/agrinet")
        for marker in forbidden
        if marker in path.read_text(encoding="utf-8")
    }
    assert not offenders


def test_archived_source_and_supported_research_packages_are_explicit() -> None:
    old_distill = ROOT / "src/agrinet/rag/distill"
    assert not any(old_distill.glob("*.py"))
    assert (ROOT / "src/agrinet/research/m1").is_dir()
    assert (ROOT / "src/agrinet/research/hcv").is_dir()
    assert (ROOT / "src/agrinet/vlm/evaluation").is_dir()
    assert (ROOT / "archive/source/rag_distill/README.md").is_file()
