from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/milvus/migrate_open_agri_v2_canonical_v3_milvus_names.py"
SPEC = importlib.util.spec_from_file_location("canonical_v3_milvus_names", SCRIPT)
assert SPEC and SPEC.loader
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


class Registry:
    source_to_canonical = {"N00002": "N00001"}

    def display_name(self, code: str, language: str) -> str:
        return {("N00001", "en"): "canonical leaf spot", ("N00001", "zh"): "规范叶斑病"}[code, language]


def test_name_migration_changes_only_display_surfaces() -> None:
    source = {
        "entry_id": "wiki::N00002", "code": "N00002",
        "english_name": "Legacy Leaf Spot", "chinese_name": "旧叶斑病",
        "similar_english_classes": ["Legacy Leaf Spot", "unmapped"],
        "similar_chinese_classes": ["旧叶斑病", "未映射"],
        "content_1": "keep this description unchanged",
        "text_vector": [0.1, 0.2], "sparse_vector": {1: 1.0},
    }
    candidates = {
        "wiki::N00002": [
            {"code": "N00002", "english_name": "Legacy Leaf Spot", "chinese_name": "旧叶斑病"},
            {"code": "N99999", "english_name": "unmapped", "chinese_name": "未映射"},
        ]
    }
    target = migration.rename_class_row(source, Registry(), candidates)
    assert target["english_name"] == "canonical leaf spot"
    assert target["chinese_name"] == "规范叶斑病"
    assert target["similar_english_classes"] == ["canonical leaf spot", "unmapped"]
    assert target["similar_chinese_classes"] == ["规范叶斑病", "未映射"]
    assert migration.changed_fields(source, target) == migration.NAME_FIELDS


def test_name_migration_refuses_to_guess_similar_class_order() -> None:
    source = {
        "entry_id": "wiki::N00002", "code": "N00002",
        "similar_english_classes": ["wrong order"], "similar_chinese_classes": ["错误顺序"],
    }
    candidates = {
        "wiki::N00002": [{"code": "N00002", "english_name": "Legacy Leaf Spot", "chinese_name": "旧叶斑病"}]
    }
    try:
        migration.rename_class_row(source, Registry(), candidates)
    except ValueError as exc:
        assert "differ from provenance" in str(exc)
    else:
        raise AssertionError("expected migration to reject mismatched relation provenance")
