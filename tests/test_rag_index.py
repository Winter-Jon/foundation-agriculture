from pathlib import Path

import pytest

from agrinet.rag.index import inspect_milvus_lite


def test_missing_milvus_db_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_milvus_lite(tmp_path / "missing.db")
