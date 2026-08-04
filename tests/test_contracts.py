from pathlib import Path

import pytest
from pydantic import ValidationError

from agrinet.common.contracts import RagSearchRequest


def test_rag_search_contract_version_and_bounds() -> None:
    request = RagSearchRequest(retrieval_type="image", query_image=Path("query.jpg"), top_k=3)
    assert request.schema_version == "agrinet.rag.search/v1"
    with pytest.raises(ValidationError):
        RagSearchRequest(retrieval_type="image", query_image=Path("query.jpg"), top_k=0)
