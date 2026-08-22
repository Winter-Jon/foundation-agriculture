from pathlib import Path

import pytest
from pydantic import ValidationError

from agrinet.common.contracts import RagSearchRequest


def test_rag_search_contract_version_and_bounds() -> None:
    request = RagSearchRequest(retrieval_type="image", query_image=Path("query.jpg"), top_k=3)
    assert request.schema_version == "agrinet.rag.search/v1"
    with pytest.raises(ValidationError):
        RagSearchRequest(retrieval_type="image", query_image=Path("query.jpg"), top_k=0)


def test_rag_search_request_enforces_image_by_mode() -> None:
    assert RagSearchRequest(retrieval_type="semantic", query_text="leaf lesion").query_image is None
    assert RagSearchRequest(retrieval_type="name", query_text="Apple Black Rot").query_image is None
    with pytest.raises(ValidationError):
        RagSearchRequest(retrieval_type="visual", query_text="leaf lesion")
    with pytest.raises(ValidationError):
        RagSearchRequest(retrieval_type="semantic", query_image=Path("query.jpg"), query_text="leaf lesion")
