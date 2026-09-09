from pathlib import Path

import pytest
import yaml

from agrinet.rag.micu_classifier_hcv_v2 import readiness, validate_contract, validate_smoke_source


CONTRACT = Path("configs/sampling/micu-classifier-hcv-v2/contract-v1.yaml")


def test_v2_contract_freezes_single_pattern_and_progress_budget() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    validate_contract(contract)
    contract["routing"]["one_primary_pattern_per_query"] = False
    with pytest.raises(ValueError, match="one pattern"):
        validate_contract(contract)
    contract["classifier"]["prediction_kind"] = "in_sample"
    with pytest.raises(ValueError, match="OOF"):
        validate_contract(contract)


def test_readiness_missing_pool_is_not_live_or_training_ready(tmp_path: Path) -> None:
    report = readiness(contract_path=CONTRACT, dataset_root=tmp_path)
    assert report["ready_for_live_collection"] is False
    assert report["training_eligible"] is False
    assert any("missing source" in blocker for blocker in report["blockers"])
    assert any("missing exclusions" in blocker for blocker in report["blockers"])


def test_readiness_rejects_wrong_rag_collections(tmp_path: Path) -> None:
    report = readiness(
        contract_path=CONTRACT,
        dataset_root=tmp_path,
        rag_health={
            "status": "ok",
            "collections": {
                "agrinet_wiki_siglip2_classes": 320,
                "agrinet_wiki_siglip2_images": 578,
            },
        },
    )
    assert any("collections mismatch" in blocker for blocker in report["blockers"])


def test_smoke_source_rejects_wrong_cardinality(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="32 rows"):
        validate_smoke_source(source, tmp_path)
