import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agrinet.rag.hermes_protocol import normalize_training_messages
from agrinet.rag.tool_schema import TOOL_NAME, validate_tool_arguments
from agrinet.research.open_agri_v2_canonical.cards import CANDIDATE_CARD_SCHEMA, candidate_card
from agrinet.research.open_agri_v2_canonical.registry import CANONICAL_VERSION, load_registry


DATASET = ROOT / "datasets/AgriNet-1K/open_agri_v2_canonical_v1"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def registry():
    return load_registry(DATASET / "taxonomy/canonical_label_registry.jsonl", DATASET / "taxonomy/approval.json")


def test_registry_is_approved_and_deterministic() -> None:
    value = registry()
    assert len(value.rows_by_code) == 211
    assert len(value.source_to_canonical) == 217
    assert value.canonical_for_source("N04031") == "N04029"
    assert value.canonical_for_source("N05037") == "N05022"
    assert value.resolve_answer("cherry healthy") == "N04029"
    assert value.resolve_answer("cydia pomonella") == "N05019"
    assert value.resolve_answer("cydia pomonella larva") == "N05020"
    assert value.resolve_answer("spodoptera frugiperda") == "N05063"
    assert value.resolve_answer("spodoptera frugiperda larva") == "N05064"
    approval = json.loads((DATASET / "taxonomy/approval.json").read_text(encoding="utf-8"))
    assert approval["status"] == "approved"
    assert approval["approved_rows"] == 211
    load_registry(DATASET / "taxonomy/canonical_label_registry.jsonl", DATASET / "taxonomy/approval.json", require_approval=True)


def test_cross_role_merges_are_known_and_public_private_are_aligned() -> None:
    classes = {row["canonical_class_code"]: row for row in rows(DATASET / "manifests/class_split.jsonl")}
    for code in ("N04029", "N04080", "N04111"):
        assert classes[code]["class_role"] == "known"
        assert classes[code]["sft_eligible"] is True
    public = {row["id"]: row for row in rows(DATASET / "vlm_data/accepted/test_public.jsonl")}
    truth = {row["id"]: row for row in rows(DATASET / "vlm_data/accepted/private/test_truth.jsonl")}
    assert set(public) == set(truth)
    assert len(public) == 1019
    assert all("canonical_class_code" in row["metadata"] for row in public.values())
    assert all("canonical_class_code" in row for row in truth.values())


def test_candidate_card_is_current_canonical_schema() -> None:
    card = candidate_card(registry(), "N05020", rank=1, score=0.9)
    assert card["schema_version"] == CANDIDATE_CARD_SCHEMA
    assert card["canonical_class_code"] == "N05020"
    assert card["canonical_english_name"] == "cydia pomonella larva"
    assert card["life_stage"] == "larva"


def test_sft_review_preview_uses_registry_and_hermes_contract() -> None:
    artifact = ROOT / "outputs/artifacts/open-agri-v2-canonical-v1-sft"
    view = rows(artifact / "views/direct-rag/data.jsonl")
    value = registry()
    assert view
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_status"] == "review_preview_only"
    for row in view:
        metadata = row["metadata"]
        final = row["messages"][-1]["content"].split("<answer>", 1)[1].split("</answer>", 1)[0]
        if metadata["question_type"] == "open":
            assert value.resolve_answer(final) == metadata["canonical_class_code"]
        else:
            assert metadata["option_codes"]["ABCD".index(final)] == metadata["canonical_class_code"]
        if metadata["route"] == "rag":
            normalize_training_messages(row["messages"], tool_name=TOOL_NAME, validate_arguments=validate_tool_arguments, require_tool_calls=True)
            assert metadata["rag_candidate_card_schema"] == CANDIDATE_CARD_SCHEMA
            tool_response = next(message for message in row["messages"] if message["role"] == "tool")
            payload = json.loads(tool_response["content"])
            assert payload["schema_version"] == CANDIDATE_CARD_SCHEMA


def test_release_manifest_declares_canonical_line() -> None:
    summary = json.loads((DATASET / "manifests/summary.json").read_text(encoding="utf-8"))
    assert summary["taxonomy_version"] == CANONICAL_VERSION
    assert summary["counts"]["source_classes"] == 217
    assert summary["counts"]["canonical_classes"] == 211
    assert summary["release_status"] == "approved"
    assert summary["counts"]["approved_classes"] == 211


def test_current_and_legacy_review_materials_are_separated() -> None:
    taxonomy = DATASET / "taxonomy"
    current = rows(taxonomy / "current/canonical_proposals.jsonl")
    legacy = rows(taxonomy / "legacy/source_label_lineage.jsonl")
    decisions = rows(taxonomy / "review/review_decisions.jsonl")
    assert len(current) == 211
    assert len(legacy) == 217
    assert len(decisions) == 211
    assert all("legacy_english_name" not in row for row in current)
    assert all("english_name" not in row for row in legacy)
    for path in ("review/current/disease.md", "review/current/pest.md", "review/legacy/disease.md", "review/legacy/pest.md"):
        assert (taxonomy / path).is_file()
