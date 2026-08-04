from pathlib import Path
from agrinet.common.contracts import ArtifactRef
from agrinet.vlm.evaluate import evaluate_predictions

def test_exact_name_evaluation(tmp_path: Path) -> None:
    source = tmp_path / "pred.jsonl"
    source.write_text('{"sample_id":"1","expected":"Apple Black Rot","predicted":"apple black rot"}\n')
    artifact = ArtifactRef(schema_version="model/v1", artifact_id="m", artifact_type="models", path=tmp_path)
    result = evaluate_predictions(source, artifact, tmp_path / "evaluation.json")
    assert result.metrics["accuracy"] == 1.0
