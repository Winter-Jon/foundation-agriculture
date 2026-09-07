from pathlib import Path
from agrinet.vlm.export import export_transformers_checkpoint

def test_export_transformers_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"; source.mkdir()
    (source / "config.json").write_text('{"architectures":["Fixture"]}')
    (source / "model.safetensors").write_bytes(b"weights")
    destination = tmp_path / "artifact"
    stats = export_transformers_checkpoint(source, destination)
    assert stats["files"] == 2
    assert (destination / "model.safetensors").read_bytes() == b"weights"
