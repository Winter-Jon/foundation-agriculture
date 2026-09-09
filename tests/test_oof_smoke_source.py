import importlib.util
from pathlib import Path


def test_smoke_builder_declares_group_and_private_pattern_logic() -> None:
    spec = importlib.util.spec_from_file_location(
        "build_oof_smoke_source", Path("scripts/vision/build_oof_smoke_source.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = {
        "canonical_class_code": "N04001",
        "prediction": {"top5": [
            {"code": "N04001", "score": 0.91},
            {"code": "N04002", "score": 0.02},
            {"code": "N04003", "score": 0.01},
            {"code": "N04004", "score": 0.01},
            {"code": "N04005", "score": 0.01},
        ]},
    }
    assert module.pattern(row) == "P1"


def test_historical_sha_exclusion_expands_to_manifest_source_and_phash_groups(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "build_oof_smoke_source", Path("scripts/vision/build_oof_smoke_source.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    manifest = tmp_path / "images.jsonl"
    manifest.write_text(
        '{"image_sha256": "contacted", "source_path": "/source/a.jpg", "phash": "aaa"}\n'
        '{"image_sha256": "peer", "source_path": "/source/b.jpg", "phash": "bbb"}\n',
        encoding="utf-8",
    )
    sources, near = module.historical_group_exclusions(
        blocked_hashes={"contacted"}, images_manifest=manifest)
    assert len(sources) == 1 and len(near) == 1
