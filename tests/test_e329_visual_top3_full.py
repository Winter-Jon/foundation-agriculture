from pathlib import Path
import pytest
from agrinet.rag.e329_visual_top3_full import prepare

ROOT=Path(__file__).resolve().parents[1]

def test_prepare_fails_closed_without_e328_gate(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(e328_source=ROOT/"outputs/artifacts/e328-classifier-full-v1/source.jsonl",e328_report=tmp_path/"missing-report.json",e328_audit=tmp_path/"missing-audit.json",e328_gate=tmp_path/"missing-gate.json",e327_source=ROOT/"outputs/artifacts/e327-visual-top3-rag-v1/source.jsonl",output_root=tmp_path/"e329")
    assert not (tmp_path/"e329").exists()
