from __future__ import annotations

import sys
from pathlib import Path


class VLMEvalKitAdapter:
    """Build evaluation commands through VLMEvalKit's public CLI."""

    def evaluate_command(self, model: str, dataset: str, work_dir: Path) -> list[str]:
        run_py = Path(__file__).resolve().parents[4] / "vlm" / "eval" / "VLMEvalKit" / "run.py"
        return [sys.executable, str(run_py), "--model", model, "--data", dataset, "--work-dir", str(work_dir)]
