from __future__ import annotations

import sys
from pathlib import Path


class MsSwiftAdapter:
    """Build commands using only ms-swift's documented module entrypoint."""

    def train_command(self, config: Path) -> list[str]:
        if not config.is_file():
            raise FileNotFoundError(config)
        return [sys.executable, "-m", "swift.cli.main", "sft", str(config)]

    def export_command(self, model: Path, output: Path) -> list[str]:
        if not model.is_dir():
            raise FileNotFoundError(model)
        return [sys.executable, "-m", "swift.cli.main", "export", "--model", str(model), "--output_dir", str(output)]
