from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from agrinet.data.io import DataError
from agrinet.common.credentials import CredentialError, yunwu_environment
from agrinet.common.network import NetworkConfigError, local_proxy_environment


class VloomTeacherDataProvider:
    """Adapter boundary around the existing project-local VLOOM runner."""

    name = "vloom"

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    def generate(self, samples_path: str, output_path: str, **options: Any) -> None:
        config_path = options.get("config_path")
        if not config_path:
            raise DataError("VLOOM generation requires parameters.config_path")
        try:
            credential_env = (
                {} if os.environ.get("YUNWU_API_KEY") or os.environ.get("OPENAI_API_KEY") else yunwu_environment()
            )
        except CredentialError as exc:
            raise DataError(f"cannot load encrypted Yunwu credential: {exc}") from exc
        try:
            proxy_env = local_proxy_environment()
        except NetworkConfigError as exc:
            raise DataError(f"cannot load local network proxy: {exc}") from exc
        command = [
            sys.executable,
            "-m",
            "tools.vloom_agrinet.run_contrast_cot",
            "--config_path",
            str(config_path),
        ]
        if options.get("max_concurrent") is not None:
            command.extend(["--max-concurrent", str(options["max_concurrent"])])
        subprocess.run(
            command, cwd=self.repository_root, check=True, env={**os.environ, **proxy_env, **credential_env}
        )
        expected = self.repository_root / output_path
        if not expected.is_file():
            raise DataError(f"VLOOM completed without expected output: {expected}")
