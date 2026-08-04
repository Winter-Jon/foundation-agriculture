from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


class CredentialError(RuntimeError):
    pass


ALLOWED_YUNWU_KEYS = {"YUNWU_API_KEY", "YUNWU_API_BASE_URL"}


def yunwu_environment(helper: Path | None = None) -> dict[str, str]:
    existing = {key: os.environ[key] for key in ALLOWED_YUNWU_KEYS if os.environ.get(key)}
    if "YUNWU_API_KEY" in existing:
        return existing
    executable = helper or Path.home() / ".apikeys" / "bin" / "apikey"
    if not executable.is_file():
        raise CredentialError(f"encrypted credential helper not found: {executable}")
    try:
        result = subprocess.run(
            [str(executable), "env", "yunwu"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        summary = (exc.stderr or "credential decryption failed").strip().splitlines()[-1]
        raise CredentialError(summary) from exc
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.startswith("export ") or "=" not in line:
            raise CredentialError("credential helper returned an unsupported command")
        name, encoded = line.removeprefix("export ").split("=", 1)
        if name not in ALLOWED_YUNWU_KEYS:
            raise CredentialError(f"credential helper returned disallowed variable: {name}")
        tokens = shlex.split(encoded, posix=True)
        if len(tokens) != 1:
            raise CredentialError(f"credential helper returned an invalid value for {name}")
        values[name] = tokens[0]
    if not values.get("YUNWU_API_KEY"):
        raise CredentialError("credential helper did not return YUNWU_API_KEY")
    return values
