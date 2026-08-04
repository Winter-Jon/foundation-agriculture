from __future__ import annotations

import os
import subprocess
from urllib.parse import urlparse


class NetworkConfigError(RuntimeError):
    pass


PROXY_KEYS = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY")
ALLOWED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}


def local_proxy_environment(shell: str = "zsh") -> dict[str, str]:
    for key in PROXY_KEYS:
        if os.environ.get(key):
            return {key: os.environ[key]}
    try:
        result = subprocess.run(
            [shell, "-ic", "proxy_on >/dev/null 2>&1; print -r -- ${ALL_PROXY:-}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NetworkConfigError("cannot load the existing local proxy helper") from exc
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    proxy = lines[-1] if lines else ""
    parsed = urlparse(proxy)
    if parsed.scheme not in ALLOWED_PROXY_SCHEMES or not parsed.hostname or not parsed.port:
        raise NetworkConfigError("local proxy helper returned an invalid proxy URL")
    # Different clients consult different conventional keys: httpx/openai accepts
    # ALL_PROXY while urllib resolves HTTP(S)_PROXY. Keep the same validated URL
    # in child-process memory only.
    return {"ALL_PROXY": proxy, "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy}
