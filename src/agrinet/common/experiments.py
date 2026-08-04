from __future__ import annotations

import re
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

RUN_ID_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{7,12}-a[0-9]{2}$")


def git_short_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_run_id(attempt: int = 1, *, now: datetime | None = None, sha: str | None = None) -> str:
    if not 0 <= attempt <= 99:
        raise ValueError("attempt must be between 0 and 99")
    timestamp = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    return f"{timestamp:%Y%m%dT%H%M%S}-{sha or git_short_sha()}-a{attempt:02d}"


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid run id: {run_id!r}")
