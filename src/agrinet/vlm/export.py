from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from agrinet.common.artifacts import sha256_file
from agrinet.vlm.inspect import inspect_model


def export_transformers_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    inspected = inspect_model(source)
    if not inspected["loadable_layout"]:
        raise ValueError(f"not a loadable Transformers checkpoint: {source}")
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    def copy_file(src: str, dst: str) -> str:
        subprocess.run(["cp", "--reflink=auto", "--preserve=mode,timestamps", src, dst], check=True)
        return dst
    shutil.copytree(source, staging, copy_function=copy_file)
    files = {}
    for item in sorted(staging.iterdir()):
        if item.is_file(): files[item.name] = sha256_file(item)
    staging.replace(destination)
    return {"files": len(files), "bytes": sum(item.stat().st_size for item in destination.iterdir() if item.is_file()), "checksums": files}
