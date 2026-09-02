#!/usr/bin/env python3
"""Stable formal-v2 entrypoint for accepted VLM supervision import.

The implementation remains in the v1.1 migration-named module for historical
lineage compatibility. This wrapper is the only new formal-v2 command to cite
in dataset cards, experiment definitions, and runbooks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_IMPLEMENTATION = Path(__file__).with_name("ingest_open_agri_v1_1_hcv_base_accepted.py")
_SPEC = importlib.util.spec_from_file_location("open_agri_v2_accepted_impl", _IMPLEMENTATION)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_MODULE, _name)


if __name__ == "__main__":
    raise SystemExit(main())
