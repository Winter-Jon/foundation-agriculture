"""Prepare a clean fixed-query successor to the frozen E3.39 campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.rag.e339_visual_top3_safe_subset import prepare as _prepare_safe

PROTOCOL = "agrinet.e340-visual-top3-rag-safe-subset-fixed/v1"
SCHEMA = "agrinet.e340-visual-top3-rag-safe-subset-fixed-manifest/v1"


def prepare(**kwargs):
    root = Path(kwargs["output_root"])
    if "e340-visual-top3-rag-safe-subset-fixed-v1" not in str(root):
        raise ValueError("E3.40 requires its dedicated fresh artifact root")
    return _prepare_safe(protocol=PROTOCOL, schema=SCHEMA, required_root_name="e340-visual-top3-rag-safe-subset-fixed-v1", **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("e328-source", "e328-report", "e328-audit", "e328-gate", "e327-source", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    value = prepare(e328_source=args.e328_source, e328_report=args.e328_report, e328_audit=args.e328_audit, e328_gate=args.e328_gate, e327_source=args.e327_source, output_root=args.output_root)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
