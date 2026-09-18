"""Prepare a clean slot-preserving successor to frozen E3.40 evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agrinet.rag.e339_visual_top3_safe_subset import prepare as _prepare_safe

PROTOCOL = "agrinet.e341-visual-top3-rag-safe-subset-slots/v1"
SCHEMA = "agrinet.e341-visual-top3-rag-safe-subset-slots-manifest/v1"


def prepare(**kwargs):
    root = Path(kwargs["output_root"])
    if "e341-visual-top3-rag-safe-subset-slots-v1" not in str(root):
        raise ValueError("E3.41 requires its dedicated fresh artifact root")
    return _prepare_safe(protocol=PROTOCOL, schema=SCHEMA, required_root_name="e341-visual-top3-rag-safe-subset-slots-v1", **kwargs)


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
