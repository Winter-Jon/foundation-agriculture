"""Prepare E3.43 full refusal inputs with a demonstrable-boundary private audit."""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agrinet.rag import e342_refusal_trajectories as base

PROTOCOL = "agrinet.e343-refusal-trajectories-boundary-audit/v1"
SCHEMA = "agrinet.e343-refusal-trajectories-manifest/v1"
ARTIFACT_ROOT_NAME = "e343-refusal-trajectories-boundary-audit-v1"
CAMPAIGN_ID = "e343-refusal-trajectories-boundary-audit-v1"


@contextmanager
def configured() -> Iterator[None]:
    values = {name: getattr(base, name) for name in ("PROTOCOL", "SCHEMA", "ARTIFACT_ROOT_NAME", "CAMPAIGN_ID")}
    base.PROTOCOL, base.SCHEMA = PROTOCOL, SCHEMA
    base.ARTIFACT_ROOT_NAME, base.CAMPAIGN_ID = ARTIFACT_ROOT_NAME, CAMPAIGN_ID
    try:
        yield
    finally:
        for name, value in values.items():
            setattr(base, name, value)


def prepare(**kwargs: object) -> dict[str, object]:
    with configured():
        return base.prepare(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("e341-source", "e341-report", "e341-audit", "e341-gate", "e341-campaign", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    value = prepare(e341_source=args.e341_source, e341_report=args.e341_report, e341_audit=args.e341_audit,
                    e341_gate=args.e341_gate, e341_campaign=args.e341_campaign, output_root=args.output_root)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
