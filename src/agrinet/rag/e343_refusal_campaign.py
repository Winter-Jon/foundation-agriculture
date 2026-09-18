"""Run E3.43 with private audit limited to demonstrable boundary violations."""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agrinet.rag import e342_refusal_audit as audit_module
from agrinet.rag import e342_refusal_campaign as base
from agrinet.rag import e342_refusal_trajectories as prepare_module
from agrinet.rag.e343_refusal_trajectories import PROTOCOL


@contextmanager
def configured() -> Iterator[None]:
    old_policy, old_protocol = base.PRIVATE_AUDIT_POLICY, base.PROTOCOL
    old_prepare_protocol, old_audit_protocol = prepare_module.PROTOCOL, audit_module.PROTOCOL
    base.PRIVATE_AUDIT_POLICY, base.PROTOCOL = "demonstrable_boundary_only", PROTOCOL
    prepare_module.PROTOCOL, audit_module.PROTOCOL = PROTOCOL, PROTOCOL
    try:
        yield
    finally:
        base.PRIVATE_AUDIT_POLICY, base.PROTOCOL = old_policy, old_protocol
        prepare_module.PROTOCOL, audit_module.PROTOCOL = old_prepare_protocol, old_audit_protocol


def validate_inputs(**kwargs: object) -> dict[str, object]:
    with configured():
        return base.validate_inputs(**kwargs)


def run_campaign(**kwargs: object) -> dict[str, object]:
    with configured():
        return base.run_campaign(**kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-model", default="gpt-5.6-sol")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--authorize-live-collection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(validate_inputs(manifest=args.manifest, source=args.source), sort_keys=True))
        return 0
    value = run_campaign(manifest=args.manifest, source=args.source, output_root=args.output_root,
                         teacher_model=args.teacher_model, timeout=args.timeout,
                         authorize_live_collection=args.authorize_live_collection)
    print(json.dumps({"rows": value["rows"], "refusal_gate_passed": value["refusal_gate_passed"]}, sort_keys=True))
    return 0 if value["refusal_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
