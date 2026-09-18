"""Prepare E3.30, a fresh full Classifier lineage with trace-bound Q1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from agrinet.rag.e328_classifier_full import FLAGS, prepare as _prepare

PROTOCOL = "agrinet.e330-classifier-full/v6"
CAMPAIGN_ID = "e330-classifier-full-v6"

def prepare(**kwargs: Any) -> dict[str, Any]:
    """Use fresh E3.30 artifact paths; never write an E3.28 destination."""
    root=Path(kwargs["output_root"])
    if "e328-classifier-full" in str(root):
        raise ValueError("E3.30 must use a fresh artifact root")
    return _prepare(**kwargs, protocol=PROTOCOL, campaign_id=CAMPAIGN_ID,
                    provenance_new="new_e330", provenance_reused="reused_e322_v3",
                    provenance_key="e330_provenance",
                    schema="agrinet.e330-classifier-full-manifest/v6",
                    shard_schema="agrinet.e330-classifier-full-shard-manifest/v6",
                    prepare_audit_schema="agrinet.e330-classifier-full-prepare-audit/v6")

def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "direct-source", "e320-source", "e322-source",
                 "e322-campaign", "e322-report", "e322-audit", "e322-gate",
                 "output-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    args=parser.parse_args(argv)
    value=prepare(queue_path=args.queue, direct_source=args.direct_source,
                  e320_source=args.e320_source, e322_source=args.e322_source,
                  e322_campaign=args.e322_campaign, e322_report=args.e322_report,
                  e322_audit=args.e322_audit, e322_gate=args.e322_gate,
                  output_root=args.output_root)
    print(json.dumps(value, sort_keys=True)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
