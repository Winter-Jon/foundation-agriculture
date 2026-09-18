"""Prepare E3.37, a fresh bounded-delivery Classifier lineage with Option-safe Q1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agrinet.rag.e328_classifier_full import prepare as _prepare

PROTOCOL = "agrinet.e337-classifier-full/v1"
CAMPAIGN_ID = "e337-classifier-full-v1"
TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION = 0.10


def prepare(**kwargs: Any) -> dict[str, Any]:
    root = Path(kwargs["output_root"])
    if "e337-classifier-full-v1" not in str(root):
        raise ValueError("E3.37 requires its dedicated fresh artifact root")
    return _prepare(**kwargs, protocol=PROTOCOL, campaign_id=CAMPAIGN_ID,
                    provenance_new="new_e337", provenance_reused="reused_e322_v3",
                    provenance_key="e337_provenance",
                    schema="agrinet.e337-classifier-full-manifest/v1",
                    shard_schema="agrinet.e337-classifier-full-shard-manifest/v1",
                    prepare_audit_schema="agrinet.e337-classifier-full-prepare-audit/v1",
                    terminal_unknown_delivery_max_fraction=TERMINAL_UNKNOWN_DELIVERY_MAX_FRACTION)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "direct-source", "e320-source", "e322-source",
                 "e322-campaign", "e322-report", "e322-audit", "e322-gate",
                 "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(queue_path=args.queue, direct_source=args.direct_source,
                             e320_source=args.e320_source, e322_source=args.e322_source,
                             e322_campaign=args.e322_campaign, e322_report=args.e322_report,
                             e322_audit=args.e322_audit, e322_gate=args.e322_gate,
                             output_root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
