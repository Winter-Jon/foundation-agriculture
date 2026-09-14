"""Public-safe checkpoint reporting for E3.21 Direct-only shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agrinet.rag.e320_four_stage_cascade import validate_e320_trajectory


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_report(*, manifest: Path, source: Path, outcomes: Path, campaign_root: Path, output: Path) -> dict[str, Any]:
    """Write an immutable public-safe report after a complete shard outcome."""
    if output.exists(): raise ValueError("E3.21 checkpoint report is immutable")
    plan=json.loads(manifest.read_text(encoding="utf-8")); result=json.loads(outcomes.read_text(encoding="utf-8"))
    if result.get("manifest_sha256") != _digest(manifest): raise ValueError("E3.21 checkpoint manifest binding mismatch")
    items=plan.get("work_items") or []; decisions=result.get("outcomes") or []
    if len(items) != len(decisions) or {str(x.get("work_id")) for x in items} != {str(x.get("work_id")) for x in decisions}:
        raise ValueError("E3.21 checkpoint outcome scope is incomplete")
    source_rows=_rows(source); item_by_work={str(item["work_id"]):item for item in items}
    disposition=Counter(); delivery=Counter(); arm=Counter(); form=Counter(); validation_errors=[]
    for decision in decisions:
        item=item_by_work[str(decision["work_id"])]; row=source_rows[str(item["sample_id"])]
        delivery[str(decision.get("delivery_status"))] += 1
        disposition[str(decision.get("disposition"))] += 1
        arm[str(row.get("arm"))] += 1; form[str(row.get("question_type"))] += 1
        parent=decision.get("parent_path")
        if decision.get("delivery_status") == "delivered" and isinstance(parent,str) and parent:
            try: validate_e320_trajectory(row,json.loads(Path(parent).read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ValueError) as exc: validation_errors.append({"sample_id":row["sample_id"],"error":str(exc)})
    budget_path=Path(campaign_root) / "token_budget.jsonl"
    events=[json.loads(x) for x in budget_path.read_text(encoding="utf-8").splitlines() if x.strip()] if budget_path.exists() else []
    prefixes={f"{work_id}:" for work_id in item_by_work}
    scoped=[e for e in events if any(str(e.get("key") or "").startswith(prefix) for prefix in prefixes)]
    reservations={str(e["key"]):int(e["uncached_input_tokens"]) for e in scoped if e.get("event") == "reserve"}
    settlements={str(e["key"]):int(e["uncached_input_tokens"]) for e in scoped if e.get("event") == "settle"}
    report={"schema_version":"agrinet.e321-checkpoint-report/v1",
            "campaign_id":plan.get("campaign_id"),"manifest":str(manifest),"manifest_sha256":_digest(manifest),
            "outcomes":str(outcomes),"outcomes_sha256":_digest(outcomes),
            "scope":{"items":len(items),"arm":dict(arm),"question_type":dict(form)},
            "delivery":dict(delivery),"disposition":dict(disposition),
            "public_validation":{"passed":not validation_errors,"errors":validation_errors},
            "token_budget":{"cap":int((plan.get("collection_controls") or {}).get("uncached_input_token_cap",0)),
                            "settled_uncached_input_tokens":sum(settlements.values()),
                            "unknown_delivery_exposure_tokens":sum(v for key,v in reservations.items() if key not in settlements),
                            "reservation_events":len(reservations),"settlement_events":len(settlements),
                            "scope":"manifest_work_items_only"},
            "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--outcomes",type=Path,required=True); parser.add_argument("--campaign-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(argv)
    print(json.dumps(checkpoint_report(manifest=args.manifest,source=args.source,outcomes=args.outcomes,campaign_root=args.campaign_root,output=args.output),ensure_ascii=False,sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
