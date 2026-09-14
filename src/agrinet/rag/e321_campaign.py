"""Managed sequential completion controller for E3.21 Direct-only collection.

This controller never dispatches a successor semantic route.  It is limited to
the frozen Direct manifests, same-stage Q1 repairs, and R1/R2 delivery recovery.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agrinet.rag.e321_direct_first import (
    write_direct_delivery_recovery_manifest, write_direct_quality_repair_manifest,
)
from agrinet.rag.e321_report import checkpoint_report, delivery_shortfall_report


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_collect(*, manifest: Path, source: Path, output: Path, campaign_root: Path,
                 budget_path: Path, private_registry: Path, teacher_model: str, timeout: int) -> None:
    if output.exists():
        return
    command=[sys.executable,"-m","agrinet.rag.e320_live","--manifest",str(manifest),
             "--source",str(source),"--output",str(output),"--output-root",str(campaign_root),
             "--budget-path",str(budget_path),"--private-registry",str(private_registry),
             "--teacher-model",teacher_model,"--timeout",str(timeout)]
    subprocess.run(command,check=True)


def _report(*, manifest: Path, source: Path, outcomes: Path, campaign_root: Path, report: Path) -> dict[str, Any]:
    if report.exists(): return _json(report)
    return checkpoint_report(manifest=manifest,source=source,outcomes=outcomes,campaign_root=campaign_root,output=report)


def _resolve_chain(*, base_manifest: Path, source: Path, outcomes: Path, campaign_root: Path, budget_path: Path,
                   private_registry: Path, teacher_model: str, timeout: int, manifest_dir: Path,
                   outcome_dir: Path, report_dir: Path) -> list[dict[str, Any]]:
    """Close every permitted Direct Q1/R1/R2 descendant of one base manifest."""
    completed=[]
    pending=[(base_manifest,outcomes)]
    seen=set()
    while pending:
        prior_manifest,prior_outcomes=pending.pop(0)
        identity=(str(prior_manifest),str(prior_outcomes))
        if identity in seen: continue
        seen.add(identity)
        prior=_json(prior_manifest)
        q1_manifest=manifest_dir / f"{prior_manifest.stem}-q1.json"
        if not q1_manifest.exists():
            write_direct_quality_repair_manifest(prior_manifest=prior_manifest,outcomes=prior_outcomes,output=q1_manifest)
        q1=_json(q1_manifest)
        if q1.get("work_items"):
            q1_out=outcome_dir / f"{q1_manifest.stem}.json"
            if not q1_out.exists():
                _run_collect(manifest=q1_manifest,source=source,output=q1_out,campaign_root=campaign_root,budget_path=budget_path,
                             private_registry=private_registry,teacher_model=teacher_model,timeout=timeout)
            _report(manifest=q1_manifest,source=source,outcomes=q1_out,campaign_root=campaign_root,report=report_dir / f"{q1_manifest.stem}.json")
            completed.append({"manifest":str(q1_manifest),"outcomes":str(q1_out)})
            pending.append((q1_manifest,q1_out))
        round_value=str(prior.get("round") or "")
        next_round="R1" if round_value in {"R0","Q1","Q1-continuation"} else "R2" if round_value == "R1" else None
        if next_round is None: continue
        recovery_manifest=manifest_dir / f"{prior_manifest.stem}-{next_round.lower()}.json"
        if not recovery_manifest.exists():
            write_direct_delivery_recovery_manifest(prior_manifest=prior_manifest,outcomes=prior_outcomes,next_round=next_round,output=recovery_manifest)
        recovery=_json(recovery_manifest)
        if recovery.get("work_items"):
            recovery_out=outcome_dir / f"{recovery_manifest.stem}.json"
            if not recovery_out.exists():
                _run_collect(manifest=recovery_manifest,source=source,output=recovery_out,campaign_root=campaign_root,budget_path=budget_path,
                             private_registry=private_registry,teacher_model=teacher_model,timeout=timeout)
            _report(manifest=recovery_manifest,source=source,outcomes=recovery_out,campaign_root=campaign_root,report=report_dir / f"{recovery_manifest.stem}.json")
            completed.append({"manifest":str(recovery_manifest),"outcomes":str(recovery_out)})
            pending.append((recovery_manifest,recovery_out))
        if next_round == "R2" and (outcome_dir / f"{recovery_manifest.stem}.json").exists():
            recovery_out=outcome_dir / f"{recovery_manifest.stem}.json"
            shortfall=report_dir / f"{recovery_manifest.stem}-delivery-shortfall.json"
            if not shortfall.exists():
                delivery_shortfall_report(manifest=recovery_manifest,outcomes=recovery_out,output=shortfall)
    return completed


def run_campaign(*, source: Path, manifest_dir: Path, outcome_dir: Path, report_dir: Path, campaign_root: Path,
                 budget_path: Path, private_registry: Path, teacher_model: str, timeout: int) -> dict[str, Any]:
    base=sorted(path for path in manifest_dir.glob("e321-direct-r0-shard-*.json")
                if re.fullmatch(r"e321-direct-r0-shard-0[0-7]\.json", path.name))
    if len(base) != 8: raise ValueError("E3.21 requires exactly eight frozen R0 shard manifests")
    outcome_dir.mkdir(parents=True,exist_ok=True); report_dir.mkdir(parents=True,exist_ok=True)
    completed=[]
    for manifest in base:
        outcome=outcome_dir / f"{manifest.stem}.json"
        if not outcome.exists():
            _run_collect(manifest=manifest,source=source,output=outcome,campaign_root=campaign_root,budget_path=budget_path,
                         private_registry=private_registry,teacher_model=teacher_model,timeout=timeout)
        report=_report(manifest=manifest,source=source,outcomes=outcome,campaign_root=campaign_root,report=report_dir / f"{manifest.stem}.json")
        descendants=_resolve_chain(base_manifest=manifest,source=source,outcomes=outcome,campaign_root=campaign_root,
            budget_path=budget_path,private_registry=private_registry,teacher_model=teacher_model,timeout=timeout,
            manifest_dir=manifest_dir,outcome_dir=outcome_dir,report_dir=report_dir)
        completed.append({"manifest":str(manifest),"outcomes":str(outcome),"checkpoint":report,"direct_descendants":descendants})
    payload={"schema_version":"agrinet.e321-direct-campaign-progress/v1","base_shards":completed,
             "executed_routes":["direct"],"frozen_not_executed":["classifier","rag","reject"],
             "training_eligible":False,"training_authorized":False,"sft_may_start":False}
    progress=campaign_root / "e321-direct-campaign-progress.json"
    progress.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","manifest-dir","outcome-dir","report-dir","campaign-root","budget-path","private-registry"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--teacher-model",default="gpt-5.6-sol"); parser.add_argument("--timeout",type=int,default=180)
    args=parser.parse_args(argv)
    result=run_campaign(source=args.source,manifest_dir=args.manifest_dir,outcome_dir=args.outcome_dir,report_dir=args.report_dir,
        campaign_root=args.campaign_root,budget_path=args.budget_path,private_registry=args.private_registry,teacher_model=args.teacher_model,timeout=args.timeout)
    print(json.dumps({"base_shards":len(result["base_shards"]),"executed_routes":result["executed_routes"],"sft_may_start":False}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
