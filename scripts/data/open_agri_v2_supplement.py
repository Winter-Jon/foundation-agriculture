#!/usr/bin/env python3
"""Human-gated Micu supplementation for the formal OpenAgri v2 dataset.

The plan, collection, audit, and review package are staging-only. The merge
operation is deliberately separate and requires human approval.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agrinet.common.credentials import yunwu_environment
from agrinet.data.io import DataError
from agrinet.research.open_agri_v2 import supplement

DATASET = ROOT / "datasets/AgriNet-1K/open_agri_v2"
DEFAULT_STAGING = ROOT / "outputs/artifacts/datasets/open-agri-v2-known-supplement-v1"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("plan", "plan-replenishment", "oracle-plan", "oracle-collect", "oracle-review", "oracle-provisional-audit", "oracle-final-export", "recover-n04094-rag", "collect-direct", "collect-rag", "stage", "review", "wait-review", "merge"))
    result.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    result.add_argument("--seed", default="open-agri-v2-known-supplement-v1")
    result.add_argument("--oversample", type=float, default=1.5)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--model", default="gpt-5.6-terra")
    result.add_argument("--rag-api", default="http://127.0.0.1:8077")
    result.add_argument("--round", type=int, default=0, help="0 is initial plan; authorized replenishment supports 1 through 11")
    result.add_argument("--authorized-replay-code", action="append", default=[])
    result.add_argument("--oracle-recovery-id", default="oracle-n04094-n04113-v1")
    result.add_argument("--oracle-code", action="append", default=[])
    result.add_argument("--allow-oracle-replay", action="store_true")
    result.add_argument("--approve-review", action="store_true")
    return result


def direct_alignment(staging: Path) -> Path:
    path = staging / "private/direct_alignment.jsonl"
    supplement.write_rows(path, supplement.direct_private_alignment(staging))
    return path


def round_root(staging: Path, round_index: int) -> Path:
    return staging if round_index == 0 else staging / "rounds" / f"v{round_index}"


def collect_direct(args: argparse.Namespace) -> None:
    staging = args.staging_root.resolve()
    root = round_root(staging, args.round)
    command = [
        sys.executable, "-m", "agrinet.research.m1.collection_runner",
        "--public-plan", str(root / "public/direct_plan.jsonl"),
        "--private-alignment", str(direct_alignment(root)),
        "--gate-config", str(ROOT / "configs/sampling/m1-direct-acceptance-gates-v2.yaml"),
        "--output-dir", str(root / "collection/direct"),
        "--ledger", str(root / "collection/direct/ledger.jsonl"),
        "--contacted-ledger", str(staging / "collection/contacted_images.jsonl"),
        "--contact-stage", "open_agri_v2_known_supplement", "--contact-round", f"v{args.round + 1}",
        "--credential-profile", "micu_slb", "--timeout", "300", "--max-tokens", "4096",
        "--image-max-side", "1536", "--workers", str(args.workers),
    ]
    replay_hashes = {
        str(row.get("image_sha256") or "")
        for row in supplement.read_jsonl(root / "public/direct_plan.jsonl")
        if row.get("special_replay")
    } - {""}
    if replay_hashes:
        contacted = {
            str(row.get("image_sha256") or ""): row
            for row in supplement.read_jsonl(staging / "collection/contacted_images.jsonl")
        }
        if replay_hashes - set(contacted):
            raise DataError("authorized replay plan contains an unregistered image")
        for digest in sorted(replay_hashes):
            prior = contacted[digest]
            command.extend([
                "--permitted-prior-ref",
                f"{prior.get('contact_stage')}|{prior.get('round')}",
            ])
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


def collect_rag(args: argparse.Namespace) -> None:
    staging = args.staging_root.resolve()
    root = round_root(staging, args.round)
    supplement.rag_service_preflight(
        root / "private/rag_plan.jsonl", root / "collection/rag/preflight.json",
        rag_api=args.rag_api,
    )
    env = yunwu_environment(profile="micu_slb")
    command = [
        sys.executable, "-m", "agrinet.research.hcv.collector",
        "--plan-file", str(root / "private/rag_plan.jsonl"),
        "--output-dir", str(root / "collection/rag"),
        "--model", args.model, "--rag-api", args.rag_api,
        "--limit", "1000000", "--max-concurrent", str(args.workers),
        "--max-tool-turns", "3", "--top-k", "3", "--teacher-timeout", "300",
        "--image-max-side", "1536",
    ]
    raise SystemExit(subprocess.run(command, cwd=ROOT, env={**os.environ, **env}, check=False).returncode)


def recover_n04094_rag(args: argparse.Namespace) -> None:
    staging = args.staging_root.resolve()
    report = supplement.build_n04094_rag_502_recovery(staging)
    root = Path(report["recovery_root"])
    supplement.rag_service_preflight(root / "private/rag_plan.jsonl", root / "collection/rag/preflight.json", rag_api=args.rag_api)
    env = yunwu_environment(profile="micu_slb")
    command = [
        sys.executable, "-m", "agrinet.research.hcv.collector",
        "--plan-file", str(root / "private/rag_plan.jsonl"),
        "--output-dir", str(root / "collection/rag"),
        "--model", args.model, "--rag-api", args.rag_api,
        "--limit", "1000000", "--max-concurrent", str(args.workers),
        "--max-tool-turns", "3", "--top-k", "3", "--teacher-timeout", "300",
        "--image-max-side", "1536",
    ]
    raise SystemExit(subprocess.run(command, cwd=ROOT, env={**os.environ, **env}, check=False).returncode)


def oracle_collect(args: argparse.Namespace) -> None:
    staging = args.staging_root.resolve()
    root = supplement._oracle_recovery_root(staging, args.oracle_recovery_id)
    supplement.rag_service_preflight(
        root / "private/rag_plan.jsonl", root / "collection/rag/preflight.json", rag_api=args.rag_api,
    )
    env = yunwu_environment(profile="micu_slb")
    command = [
        sys.executable, "-m", "agrinet.research.hcv.collector",
        "--plan-file", str(root / "private/rag_plan.jsonl"),
        "--output-dir", str(root / "collection/rag"),
        "--model", args.model, "--rag-api", args.rag_api,
        "--limit", "1000000", "--max-concurrent", str(args.workers),
        "--max-tool-turns", "3", "--top-k", "3", "--teacher-timeout", "300",
        "--image-max-side", "1536",
    ]
    raise SystemExit(subprocess.run(command, cwd=ROOT, env={**os.environ, **env}, check=False).returncode)


def oracle_review(args: argparse.Namespace) -> None:
    staging = args.staging_root.resolve()
    root = supplement._oracle_recovery_root(staging, args.oracle_recovery_id)
    status_path = root / "collection/rag/run_status.json"
    if not status_path.is_file():
        raise DataError("Oracle RAG collection has no terminal status")
    status = json.loads(status_path.read_text())
    if status.get("status") not in {"completed", "completed_with_unknown_delivery"}:
        raise DataError("Oracle RAG collection is not terminal; refusing review")
    stage = supplement.stage_rag_acceptance(staging, collection_root=root / "collection/rag", plan_root=root)
    report = supplement.write_oracle_recovery_review(staging, recovery_id=args.oracle_recovery_id)
    print(json.dumps({"stage": stage, "review": report, "formal_dataset_modified": False}, ensure_ascii=False))


def stage(args: argparse.Namespace) -> None:
    staging = args.staging_root.resolve()
    roots = [staging]
    roots.extend(sorted(path for path in (staging / "rounds").glob("v*") if path.is_dir()))
    reports = {}
    for root in roots:
        direct_root = root / "collection/direct"
        rag_root = root / ("collection/rag-v2" if root == staging else "collection/rag")
        if (direct_root / "ledger.jsonl").is_file():
            reports[f"{root.name}:direct"] = supplement.stage_direct_acceptance(
                staging, collection_root=direct_root, plan_root=root,
            )
        if (rag_root / "traces/rejected_trajectories.jsonl").is_file():
            reports[f"{root.name}:rag"] = supplement.stage_rag_acceptance(
                staging, collection_root=rag_root, plan_root=root,
            )
    report = {"rounds": reports, "formal_dataset_modified": False}
    print(json.dumps(report, ensure_ascii=False))


def review(args: argparse.Namespace) -> None:
    staging = args.staging_root.resolve()
    recovery_root = staging / "recoveries/n04094-rag-8077-502-v1"
    if (recovery_root / "collection/rag/traces/rejected_trajectories.jsonl").is_file():
        supplement.stage_rag_acceptance(staging, collection_root=recovery_root / "collection/rag", plan_root=recovery_root)
    report = supplement.write_aggregate_review(staging)
    completed_rounds = [
        round_index
        for round_index in range(1, 12)
        if (staging / f"rounds/v{round_index}/collection/rag/run_status.json").is_file()
        and json.loads((staging / f"rounds/v{round_index}/collection/rag/run_status.json").read_text()).get("status")
        in {"completed", "completed_with_unknown_delivery"}
    ]
    if completed_rounds:
        report["final_shortfall"] = supplement.write_final_shortfall_report(
            ROOT, DATASET, staging, completed_rounds=max(completed_rounds),
        )
    print(json.dumps(report, ensure_ascii=False))


def wait_review(args: argparse.Namespace) -> None:
    """Wait for staged collectors, then audit only complete local evidence."""
    staging = args.staging_root.resolve()
    root = round_root(staging, args.round)
    direct_summary = root / "collection/direct/collection_summary.json"
    rag_status = root / "collection/rag/run_status.json"
    while True:
        direct = json.loads(direct_summary.read_text()) if direct_summary.is_file() else {}
        rag = json.loads(rag_status.read_text()) if rag_status.is_file() else {}
        direct_complete = direct.get("delivery_status") in {"complete", "unknown"}
        # An explicit unknown-delivery trace is a terminal, auditable
        # rejection—not a reason to spin forever or silently replay a request.
        # Stage it so the enclosing eight-view image group is rejected, while
        # preserving the provider-ambiguous trajectory on disk.
        rag_complete = (
            rag.get("status") in {"completed", "completed_with_unknown_delivery"}
            and rag.get("delivery_status") in {"complete", "unknown"}
        )
        print(json.dumps({"direct_complete": direct_complete, "rag_status": rag.get("status"), "rag_complete": rag_complete}, ensure_ascii=False), flush=True)
        if direct_complete and rag_complete:
            stage(args)
            review(args)
            return
        if direct.get("delivery_status") == "failed" or rag.get("status") in {"terminated_unknown_delivery", "failed", "preflight_failed"}:
            raise DataError("collection did not complete cleanly; refusing to audit incomplete evidence")
        time.sleep(30)


def merge(args: argparse.Namespace) -> None:
    if not args.approve_review:
        raise DataError("merge requires --approve-review after human review")
    staging = args.staging_root.resolve()
    approval = supplement.require_human_approval(staging)
    decisions = supplement.read_jsonl(staging / "review/image_decisions.jsonl")
    accepted = {str(row["image_sha256"]) for row in decisions if row.get("selected_for_merge") is True}
    existing = supplement.existing_images(DATASET)
    known = {
        str(row["class_code"])
        for row in supplement.read_jsonl(DATASET / "manifests/class_split.jsonl")
        if row.get("class_role") == "known" and row.get("sft_eligible") is True
    }
    if len(known) != 109:
        raise DataError(f"formal v2 manifest must contain 109 SFT-eligible Known classes, found {len(known)}")
    for row in decisions:
        if row.get("selected_for_merge") is True:
            existing[str(row["class_code"])].add(str(row["image_sha256"]))
    missing = {code: 5 - len(existing[code]) for code in sorted(known) if len(existing[code]) < 5}
    if missing:
        raise DataError(f"reviewed staging does not meet the 5-image floor: {missing}")
    direct, rag = supplement.merge_sources(staging, accepted)
    command = [
        sys.executable, "scripts/data/ingest_open_agri_v2_accepted.py", "--dataset-root", str(DATASET),
        "--source", str(DATASET / "vlm_data/historical/canonical/direct.jsonl"), "--source", str(direct),
        "--source", str(DATASET / "vlm_data/historical/canonical/rag.jsonl"), "--source", str(rag), "--replace",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    print(json.dumps({"approval": approval, "accepted_images": len(accepted), "import_exit_code": result.returncode}, ensure_ascii=False))
    raise SystemExit(result.returncode)


def main() -> int:
    args = parser().parse_args()
    if args.command == "plan":
        report = supplement.build_initial_plan(ROOT, DATASET, args.staging_root.resolve(), seed=args.seed, oversample=args.oversample)
        print(json.dumps(report, ensure_ascii=False))
    elif args.command == "plan-replenishment":
        report = supplement.build_replenishment_plan(
            ROOT, DATASET, args.staging_root.resolve(), round_index=args.round,
            seed=args.seed, oversample=args.oversample,
            authorized_replay_codes=set(args.authorized_replay_code),
        )
        print(json.dumps(report, ensure_ascii=False))
    elif args.command == "oracle-plan":
        report = supplement.build_oracle_recovery_plan(
            ROOT, DATASET, args.staging_root.resolve(), seed=args.seed, oversample=args.oversample,
            recovery_id=args.oracle_recovery_id, recovery_codes=set(args.oracle_code) or None,
            allow_oracle_replay=args.allow_oracle_replay,
        )
        print(json.dumps(report, ensure_ascii=False))
    elif args.command == "oracle-collect":
        oracle_collect(args)
    elif args.command == "oracle-review":
        oracle_review(args)
    elif args.command == "oracle-provisional-audit":
        report = supplement.write_oracle_provisional_review(staging_root=args.staging_root.resolve())
        print(json.dumps(report, ensure_ascii=False))
    elif args.command == "oracle-final-export":
        report = supplement.write_oracle_final_export(staging_root=args.staging_root.resolve())
        print(json.dumps(report, ensure_ascii=False))
    elif args.command == "recover-n04094-rag":
        recover_n04094_rag(args)
    elif args.command == "collect-direct":
        collect_direct(args)
    elif args.command == "collect-rag":
        collect_rag(args)
    elif args.command == "stage":
        stage(args)
    elif args.command == "review":
        review(args)
    elif args.command == "wait-review":
        wait_review(args)
    else:
        merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
