"""Provider-free retrieval ablation over the frozen E3.26 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from agrinet.rag.e324_structured_rag import digest, rows
from agrinet.rag.e326_contract import PEST_KEYS, PROFILE_KEYS, validate_plan
from agrinet.rag.micu_classifier_hcv_v2 import local_rag_health

PROTOCOL = "agrinet.e326-semantic-retrieval-ablation/v1"
ROWS = 28
TOP_K = 8
STRATEGIES = {
    "visual_only": {"query_variant": "empty", "retrieval_type": "visual"},
    "original_rrf": {"query_variant": "original", "retrieval_type": "balanced", "ranker": "rrf"},
    "original_weighted": {"query_variant": "original", "retrieval_type": "balanced", "ranker": "weighted", "text_weight": 0.35, "image_weight": 0.65},
    "profile_rrf": {"query_variant": "profile", "retrieval_type": "balanced", "ranker": "rrf"},
    "profile_weighted": {"query_variant": "profile", "retrieval_type": "balanced", "ranker": "weighted", "text_weight": 0.35, "image_weight": 0.65},
    "morphology_rrf": {"query_variant": "morphology", "retrieval_type": "balanced", "ranker": "rrf"},
    "morphology_weighted": {"query_variant": "morphology", "retrieval_type": "balanced", "ranker": "weighted", "text_weight": 0.20, "image_weight": 0.80},
    "morphology_semantic": {"query_variant": "morphology", "retrieval_type": "semantic"},
}

def _canonical(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").split())

def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"E3.26 ablation immutable output changed: {path}")
    if not path.exists():
        path.write_text(text)

def _registry(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text())
    if not isinstance(value, list): raise ValueError("E3.26 ablation registry invalid")
    result = {str(x["canonical_class_code"]): str(x["english_name"]) for x in value}
    if len(result) < 100: raise ValueError("E3.26 ablation registry incomplete")
    return result

def _state_dir(campaign_root: Path, sample_id: str) -> Path:
    path = campaign_root / "states" / f"R0_{sample_id}_e326"
    if not path.is_dir(): raise ValueError(f"E3.26 ablation state missing: {sample_id}")
    return path

def query_variant(plan: dict[str, Any], variant: str) -> str:
    if variant == "empty": return "visual morphology"
    if variant == "original": return str(plan["query"]).strip()
    keys = sorted(PROFILE_KEYS) if variant == "profile" else ["organ_or_life_stage", "shape_structure", "surface_pattern", "color", "context"]
    values = [str(plan["visual_profile"][key]).strip() for key in keys]
    if plan.get("pest_morphology"):
        values.extend(str(plan["pest_morphology"][key]).strip() for key in sorted(PEST_KEYS))
    query = "; ".join(value for value in values if value)
    if not query: raise ValueError("E3.26 ablation query is empty")
    return query

def _request(endpoint: str, row: dict[str, Any], query: str, spec: dict[str, Any]) -> list[str]:
    body = {"retrieval_type": spec["retrieval_type"], "text": query, "top_k": TOP_K}
    if spec["retrieval_type"] != "semantic": body["image_path"] = row["image_path"]
    for key in ("ranker", "text_weight", "image_weight"):
        if key in spec: body[key] = spec[key]
    request = urllib.request.Request(endpoint.rstrip("/") + "/search", json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response: value = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"E3.26 ablation local RAG failed: {type(exc).__name__}") from exc
    evidence = value.get("evidence") if isinstance(value, dict) else None
    if value.get("schema_version") != "agrinet.rag.search/v1" or not isinstance(evidence, list): raise ValueError("E3.26 ablation RAG response invalid")
    names=[]
    for item in evidence:
        metadata=item.get("metadata") if isinstance(item,dict) else None; name=metadata.get("english_name") if isinstance(metadata,dict) else None
        if isinstance(name,str) and name.strip() and _canonical(name) not in {_canonical(x) for x in names}: names.append(name.strip())
    return names[:TOP_K]

def _evaluate_one(row: dict[str, Any], plan: dict[str, Any], truth_name: str, strategy: str, spec: dict[str, Any], endpoint: str, cache: Path) -> dict[str, Any]:
    query = query_variant(plan, spec["query_variant"])
    forbidden = {_canonical(item["name"]) for item in row["classifier"]["top5"]}
    if row["question_type"] == "open" and any(name and name in _canonical(query) for name in forbidden):
        raise ValueError(f"E3.26 ablation query contains classifier candidate: {row['sample_id']}:{strategy}")
    key = {"protocol": PROTOCOL, "sample_id": row["sample_id"], "strategy": strategy, "query": query, "spec": spec, "image_sha256": row["image_sha256"]}
    key_sha = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if cache.exists():
        value = json.loads(cache.read_text())
        if value.get("request_sha256") != key_sha: raise ValueError(f"E3.26 ablation cache binding changed: {cache}")
        return value
    names = _request(endpoint, row, query, spec)
    truth = _canonical(truth_name)
    rank = next((i for i, name in enumerate(names, 1) if _canonical(name) == truth), None)
    value = {"schema_version": "agrinet.e326-semantic-retrieval-ablation-result/v1", "protocol": PROTOCOL, "sample_id": row["sample_id"], "cell": f"{row['question_type']}/{row['task_domain']}", "strategy": strategy, "query": query, "retrieval_type": spec["retrieval_type"], "ranker": spec.get("ranker"), "returned_standard_class_names": names, "truth_rank": rank, "truth_hit_top8": rank is not None, "request_sha256": key_sha, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    _write_once(cache, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return value

def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if not total: raise ValueError("E3.26 ablation metric group empty")
    return {"rows": total, **{f"recall_at_{k}": sum(isinstance(x["truth_rank"], int) and x["truth_rank"] <= k for x in results) / total for k in (1, 3, 5, 8)}, "mrr": sum(1 / x["truth_rank"] for x in results if isinstance(x["truth_rank"], int)) / total}

def audit(*, source: Path, registry: Path, output_root: Path, output: Path) -> dict[str, Any]:
    if output.exists(): return json.loads(output.read_text())
    data = rows(source); by = {row["sample_id"]: row for row in data}; names = _registry(registry)
    results = [json.loads(line) for line in (output_root / "results.jsonl").read_text().splitlines() if line]
    expected = {(row["sample_id"], strategy) for row in data for strategy in STRATEGIES}
    observed = {(item.get("sample_id"), item.get("strategy")) for item in results}
    occurrences = []
    for item in results:
        truth = names[by[item["sample_id"]]["private"]["truth_code"]]
        if _canonical(truth) in _canonical(item["query"]): occurrences.append({"sample_id": item["sample_id"], "strategy": item["strategy"], "query_provenance": "frozen_e326_planner" if item["strategy"].startswith("original_") else "generated_public_morphology"})
    generated = [item for item in occurrences if item["query_provenance"] == "generated_public_morphology"]
    errors = []
    if len(results) != ROWS * len(STRATEGIES) or observed != expected: errors.append("strategy_sample_coverage")
    if any(item.get("provider_requests") not in {None, 0} or any(item.get(key) is not False for key in ("training_eligible", "training_authorized", "sft_may_start")) for item in results): errors.append("scope_flags")
    if generated: errors.append("generated_query_truth_name_occurrence")
    report = json.loads((output_root / "report.json").read_text())
    value = {"schema_version": "agrinet.e326-semantic-retrieval-ablation-audit/v1", "protocol": PROTOCOL, "results": len(results), "strategy_sample_pairs_unique": len(observed), "frozen_original_query_truth_name_occurrences": len(occurrences) - len(generated), "generated_query_truth_name_occurrences": len(generated), "occurrences": occurrences, "provider_requests": report.get("provider_requests"), "report_sha256": digest(output_root / "report.json"), "results_sha256": digest(output_root / "results.jsonl"), "errors": errors, "ablation_audit_passed": not errors, "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    _write_once(output, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return value

def run(*, source: Path, campaign_root: Path, registry: Path, endpoint: str, output_root: Path, workers: int = 4) -> dict[str, Any]:
    report_path = output_root / "report.json"
    if report_path.exists(): return json.loads(report_path.read_text())
    data = rows(source)
    if len(data) != ROWS or any(row.get("e39_protocol") != "agrinet.e326-structured-rag/v1" for row in data): raise ValueError("E3.26 ablation source invalid")
    names = _registry(registry)
    r0_path = campaign_root / "outcomes/r0.json"
    outcomes = json.loads(r0_path.read_text()).get("outcomes") or []
    failed = {x["sample_id"] for x in outcomes if x.get("disposition") == "future_reject"}
    if len(outcomes) != ROWS or len(failed) != 12: raise ValueError("E3.26 ablation requires authoritative 12-failure campaign")
    plans = {row["sample_id"]: json.loads((_state_dir(campaign_root, row["sample_id"]) / "plan.json").read_text()) for row in data}
    for row in data: validate_plan(plans[row["sample_id"]], row, {x["name"] for x in row["classifier"]["top5"][:3]})
    health = local_rag_health(endpoint)
    if health.get("status") != "ok": raise ValueError("E3.26 ablation local RAG unhealthy")
    tasks = [(row, strategy, spec) for row in data for strategy, spec in STRATEGIES.items()]
    def one(task):
        row, strategy, spec = task
        truth = names[row["private"]["truth_code"]]
        cache = output_root / "cache" / strategy / f"{row['sample_id']}.json"
        return _evaluate_one(row, plans[row["sample_id"]], truth, strategy, spec, endpoint, cache)
    with ThreadPoolExecutor(max_workers=min(max(workers, 1), 4)) as pool: result = list(pool.map(one, tasks))
    result = sorted(result, key=lambda x: (x["strategy"], x["sample_id"]))
    _write_once(output_root / "results.jsonl", "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in result))
    summary = {}
    for strategy in STRATEGIES:
        selected = [x for x in result if x["strategy"] == strategy]
        summary[strategy] = {"all": _metrics(selected), "e326_failures": _metrics([x for x in selected if x["sample_id"] in failed]), "open_pest": _metrics([x for x in selected if x["cell"] == "open/pest"])}
    best = max(summary, key=lambda name: (summary[name]["e326_failures"]["recall_at_8"], summary[name]["open_pest"]["recall_at_8"], summary[name]["all"]["recall_at_8"], summary[name]["e326_failures"]["mrr"], name))
    manifest = {"schema_version": "agrinet.e326-semantic-retrieval-ablation-manifest/v1", "protocol": PROTOCOL, "immutable_inputs": {"source": {"path": str(source), "sha256": digest(source)}, "campaign_r0": {"path": str(r0_path), "sha256": digest(r0_path)}, "registry": {"path": str(registry), "sha256": digest(registry)}}, "rows": ROWS, "strategies": STRATEGIES, "top_k": TOP_K, "provider_requests": 0, "workers": min(max(workers, 1), 4), "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    _write_once(output_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    report = {"schema_version": "agrinet.e326-semantic-retrieval-ablation-report/v1", "protocol": PROTOCOL, "rows": ROWS, "failure_rows": len(failed), "retrieval_calls": len(result), "provider_requests": 0, "rag_health": health, "strategy_metrics": summary, "best_strategy": best, "results_sha256": digest(output_root / "results.jsonl"), "manifest_sha256": digest(output_root / "manifest.json"), "training_eligible": False, "training_authorized": False, "sft_may_start": False}
    _write_once(report_path, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = run(source=args.source, campaign_root=args.campaign_root, registry=args.registry, endpoint=args.endpoint, output_root=args.output_root, workers=args.workers)
    audit_result = audit(source=args.source, registry=args.registry, output_root=args.output_root, output=args.output_root / "artifact-audit.json")
    print(json.dumps({"rows": result["rows"], "retrieval_calls": result["retrieval_calls"], "best_strategy": result["best_strategy"], "provider_requests": 0, "ablation_audit_passed": audit_result["ablation_audit_passed"], "sft_may_start": False}, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
