"""Bounded, sample-resumable 32-image E344 pilot and report workflow."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agrinet.rag.e344_prepare import rows, digest
from agrinet.rag.e344_pipeline import write, write_rows
from agrinet.rag.e344_runtime import Runtime
from agrinet.rag.e344_collect import collect_sample
from agrinet.rag.e344_export import make_options, export_one, summarize
from agrinet.rag.e35_ledger import DeliveryUnresolved
from agrinet.vlm.full_tool import contract_hashes


def prepare_pilot(config):
    root = Path(config["artifact_root"])
    source = root / "pilot-source.jsonl"
    if source.exists() and (root / "pilot-manifest.json").exists():
        frozen = json.loads((root / "pilot-manifest.json").read_text())
        if frozen.get("rag_binding_sha256"):
            binding_path = root / "rag-binding.json"
            if digest(binding_path) != frozen["rag_binding_sha256"]:
                raise ValueError("rag_binding_changed")
            binding = json.loads(binding_path.read_text())
            if config["rag_endpoint"] != binding["endpoint"]:
                raise ValueError("rag_endpoint_changed")
            import requests
            with requests.Session() as session:
                session.trust_env = False
                health = session.get(config["rag_endpoint"].removesuffix("/search") + "/health", timeout=120)
                health.raise_for_status()
                if health.json() != binding["health"]:
                    raise ValueError("rag_service_identity_changed")
        if digest(source) != frozen["source_sha256"] or frozen["contract"] != contract_hashes():
            raise ValueError("pilot_source_or_contract_changed")
        for path, field in ((root / "pilot-selection.jsonl", "selection_sha256"),
                            (root / "classifier-bindings.json", "bindings_sha256"),
                            (Path(config["registry"]), "registry_sha256"),
                            (root / "isolation-audit.json", "isolation_sha256")):
            if digest(path) != frozen[field]:
                raise ValueError("pilot_frozen_dependency_changed:" + field)
        prepared = list(rows(source))
        for row in prepared:
            if digest(row["image_path"]) != row["image_sha256"]:
                raise ValueError("pilot_image_changed")
            if digest(root / "prepared-cards" / (row["sample_id"] + ".json")) != row["prepared_card_sha256"]:
                raise ValueError("pilot_prepared_card_changed")
            if row["question_type"] == "option" and digest(root / "option-preparation" / (row["sample_id"] + ".json")) != row["option_preparation_sha256"]:
                raise ValueError("pilot_option_preparation_changed")
            for path, sha in row.get("option_supplement_sha256", {}).items():
                if digest(Path(path)) != sha:
                    raise ValueError("pilot_option_supplement_changed")
        return prepared
    if source.exists():
        raise ValueError("pilot_source_exists_without_manifest_inspect_before_recovery")
    selected = list(rows(root / "pilot-selection.jsonl"))
    isolation = json.loads((root / "isolation-audit.json").read_text())
    if isolation["unresolved"]:
        raise ValueError("unresolved_isolation_metadata")
    if {r["image_sha256"] for r in selected} & set(isolation["excluded_training_shas"]):
        raise ValueError("pilot_cross_split_near_duplicate")
    cells = Counter((r["arm"], r["question_type"], r["domain"]) for r in selected)
    full_batch = config.get("collection_mode") == "quota_batch"
    expected = int(config.get("sample_count", 32))
    if len(selected) != expected or (not full_batch and (len(cells) != 8 or set(cells.values()) != {4})):
        raise ValueError("pilot_eight_cell_quota_failed")
    if len({r["image_sha256"] for r in selected}) != expected:
        raise ValueError("pilot_duplicate_image")
    registry = {r["canonical_code"]: r for r in rows(config["registry"])}
    name_to_code = {r["canonical_english_name"].casefold(): code for code, r in registry.items()}
    bindings = json.loads((root / "classifier-bindings.json").read_text())
    runtime = Runtime(bindings, rag_endpoint=config["rag_endpoint"], device=config.get("classifier_device", "cuda:1"))
    prepared = []
    for original in selected:
        row = dict(original)
        sha, code = row["image_sha256"], row["canonical_class_code"]
        row["sample_id"] = "e344-" + sha[:24]
        row["private"] = {"truth_code": code, "truth_name": registry[code]["canonical_english_name"]}
        card_path = root / "prepared-cards" / (row["sample_id"] + ".json")
        if card_path.exists():
            card = json.loads(card_path.read_text())
        else:
            card = runtime.classifier(row)
            write(card_path, card)
        row["prepared_card_sha256"] = digest(card_path)
        if row["question_type"] == "option":
            analysis_path = root / "option-preparation" / (row["sample_id"] + ".json")
            if analysis_path.exists():
                retrieval = json.loads(analysis_path.read_text())
            else:
                retrieval = runtime.retrieve(row, {"retrieval_type": "name", "query": registry[code]["canonical_english_name"],
                    "top_k": 3, "rationale": "Independent private option preparation"})
                write(analysis_path, retrieval)
            similar = []
            for evidence in retrieval["evidence"]:
                meta = evidence["metadata"]
                similar.extend(name_to_code[name.casefold()] for name in meta.get("similar_english_classes", []) if name.casefold() in name_to_code)
            ranked = [x["code"] for x in card["top3"]] + similar + [x["metadata"].get("code") for x in retrieval["evidence"]]
            for mode in ("visual", "semantic", "semantic_neighbours"):
                if len({c for c in ranked + [code] if c in registry}) >= 4:
                    break
                extra_path = root / "option-preparation" / (row["sample_id"] + "." + mode + ".json")
                if extra_path.exists():
                    extra = json.loads(extra_path.read_text())
                else:
                    args = {"retrieval_type": "semantic" if mode == "semantic_neighbours" else mode, "top_k": 3,
                            "query": registry[code]["canonical_english_name"],
                            "rationale": "Independent option preparation: retrieve similar distractors"}
                    if mode == "visual":
                        args["image"] = "query_image"
                    if mode == "semantic_neighbours":
                        args["query"] = registry[code]["canonical_english_name"].split()[0] + " disease pest symptoms differential diagnosis similar conditions"
                    extra = runtime.retrieve(row, args)
                    write(extra_path, extra)
                ranked.extend(x["metadata"].get("code") for x in extra["evidence"])
                row.setdefault("option_supplement_sha256", {})[str(extra_path)] = digest(extra_path)
            options, correct = make_options(code, ranked, registry, sha)
            row["public_options"] = options
            row["private"]["correct_option"] = correct
            row["option_preparation_sha256"] = digest(analysis_path)
        prepared.append(row)
    write_rows(source, prepared)
    write(root / "pilot-manifest.json", {"source_sha256": digest(source), "contract": contract_hashes(),
          "selection_sha256": digest(root / "pilot-selection.jsonl"), "bindings_sha256": digest(root / "classifier-bindings.json"),
          "registry_sha256": digest(config["registry"]), "rows": expected, "max_workers": int(config.get("workers", 4)), "max_protocol_repairs": 1,
          "isolation_sha256": digest(root / "isolation-audit.json"),
          "full_collection_authorized": full_batch, "sft_authorized": False})
    return prepared


def run(config):
    import fcntl
    root = Path(config["artifact_root"])
    lock = (root / "pilot.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("pilot_already_running") from None
    try:
        return _run_locked(config)
    finally:
        lock.close()


def _run_locked(config):
    root = Path(config["artifact_root"])
    sources = prepare_pilot(config)
    output = root / "pilot"
    output.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(json.loads((root / "classifier-bindings.json").read_text()),
                      rag_endpoint=config["rag_endpoint"], device=config.get("classifier_device", "cuda:1"))
    def one(row):
        directory = output / "samples" / row["sample_id"]
        try:
            return collect_sample(row, root=directory, teacher=runtime.teacher, classifier=runtime.classifier,
                                  retrieve=runtime.retrieve, private_audit=runtime.private_audit)
        except DeliveryUnresolved as exc:
            result = {"training_eligible": False, "disposition": "unknown_delivery", "request_id": exc.request_id}
        except Exception as exc:
            # Keep exception classes only: network exception strings can contain secrets.
            result = {"training_eligible": False, "disposition": "execution_failure", "error_type": type(exc).__name__}
        write(directory / "outcome.json", result)
        return result
    workers = int(config.get("workers", 4))
    if not 1 <= workers <= 16:
        raise ValueError("pilot_workers_out_of_range")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(one, sources))
    qualified, lineage, exclusions = [], [], []
    for row, result in zip(sources, outcomes):
        if result.get("training_eligible"):
            try:
                student, provenance = export_one(row, result)
                import base64
                import hashlib
                image_part = next(part for message in result["messages"]
                    if isinstance(message.get("content"), list) for part in message["content"]
                    if part.get("type") == "image_url")
                image_bytes = base64.b64decode(image_part["image_url"]["url"].split(",", 1)[1], validate=True)
                image_path = output / "images" / (row["sample_id"] + ".jpg")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                if image_path.exists() and image_path.read_bytes() != image_bytes:
                    raise ValueError("export_image_changed")
                image_path.write_bytes(image_bytes)
                student["images"] = [str(image_path)]
                provenance["teacher_image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
                provenance["student_row_sha256"] = hashlib.sha256(json.dumps(student, sort_keys=True).encode()).hexdigest()
                qualified.append(student)
                lineage.append(provenance)
            except ValueError as exc:
                result["training_eligible"] = False
                result["disposition"] = "export_integrity_failure"
                result["export_error"] = str(exc)
        if not result.get("training_eligible"):
            exclusions.append({"sample_id": row["sample_id"], "disposition": result.get("disposition", "review_rejected")})
    write_rows(output / "data.jsonl", qualified)
    write_rows(output / "lineage.jsonl", lineage)
    write_rows(output / "exclusions.jsonl", exclusions)
    report = summarize(sources, outcomes, config.get("prices"))
    report["exported"] = len(qualified)
    report["source_sha256"] = digest(root / "pilot-source.jsonl")
    report["contract"] = contract_hashes()
    write(output / "report.json", report)
    from agrinet.rag.e344_report import report as rebuild_report
    report = rebuild_report(root)
    return report


def main():
    import yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())["parameters"]
    if args.prepare_only:
        print(json.dumps({"prepared": len(prepare_pilot(config))}))
    else:
        report = run(config)
        print(json.dumps({"qualified": report["qualified"], "exported": report["exported"]}))


if __name__ == "__main__":
    main()
