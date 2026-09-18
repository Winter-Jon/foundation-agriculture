"""Bound live adapters for the E344 pilot; no private fields enter generation."""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path

from agrinet.rag.e344_assets import load_classifier, predict_top3
from agrinet.rag.e344_prepare import digest
from agrinet.rag.e35_transport import transport_image
from agrinet.rag.e320_private import parse_e320_private_audit
from agrinet.research.hcv.v13_collector import _isolated_micu_request
from agrinet.vlm.full_tool import rag_request
from agrinet.rag.answer_normalization import answer_key, class_correct


class Runtime:
    def __init__(self, bindings, *, rag_endpoint="http://127.0.0.1:8077/search", device="cuda:1", timeout=180):
        self.bindings = bindings
        self.rag_endpoint, self.device, self.timeout = rag_endpoint, device, timeout
        self.lock = threading.Lock()
        self.models = {}

    def teacher(self, payload):
        base = os.environ.get("YUNWU_API_BASE_URL", "").rstrip("/")
        key = os.environ.get("YUNWU_API_KEY")
        if base != "https://api-slb.micuapi.ai/v1" or not key:
            raise RuntimeError("Micu SLB runtime credentials unavailable")
        return _isolated_micu_request(base + "/chat/completions", payload,
            {"Authorization": "Bearer " + key, "Content-Type": "application/json"}, self.timeout)

    def classifier(self, row):
        key = row["bindings"][row["arm"]]
        binding = self.bindings[key]
        with self.lock:
            if key not in self.models:
                for name in ("checkpoint", "label_map", "training_manifest"):
                    if digest(binding[name]) != binding[name + "_sha256"]:
                        raise ValueError("classifier_binding_hash_changed")
                self.models[key] = load_classifier(binding, self.device)
            model, labels = self.models[key]
            return predict_top3(row, binding, model, labels, self.device)

    def retrieve(self, row, args):
        import requests
        payload = rag_request(args, str(Path(row["image_path"]).resolve()))
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(self.rag_endpoint, json=payload, timeout=self.timeout)
            if not response.ok:
                raise RuntimeError(
                    f"rag_http_{response.status_code}:request={json.dumps(payload, ensure_ascii=False)}:"
                    f"response={response.text[:1000]}"
                )
            value = response.json()
        if not isinstance(value.get("evidence"), list) or len(value["evidence"]) > 3:
            raise ValueError("invalid_top3_retrieval_response")
        from agrinet.rag.evidence_quality import partition_evidence
        filtered, excluded = partition_evidence(value)
        if excluded:
            # Fail closed until a versioned retrieval index can replenish valid Top-3.
            # Do not silently shorten, relabel or rewrite the actual evidence.
            raise ValueError("quarantined_retrieval_evidence_requires_index_repair")
        return filtered

    def private_audit(self, row, result):
        image, _ = transport_image(Path(row["image_path"]), max_side=1024)
        messages = []
        for message in result["messages"]:
            item = dict(message)
            if isinstance(item.get("content"), list):
                item["content"] = [part for part in item["content"] if part.get("type") != "image_url"]
            messages.append(item)
        payload = {"model": "gpt-5.6-sol", "temperature": 0, "max_tokens": 512,
            "response_format": {"type": "json_object"}, "messages": [
            {"role": "system", "content": "You are an isolated private auditor. Return JSON with semantic: correct/incorrect/evidence_unavailable and quality: pass/fail. Compare the canonical final English name with private truth, using the approved canonical class and answer aliases. Independently check visible observations, candidate matches/conflicts, actual evidence grounding, nearest alternative, uncertainty, all four options when present, and each search resolving a new gap. Scores/ranks are not visual evidence. Do not require legacy HCV section headings or option letters. Never return corrections or explanations."},
            {"role": "user", "content": [{"type": "text", "text": json.dumps({"private": row["private"],
                "public_options": row.get("public_options", []), "messages": messages}, ensure_ascii=False)}, image]}]}
        raw = self.teacher(payload)
        review = parse_e320_private_audit(raw)
        if not class_correct(result["validation"]["answer"], row["private"]["truth_name"]):
            review["semantic"] = "evidence_unavailable" if result["validation"]["insufficient_evidence"] else "incorrect"
        else:
            review["semantic"] = "correct"
        return {**review, "raw": raw}
