"""Registered live adapter for the frozen E3.5 audit manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any
from argparse import Namespace

from agrinet.common.credentials import yunwu_environment
from agrinet.rag.e35_cascade_collect import collect_round
from agrinet.rag.e35_private import run_private_parent
from agrinet.rag.e35_transport import transport_image
from agrinet.rag.micu_classifier_hcv_v2 import local_rag_health
from agrinet.rag.micu_classifier_hcv_v2_collect import execute_rag
from agrinet.research.hcv.collector import post_teacher_json

def available_models(environment: dict[str, str], *, timeout: int) -> set[str]:
    request = urllib.request.Request(environment["YUNWU_API_BASE_URL"].rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {environment['YUNWU_API_KEY']}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list): raise ValueError("teacher model capability response is invalid")
    return {str(item.get("id")) for item in rows if isinstance(item, dict) and isinstance(item.get("id"), str)}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rag-endpoint", required=True)
    parser.add_argument("--private-registry", type=Path, required=True)
    parser.add_argument("--teacher-model", default="gpt-5.6-terra")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(args.manifest.read_text(encoding="utf-8"))
    registry_rows = [json.loads(line) for line in args.private_registry.read_text(encoding="utf-8").splitlines() if line.strip()]
    truth_names = {str(row.get("canonical_code")): str(row.get("canonical_english_name") or "") for row in registry_rows}
    if len(truth_names) != len(registry_rows) or any(not name for name in truth_names.values()):
        raise ValueError("E3.5 private registry is malformed")
    e39 = plan.get("protocol") in {"agrinet.e39-hcv-cascade/v1", "agrinet.e310-hcv-cascade/v1", "agrinet.e311-hcv-cascade/v1", "agrinet.e312-hcv-cascade/v1", "agrinet.e313-hcv-cascade/v1", "agrinet.e314-hcv-cascade/v1", "agrinet.e315-option-format-repair/v1", "agrinet.e316-rag-discriminator/v1", "agrinet.e316-rag-discriminator-canary/v1", "agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"}
    if plan.get("round") not in {"R0", "R1", "R2"} or len(plan.get("work_items", [])) > (1038 if e39 else 32):
        raise ValueError("live E3.5 entrypoint permits only the frozen 32-image audit lineage")
    if plan.get("schema_version") not in ({"agrinet.e39-hcv-cascade-manifest/v1", "agrinet.e310-hcv-cascade-manifest/v1", "agrinet.e311-hcv-cascade-manifest/v1", "agrinet.e312-hcv-cascade-manifest/v1", "agrinet.e313-hcv-cascade-manifest/v1", "agrinet.e314-hcv-cascade-manifest/v1", "agrinet.e315-option-format-repair-manifest/v1", "agrinet.e316-rag-discriminator-manifest/v1", "agrinet.e316-rag-discriminator-canary-manifest/v1", "agrinet.e317-all-unknown-rag-audit-manifest/v1", "agrinet.e318-all-unknown-512-rag-audit-manifest/v1", "agrinet.e319-rag-closure-audit-manifest/v1"} if e39 else {"agrinet.e35-cascade-manifest/v1", "agrinet.e35-cascade-manifest/v2"}):
        raise ValueError("live E3.5 manifest schema is unsupported")
    if plan.get("round") == "R0":
        rows_expected = plan.get("source_rows_expected")
        if e39 and not ((plan.get("protocol") == "agrinet.e316-rag-discriminator-canary/v1" and plan.get("audit_only") is True and plan.get("canary_only") is True and rows_expected == 4) or (plan.get("protocol") == "agrinet.e315-option-format-repair/v1" and plan.get("audit_only") is True and rows_expected == 5) or (plan.get("protocol") in {"agrinet.e317-all-unknown-rag-audit/v1", "agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"} and plan.get("audit_only") is True and plan.get("all_simulated_unknown") is True and rows_expected == 32) or (plan.get("audit_only") is True and rows_expected == 32) or (plan.get("audit_only") is False and rows_expected == 1038)):
            raise ValueError("E3.9 R0 requires its 32-image audit or 1,038-image post-gate manifest")
        if e39:
            rows_expected = None
        valid_initial = plan.get("audit_only") is True and rows_expected == 32
        valid_reauthorization = (plan.get("delivery_reauthorization") is True and
                                 plan.get("audit_only") is True and isinstance(rows_expected, int) and 1 <= rows_expected <= 32)
        valid_v9_continuation = (plan.get("schema_version") == "agrinet.e35-cascade-manifest/v2" and
                                 plan.get("continuation") is True and rows_expected in {4, 6, 10})
        if not e39 and not (valid_initial or valid_reauthorization or valid_v9_continuation):
            raise ValueError("live E3.5 entrypoint requires an audit R0 or explicit delivery reauthorization manifest")
    if plan.get("round") in {"R1", "R2"} and any(item.get("attempt_ordinal") not in {1, 2} or not item.get("predecessor_request_id") for item in plan.get("work_items", [])):
        raise ValueError("E3.5 recovery manifest lacks immutable predecessor lineage")
    controls = plan.get("collection_controls") or {}
    if controls:
        if e39:
            expected_cap = 120000 if plan.get("protocol") == "agrinet.e316-rag-discriminator-canary/v1" else (1200000 if plan.get("protocol") == "agrinet.e317-all-unknown-rag-audit/v1" else (8000000 if plan.get("protocol") in {"agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"} else (300000 if plan.get("audit_only") else 8000000)))
            expected_side = 512 if plan.get("protocol") in {"agrinet.e318-all-unknown-512-rag-audit/v1", "agrinet.e319-rag-closure-audit/v1"} else 1024
            required = {"uncached_input_token_cap": expected_cap, "transport_image_max_side": expected_side, "max_rag_searches": 3}
            reserve = controls.get("reservation_uncached_tokens") or {}
            if any(controls.get(key) != value for key, value in required.items()) or int(controls.get("max_public_turns_per_route", 0)) < 8 or reserve.get("generation") != {"direct": 5000, "classifier": 8000, "rag": 12000} or reserve.get("private_audit") != 18000:
                raise ValueError("E3.9 collection controls are not frozen")
        else:
            required = {"uncached_input_token_cap": 200000, "transport_image_max_side": 1024,
                        "max_public_turns_per_route": 2}
            if any(controls.get(key) != value for key, value in required.items()):
                raise ValueError("E3.5 v9 collection controls are not frozen")
            reserve = controls.get("reservation_uncached_tokens") or {}
            if reserve.get("generation") != {"classifier": 2742, "rag": 2775} or reserve.get("private_audit") not in {13542, 20000}:
                raise ValueError("E3.5 token reservations are not frozen")
        selected = {str(item.get("sample_id") or "") for item in plan.get("work_items", [])}
        source_rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in source_rows:
            if str(row.get("sample_id") or "") not in selected:
                continue
            # Materialize only in memory; this catches invalid image bindings
            # before credentials or provider intents are touched.
            # Preflight must use the same deterministic transport view that the
            # provider will receive.  In particular, E3.18 is a 512px budget
            # audit, so materializing a 1024px view here would make the check
            # operationally inconsistent with the immutable manifest.
            transport_image(Path(row["image_path"]), max_side=int(controls["transport_image_max_side"]))
        health = local_rag_health(args.rag_endpoint)
        if health.get("status") != "ok":
            raise ValueError("E3.5 local RAG health preflight failed")
    if args.dry_run:
        print(json.dumps({"ready": True, "rows": len(plan.get("work_items", [])), "teacher_requests": 0,
                          "private_registry_sha256": hashlib.file_digest(args.private_registry.open("rb"), "sha256").hexdigest(),
                          "sft_may_start": False, "manifest": str(args.manifest)})); return 0
    # Resolving the credential happens only after every immutable-input guard.
    environment = yunwu_environment(profile="micu_slb")
    base_url = environment.get("YUNWU_API_BASE_URL")
    if not isinstance(base_url, str) or not base_url.startswith("https://"):
        raise ValueError("E3.5 live teacher endpoint is missing or not HTTPS")
    models = available_models(environment, timeout=min(args.timeout, 20))
    if args.teacher_model not in models:
        raise ValueError("configured E3.5 teacher model is unavailable to this credential")
    def teacher(payload: dict[str, Any]) -> dict[str, Any]:
        return post_teacher_json(base_url.rstrip("/") + "/chat/completions", payload,
            {"Authorization": f"Bearer {environment['YUNWU_API_KEY']}", "Content-Type": "application/json"},
            Namespace(teacher_retries=0, teacher_timeout=args.timeout, teacher_retry_sleep=0.0))
    def audit(row: dict[str, Any], trajectory: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        private = row.get("private") or {}
        code = str(private.get("truth_code") or "")
        if code not in truth_names: raise ValueError("private truth code is absent from frozen registry")
        private_row = {**row, "private": {**private, "truth_name": truth_names[code]}}
        # The controller validates audit-protocol controls after this response is
        # durably recorded; callback exceptions must remain delivery ambiguity.
        responses: list[dict[str, Any]] = []
        def call(payload: dict[str, Any]) -> dict[str, Any]:
            response = teacher(payload); responses.append(response); return response
        value = run_private_parent(private_row, trajectory, call=call, model=args.teacher_model, enforce_protocol=False,
                                   transport_max_side=int(controls.get("transport_image_max_side", 0)))
        work_id = context.get("work_id")
        if not isinstance(work_id, str) or not work_id or context.get("route") != trajectory["route"]:
            raise ValueError("private audit lacks immutable attempt binding")
        return {**value, "private_registry_sha256": hashlib.file_digest(args.private_registry.open("rb"), "sha256").hexdigest(),
                "_provider_response": responses[0]}
    def rag(row: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
        public = {"image_path": row["image_path"]}
        return execute_rag(args.rag_endpoint, public, arguments)
    result = collect_round(manifest=args.manifest, source=args.source, output=args.output, output_root=args.output_root,
                           teacher=teacher, private_audit=audit, rag_search=rag, model=args.teacher_model)
    print(json.dumps({"round": result["round"], "outcomes": len(result["outcomes"]), "sft_may_start": False}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
