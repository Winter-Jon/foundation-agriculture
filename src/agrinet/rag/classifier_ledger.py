"""Durable per-trajectory request accounting, independent of provider transport.

Only public generation/tool payloads belong here. Private audit requests must
use their private directory; credentials belong in the callable, never payloads.
An interrupted intent blocks further requests until external delivery review.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable
import uuid

from agrinet.rag.classifier_distill import validate_contract


class DeliveryUnresolved(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


def _encoded(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestLedger:
    """One directory per image/view; lock spans the call, intent commits first.

    Request keys are stable logical call IDs supplied by orchestration. A
    delivered key returns its recorded result without invoking transport. A
    changed payload, ambiguous delivery, or a truncated response cannot resume.
    """

    def __init__(self, directory: Path, *, contract: dict, image_group_id: str,
                 view: str, channel: str = "public"):
        validate_contract(contract)
        if view not in contract["views"] or not image_group_id:
            raise ValueError("invalid image/view binding")
        if channel not in {"public", "private"}:
            raise ValueError("channel must be public or private")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self.binding = {
            "image_group_id": image_group_id, "view": view, "channel": channel,
            "contract_sha256": hashlib.sha256(_encoded(contract).encode()).hexdigest(),
            "teacher": deepcopy(contract["teacher"]),
        }
        limits = contract["budgets"]
        self.limits = ({"generation": limits["micu_requests_per_trajectory"],
                        "rag": limits["rag_calls_per_trajectory"]}
                       if channel == "public" else
                       {"audit": limits["private_audits_per_trajectory"]})
        with self._locked():
            events = self._read()
            if not events:
                self._append({"event": "binding", **self.binding})
            elif events[0] != {"event": "binding", **self.binding}:
                raise ValueError("ledger is bound to a different image/view/channel/contract")

    @contextmanager
    def _locked(self):
        with (self.directory / "ledger.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with self.path.open(encoding="utf-8") as stream:
                events = []
                for line in stream:
                    if not line.endswith("\n"):
                        raise ValueError("incomplete event")
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("invalid event")
                    events.append(item)
                if not events:
                    raise ValueError("existing ledger is empty")
                if events[0] != {"event": "binding", **self.binding}:
                    raise ValueError("ledger binding changed")
                intents, completed = {}, set()
                for event in events[1:]:
                    key = event.get("request_key")
                    if not isinstance(key, str) or not key:
                        raise ValueError("missing request key")
                    if event.get("event") == "intent":
                        if key in intents or event.get("kind") not in self.limits:
                            raise ValueError("duplicate intent or invalid request kind")
                        if not isinstance(event.get("request_id"), str) or not event["request_id"]:
                            raise ValueError("missing request ID")
                        actual = hashlib.sha256(_encoded(event["payload"]).encode()).hexdigest()
                        if actual != event.get("payload_sha256"):
                            raise ValueError("request payload digest mismatch")
                        intents[key] = event
                    elif event.get("event") == "result":
                        if key not in intents or key in completed:
                            raise ValueError("orphan or duplicate result")
                        if event.get("request_id") != intents[key]["request_id"]:
                            raise ValueError("result request ID mismatch")
                        if event.get("status") not in {"delivered", "truncated", "unknown_delivery", "invalid_response"}:
                            raise ValueError("invalid result status")
                        if event["status"] != "unknown_delivery" and not isinstance(event.get("response"), dict):
                            raise ValueError("missing raw response")
                        completed.add(key)
                    else:
                        raise ValueError("unknown ledger event")
                return events
        except (ValueError, UnicodeError, KeyError, TypeError) as exc:
            raise DeliveryUnresolved("ledger incomplete or corrupt; inspect delivery before recovery") from exc

    def _append(self, event: dict) -> None:
        payload = _encoded(event) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def call(self, kind: str, request_key: str, payload: dict,
             invoke: Callable[[dict], dict]) -> dict:
        if kind not in self.limits or not isinstance(request_key, str) or not request_key:
            raise ValueError("invalid request kind or key for this channel")
        if not isinstance(payload, dict):
            raise ValueError("request payload must be an object")
        serialized = _encoded(payload)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        with self._locked():
            events = self._read()
            intents = {e["request_key"]: e for e in events if e["event"] == "intent"}
            results = {e["request_key"]: e for e in events if e["event"] == "result"}
            for key in intents:
                if key not in results or results[key]["status"] != "delivered":
                    raise DeliveryUnresolved("unresolved or truncated request; automatic replay is disabled")
            if request_key in intents:
                original = intents[request_key]
                if original["kind"] != kind or original["payload_sha256"] != digest:
                    raise ValueError("request key reused with different payload or kind")
                return results[request_key]["response"]
            if sum(e["kind"] == kind for e in intents.values()) >= self.limits[kind]:
                raise BudgetExhausted(f"{kind} request budget exhausted")
            request_id = uuid.uuid4().hex
            self._append({"event": "intent", "request_key": request_key,
                          "request_id": request_id, "kind": kind,
                          "payload_sha256": digest, "payload": payload, "time": _stamp()})
            start = time.monotonic()
            try:
                response = invoke(json.loads(serialized))
                if not isinstance(response, dict):
                    raise ValueError("transport must return a raw response object")
                _encoded(response)
            except Exception as exc:
                # Exception text may contain credentials or URLs; retain its type only.
                self._append({"event": "result", "request_key": request_key,
                              "request_id": request_id, "status": "unknown_delivery",
                              "error_type": type(exc).__name__,
                              "latency_seconds": time.monotonic() - start, "time": _stamp()})
                raise DeliveryUnresolved("request failed after intent; inspect delivery, do not replay") from None
            choices = response.get("choices", [])
            invalid = ("error" in response or not isinstance(choices, list) or
                       any(not isinstance(c, dict) for c in choices))
            truncated = not invalid and any(c.get("finish_reason") == "length" for c in choices)
            status = "invalid_response" if invalid else "truncated" if truncated else "delivered"
            self._append({"event": "result", "request_key": request_key,
                          "request_id": request_id,
                          "status": status,
                          "response": response, "usage": response.get("usage"),
                          "latency_seconds": time.monotonic() - start, "time": _stamp()})
            if status != "delivered":
                raise DeliveryUnresolved("invalid or truncated response retained; trajectory requires review")
            return response
