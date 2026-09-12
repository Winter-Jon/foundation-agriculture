"""Small E3.5-specific atomic provider ledger.

It deliberately does not reuse E2's contract-bound ledger: the E3.5 public
and private channels have a different immutable contract.  An intent is fsync'd
before dispatch; an interrupted call is therefore unresolved, never replayed.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
import uuid

class DeliveryUnresolved(RuntimeError):
    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message); self.request_id = request_id

def _stamp() -> str: return datetime.now(timezone.utc).isoformat()
def _json(value: Any) -> str: return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

class E35Ledger:
    def __init__(self, directory: Path, *, work_id: str, attempt_ordinal: int, intent_limit: int) -> None:
        if not work_id or attempt_ordinal not in {0, 1, 2} or intent_limit <= 0: raise ValueError("invalid E3.5 ledger binding")
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"
        self.binding = {"event": "binding", "schema_version": "agrinet.e35-ledger/v1",
                        "work_id": work_id, "attempt_ordinal": attempt_ordinal, "intent_limit": intent_limit}
        with self._lock():
            if not self.path.exists(): self._append(self.binding)
            elif self._read()[0] != self.binding: raise ValueError("E3.5 ledger binding changed")
    @contextmanager
    def _lock(self):
        with (self.directory / "ledger.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(stream, fcntl.LOCK_UN)
    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(_json(event) + "\n"); stream.flush(); os.fsync(stream.fileno())
    def _read(self) -> list[dict[str, Any]]:
        try:
            rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
            if not rows or rows[0] != self.binding: raise ValueError
            intents, results = {}, {}
            for row in rows[1:]:
                key = row.get("key")
                if row.get("event") == "intent" and isinstance(key, str) and key not in intents: intents[key] = row
                elif row.get("event") == "result" and isinstance(key, str) and key in intents and key not in results and row.get("request_id") == intents[key].get("request_id"): results[key] = row
                else: raise ValueError
            if any(key not in results or results[key].get("status") != "delivered" for key in intents):
                unresolved = next(intents[key] for key in intents if key not in results or results[key].get("status") != "delivered")
                raise DeliveryUnresolved("E3.5 ledger has unresolved provider intent; automatic replay is forbidden", request_id=str(unresolved.get("request_id") or ""))
            return rows
        except DeliveryUnresolved: raise
        except Exception as exc: raise DeliveryUnresolved("E3.5 ledger is incomplete or corrupt") from exc
    def call(self, *, kind: str, key: str, payload: dict[str, Any], invoke: Callable[[], dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        if kind not in {"generation", "private_audit", "rewrite", "rewrite_audit"} or not key or not isinstance(payload, dict): raise ValueError("invalid E3.5 ledger call")
        digest = hashlib.sha256(_json(payload).encode()).hexdigest()
        with self._lock():
            rows = self._read(); intents = {row["key"]: row for row in rows[1:] if row["event"] == "intent"}; results = {row["key"]: row for row in rows[1:] if row["event"] == "result"}
            if key in intents:
                if intents[key].get("payload_sha256") != digest: raise ValueError("E3.5 request key payload changed")
                return str(intents[key]["request_id"]), dict(results[key]["response"])
            if len(intents) >= self.binding["intent_limit"]: raise ValueError("E3.5 global Micu intent budget exhausted")
            request_id = uuid.uuid4().hex
            self._append({"event": "intent", "key": key, "kind": kind, "request_id": request_id, "payload_sha256": digest, "payload": payload, "time": _stamp()})
            try:
                response = invoke()
                if not isinstance(response, dict): raise ValueError("provider response is not an object")
                _json(response)
            except Exception as exc:
                self._append({"event": "result", "key": key, "request_id": request_id, "status": "unknown_delivery", "error_type": type(exc).__name__, "time": _stamp()})
                raise DeliveryUnresolved("provider delivery is unresolved", request_id=request_id) from exc
            self._append({"event": "result", "key": key, "request_id": request_id, "status": "delivered", "response": response, "time": _stamp()})
            return request_id, response
