"""Atomic token-authorization accounting for E3.5 live collection.

Provider usage is only known after a delivered response.  This ledger therefore
reserves a conservative uncached-input allowance before dispatch, settles it to
observed usage when available, and deliberately keeps the full reservation for
unknown delivery.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any


class BudgetExhausted(RuntimeError):
    """No new provider intent may be created within the authorized cap."""


class BudgetObservedOverrun(BudgetExhausted):
    """A delivered provider response makes the campaign exceed its cap."""


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class E35TokenBudget:
    """File-backed, process-safe token budget for one immutable campaign."""

    def __init__(self, path: Path, *, uncached_input_token_cap: int) -> None:
        if uncached_input_token_cap <= 0:
            raise ValueError("E3.5 uncached input-token cap must be positive")
        self.path = Path(path)
        self.cap = int(uncached_input_token_cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _lock(self):
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        with lock.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _state(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for event in events:
            key = event.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("E3.5 token budget has an invalid event key")
            if event.get("event") == "reserve":
                if key in state:
                    raise ValueError("E3.5 token budget duplicated a reservation")
                state[key] = {"reserved": int(event["uncached_input_tokens"]), "settled": None}
            elif event.get("event") == "settle":
                if key not in state or state[key]["settled"] is not None:
                    raise ValueError("E3.5 token budget has an invalid settlement")
                state[key]["settled"] = int(event["uncached_input_tokens"])
            else:
                raise ValueError("E3.5 token budget has an invalid event")
        return state

    def reserve(self, key: str, *, uncached_input_tokens: int, metadata: dict[str, Any]) -> None:
        if not key or uncached_input_tokens <= 0:
            raise ValueError("E3.5 token reservation is invalid")
        with self._lock():
            state = self._state(self._events())
            if key in state:
                return
            committed = sum(item["settled"] if item["settled"] is not None else item["reserved"] for item in state.values())
            if committed + uncached_input_tokens > self.cap:
                raise BudgetExhausted(f"E3.5 token budget exhausted: {committed}/{self.cap}")
            self._append({"event": "reserve", "key": key, "uncached_input_tokens": uncached_input_tokens,
                          "metadata": metadata, "time": _stamp()})

    def settle(self, key: str, *, uncached_input_tokens: int) -> None:
        if uncached_input_tokens < 0:
            raise ValueError("E3.5 token settlement is invalid")
        with self._lock():
            state = self._state(self._events())
            if key not in state:
                raise ValueError("E3.5 token settlement has no reservation")
            if state[key]["settled"] is not None:
                return
            self._append({"event": "settle", "key": key, "uncached_input_tokens": uncached_input_tokens,
                          "time": _stamp()})
            # Reservations make concurrent dispatch safe, but a conservative
            # per-call estimate is not itself a campaign hard cap. Once actual
            # usage is known, enforce the authorized total against all other
            # settled values and unresolved reservations.
            updated = self._state(self._events())
            committed = sum(item["settled"] if item["settled"] is not None else item["reserved"]
                            for item in updated.values())
            if committed > self.cap:
                raise BudgetObservedOverrun(f"E3.5 token budget exceeded after settlement: {committed}/{self.cap}")

    def report(self) -> dict[str, int]:
        with self._lock():
            state = self._state(self._events())
        reserved = sum(item["reserved"] for item in state.values())
        settled = sum(item["settled"] for item in state.values() if item["settled"] is not None)
        unknown_exposure = sum(item["reserved"] for item in state.values() if item["settled"] is None)
        committed = settled + unknown_exposure
        return {"uncached_input_token_cap": self.cap, "reserved_uncached_input_tokens": reserved,
                "settled_uncached_input_tokens": settled, "unknown_delivery_exposure_tokens": unknown_exposure,
                "committed_uncached_input_tokens": committed, "remaining_uncached_input_tokens": self.cap - committed}


def uncached_input_tokens(response: dict[str, Any]) -> int | None:
    """Extract provider-reported uncached input tokens without guessing."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details")
    cached = usage.get("cached_tokens", details.get("cached_tokens", 0) if isinstance(details, dict) else 0)
    if not isinstance(prompt, (int, float)) or not isinstance(cached, (int, float)):
        return None
    return max(0, int(prompt) - int(cached))
