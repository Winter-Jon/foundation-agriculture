"""Inspected per-request R1/R2 successors; never replay an unresolved intent."""
import json
from pathlib import Path
from agrinet.rag.e35_ledger import E35Ledger, DeliveryUnresolved
from agrinet.rag.e344_prepare import digest


class SampleLedger:
    def __init__(self, directory, *, work_id, attempt_ordinal=0, intent_limit=9):
        self.directory, self.work_id = Path(directory), work_id
        self.intent_limit = intent_limit

    def call(self, *, kind, key, payload, invoke):
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        parent = None
        for ordinal in range(3):
            directory = self.directory / key_hash / f'R{ordinal}'
            if ordinal:
                # The predecessor is a terminal local failure with no delivered result.
                # A fresh successor is bounded and explicitly linked, not a replay.
                decision = directory / 'recovery-decision.json'
                directory.mkdir(parents=True, exist_ok=True)
                binding = {'predecessor': str(parent), 'predecessor_sha256': digest(parent),
                           'request_key': key, 'round': ordinal,
                           'inspection': 'terminal unknown_delivery result; no delivered response in predecessor'}
                if decision.exists() and json.loads(decision.read_text()) != binding:
                    raise ValueError('recovery_predecessor_changed')
                decision.write_text(json.dumps(binding, indent=2))
            try:
                ledger = E35Ledger(directory, work_id=f'{self.work_id}:{key}:R{ordinal}',
                                   attempt_ordinal=ordinal, intent_limit=1)
                return ledger.call(kind=kind, key=key, payload=payload, invoke=invoke)
            except DeliveryUnresolved:
                path = directory / 'events.jsonl'
                if not path.exists():
                    raise
                events = [json.loads(line) for line in path.read_text().splitlines()]
                intents = [e for e in events if e.get('event') == 'intent']
                results = [e for e in events if e.get('event') == 'result']
                # An interrupted process with no terminal receipt remains unresolved.
                if len(intents) != 1 or len(results) != 1 or results[0].get('status') != 'unknown_delivery':
                    raise
                if ordinal == 2:
                    raise
                parent = path
