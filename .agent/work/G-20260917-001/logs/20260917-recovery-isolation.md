# Recovery and isolation verification
Action: Added inspected R1/R2 successor ledgers, sample locking, reporting, candidate union gates, replenishment selection and tests.
Observation: All six checkpoint label maps match actual entities. All evaluation source paths and pHashes resolve; 14 cross-split near duplicates were already excluded. Forty-one focused/CLI tests pass. Formal encoder pass remains active.
Evidence: EXP-20260917-002, isolation-audit.json, checkpoint-embedded-label-audit.json, test output.
Decision: Continue bounded 32-image pilot after preparation; no need for new authorization. Monetary cost remains unknown pending verified rates.
Resume: STATE latest checkpoint; paid pilot still unstarted.
