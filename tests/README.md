# Test Suite Contract

tests/ is the only repository test suite. It is collected by pytest through the project configuration in pyproject.toml.

## Scope

| Location | Purpose | Execution rule |
| --- | --- | --- |
| tests/ | Ignored local automated tests for the agrinet package and supported workflows. | Run with .venv/bin/python -m pytest when the local suite is present. |
| tests/fixtures/ | Local small inputs required by automated tests. | Keep deterministic and minimal. |
| tools/legacy/data-preparation/ | Ignored legacy local data utilities from an earlier project phase. | Never collected or treated as test coverage. Do not run without inspecting inputs and side effects. |
| tools/ | Ignored local executable and diagnostic tools. | Do not infer that a clean clone contains them. |

The six current tools/legacy/data-preparation/ utilities use fixed local dataset or output paths. The
latest static audit identifies three with mutation signals and three that first
need path parameterization review. At least create_index.py and
pack_missing_file.py can move or package files. They are unversioned because
.gitignore excludes tools/ and tests/. Their content and effects are therefore unknown to a
clean checkout; they must not be silently deleted, renamed, or promoted to CI.

## Target organization

The test suite will be grouped gradually without changing test semantics:

    tests/
    ├── unit/          pure functions and contracts
    ├── integration/   CLI, local runtime, adapters, and service boundaries
    ├── regression/    protocol, gate, and reproducibility regressions
    ├── fixtures/      tiny committed data only
    └── helpers/       shared test-only builders and assertions

Existing flat test modules remain supported until they are moved one group at a time with import, collection, and test-name checks. New tests should use the target grouping when the appropriate category is clear.

Until files move, pytest markers provide the executable grouping boundary:

    .venv/bin/python -m pytest -m unit -q
    .venv/bin/python -m pytest -m integration -q
    .venv/bin/python -m pytest -m regression -q

Unit covers deterministic local contracts. Integration covers the CLI, subprocess,
runtime, adapter, and local-service boundaries. Regression covers research data,
protocol, evaluation, and promotion-gate behavior. Each collected test receives
exactly one of these markers; the source files remain flat until a later,
separately validated move.

## Test data and output rules

- Use tmp_path or pytest temporary directories for generated files. Never write to the real outputs/, datasets/, or models/ roots during automated tests.
- A fixture must be small enough to review in Git. Put bulky inputs under local outputs/ and test only their metadata contract through a controlled fixture.
- Network, credentials, GPU, and external service calls must be opt-in and clearly marked. Normal pytest collection must remain local and deterministic.
- Add a regression test when a result protocol, experiment manifest, path contract, or no-leakage gate is repaired.

## Migration of legacy utilities

1. Inventory each ignored tools/legacy/data-preparation/ script: owner, input paths, output paths, dependencies, and destructive behavior.
2. Classify it as a supported data tool, a one-off historical tool, or disposable local scratch work.
3. For a supported tool, parameterize paths, add a dry-run where it mutates data, document it under scripts/data/, and add focused tests.
4. For a historical tool, move a reviewed copy to scripts/archive/ with a README mapping only after confirming no active use.
5. Delete nothing from tools/legacy/data-preparation/ until the user explicitly approves exact paths and any outputs have been audited.

## Verification commands

    .venv/bin/python -m pytest --collect-only -q
    .venv/bin/python -m pytest -q
    .venv/bin/python tools/repository_audit.py --root .

Use a targeted module or keyword while iterating. The full suite is the acceptance gate after a structural migration.
