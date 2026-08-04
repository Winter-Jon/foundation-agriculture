# AgriNet Data Workflows

The Data domain prepares AgriNet samples, invokes teacher-data providers, converts teacher records to student SFT, and validates every versioned boundary. The stable entrypoint is `agrinet data`; callers do not import VLOOM registries.

## Discover experiments

```bash
.venv/bin/agrinet data list
.venv/bin/agrinet data show data-generate-agrinet-vloom-teacher-v1
```

## Local operations

```bash
.venv/bin/agrinet data prepare EXPERIMENT_ID
.venv/bin/agrinet data generate EXPERIMENT_ID
.venv/bin/agrinet data convert EXPERIMENT_ID
.venv/bin/agrinet data validate PATH --kind prepared
.venv/bin/agrinet data validate PATH --kind teacher
.venv/bin/agrinet data validate PATH --kind sft
.venv/bin/agrinet data validate ARTIFACT_DIR --kind bounded
```

The current bounded preparation baseline is discoverable as `data-prepare-agrinet-bounded-contrast-v1`. It reads the current `wiki/base.json` schema, selects four disease and four pest classes with seed 42, and keeps all three negative candidates in the same domain. This is a deterministic migration baseline, not a reproduction of the legacy feature-based hard-negative pipeline whose pair and representative artifacts are absent from the current filesystem.

Every experiment command accepts repeatable `--config-override key=value`. Explicit environment credentials are supported; otherwise the VLOOM adapter decrypts Yunwu credentials through `~/.apikeys/bin/apikey env yunwu`. Decrypted values are never put in YAML, command previews, manifests, or logs.

If GPG reports that the private key is locked, unlock it once in an interactive terminal without printing the secret, then retry the AgriNet command:

```bash
~/.apikeys/bin/apikey env yunwu >/dev/null
```

The VLOOM adapter also loads the existing interactive-shell `proxy_on` definition when no proxy environment is already present. The proxy value is strictly URL-validated and, like credentials, is passed only in child-process memory.

## Local runs

Preview, run in the foreground, or explicitly detach a long operation:

```bash
.venv/bin/agrinet data submit EXPERIMENT_ID --operation generate --dry-run
.venv/bin/agrinet data submit EXPERIMENT_ID --operation generate
.venv/bin/agrinet data submit EXPERIMENT_ID --operation generate --detach
```

Foreground and detached runs both write `manifest.yaml`, `config.resolved.yaml`, and `status.json` under `outputs/runs/data/<experiment-id>/<run-id>/`. Detached runs additionally persist stdout/stderr under `logs/`; foreground output remains attached to the terminal. Data artifacts are separate program outputs defined by the experiment. Existing `slurm/` files are pre-migration history.

## Contracts

- Prepared samples: `agrinet.data.sample/v1`
- Teacher records: `agrinet.data.teacher/v1`
- Student SFT: `agrinet.sft.student/v1`

The VLOOM-specific dependency is contained in `agrinet.data.providers.vloom`. Existing pre-refactor scripts remain temporarily available until production teacher-output conversion is validated; new callers should use `agrinet data`.

VLOOM is installed from `vendor/VLooM`, whose provenance and currently unresolved upstream license status are recorded in `vendor/VLooM/README.agrinet.md`.
