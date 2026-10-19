# Experiment procedures

Run the following commands from `funcdroid/` with the repository virtual environment (`../.venv/bin/python`). Supply your own APK directory. Historical experiment directories and screenshots are preserved as original evidence and are not acceptance results for the current code.

## Configuration

Configure `API_KEY`, `BASE_URL`, and `MODEL` in the ignored `.env`. Use the provider's OpenAI-compatible API path. Never commit credentials. Process environment variables take precedence over `.env`; restart after changing configuration. Reasoning settings must be supported by the selected model; the execution role can use its own setting. See `.env.example` and the effective startup diagnostics.

## Offline APK inventory

```sh
../.venv/bin/python scripts/batch_test_apks.py \
  --mode compliance --apk-root /path/to/apks \
  --manifest-only --output-dir experiments/output/compliance-batch
```

This step does not connect a device, install apps, or call a model. It locates `aapt` through SDK configuration or uses `pyaxmlparser`. `manifest.json` records each APK's path, SHA-256, package, version, minimum SDK, and native ABIs. Parse failures remain in `rejected` and cause a nonzero exit status. Device compatibility is checked during live execution.

## Independent live batches

Verify the device ID with `adb devices`. Batch installation clears/uninstalls existing copies of each selected package, installs the requested APK, and cleans it up after testing. Use a dedicated test device and preserve any needed account/data state beforehand.

```sh
../.venv/bin/python scripts/batch_test_apks.py \
  --mode compliance --device-id DEVICE_ID --apk-root /path/to/apks \
  --total-time 30 --limit 1 --output-dir experiments/output/compliance-batch

../.venv/bin/python scripts/batch_test_apks.py \
  --mode defect --device-id DEVICE_ID --apk-root /path/to/apks \
  --total-time 30 --limit 1 --output-dir experiments/output/defect-batch
```

Start with one app and inspect its evidence before expanding the batch. Do not run two queues against the same device. Budgets are per APK. `--process-timeout` overrides the process deadline in minutes, including cleanup considerations. Keep each mode's output, FFG, evidence, and completion state separate.

## Two stages on the same installation

```sh
../.venv/bin/python -m scripts.staged_apk_experiment \
  --apk /path/to/app.apk --device-id DEVICE_ID --require-network \
  --compliance-minutes 5 --defect-minutes 5 \
  --output-dir experiments/output/staged-app
```

The package must not already be installed and the output directory must be new. The runner installs once, completes compliance testing, starts functional testing, and cleans up that installation. Each stage has separate budgets and outputs. Background review can outlast exploration; process deadlines include 300 seconds of finalization allowance. A failed stage prevents the next one.

`--seed-ffg` inherits pages, controls, transitions, resource dependencies, and screenshots from a successfully finalized run for the same package. It resets operation coverage, scenario counts, and completion states while preserving app data. The staged runner guarantees the same installation; direct callers must verify the APK version themselves. The source output directory is not modified.

## Initial app state

- `--reset-app-data` defaults off for defect mode and on for compliance mode. Seeding and resuming preserve data.
- `--grant-permissions` defaults on for defect mode and off for compliance mode. It grants eligible runtime permissions before launch so dialogs do not consume functionality-testing steps.
- Individual permission failures do not abort execution. `run_result.json` records `granted`, `missing`, `failed`, and `unavailable` under `permission_grant_report`.
- Pre-granting changes initial conditions; do not describe functional runs as first-launch consent tests.
- `--require-network` checks for a default device network, not target-backend reachability. Omit it for offline apps.

`experiment.json` records APK identity, code fingerprint, network state, stage status, and metrics. Each stage stores a source ZIP without `.env`. `run_result.json` records seed provenance. Automatic `bugs.json` candidates are not independently reproduced bug counts. Official-notice comparisons require version and evidence review.

## Saved-trace review

```sh
../.venv/bin/python -m scripts.review_compliance_trace \
  --trace RUN/compliance_traces/TRACE_ID/trace.json \
  --rules RUN/compliance_rules.json --output-dir experiments/output/review-new
```

This calls a model and records usage but does not operate the device or overwrite the original conclusion. It is a new review of existing evidence, not an independent device reproduction.

## Failure handling and artifacts

`batch-results.jsonl` records installation, testing, validation, cleanup, and timing. Failed tests or unconfirmed cleanup stop the queue with a nonzero exit status. Incompatible APKs are recorded and skipped, with the batch still returning nonzero.

Resume skips only successful, complete runs with confirmed cleanup. Failed apps restart from a fresh installation; earlier outputs move into `previous-attempts/`. `--no-resume` reruns selected APKs while preserving old evidence. This restarts testing rather than restoring device state.

Both `run_result.status=success` and `artifact_manifest.status=complete` are required for completion. Completion does not establish exhaustive exploration or passing every compliance rule. Inspect `compliance_report.json/md` for compliance and `bugs.json` for functional bugs, along with traces, screenshots, model usage, and logs.

Background queues primarily live in memory, so forced termination cannot guarantee complete offline replay. Device snapshots and account restoration are not automated. Keep these limits distinct from ordinary checkpoint recovery.
