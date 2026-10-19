# GUI compliance screening

This extension is available on the full implementation branch. It dynamically exercises Android app interfaces, including consent dialogs, permission requests, login/registration, privacy settings, user-rights entrypoints, and policy text. It preserves actions and screenshots for per-rule review.

## Scope and evidence

A compliant result is limited to observed, applicable GUI paths. A violation requires direct evidence from visible content, control state, or an executed interaction. Network traffic, background processing, server-side deletion, and long-term operational practices are outside direct GUI evidence. Screening supports investigation and manual review; it does not replace code/network auditing or legal assessment.

The active library uses 20 problem rules, P01-P20, as the unit for scenarios and counts. Any evidenced relevant manifestation may identify a problem under a rule. Missing login access or an unvisited page without violation evidence remains undetermined. Historical 57-rule resources remain available for historical review, not as additional current checkpoints. See [rule definitions](compliance-rules.md).

## Execution and rechecks

Related rules share seven evidence-gathering groups: initial launch and consent, permission refusal, separate consent, registration forms, settings disclosures, user rights, and policy text. A before-action screen, actual action, and subsequent result form the evidence chain. A visible refusal button alone does not establish that refusal works.

Capture defaults use a maximum image side of 1600 pixels and JPEG quality 95, configurable through `SCREENSHOT_MAX_SIDE` and `SCREENSHOT_QUALITY`. Small images are not enlarged. Review and persisted traces reuse the captured image, without applying FFG-specific recompression. Image coordinates map to screen coordinates as `round(x_image / (image_width - 1) * (screen_width - 1))`, similarly for y. Traces retain image/screen sizes and original/mapped coordinates.

The FFG provides navigation context and signals new evidence relevant to deferred checks. Evidence-insufficient checks yield to other work; new related pages, controls, paths, or states can trigger a recheck. Lower-attempt checks receive priority. Historical violations are not silently overwritten by later locally compliant observations.

| Verdict | Meaning |
| --- | --- |
| `compliant` | Observed applicable behavior satisfies the checked requirement. |
| `violation` | Direct GUI or interaction evidence demonstrates a problem. |
| `undetermined` | Required observations are missing, unreadable, or contradictory. |
| `not_applicable` | Positive evidence excludes the rule's applicability. |

Scheduling every rule does not mean all rules have conclusive verdicts. A completed run also does not mean every rule passed.

## Targeted review and outputs

Use `--mode compliance --compliance-targeted --rules PATH` to check only the selected library and stop after those checks receive review, retaining undetermined outcomes. For an already logged-in app, also use `--no-reset-app-data --no-grant-permissions`.

Each run writes `compliance_report.json`, `compliance_report.md`, and referenced evidence under `compliance_traces/`. Inspect `run_result.json` for process status and `artifact_manifest.json` for artifact completeness. See [experiment procedures](experiments.md).

Login barriers, CAPTCHA, identity verification, payments, visually similar states, and model errors can limit coverage. Critical compliant and violation verdicts require manual evidence review. Original experiment data and screenshots retain their source language and are not rewritten by the English source conversion.
