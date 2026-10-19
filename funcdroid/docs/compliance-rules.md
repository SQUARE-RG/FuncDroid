# GUI compliance problem rules

Library version: `funcdroid-compliance-20-v1`.

P01-P20 are the units for scenario generation, verdicts, and counts. Each app version has at most 20 distinct rule-level findings. A directly evidenced relevant manifestation can establish a problem under a rule; all subconditions need not fail together. Missing access, blocked login, or an untriggered target without violation evidence remains undetermined.

Keep compliant, violation, undetermined, and not-applicable results separate. Multiple screenshots or source clauses do not multiply a finding. The historical 57-rule library remains supporting material, not extra current checkpoints. Keep first-launch checks separate from logged-in rechecks; use `--no-reset-app-data` when preserving an existing session. Do not log out, delete accounts, pay, or subscribe.

## Rules and evidence requirements

### P01: Missing public data-collection rules

Inspect first-launch notices and privacy-policy entrypoints; record missing, inaccessible, or unreadable disclosures.

Historical references: R023, R033.

### P02: Incomplete personal-information processing disclosures

Read purposes, methods, categories, uses, retention, and prominent sensitive-data notices.

Historical references: R015, R024, R025, R027, R034, R061, R075, R092.

### P03: Collection or use without consent

Observe default selections, bundled or induced consent, and actual processing after refusal.

Historical references: R014, R028, R035, R058, R091.

### P04: Excessive personal-information collection

Check business-specific registration/profile requirements for unrelated or unnecessary collection.

Historical references: R013, R029, R035, R036, R040, R041, R042, R043, R044, R045, R046, R047, R048, R049, R050, R074, R083, R094.

### P05: Missing separate or guardian consent

Exercise applicable sensitive-data or minor-related paths and inspect the required confirmation.

Historical references: R018, R082, R095.

### P06: Forced, frequent, or excessive permission requests

Trigger and refuse permissions, then observe repeated, bundled, unnecessary, or mistimed requests.

Historical references: R030, R036, R055, R076, R083.

### P07: Basic functionality restricted after refusing unnecessary access

Actually refuse nonessential information/permissions and execute the basic functionality afterward.

Historical references: R028, R036, R039, R074, R093.

### P08: Forced targeted recommendations

Inspect and operate available personalization/recommendation/ad controls; record missing or ineffective opt-outs.

Historical references: R019, R054.

### P09: Incomplete personal-information or third-party inventories

Inspect personal-information and third-party/SDK disclosures for missing access, recipients, or purposes.

Historical references: R025, R027, R063.

### P10: Ineffective account cancellation

Inspect account/security/privacy/help entrypoints and conditions up to safe pre-submission steps; never cancel the account.

Historical references: R031, R038, R084, R096.

### P11: Missing information-management or consent-withdrawal channels

Inspect access, correction, deletion, and withdrawal channels without deleting user data or ending the current session.

Historical references: R031, R035, R038, R084, R096.

### P12: Incomplete complaint or customer-support channels

Inspect help, feedback, about, contact details, and stated response times.

Historical references: R032, R038, R065, R077.

### P13: Unreasonable privacy-policy terms

Quote specific clauses that disclaim responsibility, restrict rights, or expand authorization unreasonably.

Historical references: R026.

### P14: Information windows cannot be closed

Attempt available close controls on launch screens, dialogs, and ads; record missing or ineffective controls.

Historical references: R062, R070.

### P15: Frequent popups or induced redirects

Record repeated interruptions, induced interaction, click targets, and actual unrelated redirects or downloads.

Historical references: R062, R070.

### P16: Improper automatic renewal

Inspect subscription defaults, renewal notices, and cancellation instructions without purchasing or changing a subscription.

Historical references: R073.

### P17: Undisclosed service charges or conditions

Inspect prices, billing periods, additional conditions, and service restrictions without paying.

Historical references: R071.

### P18: Missing permission-use records or notices

Inspect available permission histories and usage-time notices; state the exact paths checked.

Historical references: R080.

### P19: Improper self-start or associated startup

Inspect relevant settings and available execution records. A missing switch is not proof of frequent background starts.

Historical references: R081.

### P20: Missing device-identifier reset

Inspect device/ad identifier and privacy settings for accessible, effective reset mechanisms.

Historical references: R081.

## Official-notice comparisons

Record the app, package, version, original notice text, and relevant P identifiers. The reported version and later versions answer different questions and must be counted separately. A related detected manifestation can be reported as a category-level match under this project's comparison convention; it is not proof that the exact official path was independently reproduced. Describe the actual evidence.

A notice item remains one denominator item even when it maps to multiple P rules. Additional findings are deduplicated by P rule outside the notice's mapped scope; “additional” does not establish that a newer release introduced the problem. Keep original R-rule reports and snapshots, mapping compound findings only where supported by their descriptions.

| Notice category | Related rules |
| --- | --- |
| Information window cannot be closed | P14 |
| Forced targeted recommendations | P08 |
| Improper personal-information collection | P01, P02, P03, P04, P05, P06, P07, P09 |
| Improper personal-information use | P01, P02, P03, P05, P08, P09, P11 |
| Forced/frequent/excessive permission requests | P06, P07 |
| Frequent self-start or associated startup | P19 |
| Improper automatic renewal | P16 |
| Missing public collection/use rules | P01 |
| Frequent unnecessary permission requests | P06, P07 |
| Incomplete SDK listing | P09 |
| Ineffective account cancellation | P10 |

These mappings define comparison scope, not official confirmation of every listed rule. See [active rules](../compliance/rules.json) and [historical references](../compliance/legacy_rules_v1.json).
