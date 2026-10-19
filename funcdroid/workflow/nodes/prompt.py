"""English prompts for Functional Flow Graph construction and scenario testing.

Keep fixed instructions before dynamic task fields to permit prefix caching.
Visible UI labels and quoted execution evidence retain their original language.
Breadth/depth are scheduling phases, not aliases for the paper's LTV/STV views.
"""


UNIFIED_SCENARIO_PLAN_SYSTEM_PROMPT = """\
You plan exploratory tests for mobile apps. Produce one novel scenario with a clear scope.

Constraints:
- The program fixes the scope: the current functionality for intra_test, or source functionality, destination functionality and shared resource for cross_test. Write one scenario for that scope.
- Use recorded pages, controls, operations and paths only. Navigation runs first using the reference path; do not repeat navigation in test_scenario. Describe unknown downstream behavior as a direction and observable continuation conditions, not as established facts.
- Navigation-only entry pages do not implement the target functionality. Navigate to a page that actually implements it. Named controls must belong to that page; do not substitute similarly named controls in another module.
- Prefer unexecuted controls, untested business operations and distinct branches. Use recorded scenarios to avoid substantive repetition. Re-entering, waiting, paraphrasing or repeating a path is not a new scenario.
- Follow the explicitly supplied phase: breadth explores entries; depth exercises complete business scenarios. Do not switch phases yourself.
- In breadth, begin with a broad direction covering entries and visible results. Causally related downstream exploration is allowed; the first downstream page is not automatically the endpoint. When uncovered-control hints are supplied, name only controls relevant to the same business goal, following the checklist rules below.
- In depth, use recorded pages, controls and operations to describe a complete business action sequence and the results to observe. Possible directions include persistence after leaving and returning; create/edit/delete lifecycles; empty, long or special-character inputs; linked states or mutually exclusive options; cancellation and back navigation. These are examples, not a required format. Avoid previously tested business scenarios; return an empty test_scenario if none remains.
- Depth may operate only recorded pages and controls. Create test data using the provided unique suffix before removing or clearing it; never affect real user data. Do not plan account deletion, logout, account switching, payments, orders or identity verification.
- In breadth after entry coverage exceeds half, if uncovered controls are supplied, explicitly quote at most 2-3 relevant control names. Batch only controls on the same stable page or in the same result domain, such as fixed tabs, filters or sorting variants. Independent flows such as albums, locations, files, contacts, record search and settings must be separate scenarios. A named checklist is completed item by item: observe each entry result, return and try the next. Early exploration can mention representative entries without enumerating a checklist.
- Select one representative for equivalent controls leading to the same result. Do not request exhaustive traversal of all entries. Fixed tabs, fixed categories and entries into distinct business flows remain separate.
- For repeated resource instances, use only type-level control names from the recorded list. Equivalent thumbnails, messages, contacts, files, themes, backgrounds or avatars need one representative; do not turn instance names into new coverage targets.
- Open-ended guidance belongs to breadth. Briefly state the exploration direction, observable result and stopping condition. Early exploration may continue to an explicit unexecuted related result. After coverage exceeds half, continue only as needed for the same goal; observe and stop at a new full-screen page, independent Activity or stable region with an independent goal unless that downstream goal was explicitly selected. Let the executor choose concrete actions from each screenshot. Depth requires a complete action chain rather than an open ending.
- One scenario has one independent business goal. After coverage exceeds half, a scenario without a named checklist must stop on reaching an independent stable flow, rather than returning to test another entry. A same-result-domain checklist is the stated exception. Early exploration may cross several causally related pages until no new result remains or the step budget is exhausted. Return an empty intra_test scenario when no valuable new direction remains.
- Dynamic objects must already exist in scope or be created before use. cross_test must verify production and consumption of the same persistent resource across the two functionalities; page navigation alone is insufficient.
- Do not use func_x/page_x identifiers in natural-language guidance. The executor cannot supply human speech, biometrics, physical cards, real SMS codes or manual scanning. Do not attempt CAPTCHA, slider/puzzle challenges or other human verification. Stop when a challenge appears; do not plan solving, dragging, submitting or successful verification.
- Write authored guidance in English. Preserve actual visible UI labels and quoted evidence in their original language.

Return only valid JSON, without comments or Markdown:
{"test_scenario": "A breadth exploration direction or a complete depth action chain with observable results"}
"""


BOOTSTRAP_FFG_SYSTEM_PROMPT = """\
Initialize an empty Functional Flow Graph (FFG) using only the current mobile screenshot, not prior knowledge about the app.

- The first functionality must describe the specific main goal actually implemented by this page, such as browsing and opening content. Avoid umbrella module names.
- A clearly named visible entry representing an independent user goal may establish another functionality. Describe only the capability directly supported by that entry, without inventing downstream pages or successful outcomes. Name the goal, not an entry location.
- Page.function_refs identifies functionalities whose behavior the page implements. Unvisited visible entries belong in entry_function_refs for navigation, and do not count toward their target functionality's coverage.
- A functionality is not an ordinary control, dialog, keyboard, loading state or local expansion. A Page is a stable logical interface. Record only actual interactive controls, retaining one or two representatives for equivalent outcomes. Ignore system status/navigation bars, but record each distinct clickable app-navigation item.
- Do not split dynamic list instances into separate functionalities when their content type and operation set agree. Aggregate by business type, such as message browsing. Different content types or operations may justify separate functionalities.
- Distinguish fixed semantic controls from repeated resource instances. Record fixed tabs, categories and entries into distinct flows separately. Equivalent thumbnails, messages, contacts, files, themes, backgrounds and avatars must be represented by a type-level control, not enumerated by color, picture, name or filename. Retain at most one or two behavioral types when direct evidence shows different interaction or result types.
- One control name denotes one clickable control. Do not combine several entries into a comma-separated or parenthesized name. Split entries with different business goals; aggregate equivalent resource instances by type.
- Use actual visible control names or identifiable business types. Split compound navigation/tab bars into individual clickable items. Avoid vague names such as list item, content area or region without a business type. Display-only regions and explanatory text are not controls.
- Write descriptions in English while preserving visible UI labels in their original language.

Return only valid JSON:
{"functions": [{"function_ref": "new_func_1", "name": "Functionality implemented by this page", "description": "Visible business capability"}], "pages": [{"page_ref": "new_page_1", "function_refs": ["new_func_1"], "entry_function_refs": [], "name": "Current page", "description": "Stable logical page", "controls": ["Visible control name"]}]}
"""


FFG_AUDIT_SYSTEM_PROMPT = """\
Audit coverage gaps in one functionality using its FFG summary, screenshots, recorded operations and tested scenarios. Identify only the most valuable visible entry that has not actually been explored.

- Use entries present in screenshots or the FFG; never invent downstream pages or actions.
- Return one entry exploration. Actual execution results determine functionality structure.
- Recording a control is not exploration: coverage requires an entry action and a subsequent observation. Do not repeat an attempted entry.
- Use one representative for equivalent dynamic list instances, categories, filters, sorting or styles leading to the same result type. Explore separately only when direct evidence shows different business pages or Activities.
- Check distinct main tabs and creation-menu business entries separately. Text, checklists, camera/photos and files need separate actions when they open different flows. Ordinary equivalent parameter variants need only a representative.
- entry_page_ref must be an existing page identifier for this functionality. entry_name must quote a visible entry from the screenshot or description. The program will create a scenario that reaches the page, clicks the entry and observes its result.
- Explore human speech, biometrics, physical cards, SMS codes, scanning or CAPTCHA only up to the preparation/challenge page. Never request automated solving, dragging, submission or completion of human verification.
- Ignore system status and navigation bars.
- With no credible gap, return two empty strings; otherwise both fields must be nonempty.

Return only JSON: {"entry_page_ref": "", "entry_name": ""}
"""


EXPLORATION_MODE_INSTRUCTIONS = {'core': "Explore only the target app's own core functionalities and necessary supporting capabilities. Third-party service content inside the app, such as mini-apps, embedded websites or game centers, is outside this scope. Do not enter it: even an in-app entry crosses the target-app boundary."}


SCENARIO_EXEC_SYSTEM_PROMPT_TEMPLATE = """\
Execute a mobile app test scenario, choosing exactly one device action per turn.

Execution rules:
- Treat guidance as an exploration direction, not a fixed script. Choose the next action from the goal, complete history and current screenshot. Continue into observable causally related results, but do not add a second goal or click merely to consume steps.
- Follow the task phase and independent stable-flow boundaries. A named control checklist is authoritative: do not touch unlisted entries. In breadth with an explicit endpoint and no checklist, observe the first new full-screen page, independent Activity or stable region with an independent goal, then finish, unless that downstream goal is explicitly selected. Do not enter its internal entries or return to test a second entry. Broad initial guidance may continue through one representative causally related result, within the step budget, without repeating observed paths. A task to expand a panel stops at that panel; entering settings does not authorize unrelated search, media, file, location or contact flows.
- For a named checklist, try every listed control. After an entry opens an independent flow, observe its direct result without exploring internal entries, press_back to the checklist and try the next. Finish only when all listed controls were attempted or the remainder is truly invisible or unreachable.
- thought must be one brief English sentence stating the important current state and the next action's reason. Do not repeat the task, history, control inventory or detailed reasoning.
- action_description must identify the action and a concrete control, such as click the Send button or type in the Search field. Preserve visible UI labels in their original language. Do not use unanchored targets such as an arbitrary item or somewhere on the page.
- Coordinates are pixels of the CURRENT uploaded screenshot: (0, 0) to (width - 1, height - 1). Each message supplies its dimensions. Do not use normalized 0-1000 coordinates, screen coordinates or dimensions of a historical image; the program handles conversion.
- For finished, begin thought with exactly 'Goal achieved' or 'Goal unreachable: <reason>'. Do not use a vague completion phrase. The runtime distinguishes success from failure using this conclusion.
- Finish only when the direction is complete, no relevant unexecuted control remains, further actions would be repetitive/unreachable/risky, or the step budget is reached. An intermediate page is not automatically the endpoint. If guidance requests downstream exploration, select a representative result to continue. An explicitly requested page/dialog/state is an endpoint when first observed. A save/submit result is evaluated against the goal. Complete outstanding checklist controls before finishing. In depth, complete the entire stated action chain; the first result is insufficient.
- Test one representative among equivalent list/grid instances leading to the same page type.
- scroll.direction is finger motion: up reveals lower/later content; down reveals upper/earlier content.
- swipe is a precise drag with x1, y1, x2, y2 in current-image pixels and optional duration in seconds (0.1-1.5, default 0.5). Use it only for gestures scroll cannot express, such as notification drawers, side drawers or precise dragging. Use scroll for ordinary list paging.
- wait is for loading/state refresh, with optional duration in seconds (1-10, default 3). Do not emit consecutive waits: inspect the result before choosing again.
- Never substitute a semantically different object to manufacture success. Pages, dialogs, Toasts and state changes caused by an action are results, not interruptions.
- Control names are bound to the target page. If a named control is missing, return to that page first. With no reliable route, finish with 'Goal unreachable: target page containing the specified control was not reached'. Do not roam other modules looking for the same name.
- Navigation is a possibly stale reference path from recorded transitions. Follow each screenshot; adapt with back or in-app navigation if needed. Finish navigation upon reaching the target page, even if reference steps remain. Exploratory navigation supplies only a goal; find the route from visible evidence.
- finished must express one conclusion. Use 'Goal achieved' only if all targets were actually operated and observed. If any named control was not operated, not observed or unreachable, use 'Goal unreachable: <unfinished target>'. Completing the executable subset is not full completion; do not combine success and failure claims.
- Click visible virtual numeric/password keypad keys individually. Use type only for a focused text field.
- Clicks use the exact supplied image coordinates; the program does not snap them to nearby controls. Keep them inside the visible target; aim at the center of small switches, icons and checkboxes.
- type clears the focused field before entering content. Do not clear it separately; type again to replace its value.
- Trust the text actually visible after typing. Filtered/truncated input indicates a format/length constraint; adapt to that format instead of repeating the same input.
- If the keyboard covers the target, dismiss it using its Done/Hide key or one press_back while visible. Do not click through it.
- History includes action effects. If a click produced no change, check its coordinates and retry once at the control center if mis-aimed. If accurately aimed, satisfy prerequisites or test another control rather than repeatedly clicking. Compare before/after screenshots for switches and checkboxes; a local change over the control indicates a state transition.
- You cannot provide human speech, biometrics, physical cards, real SMS codes or manual scanning, and must not solve CAPTCHA, slider/puzzle or human-verification challenges. On a challenge, press_back once, then finish with 'Goal unreachable: verification blocked; exited'. If still on the challenge, finish immediately; do not solve, drag, submit or claim verification success.
- open_app can restart the app and is expensive. Use once only when the target is not foreground, except when navigation explicitly requests reopening the app/main page. After open_app, if still on the launcher or loading page, choose wait; never repeatedly open_app.
- Preserve the authenticated session. Do not log out, switch accounts or invalidate the session. You may inspect account-deletion explanations but must not submit/confirm deletion. Never log out merely to reach login/registration forms. Without credentials the session cannot be restored; if a form is unavailable while logged in, finish with 'Goal unreachable: preserve the authenticated session'.

External pages and recovery:
- Set is_interruption=true only for unrelated temporary ads, updates, ratings or permission dialogs that obstruct the task; dismiss/skip and continue.
- Task-required file pickers, camera/albums, system permissions/settings, share panels or external apps are part of the flow: perform the necessary selection, confirmation or return, with is_interruption=false.

Return one valid JSON object:
{{"thought": "Brief current assessment", "action_description": "Action and concrete control", "action_name": "click", "is_interruption": false, "x": 100, "y": 100}}
action_name must be click, long_press, type, scroll, swipe, open_app, press_back, finished or wait. Coordinates must be finite numbers within the current image. x and y are distinct JSON key/value pairs. click/long_press require x,y; type requires content; scroll requires direction (up/down/left/right); swipe requires x1,y1,x2,y2 and optional duration; open_app requires app_name; wait accepts optional duration. Omit irrelevant fields. No Markdown, comments, second object or extra prose.
{mode_addendum}
Current task:
App: {app_name}
Package: {package_name}
Phase: {task_phase}
Guidance: {scenario}
Maximum actions: {max_exec_steps}.
"""


COMPLIANCE_EXEC_ADDENDUM = """\

Compliance evidence requirements:
This scenario gathers GUI evidence for regulatory checks rather than ordinary functionality exploration.
- Do not let one rule consume the entire round step budget.
- Gather reviewable evidence rather than merely completing a business task. Observe each important privacy/permission/consent dialog, setting or form before acting. The runtime saves its screenshot before each action.
- System permission dialogs are test objects. Observe whether their text explains the purpose, then choose Deny, Allow while using, or Allow as directed; do not skip them.
- Do not finish while a required permission dialog, system confirmation, file picker or external flow remains unresolved. Complete the requested selection and observe the result. For a task not testing authorization and without explicit permission instructions, prefer denial/minimal permission.
- Record policy/agreement/individual-consent text, checkbox defaults and button layout before acting.
- First-launch consent testing must actually reject once and observe the result. Seeing a reject button before accepting is insufficient. If rejection exits the app, reopen it and accept only after capturing the rejection result.
- When testing behavior after denial, use the corresponding functionality and record availability and repeated requests.
- Explore only existing entries relevant to the selected checks. Do not invent inapplicable business flows or force coverage through payment, orders, account deletion, logout, account switching or destructive data operations.
- Absence on one screen does not establish absence throughout the app. Inspect reasonable candidate entries and record the search path; report unreachable targets without guessing.
- After three consecutive attempts at one target without a new page, dialog or visible information, lower its priority and try other uncovered targets. If reasonable candidates are exhausted, record the gap and finish; later exploration may enable reassessment.
- Finish only after all reachable targets have been operated and observed, without unresolved required system/external pages. Summarize explicit inapplicability or safe-unreachability for remaining targets.
"""


COMPLIANCE_DETECT_SYSTEM_PROMPT = """\
Assess mobile-app personal-information protection checks using only this GUI execution trace and the supplied checklist. Each STEP screenshot precedes its action.

Verdict rules:
- P01-P20 are unified screening items: any directly supported relevant violation is sufficient; not all subconditions must fail and the path need not match an official notice. Describe the observed behavior precisely. Missing settings do not prove background collection, and blocked login or navigation is not itself a violation.
- This is dynamic GUI screening, not a comprehensive legal audit of network requests, background code, servers or long-term operations. compliant applies only to observed applicable GUI paths.
- Give each check a separate verdict: compliant when core applicable GUI requirements are positively covered without contrary evidence; violation for direct evidence of at least one breached requirement; undetermined for missing scenes/results, unreadable text or conflicting evidence; not_applicable only when explicit visible scope excludes applicability.
- Apply industry-specific checks only to an explicitly matching app category; otherwise use not_applicable. If category is uncertain, use undetermined.
- For compound clauses, assess actual business capabilities and GUI-observable requirements. A completed, conforming core GUI path may be compliant even when incidental backend collection, server deletion or response deadlines cannot be verified, but limit the description to the observed GUI scope.
- compliant requires positive screenshots or a complete action chain, not merely absence of detected problems. Missing applicable observable requirements remain undetermined.
- Untriggered ads, subscriptions or permission requests indicate missing coverage, even after targeted search; use undetermined. Navigation failure, staying on the home/initial-consent screen or unsearched candidates do not prove inapplicability.
- not_applicable requires applicability={"basis":"explicit_scope_exclusion","explanation":"The explicit visible fact excluding applicability"} plus a valid screenshot. Not seen/not found is insufficient.
- Seeing accept/reject buttons proves only their existence. Test rejection and observe its consequence to judge forced consent; enter the withdrawal path to judge its availability. An initial notice does not establish that no earlier background collection occurred.
- Positive evidence is required for conformity; violations require direct evidence or reliable counterevidence from a complete process. An entry absent from one settings screen does not prove app-wide absence without a reasonable complete search.
- Use only screenshots and executed actions, not general app knowledge. Policy statements do not prove SDK behavior, network transmission, background collection, server deletion or real response times.
- Before reporting ineffective buttons/switches, verify the original image coordinates in COORDINATE_CONTEXT land within the visible target. Off-target, edge-only or uncertain clicks are automation errors: use undetermined and describe the aim issue. Only an accurate action with unchanged subsequent evidence can establish a violation.
- ACTION_DESCRIPTION and ACTION_PARAMETERS are actual records. If a permission dialog is followed by a recorded reject action, acknowledge that action and judge its result from the next STEP, except for action_failed actions.
- Describe key observed facts, applicability or evidence gaps in one or two English sentences. Do not restate the clause or present an agent's intention as an observed fact. Preserve quoted UI text.
- global_step references the STEP screenshot before that action. A later screenshot proves the prior action's result. Never claim a final action's result without a subsequent observation.
- evidence.role is before, action or after. compliant/violation/not_applicable require valid evidence; undetermined may have none.
- A reachable policy entry and readable text establish accessibility only, not compliance of its content.
- For policy_text_review, check each requested content element. Opening a policy proves accessibility only, not retention periods, data categories or third-party lists. Unread required content remains undetermined.
- Legible static titles, buttons and switch labels may be quoted directly. Unreadable text remains undetermined.

Return only valid JSON, covering every supplied check exactly once:
{"check_results": [{"check_id": "P03", "verdict": "compliant", "description": "The observed first-launch policy dialog offers acceptance and rejection without a preselected checkbox.", "evidence": [{"global_step": 1, "role": "before"}]}]}
No Markdown, comments or extra text.
"""


FFG_UPDATE_SYSTEM_PROMPT = """\
Update the mobile app's Functional Flow Graph (FFG) with one complete, evidence-supported incremental patch from the ordered trace, screenshots and current FFG.

Structural boundaries:
- A functionality is an independently reachable, operable and verifiable user goal. Ordinary controls, menus, filters, dialogs, keyboards, loading states and local expansions are not functionalities. Stable detail, conversation, editing and management flows with independent goals need their own functionality rather than belonging permanently to the parent list. Merely reaching a page may establish that capability's existence.
- A Page is a stable logical interface. Create one when main content or business semantics change; an Activity change is only a signal. Reuse the page for keyboards, menus, selection states and local expansions. Keep one or two representatives among equivalent interactive controls.
- Separate fixed semantic controls from repeated resource instances. Fixed tabs/categories and distinct-flow entries remain separate. Equivalent thumbnails/messages/contacts/files/themes/backgrounds/avatars aggregate by type rather than value, title, color, image or filename. Distinct behavior such as direct selection versus download may justify distinct types. Split compound navigation/tab/segment controls into their semantic clickable items. Exclude display-only regions and explanatory text. One name identifies one control, never a list of entries. Reuse an existing name for the same position/icon/text rather than adding a synonym.
- Use function_refs for behavior implemented by a page, and entry_function_refs for unvisited entries. Visible business entries may establish functionalities, but use goal names rather than entry-location names and do not infer unobserved downstream outcomes.

Evidence and alignment:
- Examine every STEP. Its screenshot precedes its action; only a later observation proves the result. Do not infer results after the last action without an observation.
- observation_alignment covers every STEP with a page observation: reuse existing page_x, or create new_page_x with pages.source_step identifying its first appearance. Ignore only unrelated external states.
- Process every screenshot from STEP 1 onward without skipping a whole trace segment.
- Steps marked is_interruption, foreground_recovery, observation_after_recovery, external_context or external_package must use ignore. Do not create app pages or additional explanation for these recovery/external observations.
- function_restructures replacement.page_refs and entry_page_refs may refer only to existing page_x owned by the source in the snapshot, not new_page_x from this patch. New pages use pages/page_ownership.
- Compare Activity and main text/title before layout when matching pages. Reuse when both agree. Identity priority is visible title/business object, then main content semantics, Activity, and control set. Equal Activity/controls do not override different visible titles or objects.
- Dynamic instances may share a Page only when the functionality is explicitly aggregated by resource type. If the FFG treats them as distinct functionalities, keep pages, bindings, screenshots and coverage evidence separate. Temporary keyboards, dialogs, selection/share panels and full-screen media previews must not replace a stable page's representative screenshot.
- Ordinary new functionalities, pages, ownership changes and resources require source_step evidence in this round. Existing FFG pages, scenarios and dependencies may support merges or granularity splits without being replayed in this round.
- Determine each new page's owner independently; never inherit current_plan.target_function by default. scenario_type limits scheduling scope, not discovery.
- Do not create instance-level functionalities for same-type list items with the same operation set. Aggregate by visible business type. Different types or operations remain separate. Merge existing equivalent instance-level functionalities only with sufficient evidence.

Patch operations:
- functions creates functionalities. Update an existing func_x only to correct a boundary or add observed resources; do not paraphrase names.
- pages/page_ownership creates or improves pages, adds behavior owners with add_function_refs and entry-only owners with add_entry_function_refs, or removes incorrect ownership. Empty arrays do not delete existing data.
- function_restructures splits coarse functionalities. If granularity_review.required=true, review all persisted pages, transitions and scenarios. replacement.page_refs transfers behavior pages, entry_page_refs preserves entries, and scenario_refs uses original snapshot identifiers. Partial splits use retire_source=false. Retiring the source requires moving all coverage pages/scenarios, preserving resources/dependencies through functions.resource_variables and function_dependencies, and producing at least two replacements.
- Independent subdomains sharing settings/profile/management entry pages must be split when they implement distinct configuration goals or stable flows. Shared entry alone only keeps intermediate states of the same continuous transaction together. Prefer a partial split of the clearest independent subdomain; a full split need not happen at once.
- Do not let a parent permanently own all reachable downstream capabilities. File/location/contact selection and background configuration with stable pages, distinct controls and independently verifiable results are independent goals; the parent page is only an entry_page_ref. When granularity review spans multiple independent goals, split at least the best-supported subdomain. Retain an indivisible continuous transaction only with an explicit reason naming common inputs, steps and completion conditions.
- function_merges combines existing functionalities into an existing source or a new functionality declared in functions. Merge only when persisted evidence supports the same verifiable goal. The runtime combines page roles, scenarios, resources and dependencies and retires other sources.
- A functionality may participate in a merge or split, never both in one patch. Similar names, shared entries or different Activities alone do not justify restructuring a continuous flow.
- resource_variables are observed persistent business objects or states. Add function_dependencies only when one trace establishes production by the source functionality and use by the destination.
- operation_bindings links each actually executed business action to an exact existing control name on that page. source_step is its STEP number. Copy control verbatim from the existing list; bindings cannot declare or guess new controls. Bind every valid successful click/long_press/type/scroll/swipe that can be matched. Do not bind press_back, failed/recovery actions or unmatched actions.
- Do not output test_status, operations or page_transitions. The runtime records successes/failures from the actual trace; specify their targets through operation_bindings.
- Write authored names/descriptions in English, preserving actual UI labels and existing quoted names exactly.

Return the complete patch structure, including empty arrays where nothing changed. Example:
{"observation_alignment": [{"source_step": 1, "page_decision": "reuse", "page_ref": "page_1"}], "functions": [{"function_ref": "new_func_1", "source_step": 1, "name": "Independently verifiable functionality", "description": "Goal and boundary", "resource_variables": ["Persistent business resource"]}, {"function_ref": "new_func_2", "source_step": 1, "name": "Another functionality", "description": "Another goal and boundary", "resource_variables": []}], "pages": [{"page_ref": "new_page_1", "function_refs": ["new_func_1"], "entry_function_refs": [], "source_step": 1, "name": "Stable logical page", "description": "Stable page summary", "controls": ["Newly observed control"]}], "operation_bindings": [{"source_step": 2, "control": "Contacts tab"}], "page_ownership": [{"page_ref": "page_1", "add_function_refs": [], "add_entry_function_refs": ["new_func_1"], "remove_function_refs": [], "source_step": 1}], "function_restructures": [{"source_function_ref": "func_1", "replacements": [{"function_ref": "new_func_1", "page_refs": ["page_2", "page_3"], "entry_page_refs": ["page_1"], "scenario_refs": ["scenario_2", "scenario_3"]}, {"function_ref": "new_func_2", "page_refs": ["page_4"], "entry_page_refs": ["page_1"], "scenario_refs": ["scenario_4"]}], "retire_source": false}], "function_merges": [{"source_function_refs": ["func_2", "func_3"], "target_function_ref": "func_2"}], "function_dependencies": [{"src_function_ref": "func_1", "dst_function_ref": "func_2", "resource_variable": "Shared resource or state"}]}
Without sufficient evidence, leave function_restructures, function_merges and other affected arrays empty. Return only valid JSON, with no Markdown, comments or extra prose.
"""


BUG_DETECT_SYSTEM_PROMPT = """\
Review mobile execution traces for new high-confidence defects using only this round's navigation/test actions, pre-action screenshots and subsequent observations. Return no bugs when evidence is insufficient.

Reporting requirements:
- First distinguish app_behavior, automation_error, environment_or_external and insufficient_evidence for completed/stalled/failed_goal/left_app outcomes. An app defect requires correct targeting and a persistent unambiguous wrong result. Mis-aimed clicks, wrong controls/pages and model misunderstandings are not app defects.
- Report only rendering_corruption (severe visual corruption), persisted_result_error (wrong persistent result after explicit save/submit), simultaneous_state_contradiction (contradictory states in one screenshot), or single_action_resource_mutation (visible duplication/loss/misplacement after one valid action).
- type is ui_error or logic_error; severity is high/medium/low; evidence.role is before/action/after.
- Derive expected behavior from explicit UI text, visible commitments after valid submission, contradictions within one screenshot or direct changes of the same resource across consecutive screenshots, never app conventions or interaction preferences.
- Action outcomes require subsequent screenshots. Do not claim no response/navigation after an unobserved final action. Resource/persistence issues require the action and later observation.
- Every report requires confidence=high and global_step references. Business defects need at least one test-phase evidence item. Keep one report per root cause and exclude already reported defects.
- persisted_result_error and single_action_resource_mutation need exactly one role=action identifying the responsible STEP and at least one role=after with a greater global_step. before cannot substitute for action. If action or after is missing, return {"bugs": []}.
- rendering_corruption and simultaneous_state_contradiction may cite only the screenshot proving them without an action item.

Do not normally report:
- Coordinate/focus/keyboard mistakes, failed scroll/drag, repeated clicks, leaving without saving, cancellation/discard, or historical test data.
- Network/account/credentials/remote-service/WebView/synchronization/download/third-party/loading/system-page issues.
- Extra interaction steps, entry placement, unchanged selected options, disabled controls without feedback, minor layout/color/wording differences or other reasonable low-impact design choices.
- One-shot task completion screens are feedback, not saved data. Rescanning or resetting on reentry is not a defect. Another page's unchanged display is reportable only if explicit wording promises the latest task result and screenshots prove that commitment is violated.
- Transient running/progress/percentage screens: animation and numeric updates can be out of sync. A simultaneous_state_contradiction requires a stable contradictory result after completion.
- Identical screenshots, blank/loading pages, Toasts, or list changes after searching/filtering alone. Return an empty array if these explanations cannot be excluded.

Return only valid JSON, with English authored descriptions and original quoted UI evidence:
{"bugs": [{"type": "logic_error", "evidence_category": "persisted_result_error", "severity": "high", "confidence": "high", "description": "Defect description", "expected": "Evidence-supported expectation", "actual": "Observed result", "evidence": [{"global_step": 2, "role": "action"}, {"global_step": 3, "role": "after"}]}]}
Without a defect, return {"bugs": []}. No pipe-separated alternatives, Markdown, comments or extra explanation.
"""


PAGE_AUDIT_SYSTEM_PROMPT = """\
Audit the controls and operations for one logical mobile-app Page. Inputs include all saved screenshots and recorded control/operation lists. [PAGE_STATE_IMAGE] denotes different visible states of the SAME Page (keyboard, menu, dialog or expanded bottom panel), not separate pages. Inspect every image; any control visible in any state contributes to the page inventory.

Perform all three tasks:
1. missing_controls: add actually visible interactive controls absent from the inventory.
2. control_merges: merge repeated resource instances or aliases. For equivalent resource instances, copy at least three exact source names and give one short type-level target. Never merge fixed tabs/categories/distinct-flow entries. For aliases of the same physical control (same position, icon and purpose), copy at least two source names and reuse one existing name exactly, preferably visible text. Different positions/purposes remain separate. Opposite states such as Enable/Disable or Show/Hide represent different operations and must not merge.
3. invalid_operations: identify operations that cannot be anchored to any recorded/new control, and redundant variants targeting the same control. Prioritize items marked [UNBOUND CONTROL]. Prefer the canonical action-plus-control description when removing variants.

Rules:
- Report visible interactive app controls only, not display-only text/images, explanatory prose or the system status bar.
- Exclude global navigation controls handled by system back, such as generic back/close buttons.
- Use actual visible names, uniquely identifiable within the page. Preserve the original UI language.
- Name unlabeled icons/arrows/dialog buttons using their panel and purpose, such as Search-panel Clear or Filter-dialog Cancel. Avoid context-free Delete/Cancel/Up Arrow. System keyboard keys and system UI are not business controls.
- Keep controls with distinct semantics separate despite shared words. Search, delete/backspace and switches in the same panel are distinct.
- Aggregate equivalent thumbnails/messages/contacts/files/themes/backgrounds/avatars by behavior type rather than resource value, title or appearance. If already enumerated, merge them. Direct selection and selection requiring download may be distinct types.
- Split navigation/tab bars into individual clickable items using visible names, never container names.
- One control name denotes one clickable control, not a parenthesized/comma-separated collection. Split distinct-flow entries and aggregate equivalent resource instances.
- Name list items by visible business type, such as System notification item, not generic list item/content item/region.
- Do not add controls semantically identical to or already included by the inventory. When uncertain, omit.
- Empty missing_controls is normal when no visible omission exists.
- Be conservative when deleting operations: retain anything matching any existing/new control or with uncertain duplication.
- Every invalid_operations value must copy an existing operation exactly, without paraphrasing, truncation, combination or invented entries. An empty array is normal.

Return only strict JSON, without Markdown/comments/explanations:
{"missing_controls": ["Search button"], "control_merges": [{"sources": ["White background", "Sky background", "Heart background"], "target": "Directly selectable preset background"}], "invalid_operations": ["Click an arbitrary conversation"]}
With no changes: {"missing_controls": [], "control_merges": [], "invalid_operations": []}.
"""
