get_widgets_from_page_prompt = """
You are a professional Mobile GUI Analysis Assistant. You will receive:
1. A screenshot before action.
2. A screenshot after action (current page).
3. The user action performed.

### Core Task
Analyze the screenshots to accurately identify all interactive widgets on the current page, adhering to the specified criteria.

### 1. Widget Recognition Standards
#### 1.1 Definition of Interactive Widgets
- Interactive widgets refer to elements supporting user operations, including but not limited to: clickable buttons, input fields, state-toggle widgets (Switch/Checkbox), dropdown menu options, and operation buttons in popups.

#### 1.2 Core Scope (Based on Screenshot Comparison)
- Compare the "screenshot before action" and "current page screenshot" strictly; **only identify interactive widgets that are newly added or newly visible on the current page**.
- Fully ignore interactive widgets that already existed and remained unchanged in the "screenshot before action".

#### 1.3 Deduplication & Batch Widget Handling Rules
- For batch identical-function widgets (e.g., homogeneous cards in a list, multiple identical buttons) on the current page: select **only one representative** to avoid redundant entries.
- For functionally differentiated widgets (e.g., different options in a dropdown menu, buttons with distinct labels): retain all without deduplication.
- For long homogeneous data lists (e.g., list rows with identical structure): extract **only one representative item** (do not identify all entries).

#### 1.4 Exclusion Rules
- Ignore system-level UI elements (e.g., system Back/Home buttons) unless the current page has no other interactive widgets.
- Ignore persistent static elements (e.g., top navigation bar, bottom tab bar) that show no visual changes compared to the "screenshot before action" (do not re-identify them).

### 2. Field Definitions (Precise Guidelines)
- is_leaf:
  - `true`: Widgets that toggle states (Switch, Checkbox) or select options **without triggering page navigation/redirection**.
  - `true`: Log out buttons or exit actions that close the app without navigating to another page.
  - `false`: Widgets that initiate page navigation, confirm actions (Login, Search, Submit), or **open new layers/pages** (modals, tabs).
- content:
  - Input Fields: Generate a **contextually relevant, realistic sample value** aligned with the field’s label/hint (avoid generic placeholders).
    - Examples: Username → "jane_doe89"; Password → "SecurePass123!"; Search → "wireless headphones"; Phone → "+1-555-123-4567".
  - Clickable Elements (buttons, links, toggles): Return an empty string "".
- postcondition:
  - Provide a concise prediction of the immediate behavior after interacting with the widget (≤30 words).

### Important Notes
- Note Bottom Tabs Control.
- Representative Widget per Category: If multiple widgets belong to the same category/type on the current page (e.g., repeated items in a list/grid), identify only one representative widget and ignore the rest.

### Output Format (Strict Compliance)
- Return **exactly one JSON object** (no markdown code fences, comments, or extra text).
- All string fields must adhere to length limits; ensure coordinates are logical relative to the screenshot.

{
  "function_description": "Concise summary of the page’s core function (≤20 words)",
  "widgets": [
    {
      "description": "Clear, functional description of the widget (≤20 words)",
      "action": "click" or "input",
      "content": "Sample value if 'input'; empty string '' if 'click'",
      "position": [x, y], // Center coordinates (normalized 0-1000 for consistency)
      "is_leaf": true/false, // Strict boolean (no strings)
      "postcondition": "Concise prediction of immediate post-interaction behavior (≤30 words)"
    }
  ]
}
"""


initial_page_prompt = """
You are a professional Mobile GUI Analysis Assistant. You will receive a screenshot of a mobile app's current page.


### Core Task
Analyze the screenshot to accurately identify all interactive widgets, adhering to the specified criteria.

### 1. Widget Recognition Standards
- Define "interactive widgets" as elements supporting user actions (e.g., click, input, state toggle).
- For batch identical-function widgets (e.g., repetitive cards in a list/grid), select **only one representative** (avoid redundant entries),
  **except for Bottom Tab bars**.
- Ensure no omission of unique interactive elements; prioritize functional distinctiveness over visual duplication.

### 2. Bottom Tab Bar (Special Rule — MUST FOLLOW)
- **All Bottom Tab bar items MUST be identified individually**, even if they share similar structure or appearance.
- Do NOT deduplicate Bottom Tab items. Each tab (e.g., Home, Search, Favorite, Settings) must be listed as a separate widget.
- Treat Bottom Tab items as navigation controls (`is_leaf = false`).

### 3. Field Definitions (Precise Guidelines)
- is_leaf:
  - `true`: Widgets that toggle states (Switch, Checkbox) or select options **without triggering page navigation/redirection**.
  - `false`: Widgets that initiate page navigation, confirm actions (Login, Search, Submit), or **open new layers/pages**
    (including Bottom Tab switches, modals, and tab changes).
- content:
  - Input Fields: Generate a **contextually relevant, realistic sample value** aligned with the field’s label/hint (avoid generic placeholders).
    - Examples: Username → "jane_doe89"; Password → "SecurePass123!"; Search → "wireless headphones"; Phone → "+1-555-123-4567".
  - Clickable Elements (buttons, links, toggles): Return an empty string "".

### Important Notes
- Bottom Tab bar widgets are an exception to the representative-widget rule and must be **fully enumerated**.
- Representative Widget per Category applies ONLY to non-tab repetitive items (e.g., list cards, grid items).

### Output Format (Strict Compliance)
- Return **exactly one JSON object** (no markdown code fences, comments, or extra text).
- All string fields must adhere to length limits; ensure coordinates are logical relative to the screenshot.

{
  "function_description": "Concise summary of the page’s core function (≤20 words)",
  "widgets": [
    {
      "description": "Clear, functional description of the widget (≤20 words)",
      "action": "click" or "input",
      "content": "Sample value if 'input'; empty string '' if 'click'",
      "position": [x, y], // Center coordinates (normalized 0–1000)
      "is_leaf": true/false, // Strict boolean
      "postcondition": "Concise prediction of immediate post-interaction behavior (≤30 words)"
    }
  ]
}
"""


test_function_prompt = '''You are a GUI testing agent.  Your goal is to test a specific function on the current app screen according to the user's instruction.

## Action Space
You can only use these actions:
- `click(point='<point>x y</point>')`: Click a coordinate.
- `long_click(point='<point>x y</point>')`: Long click a coordinate.
- `input(content='...')`: Input text (use "\\n" to confirm or submit).
- `press_back()`: Press the back button.
- `finished(content='xxx')`: Mark the task as fully complete. Use escape characters (\\', \\", \\n) if needed.

## Important Notes
If an action causes **any abnormal situation**, **immediately stop** and output a `finished` action with a short reason.  
  Abnormal situations include:
  - App crash or unexpected close.
  - Wrong or unexpected page jump.
  - Page freeze or no response to interaction.
  - UI layout corruption or broken display.
  - Any unexpected or undefined behavior.
  Example:  
  `Action: finished(content='App crashed after clicking the button.')`

## Output Format
Return exactly in the following format:
```
Thought: [Brief reasoning in {language} about what to do next]
Action: [A single valid action from the Action Space]
Description: [A short natural-language description of what this action does, e.g. "Tap the settings button"]
```

### **Examples**
- `Action: click(point='<point>123 456</point>')`
- `Action: long_click(point='<point>123 456</point>')`
- `Action: input(content='hello world\\n')`
- `Action: press_back()`
- `Action: finished(content='Task complete.')`

## Function to Test
{instruction}
'''


path_bug_detection_prompt = """
You are a Mobile App QA Expert.
You are given a sequential execution log (Path Record) of an automated test.
Each step contains a screenshot and a description of the state/action.

Your task:
Decide whether there is a bug.

You should focus on two categories:
1) Crash bugs
2) Functional bugs

### 1. Crash Bugs
Crash bugs include (but are not limited to):
- App crash dialogs (e.g., "App has stopped", "App keeps stopping").
- System error dialogs related to the app.
- Completely blank screen where the app appears dead and does not recover.
- Fully frozen UI where nothing responds even after reasonable waiting.

### 2. Functional Bugs
A functional bug means: the app runs, but the **functional outcome** clearly does NOT match the expected behavior of the feature being tested.

Typical functional bugs include:
- An operation that should update something, but **no update happens at all**
  (e.g., submit/add/save button is clicked, but the item is not created or the data is not saved).
- Data is updated **incorrectly**
  (e.g., wrong value displayed, calculation result obviously wrong, incorrect total or count).
- Data is **missing or duplicated**
  (e.g., the newly added item does not appear in the list, or the same item appears twice unexpectedly).
- A button or control that is clearly supposed to be available is **disabled or unresponsive**
  (e.g., "Submit" button stays disabled after all required fields are filled; tapping a visible button has no effect).
- A confirmation result/message is clearly inconsistent with the performed action
  (e.g., "Delete successful" is shown but the item is still there).

IMPORTANT:
- Do NOT treat minor visual/layout changes as functional bugs.
- Do NOT treat “navigated to an unexpected page” or “landed on a different screen than expected”
  as a functional bug **by itself**.
- Only consider navigation-related issues as a bug if they lead to a crash or a completely broken state
  (e.g., navigation causes an immediate crash or an unusable blank/frozen page).

Return ONLY one JSON object in this format:

{
  "has_bug": true or false,
  "bug_type": "crash" | "functional" | "none",
  "bug_description": "Short description (<=50 words)"
}

Rules:
- If you see any crash dialog, system error, blank screen, or frozen UI
  -> has_bug = true, bug_type = "crash".
- If the app keeps running but the functional outcome clearly mismatches the expected behavior
  (according to the described actions and context)
  -> has_bug = true, bug_type = "functional".
- Do NOT report navigation-only mismatches (wrong page / unexpected screen) as functional bugs,
  unless they cause a crash or a completely unusable state.
- If everything looks consistent with the expected behavior -> has_bug = false, bug_type = "none".
- If you are not sure, be conservative and set has_bug = false.
- Output only valid JSON, with no extra text.
"""


get_widget_test_prompt = """
You are a mobile app GUI testing assistant.

You will receive:
1) A screenshot of the current page.
2) A text description of ONE target widget to test.

Your task:
Based on the screenshot and the target widget, design simple test cases.
Each test case MUST contain exactly:
- ONE action (click or input).
- ONE context (only used when action = "input", otherwise MUST be an empty string "").
- ONE postcondition (the expected result after this operation).

Rules for action:
- The "action" field must be exactly one of: "click", "input".
- Across ALL test cases:
  - "click" can appear AT MOST ONCE.
  - "input" can appear AT MOST ONCE.
- If action = "input", then "context" MUST describe what text is entered and MUST be non-empty.
- If action = "click", then "context" MUST be exactly "" (empty string).

Rules for postcondition:
- Clearly describe what should happen after this single operation.
- Do NOT describe multiple steps, only the immediate result of THIS operation.

Output format:
- Return ONLY a JSON object with NO extra text.
- The object MUST have exactly one field: "tests".
- "tests" MUST be an array of test cases.
- Example:

{
  "tests": [
    {
      "action": "click",
      "context": "",
      "postcondition": "The details dialog is shown for this item."
    },
    {
      "action": "input",
      "context": "typed empty title",
      "postcondition": "An error message is shown indicating that the title cannot be empty."
    }
  ]
}
"""


get_position_prompt = '''You are a mobile app GUI analysis expert.

You will receive:
1) A screenshot of the current app page.
2) A text description of a target widget on this page.

Your task:
Determine the precise center coordinates of the target widget on the screenshot.

Rules:
- If the target widget is found, return its center coordinates as [x, y],
  where x and y are integers representing pixel positions.
- If the target widget cannot be found or cannot be confidently identified,
  return [0, 0].

Output format:
- Return ONLY a JSON object with NO extra text.
- The JSON object MUST contain the key "position".
- Example when found:
{
  "position": [123, 456]
}
- Example when not found:
{
  "position": [0, 0]
}

'''


event_llm_prompt = '''You are a GUI agent designed to follow instructions and interact with a user interface. Your task is to select the next action to perform based on the user's instruction and the current screen.

## Action Space
Your **only** available actions are:
- `click(point='<point>x y</point>')`: Clicks a specific coordinate on the screen.
- `long_click(point='<point>x y</point>')`: Performs a long click at the specified coordinate.
- `input(point='<point>x y</point>', content='...')`: Inputs the given content. To submit after typing, append "\\n".

## Output Format
You **MUST** return your response in the following format. The `Action:` line must be one of the functions from the Action Space.

```
Action: [A single action from the Action Space]
```

### **Examples of Valid Actions:**
- `Action: click(point='<point>123 456</point>')`
- `Action: long_click(point='<point>200 300</point>')`
- `Action: input(point='<point>150 250</point>', content='hello world\\n')`

## User Instruction
{instruction}
'''


bug_detection_prompt = """
You are a mobile app testing assistant.

You will receive:
- Screenshot before a user action.
- Screenshot after the user action.
- A description of the user action.

Your task:
Decide whether there is a bug.

You should focus on two categories:
1) Crash bugs
2) Functional bugs

### 1. Crash Bugs
Crash bugs ONLY include the following cases:
- Explicit app crash dialogs (e.g., "App has stopped", "App keeps stopping").
- The app process exits after the action and the system home screen (launcher/desktop) is shown.

### 2. Functional Bugs
A functional bug means: the app runs, but the actual result after this action clearly does NOT match the expected state.

Typical functional bugs include:
- An operation that should update something, but no update happens at all
  (e.g., user taps submit/add/save, but the item is not created or the data is not saved).
- Data is updated incorrectly
  (e.g., wrong value displayed, obviously wrong calculation result, incorrect total or count).
- Data is missing or duplicated
  (e.g., the newly added item does not appear in the list, or the same item appears twice unexpectedly).

IMPORTANT:
- Do NOT treat minor visual or layout differences as functional bugs.
- Do NOT treat navigation to an unexpected page or screen as a functional bug by itself.
- Do NOT treat informational messages, reminders, warnings, tips, or guidance prompts shown by the app (e.g., toast messages, usage hints, permission reminders, confirmation prompts) as bugs, as long as the core functionality behaves correctly. 
- Do NOT treat blank or empty pages resulting from incomplete loading as functional bugs.
Only consider an issue a functional bug if the application’s functional behavior or data state clearly violates the expected outcome.

Return ONLY one JSON object in this format:
{
  "has_bug": true or false,
  "bug_type": "crash" | "functional" | "none",
  "bug_description": "Short description (<=50 words)"
}
"""


bug_recheck_prompt = """
You are a mobile app testing assistant.

You will receive:
- Screenshots before and after a user action for the CURRENT case.
- A description of the user action.
- Several NON-BUG reference examples from a knowledge base.
  These reference examples represent NORMAL and CORRECT app behaviors and MUST NOT be considered defects.

Your task:
Re-evaluate whether the CURRENT case is truly a bug, by comparing it with the NON-BUG reference examples.

You should be MORE conservative than the initial detection.

Decision principles:
- If the CURRENT case is semantically or visually similar to any NON-BUG reference example, then it is NOT a bug.
- Only report a bug if the CURRENT case clearly violates the expected state AND is clearly different from all NON-BUG reference examples.


### Functional Bugs
A functional bug means: the app runs, but the actual result after this action clearly does NOT match the expected state.

Typical functional bugs include:
- An operation that should update something, but no update happens at all.
- Data is updated incorrectly (wrong value, wrong calculation, incorrect count).
- Data is missing or duplicated unexpectedly.

IMPORTANT:
- Do NOT treat minor visual or layout differences as functional bugs.
- Do NOT treat navigation to an unexpected page or screen as a functional bug by itself.
- Do NOT treat informational messages, reminders, warnings, tips, guidance prompts, or confirmation dialogs as bugs, if the core functionality is correct.
- Do NOT treat blank or empty pages caused by incomplete loading as functional bugs.

Final decision rule:
- If there is ANY reasonable doubt, choose NOT A BUG.

Return ONLY one JSON object in this format:
{
  "has_bug": true or false,
  "bug_description": "Short description (<=50 words)"
}
"""


FDG_function_description_prompt = """You are a mobile app functional analysis expert. The app you are analyzing is {app_name}.

You will be given:
1. The full navigation path that led to the current page (including pages and user actions).
2. A screenshot of the current page.

Your task:
Summarize the function of the current page in the context of the entire navigation path.

Rules:
- Do NOT describe individual buttons or UI components.
- Do NOT repeat the navigation path.
- Do NOT include explanations or reasoning.

Navigation Path:
{path_description}

Output Format:
Return only a concise, high-level functional description.
No explanations, reasoning, or JSON structure.
"""


page_to_page_prompt = """You are a mobile app interaction expert.
Given a user action description and two screenshots (before and after the action), analyze the interaction to determine functional changes and specific data flow.

### Part 1: New Functional Point Detection
A "New Functional Point" represents a distinct change in the functional context or the emergence of a new interactive layer.

**Return `true` (New Functional Point) if:**
1.  **Page Navigation:** Full-screen transition (e.g., List -> Detail, Home -> Settings).
2.  **Tab/Module Switching:** Switching major modules (e.g., Feed -> Profile).
3.  **Modal/Dialogs:** Popup/Alert requiring attention and blocking background.
4.  **Bottom Sheets/Action Panels:** Panels offering distinct actions.
5.  **Side Drawers:** Navigation drawer slides in.

**Return `false` (Same Functional Point) if:**
1.  **Inline UI Changes:** Expanding accordions, "read more".
2.  **Simple Inputs:** Typing, toggling switches, checkboxes.
3.  **Dropdowns/Menus:** Simple selection within a form.
4.  **Transient UI:** Toasts, loading spinners.
5.  **Scroll/Swipe:** Moving viewport without structural change.

### Part 2: Concrete Data Flow Analysis (STRICT)
Identify the **SPECIFIC** data entities consumed or produced.
**DO NOT use vague terms** like "User Data", "Settings", "Info", "Input". You must name the specific object or state.

- **data_in**: List specific entities used/required to perform the action.
  - *Bad:* "Input", "Selection", "System Data".
  - *Good:* "Note Title", "Search Keyword", "Selected Photo", "Login Credentials", "Current GPS Location".
- **data_out**: List specific entities created, modified, or loaded.
  - *Bad:* "New Data", "Updated Page", "Result".
  - *Good:* "Created Note Entry", "Flight Ticket Confirmation", "Filtered Product List", "WiFi State: On", "Deleted Message".
- **Empty Rule**: If the action is purely navigational (e.g., "Back" button) or visual (e.g., "Scroll down") with no data logic, return `[]`.

### Input
Action description: {action}

### Output format
Return exactly one JSON object:
{{
  "new_functional_point": true or false,
  "data_in": ["string", ...],
  "data_out": ["string", ...]
}}
Do not include explanations or any extra text.
"""


page_to_widget_prompt = """You are a mobile app GUI functional analysis expert.

You will be given:
1. A screenshot of the current page.
2. The functional description of a control on this page.

Your task is to determine if this control represents a new functional point and analyze its specific data usage.

### Part 1: Functional Point Determination
- Return `true` if the control **leads to a new functional module or distinct feature page** (e.g., Opening Settings, Toggling Dark Mode independent feature).
- Return `false` if the control **changes a state, option, or setting inside an existing feature page** (e.g., Selecting "English", Typing text).
- **Group Rule**: If multiple controls belong to the same selection group, they are NOT separate functional points.

### Part 2: Concrete Data Flow Analysis (STRICT)
Identify the **SPECIFIC** data entities this control reads or modifies.
**DO NOT use generic terms.** Be specific about *what* is being handled.

- **data_in**: What specific data does this widget take in?
  - *Examples:* "Credit Card Number" (for input), "Selected Date" (for calendar), "City Name" (for weather search).
- **data_out**: What specific data does this widget produce or change?
  - *Examples:* "Saved Contact", "Alarm Time: 08:00", "Volume Level", "Bluetooth State: Off", "Sent Email".
- **Note**:
  - If it's a simple navigation button (e.g., "Go to Home"), data lists should be `[]`.
  - System settings must be specific (e.g., "Screen Brightness", not "System Setting").

### Input
Control functional description: "{function_description}"

### Output format
Return in JSON:
{{
  "new_functional_point": true or false,
  "data_in": ["string", ...],
  "data_out": ["string", ...]
}}
Do not include explanations or any extra text.
"""


task_planning_prompt = """
You are a Mobile App QA Automation Expert.

Your Task:
Based on the function description and the provided UI context, generate test paths for: "{task_description}".

You must generate two types of paths:
1. **One Main Path (Happy Path)**: The standard, successful sequence of actions to complete the task perfectly.
2. **2-3 Variant Paths (Edge Cases/Negative Tests)**:
   - Deviate from the main path to test robustness.
   - Examples of variations: Skipping optional steps, omitting required inputs, attempting actions in wrong order, or submitting empty forms.

### Output Format
Return a single JSON object with the following structure:
{{
  "main_path": "String describing the steps. Format: 1. [Action]... 2. [Action]...",
  "variant_paths": [
    "String for Variant 1. Format: 1. [Action]...",
    "String for Variant 2..."
  ]
}}

### Example Strategy
If the task is "Login":
- "main_path": "1. Input username. 2. Input password. 3. Tap Login."
- "variant_paths": [
    "1. Input username. 2. Tap Login (Skip password).",
    "1. Tap Login (Empty fields)."
  ]

Do NOT include markdown code blocks (```json). Return raw JSON only.
"""


filter_fdg_prompt = """
You are a mobile app testing expert.

You are given a list of FDG nodes (functional points). Each node has:
- an integer index
- a short text description of what this node does in the app

List of nodes:
{fdg_info_text}

Your goal is to select which nodes are WORTH TESTING as independent test targets.

## Selection Rules

1. KEEP a node (include it in the result) if its description shows a **concrete, user-visible functionality**, for example:
   - Performs an operation: create / add / edit / delete / submit / save / confirm / search / filter / sort / share / export / import / login / register / sync, etc.
   - Changes or uses important data or state: user profile, account info, business records, configuration that affects other pages, etc.

2. REMOVE a node (do NOT include it) if its description is **only navigation or very low-value**, for example:
   - Pure navigation: open page, go to details, back, close, open tab, open menu, open drawer, switch page, simple list browsing, etc.
   - Generic container / entry: home page, main menu, category list, generic dashboard, placeholder pages without specific behavior.

3. If a node description is **too vague or unclear** and you cannot see a concrete operation or business logic, treat it as low-value navigation and REMOVE it.

## Output Requirements

- Return ONLY a JSON object.
- The JSON must contain a single field "to_test_nodes".
- "to_test_nodes" is a list of indices (integers) of nodes that should be tested.

Output format (no comments, no extra text):

{{ 
  "to_test_nodes": [0, 3, 5]
}}

Do not add explanations, comments, or any other fields.
"""


data_flow_prompt = '''
You are an expert in mobile app functional dependency analysis.

You are given several functional nodes. Each node has:
- index: integer
- description: what this function does
- data_in: data it uses or requires (may come from other functions)
- data_out: data it produces or updates (may be used by other functions)

Your task:
Infer **data dependencies for each node**.

Definition: data_dependencies (j depends on i)
- "i → j" means: Node j uses/reads/edits/displays data that could be produced/updated by Node i.
- Equivalently: Node j has a data dependency on Node i.
- You must output **for each consumer node j**, which producer nodes i it depends on.

Core rule (must use):
- If j.data_in contains an abstract data entity that is the same as (or semantically equivalent to) an entity in i.data_out,
  then include i in j.data_dependencies.

You MUST use BOTH:
1) The semantics of the two descriptions (what each function does).
2) The relationship between i.data_out and j.data_in (data matching).

Relaxed rules (high recall):
- Treat semantically similar / renamed data as the same entity
  (e.g., "expense record" vs "expense list", "deck" vs "deck details",
        "note" vs "note details", "task" vs "task item").
- If j logically continues, views, edits, deletes, filters, exports, shares,
  or shows statistics/history/summary for data created or updated by i,
  then include i → j even if the data_in / data_out strings are not exactly the same.
- Ignore pure navigation or UI-only actions with no real data usage.

Examples:
- Node 0: "Create a new record" (data_out: ["new record"])
- Node 2: "Show record details" (data_in: ["record"])
Then node 2 depends on node 0, output: "2": [0].

- If node 1 also uses the record, output: "2": [0], "1": [0].

Functional nodes:
{fdg_info_text}

Output format:
Return ONLY one valid JSON object:

{{
  "data_dependencies": {{
    "1": [0],
    "2": [0, 3],
    "4": [3]
  }}
}}

Rules:
- data_dependencies is directed: "j": [i1, i2] means consumer node j depends on producers i1 and i2.
- Keys MUST be strings of the consumer indices: "0", "1", "2", ...
- Values are lists of integers (producer node indices).
- Do NOT include a node j if it has no dependencies.
- Do NOT include self-dependency (do not put j in its own list).
- Prefer INCLUDING a dependency when it is reasonably likely based on description + data_in/data_out.
- Do NOT add any text, comments, or code fences outside the JSON.
'''


page_exist_prompt = f'''You are a professional mobile app GUI comparison expert. You will be provided with:

1. A new screenshot (current page to verify).
2. The most similar known candidate page (from historical screenshot library).

### Core Task
Determine whether the new screenshot represents the **same page** as the candidate page, following the strict judgment rule:  
**Pages with identical structural layout (regardless of content differences) are considered the same page.**

### Judgment Criteria 
1. **Structural Identity (Primary Criterion)**:
   - Evaluate overall page layout, main functional sections (e.g., header, content area, footer), key widget placement, and navigation structure.
   - If the core structural framework (section division, key element positions) is consistent, the pages are deemed the same, regardless of content changes.

2. **Ignorable Differences (Do NOT affect judgment)**:
   - Dynamic content updates (e.g., text, images, list items, numerical values).
   - Minor temporary elements (e.g., pop-up tips, loading spinners, notification badges).
   - Cosmetic changes (e.g., slight color adjustments, font size variations) that do not alter layout structure.

3. **Differences That Change Page Identity**:
   - Alteration of main section layout (e.g., content area replaced, header/footer structure modified).
   - Addition/removal of core functional sections (e.g., new sidebar, missing bottom navigation bar).
   - Fundamental shift in key widget placement (e.g., search bar moved from top to bottom).

### Output Requirements
Return ONLY a single JSON object with no explanations, comments, or extra text:
{{
  "is_same_page": true or false
}}
- `true`: New screenshot has identical structural layout to the candidate page (even with content differences).
- `false`: New screenshot has different core structure (represents a new page).
'''



data_flow_prompt_without_data = '''
You are an expert in mobile app functional dependency analysis.

You are given several functional nodes. Each node has:
- index: integer
- description: what this function does

Your task:
Infer **data dependencies for each node** based ONLY on the descriptions.

Definition: data_dependencies (j depends on i)
- "i → j" means: Node j likely uses/reads/edits/displays data or persistent state that is directly created/updated/deleted by Node i.
- You must output **for each consumer node j**, which producer nodes i it depends on.

STRICT extraction policy (precision-first):
- Extract as FEW dependencies as possible.
- Add a dependency ONLY when the relationship is STRONGLY supported by the descriptions (high confidence).
- If the relationship is ambiguous or only weakly related, DO NOT include it.

Strong-evidence rules (must satisfy at least ONE):
1) Same explicit entity
- i and j clearly mention the same concrete entity type (e.g., note, task, file, record, profile, playlist, message, deck),
  and i creates/updates/deletes it while j views/edits/shares/exports/searches/filters/sorts/history/stats of it.

2) Explicit create→consume pipeline
- i clearly produces an output that j clearly consumes
  (e.g., "create note" → "view note details" / "edit note" / "export note").

3) Specific setting effect
- i changes a clearly named setting/toggle, and j clearly indicates behavior/UI that depends on that exact setting.
  (Do NOT treat “Settings” as a single dependency source; only link via specific settings.)

Representative-only rule for similar dependencies (IMPORTANT: dedup):
- Many apps have multiple nodes that are functionally equivalent (e.g., a list of options where each option triggers the same type of operation).
- If multiple candidate dependencies are essentially the SAME kind of relationship, output ONLY ONE representative dependency and ignore the rest.

How to deduplicate:
A) Similar consumer nodes:
- If several consumer nodes j are semantically equivalent (same action type + same entity, e.g., "View X details" repeated for different tabs/options),
  then choose ONE representative consumer (prefer the most specific description; if tied, the smallest index),
  and ONLY output dependencies for that representative. Ignore the other similar consumers.

B) Similar producer nodes:
- If several producers i are semantically equivalent for the same entity/action (e.g., multiple "Create X" variants),
  then choose ONE representative producer (prefer the most specific; if tied, smallest index),
  and ONLY include that producer in the dependency list.

C) Similar dependency patterns:
- If the dependency pattern (producer type → consumer type on the same entity) repeats across many nodes,
  keep only ONE instance of that pattern.

Goal:
- Do NOT generate too many dependencies.
- Prefer fewer, high-confidence, representative dependencies over exhaustive coverage.

Exclusions (do NOT link):
- Pure navigation or UI-only actions (open menu, go back, switch tab) with no persistent data/state effect.
- Generic or vague descriptions ("open settings", "manage", "browse", "general") unless the same entity is explicitly shared.
- Do NOT infer dependencies based on being in the same feature area alone.
- Do NOT include self-dependency.

Functional nodes:
{fdg_info_text}

Output format:
Return ONLY one valid JSON object:

{{
  "data_dependencies": {{
    "1": [0],
    "2": [0, 3],
    "4": [3]
  }}
}}

Output rules:
- data_dependencies is directed: "j": [i1, i2] means consumer node j depends on producers i1 and i2.
- Keys MUST be strings of the consumer indices: "0", "1", "2", ...
- Values are lists of integers (producer node indices).
- Do NOT include a node j if it has no dependencies.
- Do NOT add any text, comments, or code fences outside the JSON.
'''








