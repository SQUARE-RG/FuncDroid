import base64
from datetime import datetime
from queue import Queue
import re
import threading
import cv2, copy
from hmbot.explorer.fdg import *
from hmbot.device.device import Device
from hmbot.app.app import App
from hmbot.explorer.llm import *
from hmbot.explorer.prompt import *
import json, time
from hmbot.explorer.utils import *
from hmbot.explorer.action import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from hmbot.model.vht import VHTParser
from hmbot.utils.cv import encode_image
from hmbot.utils.proto import PageInfo
from collections import deque
from hmbot.explorer.knowledge import KnowledgeBaseRetriever
from pathlib import Path


VHT_WEIGHT = 0 
IMG_WEIGHT = 1  
SIMILARITY_THRESHOLD_MATCH = 0.95


def get_current_app_package() -> str:
    def _run(cmd: str) -> str:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (p.stdout or "").strip()

    out = _run(r'adb shell dumpsys window | findstr mCurrentFocus')
    m = re.search(r"\s([a-zA-Z0-9_\.]+)\/", out)
    if m:
        return m.group(1)

    return ""


class Explorer:
    def __init__(self, device: Device, app_name, app: App=None):
        self.device = device
        # self.app = app
        self.app_name = app_name
        # self.app_bundle = app.package_name
        self.app_bundle = get_current_app_package()
        self.app_abilities = app.abilities if app else []
        self.explored_abilities = []

        self.page_nodes: list[PageNode] = []
        self.FDG: list[FDGNode] = []

        self.executor = ThreadPoolExecutor(max_workers=4)

        self.path = []

        self.bug_queue: Queue = Queue()
        self._bug_detector_running = True

        self.bug_detector_thread = threading.Thread(
            target=self.dectect_bug, daemon=True
        )
        self.bug_detector_thread.start()
        self.bug_counter = 0

        self.lock = threading.Lock()

        self.output_dir = ""
        self.start_time = time.time()
        self.time_limit_seconds = 60 * 60  
        self.stop_exploration = False

        self.depth_limit = 10

        # self.kb_retriever = KnowledgeBaseRetriever()

        self._declared_activities = set()   # manifest declared
        self._visited_activities = set()    # reached during exploration

        self._last_act_cov_ts = 0.0
        self._act_cov_interval_sec = 60

        # 输出文件路径（在 explore() 设置 output_dir 后再初始化也行）
        self._act_cov_path = None
        self._act_cov_hist_path = None


    def _time_exceeded(self):
        if time.time() - self.start_time >= self.time_limit_seconds:
            self.stop_exploration = True
            return True
        return False


    def explore(self, output_dir: str):
        self.output_dir = output_dir
        self._declared_activities = set(self.app_abilities or [])
        out = Path(self.output_dir)
        self._act_cov_path = out / "activity_coverage.json"
        self._act_cov_hist_path = out / "activity_coverage_history.jsonl"

        grant_all_permissions(self.app_bundle)

        self.root_page_node = PageNode(index=-1, page=None)
        self.root_page_node.edges.append({
            "description": "",
            "action": "click",
            "position": [0, 0],
            "content": "",
            "is_leaf": False,
            "page_node": None
        })
        try:
            self._explore(self.root_page_node)
            logger.info("Exploration completed successfully.")
        except Exception as e:
            logger.exception(f"Error occurred in explore: {e}")
        finally:
            logger.info("Exploration finished. Saving PTG...")
            self._bug_detector_running = False
            self.save_PTG(output_dir)
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            token_path = out_dir / "LLM-Token-Stats.json"
            token_stats = globals().get("TOKEN_STATS", None)
            payload = {
                "ts": time.time(),
                "app_bundle": getattr(self, "app_bundle", ""),
                "output_dir": str(out_dir),
                "token_stats": token_stats or {},
            }

            token_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"[TOKENS] saved to: {token_path}")

    
    def build_FDG(self, ptg_file_path: str):
        count = 0
        FDG_EDGE_CLASSIFY_PROMPT = """
You are identifying functional points and their boundaries in a mobile app.

Task:
Given the DESCRIPTION of the CURRENT functional point and the UI screenshots BEFORE and AFTER an action, decide whether this action starts a NEW functional point, or it is still part of the CURRENT functional point.

What is a "functional point"?
A functional point is a self-contained user goal that the user perceives as ONE function/goal. A functional point can be completed by a single action (one-edge function) or multiple actions (a workflow).

Core principle (do NOT be too broad):
- Do NOT merge many unrelated things into one big functional point (especially on Settings pages).
- A functional point must have a clear and specific purpose/effect (what is achieved or what changes), not a vague umbrella like “Settings” or “General operations”.

Boundary rules:

1) Navigation / feature entry
- If the action clicks a feature item on a navigation list / home menu / drawer / tab list and enters that feature,
  then that clicked feature item should be treated as the start of a NEW functional point.
  Example: "Notes" / "Calendar" / "Search" / "Settings" items in a navigation list → each item is a new functional point.

2) Goal completion within the same function (one workflow)
- If the action is a step to complete the same user goal within ONE feature page/workflow, it is NOT a new functional point.
  Example 1: On a "Create Note" page, typing title/body, adding tags, choosing a folder, and tapping "Save" are all part of the same functional point: Create Note.
  Example 2: When saving/exporting a file, the app may navigate to a "File picker / Choose location" page. Selecting a folder/location and confirming the save are still part of the SAME functional point (Save/Export File), and should NOT be split into a separate "File selection" functional point.

3) Settings page rule (MUST be fine-grained)
- On a Settings page, each distinct setting item (including toggles/switches) is its own functional point.
  Example: "Theme", "Font size", "Sync", "Notifications", "Enable Backup", "Dark Mode" → each setting item is a NEW functional point.
- A setting item can be a one-edge functional point even if the UI does NOT navigate to a new page.
  Example: Toggling "Dark Mode" ON/OFF or enabling "Sync" is a NEW functional point because it has a clear effect.
- The options/configurations within the SAME setting item are NOT new functional points.
  Example: Inside "Theme", choosing "Light/Dark/System" are all within the same functional point: Change Theme.

Also extract abstract data:

Definitions (abstract data entities):
- "data_in": abstract data entities that this action/functional point uses/loads/edits/operates.
  Examples: a note, a task item, a diary record, a user profile, a file, a reminder entry, a playlist, a message thread/post, a login session, a setting item.
- "data_out": abstract data entities that this action/functional point produces or updates.
  Examples: a newly created note/task/reminder, an updated record, a generated report, a saved setting/profile change, an exported file, an updated toggle state.

Rules for data_in/data_out:
- Only include ABSTRACT entities (not UI widget names, not button text, not coordinates, not view ids, not program variables or implementation details).
- Use short noun phrases; de-duplicate; do not invent data that is not implied by the UI.
- If the action is a settings toggle/switch:
  - data_in should include the setting item entity (e.g., "dark mode setting", "sync setting").
  - data_out should reflect the state change (e.g., "dark mode setting enabled/disabled").

Output (STRICT JSON ONLY):
{
  "new_functional_point": true/false,
  "data_in": ["<abstract entity>", ...],
  "data_out": ["<abstract entity>", ...]
}

Additional requirement:
- Do NOT output any extra text, explanation, comments, or code fences outside the JSON.
"""



        FDG_FUNCTION_DESCRIPTION_PROMPT = """
App: {app_name}

You are given a navigation path description to reach the current page. Summarize the functionality of the current page as a concise function description (one sentence).

Path:
{path_description}

Return only plain text.
"""


        FDG_CORE_LOGIC_PROMPT = """
You are extracting the CORE EXECUTION LOGIC of a mobile app functional point.

App: {app_name}

Functional point description (high-level):
{function_description}

Below is the set of actions that belong to this functional point.
Each action is a PTG edge reference, and contains:
- action_ref: [page_idx, edge_idx]
- src_page_idx
- dst_page_idx (may be null)
- action (e.g., click / input)
- description / content / position / is_leaf

Your task:
1) Build a FLOWCHART-LIKE structure that captures the core execution logic of this functional point:
   - ordered steps (sequence)
   - branch points (if-else / loop) when the flow can diverge or repeat
2) Every step MUST correspond to EXACTLY ONE provided action_ref (no invented steps).
3) The final output must be STRICT JSON that follows the schema below.

CRITICAL CONSTRAINTS (MUST FOLLOW):
A) No hallucination:
   - You MUST NOT invent steps/actions that are not backed by the given actions list.
   - Every "steps[i].action_ref" MUST appear in the given actions list.
B) Reference integrity:
   - All step ids referenced in "flow_edges" and "branch_points" MUST exist in "steps".
   - "flow_edges[].from/to" must be valid step ids.
   - "branch_points[].at_step" must be a valid step id.
   - Every "branches[].next_step" must be a valid step id.
C) Branch conditions:
   - If you describe conditions (in summaries), they MUST be UI-observable
     (e.g., "toggle is off", "input is empty", "dialog appears"),
     NOT program variables, NOT internal states, NOT coordinates.
D) Output field "logic" definition:
   - "logic" MUST be a NATURAL-LANGUAGE description of the CORE EXECUTION LOGIC (2–6 sentences).
   - It must summarize the main flow in order, and explicitly mention major decision/loop points if any.
   - Do NOT include testing oracles, assertions, mutations, or implementation details.
   - Do NOT mention coordinates, view ids, or program variables.

Guidance on structuring:
- If the logic is a straight-line flow, leave "branch_points" as an empty list.
- Use "type": "if-else" for mutually exclusive branches (A or B).
- Use "type": "loop" if steps repeat under some UI-observable condition.
- Keep step summaries short and action-focused.

Output STRICT JSON (NO extra text, NO markdown):
{{
  "entry_page": <int|null>,
  "logic": "<natural-language description of the core execution logic, 2-6 sentences>",
  "steps": [
    {{
      "id": "S1",
      "action_ref": [page_idx, edge_idx],
      "src_page_idx": <int>,
      "dst_page_idx": <int|null>,
      "action": "<string>",
      "summary": "<what this step does>"
    }}
  ],
  "flow_edges": [
    {{"from": "S1", "to": "S2"}}
  ],
  "branch_points": [
    {{
      "at_step": "S2",
      "type": "if-else" or "loop",
      "branches": [
        {{"next_step": "S3"}},
        {{"next_step": "S4"}}
      ]
    }}
  ]
}}
"""


        def safe_json_loads(raw: str, default):
            text = (raw or "").strip()
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text).strip()
            text = re.sub(r"\s*```$", "", text).strip()
            try:
                return json.loads(text)
            except Exception:
                return default

        def build_path_description(page_path: list):
            parts = []
            for i in range(len(page_path) - 1):
                src = page_path[i]
                dst = page_path[i + 1]
                edge = next((e for e in src.edges if e.get("page_node") == dst), None)
                if edge:
                    action_desc = edge.get("description") or edge.get("content") or "N/A"
                    if getattr(src, "function_description", ""):
                        parts.append(f"Page {src.index}: {src.function_description} → Action: {action_desc}")
                    else:
                        parts.append(f"Page {src.index} → Action: {action_desc}")
            return "\n".join(parts) or "Root page"

        def get_function_description(page_path: list, current_page_node):
            path_description = build_path_description(page_path)
            content = [
                {
                    "type": "input_text",
                    "text": FDG_FUNCTION_DESCRIPTION_PROMPT.format(
                        app_name=self.app_name,
                        path_description=path_description
                    )
                },
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{current_page_node.page.encoded_img}"},
            ]
            return ask_llm(content=content).strip()

        def classify_edge_with_llm(prev_node, edge: dict, next_node):
            action_text = edge.get("description") or edge.get("content") or "N/A"
            before_b64 = prev_node.page.encoded_img if prev_node.page else ""
            after_b64 = next_node.page.encoded_img if (next_node and getattr(next_node, "page", None)) else ""

            llm_content = [
                {"type": "input_text", "text": FDG_EDGE_CLASSIFY_PROMPT},
                {"type": "input_text", "text": f"Action: {action_text}"},
                {"type": "input_text", "text": "Screenshot before action:"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{before_b64}"},
                
            ]
            if after_b64:
                llm_content.extend([
                    {"type": "input_text", "text": "Screenshot after action:"},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{after_b64}"},
                ])

            raw = ask_llm(content=llm_content)
            j = safe_json_loads(raw, default={"new_functional_point": False, "data_in": [], "data_out": []})
            return (
                bool(j.get("new_functional_point", False)),
                (j.get("data_in") or []),
                (j.get("data_out") or []),
                raw
            )

        def extract_core_logic_for_fdg(fdg_node: FDGNode):
            if not fdg_node.action_refs:
                return None

            actions_info = []
            img_payload = []
            max_imgs = 6

            for (pidx, eidx) in fdg_node.action_refs:
                src = self.page_nodes[pidx]
                edge = src.edges[eidx]
                dst = edge.get("page_node")
                dst_idx = dst.index if dst else None

                actions_info.append({
                    "action_ref": [pidx, eidx],
                    "src_page_idx": pidx,
                    "dst_page_idx": dst_idx,
                    "action": edge.get("action", "click"),
                    "description": edge.get("description", ""),
                    "content": edge.get("content", ""),
                    "position": edge.get("position", [0, 0]),
                    "is_leaf": bool(edge.get("is_leaf", False)),
                })

            src_count = {}
            for a in actions_info:
                src_count[a["src_page_idx"]] = src_count.get(a["src_page_idx"], 0) + 1

            candidate_pages = []
            candidate_pages.append(actions_info[0]["src_page_idx"])
            for p, c in sorted(src_count.items(), key=lambda x: -x[1]):
                if c >= 2:
                    candidate_pages.append(p)
            last_dst = actions_info[-1]["dst_page_idx"]
            if last_dst is not None:
                candidate_pages.append(last_dst)

            seen_pages = set()
            for pidx in candidate_pages:
                if len(img_payload) >= max_imgs:
                    break
                if pidx in seen_pages:
                    continue
                seen_pages.add(pidx)
                pnode = self.page_nodes[pidx]
                if getattr(pnode, "page", None) and getattr(pnode.page, "encoded_img", None):
                    img_payload.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{pnode.page.encoded_img}"
                    })

            content = [
                {
                    "type": "input_text",
                    "text": FDG_CORE_LOGIC_PROMPT.format(
                        app_name=self.app_name,
                        function_description=fdg_node.function_description or "",
                    )
                },
                {"type": "input_text", "text": "Actions JSON:"},
                {"type": "input_text", "text": json.dumps(actions_info, ensure_ascii=False)},
            ]
            if img_payload:
                content.append({"type": "input_text", "text": "Reference screenshots of key pages:"})
                content.extend(img_payload)

            raw = ask_llm(content=content)
            core = safe_json_loads(raw, default=None)

            return core
        
        self.read_PTG(ptg_file_path)
        start_node = self.page_nodes[0]

        self.FDG = []
        root_desc = start_node.function_description or "Root"
        self.FDG.append(FDGNode(root_desc))

        visited_pages = set([start_node])

        # BFS queue: (page_node, current_fdg_id, page_path)
        q = Queue()
        q.put((start_node, 0, [start_node]))


        def _process_edge_parallel(task):
            """
            task = (prev_node, cur_fdg_id, page_path, edge_idx)
            """
            prev_node, cur_fdg_id, page_path, edge_idx = task
            edge = prev_node.edges[edge_idx]
            next_node = edge.get("page_node")

            is_new, data_in, data_out, _raw = classify_edge_with_llm(prev_node, edge, next_node)

            func_desc = None
            if is_new and next_node is not None and getattr(next_node, "page", None) is not None:
                func_desc = get_function_description(page_path + [next_node], next_node)
            elif is_new:
                func_desc = edge.get("description") or edge.get("content") or "New Function"

            return {
                "prev_node": prev_node,
                "cur_fdg_id": cur_fdg_id,
                "page_path": page_path,
                "edge_idx": edge_idx,
                "next_node": next_node,
                "is_new": is_new,
                "func_desc": func_desc,
                "data_in": data_in,
                "data_out": data_out,
            }


        processed_edges = set()                 # (src_page_idx, edge_idx)
        expanded_state = set([(start_node.index, 0)])  # (page_idx, fdg_id)

        with ThreadPoolExecutor(max_workers=10) as executor:
            while not q.empty():
                batch = []
                while not q.empty():
                    batch.append(q.get())

                tasks = []
                for prev_node, cur_fdg_id, page_path in batch:
                    for edge_idx, edge in enumerate(prev_node.edges):
                        edge_key = (prev_node.index, edge_idx)
                        if edge_key in processed_edges:
                            continue
                        processed_edges.add(edge_key)

                        tasks.append((prev_node, cur_fdg_id, page_path, edge_idx))

                if not tasks:
                    continue

                futures = [executor.submit(_process_edge_parallel, t) for t in tasks]
                results = []
                for f in as_completed(futures):
                    try:
                        results.append(f.result())
                    except Exception as e:
                        print(f"[Error] edge task failed: {e}")

                results.sort(key=lambda r: (r["prev_node"].index, r["edge_idx"]))

                for r in results:
                    prev_node = r["prev_node"]
                    cur_fdg_id = r["cur_fdg_id"]
                    page_path = r["page_path"]
                    edge_idx = r["edge_idx"]
                    next_node = r["next_node"]
                    is_new = r["is_new"]
                    func_desc = r["func_desc"]
                    data_in = r["data_in"]
                    data_out = r["data_out"]

                    action_ref = (prev_node.index, edge_idx)

                    if is_new:
                        new_fdg = FDGNode(func_desc or "New Function")
                        new_fdg.action_refs.append(action_ref)
                        new_fdg.data_in.extend(data_in)
                        new_fdg.data_out.extend(data_out)
                        self.FDG.append(new_fdg)
                        new_fdg_id = len(self.FDG) - 1

                        if next_node is not None:
                            if next_node in page_path:
                                continue

                            st = (next_node.index, new_fdg_id)
                            if st not in expanded_state:
                                expanded_state.add(st)
                                q.put((next_node, new_fdg_id, page_path + [next_node]))

                    else:
                        fdg = self.FDG[cur_fdg_id]
                        fdg.action_refs.append(action_ref)
                        fdg.data_in.extend(data_in)
                        fdg.data_out.extend(data_out)

                        # ✅ 页面扩展入队：去重 + 防环
                        if next_node is not None:
                            if next_node in page_path:
                                continue

                            st = (next_node.index, cur_fdg_id)
                            if st not in expanded_state:
                                expanded_state.add(st)
                                q.put((next_node, cur_fdg_id, page_path + [next_node]))

        for fdg in self.FDG:
            if not fdg.action_refs:
                continue
            fdg.core_logic = extract_core_logic_for_fdg(fdg)
            fdg.to_test = True

        def _dedup_keep_order(lst):
            if not lst:
                return []
            seen = set()
            out = []
            for x in lst:
                key = x  
                if key not in seen:
                    seen.add(key)
                    out.append(x)
            return out

        for fdg in self.FDG:
            fdg.data_in = _dedup_keep_order(getattr(fdg, "data_in", []))
            fdg.data_out = _dedup_keep_order(getattr(fdg, "data_out", []))

        dir_path = os.path.dirname(ptg_file_path)
        new_path = os.path.join(dir_path, "fdg.json")
        self.save_FDG(new_path)
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        token_path = out_dir / "LLM-Token-Stats.json"
        token_stats = globals().get("TOKEN_STATS", None)
        payload = {
            "ts": time.time(),
            "app_bundle": getattr(self, "app_bundle", ""),
            "output_dir": str(out_dir),
            "token_stats": token_stats or {},
        }

        token_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[TOKENS] saved to: {token_path}")


    def build_FDG_with_dependency(self, ptg_file_path: str, fdg_file_path: str):
        self.read_PTG(ptg_file_path)
        self.read_FDG(fdg_file_path)

        # fdg_info_lines = []
        # for i, node in enumerate(self.FDG):
        #     line = f"{i}: {node.function_description}"
        #     fdg_info_lines.append(line)
        # fdg_info_text = "\n".join(fdg_info_lines)

        # print(fdg_info_text)

        # response = ask_llm(
        #     content=[
        #         {
        #             "type": "input_text",
        #             "text": filter_fdg_prompt.format(fdg_info_text=fdg_info_text)
        #         }
        #     ]
        # )
        # raw_json = response.strip()

        # cleaned_json = re.sub(r"^```[a-zA-Z]*|```$", "", raw_json, flags=re.MULTILINE).strip()
        # try:
        #     result = json.loads(cleaned_json)
        # except json.JSONDecodeError:
        #     print(f"[Warning] JSON parsing failed. Raw response:\n{raw_json}")
        #     result = {"to_test_nodes": []}

        # to_test_indices = result.get("to_test_nodes", [])
        # print(f"[Test-worthy node indices] {to_test_indices}")

        # for i, node in enumerate(self.FDG):
        #     node.to_test = i in to_test_indices

        fdg_info_lines = []
        for i, node in enumerate(self.FDG):
            if not node.to_test:
                continue
            if not node.data_in and not node.data_out:
                continue

            data_in_unique = list(dict.fromkeys(node.data_in)) if node.data_in else []
            data_out_unique = list(dict.fromkeys(node.data_out)) if node.data_out else []

            line = [
                f"index: {i}",
                f"description: {node.function_description}",
                f"data_in: {data_in_unique}",
                f"data_out: {data_out_unique}",
            ]
            fdg_info_lines.append("\n".join(line))

        fdg_info_text = "\n\n".join(fdg_info_lines)

        response = ask_llm(
            content=[
                {
                    "type": "input_text",
                    "text": data_flow_prompt.format(fdg_info_text=fdg_info_text)
                }
            ]
        )
        raw_json = response.strip()
        cleaned_json = clean_llm_json(raw_json).strip()

        try:
            result = json.loads(cleaned_json)
            data_flow = result.get("data_dependencies", {})
            print(f"[Info] Parsed data_flow: {data_flow}")
        except json.JSONDecodeError:
            print(f"[Warning] JSON parsing failed in data_flow. Raw response:\n{raw_json}")
            data_flow = {}

        for node in self.FDG:
            node.data_dependencies = []

        for producer_idx_str, consumers in data_flow.items():
            try:
                producer_idx = int(producer_idx_str)
            except (TypeError, ValueError):
                continue

            if producer_idx < 0 or producer_idx >= len(self.FDG):
                continue

            if not isinstance(consumers, (list, tuple)):
                continue

            for consumer_idx in consumers:
                try:
                    consumer_idx_int = int(consumer_idx)
                except (TypeError, ValueError):
                    continue

                if consumer_idx_int < 0 or consumer_idx_int >= len(self.FDG):
                    continue

                if producer_idx not in self.FDG[consumer_idx_int].data_dependencies:
                    self.FDG[consumer_idx_int].data_dependencies.append(producer_idx)

        for node in self.FDG:
            node.data_dependencies = sorted(set(node.data_dependencies))

        print("[Info] Data-flow dependencies updated in FDG (consumer -> producers).")

        base, ext = os.path.splitext(fdg_file_path)
        if not ext:
            ext = ".json"

        new_fdg_file_path = f"{base}_with_data_dep{ext}"
        self.save_FDG(new_fdg_file_path)
        print(f"[Info] FDG with data dependencies saved to {new_fdg_file_path}")
        out_dir = Path("C:\\Users\\23314\\Desktop\\Fim\\output")
        out_dir.mkdir(parents=True, exist_ok=True)
        token_path = out_dir / "LLM-Token-Stats.json"
        token_stats = globals().get("TOKEN_STATS", None)
        payload = {
            "ts": time.time(),
            "app_bundle": getattr(self, "app_bundle", ""),
            "output_dir": str(out_dir),
            "token_stats": token_stats or {},
        }

        token_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[TOKENS] saved to: {token_path}")


    def _explore(self, page_node: PageNode):

        if self.stop_exploration or self._time_exceeded():
            logger.info("Time limit reached. Stop exploration.")
            return
        
        cur_act = getattr(self.device, "current_activity", None)
        if callable(cur_act):
            cur_act = cur_act()
        if cur_act:
            self._visited_activities.add(cur_act)

        self._maybe_dump_activity_coverage()

        
        self._excute_edges(page_node)

        for edge in page_node.edges:
            if self.stop_exploration or self._time_exceeded():
                logger.info("Time limit reached during edge exploration.")
                return
            
            if edge["is_leaf"] or edge["page_node"].is_visited or edge["page_node"].edges == [] or len(self.path) >= self.depth_limit:
                continue

            if page_node.index != -1:
                if self._excute_action(edge) is False:
                    continue
                self.path.append((page_node.index, edge))

            self._explore(edge["page_node"])

            if self.path != []:
                self.path.pop()

            success = False
            for _ in range(3):
                self.device.back()
                time.sleep(2)
                new_page = self.device.dump_page(refresh=True)
                new_page_index = self._is_page_exist(new_page)

                if new_page_index == page_node.index:
                    success = True
                    break

                found_index = -1
                for idx, p in enumerate(self.path):
                    if p[0] == new_page_index:
                        found_index = idx
                        break

                if found_index != -1:
                    for i in self.path[found_index:]:
                        self.page_nodes[i[0]].page = self.device.dump_page(refresh=True)
                        edge = i[1]
                        self._excute_action(edge)
                    success = True
                    break  
                    
            if not success:
                logger.info("Return attempts failed, restarting app.")
                self.device.restart_app_by_bundle(self.app_bundle)
                time.sleep(15)
                for i in self.path:
                    self.page_nodes[i[0]].page = self.device.dump_page(refresh=True)
                    edge = i[1]
                    self._excute_action(edge)

        logger.info(f"Explore {page_node.index} finished.")


    def _excute_edges(self, page_node: PageNode): 
        if self.stop_exploration or self._time_exceeded():
            logger.info("Time limit reached before executing edges.")
            return
    
        futures = [] 

        page_node.is_visited = True

        print(f"Executing edges for page {page_node.index}")
        for idx, edge in enumerate(page_node.edges):
            print(f"{idx}: {edge['description']}")

        for edge in page_node.edges:
            if self.stop_exploration or self._time_exceeded():
                logger.info("Time limit reached before executing edges.")
                return
            
            if edge["is_leaf"]:
                print("leaf node reached.")
                new_page_node = PageNode(index=len(self.page_nodes), page=page_node.page)
                new_page_node.type = "widget"
                self.page_nodes.append(new_page_node)
                new_page_node.function_description = edge["description"]
                edge["page_node"] = new_page_node
                continue

            print(f"Executing edge action: {edge['action']} at position: {edge['position']}")

            if page_node.index != -1:
                if self._excute_action(edge) is False:
                    edge["is_leaf"] = True
                    edge["page_node"] = page_node
                    continue

            try:
                new_page = self.device.dump_page(refresh=True)
            except Exception as e:
                self.device.back()
                edge["is_leaf"] = True
                edge["page_node"] = page_node
                time.sleep(3)
                continue
            index = self._is_page_exist(new_page)

            if index == page_node.index:
                page_node.page = new_page
                edge["is_leaf"] = True
                edge["page_node"] = page_node
                continue

            # detect bug
            task = {
                "page_before": page_node.page,
                "page_after": new_page,
                "action": edge["description"],
                "expected_state": edge["postcondition"] if "postcondition" in edge else "",
            }
            self.bug_queue.put(task)

            if index == len(self.page_nodes):
                new_page_node = PageNode(index=index, page=new_page)
                self.page_nodes.append(new_page_node)
                edge["page_node"] = new_page_node
                
                future = self.executor.submit(
                    self.get_widgets_from_page,
                    page_node,
                    new_page_node,
                    edge["description"],
                )
                futures.append(future)
            else:
                self.page_nodes[index].page = new_page  
                edge["page_node"] = self.page_nodes[index]

            if page_node.index == -1:
                continue

            found_index = -1
            for idx, p in enumerate(self.path):
                if p[0] == index:
                    found_index = idx
                    break

            if found_index != -1:
                for i in self.path[found_index:]:
                    self.page_nodes[i[0]].page = self.device.dump_page(refresh=True)
                    edge = i[1]
                    self._excute_action(edge)
                continue

            success = False
            for _ in range(3):
                self.device.back()
                time.sleep(3)
                new_page = self.device.dump_page(refresh=True)
                new_page_index = self._is_page_exist(new_page)

                if new_page_index == page_node.index:
                    success = True
                    break

                found_index = -1
                for idx, p in enumerate(self.path):
                    if p[0] == new_page_index:
                        found_index = idx
                        break

                if found_index != -1:
                    for i in self.path[found_index:]:
                        self.page_nodes[i[0]].page = self.device.dump_page(refresh=True)
                        edge = i[1]
                        self._excute_action(edge)
                    success = True
                    break  

            if not success:
                logger.info("Return attempts failed, restarting app.")
                # self.device.restart_app(self.app)
                self.device.restart_app_by_bundle(self.app_bundle)
                time.sleep(15)
                for i in self.path:
                    self.page_nodes[i[0]].page = self.device.dump_page(refresh=True)
                    edge = i[1]
                    self._excute_action(edge)

        if futures:
            for f in futures:
                try:
                    f.result()  
                except Exception as e:
                    logger.error(f"[ERROR] get_widgets_from_page task failed: {e}")


    def _excute_action(self, edge: dict):
        page = self.device.dump_page(refresh=True)
        content = [
            {"type": "input_text", "text": get_position_prompt},
            {"type": "input_text", "text": "Current page screenshot:"},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{page.encoded_img}",
            },
            {"type": "input_text", "text": f"Widget description: {edge['description']}"},
        ]

        def _sanitize_json(s: str) -> str:
            # remove trailing commas before } or ]
            import re
            prev, cur = None, s or ""
            while prev != cur:
                prev = cur
                cur = re.sub(r',(\s*[}\]])', r'\1', cur)
            return cur

        def _parse_position_result(raw_text: str):
            cleaned = clean_llm_json(raw_text)
            cleaned = _sanitize_json(cleaned)
            result = json.loads(cleaned)

            if not isinstance(result, dict) or "position" not in result:
                raise ValueError(f"Missing 'position' in model output: {result}")

            pos = result["position"]
            if isinstance(pos, str):
                parts = [p.strip() for p in pos.strip("[]()").split(",") if p.strip()]
                if len(parts) != 2:
                    raise ValueError(f"Invalid position string: {pos}")
                parsed_pos = [int(parts[0]), int(parts[1])]
            elif isinstance(pos, list) and len(pos) == 2:
                parsed_pos = [int(pos[0]), int(pos[1])]
            else:
                raise ValueError(f"Invalid position type: {type(pos)} => {pos}")

            return parsed_pos  # [x, y] in 0~1000 space

        max_retries = 3
        last_err = None
        parsed_pos = None

        for attempt in range(1, max_retries + 1):
            try:
                raw = ask_uitars_without_thinking(content)
                parsed_pos = _parse_position_result(raw)
                last_err = None
                break
            except Exception as e:
                last_err = e
                logger.warning(f"[LLM][pos] attempt {attempt}/{max_retries} failed: {e}")
                time.sleep(0.6 * attempt)  # small backoff

        if parsed_pos is None:
            logger.warning(f"[LLM][pos] all retries failed, skip action. last_err={last_err}")
            return

        real_pos = None
        image_height, image_width = page.img.shape[:2]
        real_pos = (
            int(parsed_pos[0] / 1000 * image_width),
            int(parsed_pos[1] / 1000 * image_height),
        )

        if edge["action"] == "click":
            if real_pos and real_pos != (0, 0):
                edge["position"] = real_pos
                self.device.click(real_pos[0], real_pos[1])
                time.sleep(5)
                return True
            else:
                # self.device.click(edge["position"][0], edge["position"][1])
                return False

        # elif edge["action"] == "long_click":
        #     self.device.long_click(edge["position"][0], edge["position"][1])

        elif edge["action"] == "input":
            if real_pos and real_pos != (0, 0):
                edge["position"] = real_pos
                self.device.click(real_pos[0], real_pos[1])
                time.sleep(5)
            else:
                # self.device.click(edge["position"][0], edge["position"][1])
                return False

            time.sleep(1)
            try:
                if edge["content"]:
                    self.device.input(edge["content"])
                else:
                    self.device.input("test input")
                time.sleep(5)
                return True
            except Exception as e:
                print(f"Input action failed: {e}")
                return False
            

    def get_widgets_from_page(self, page_node_before: PageNode, page_node_after: PageNode, action: str):
        if page_node_after.page.info.bundle != self.app_bundle:
            return None

        if len(self.page_nodes) == 1:
            prompt = initial_page_prompt
        else:
            prompt = get_widgets_from_page_prompt

        human_content = [
            {"type": "input_text", "text": prompt},
        ]

        if page_node_before is not None and page_node_before.page is not None:
            human_content.extend([
                {"type": "input_text", "text": "\nScreenshot before action:"},
                {
                    "type": "input_image",
                    "image_url":f"data:image/jpeg;base64,{page_node_before.page.encoded_img}",
                },
            ])

        human_content.extend([
            {"type": "input_text", "text": "\nScreenshot after action (current page):"},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{page_node_after.page.encoded_img}",
            }
        ])

        if page_node_before is not None and action:
            human_content.append(
                {"type": "input_text", "text": f"\nUser action: {action}"}
            )


        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                raw = ask_uitars(human_content)
                print("=======================get_widgets_from_page========================")
                print("LLM raw output:", raw)
                print("====================================================================")
            except Exception as e:
                print(f"[ERROR] get_widgets_from_page: LLM call failed on attempt {attempt}/{max_retries}: {e}")
                if attempt == max_retries:
                    return None
                continue

            cleaned = clean_llm_json(raw)

            try:
                result = json.loads(cleaned)
                page_node_after.function_description = result["function_description"]

                # if page_node_after.page is not None:
                #     if page_node_after.page.info.ability not in self.app_abilities and page_node_after.page.info.ability != "PopupWindow":
                #         return None

                for widget in result["widgets"]:
                    pos = widget["position"]
                    parsed_pos = None
                    if isinstance(pos, str):
                        try:
                            parsed_pos = [int(p.strip()) for p in pos.strip("[]()").split(",")]
                        except Exception as e:
                            logger.warning(f"Failed to parse position: {pos}, err={e}")
                            parsed_pos = None
                    elif isinstance(pos, list) and len(pos) == 2:
                        parsed_pos = [int(pos[0]), int(pos[1])]
                    else:
                        parsed_pos = None
                    real_pos = None
                    if parsed_pos:
                        image_height, image_width = page_node_after.page.img.shape[:2]  
                        real_pos = (
                            int(parsed_pos[0] / 1000 * image_width),
                            int(parsed_pos[1] / 1000 * image_height),
                        )
         
                    page_node_after.edges.append({ 
                        "description": widget["description"],
                        "action": widget["action"],
                        "position": real_pos,
                        "content": widget.get("content", ""),
                        "is_leaf": widget["is_leaf"],
                        "postcondition": widget["postcondition"],
                        "page_node": None
                    })

                # print("page_node {}".format(page_node_after.index))
                # for idx, edge in enumerate(page_node_after.edges):
                #     print(f"  Edge {idx}: {edge['description']}, action: {edge['action']}, position: {edge['position']}, is_leaf: {edge['is_leaf']}")

                return None
            
            except Exception as e:
                print(f"[ERROR] get_widgets_from_page: JSON parse failed on attempt {attempt}/{max_retries}: {e}")
                print("Raw LLM output:", raw)
                if attempt == max_retries:
                    return None
                
        return None


    def widget_level_test(self, widget_node: PageNode):

        def _find_parent_page_and_edge(widget_node_inner):
            for page_node in self.page_nodes:
                if getattr(page_node, "type", None) != "page":
                    continue
                for edge in getattr(page_node, "edges", []):
                    if edge.get("page_node") is widget_node_inner:
                        return page_node, edge
            return None, None

        def _generate_test_flow(page_node, widget_node_inner):
            page_img_b64 = page_node.page.encoded_img
            widget_desc = getattr(widget_node_inner, "function_description", "") or getattr(
                widget_node_inner, "description", ""
            )

            prompt = f"""
        You are a mobile app GUI testing assistant.

        You are given:
        1) A screenshot of a page.
        2) The natural-language description of ONE interactive widget on this page.

        Your task:
        1. Design a short TEST FLOW (string description) that exercises ONLY this widget.
        2. Predict the EXPECTED RESULT (string description) of this flow.

        Rules:
        - Focus ONLY on this widget.
        - The flow should be a sequence of actions described in natural language (e.g., "1. Click X. 2. Input Y.").
        - Use realistic example text if input is needed.
        - Keep it concise (3-5 steps max).

        Return ONLY one JSON object with exactly two keys: "steps" and "expected_result".

        Format example:
        {{
            "steps": "1. Tap the search bar. 2. Input 'test item'. 3. Tap the search icon.",
            "expected_result": "The search results page is displayed showing items related to 'test item'."
        }}

        Widget description: "{widget_desc}"
        """

            content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_text", "text": "Current page screenshot:"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{page_img_b64}",
                },
            ]

            resp = ask_llm(content=content)
            raw = resp.strip()

            cleaned = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw, flags=re.DOTALL).strip()

            default_result = {
                "steps": "Failed to generate steps",
                "expected_result": "Unknown"
            }

            try:
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    final_steps = data.get("steps", "")
                    final_expected = data.get("expected_result", "")
                    
                    if isinstance(final_steps, list):
                        final_steps = "; ".join([str(s) for s in final_steps])
                    
                    return {
                        "steps": str(final_steps),
                        "expected_result": str(final_expected)
                    }
                else:
                    print(f"[widget_level_test] Parsed data is not a dict: {data}")
                    return default_result

            except json.JSONDecodeError as e:
                print(
                    f"[widget_level_test] JSON parse failed for widget {getattr(widget_node_inner, 'index', '?')}: {e}\nraw={raw}"
                )
                return default_result
            except Exception as e:
                print(f"[widget_level_test] Unexpected error: {e}")
                return default_result

        if getattr(widget_node, "type", None) != "widget":
            print(f"[widget_level_test] Node {getattr(widget_node, 'index', '?')} is not a widget, skip.")
            return

        widget_idx = getattr(widget_node, "index", "?")
        print(f"[widget_level_test] ==== Start widget {widget_idx} ====")

        parent_page, _ = _find_parent_page_and_edge(widget_node)

        task_description = _generate_test_flow(parent_page, widget_node)
        print(f"[widget_level_test] Generated test flow for widget {widget_idx}:")
        print(json.dumps(task_description, indent=2, ensure_ascii=False))

        self._replay_to_page(parent_page)

        path_record = self._test_function(task_description)

        self.detect_bug_from_path_record(path_record)

        print(f"[widget_level_test] ==== Done widget {widget_idx} ====")


    def task_level_test(self):

        TASK_MUTATION_PROMPT = """
You are generating MUTATION test cases for a mobile app functional point.

You will be given:
- Functional point description
- Core logic summary (natural language)
- widget_descriptions: a list of UI widgets/actions descriptions available within this functional point (strings)

HARD CONSTRAINT (MUST FOLLOW):
- You can ONLY interact with widgets that appear in widget_descriptions.
- When you reference a widget, you MUST copy its description EXACTLY and wrap it using <....>.
  Example: Tap <Add note>, Long-press <Item options>, Input "abc" into <Title field>.
- Do NOT mention any other buttons/menus/settings not in widget_descriptions.
- If you cannot generate valid mutations using only these widgets, return an empty list.

Mutation goals (high-risk):
1) Branch mutation: choose alternative widgets/options among the list.
2) Order mutation: swap two independent steps if plausible.
3) Repeat/omit: repeat an operation or omit a step (e.g., try saving without input).
4) Boundary input: empty / very long / special chars ONLY if input is plausible (e.g., a text field widget exists).

Output STRICT JSON ONLY:
{
  "variant_paths": [
    "natural language task 1",
    "natural language task 2"
  ]
}

The length of variant_paths must be less than or equal to 3.
"""


        def _safe_json(raw: str, default):
            text = (raw or "").strip()
            text = re.sub(r"^\s*```[a-zA-Z]*\s*", "", text).strip()
            text = re.sub(r"\s*```\s*$", "", text).strip()
            try:
                return json.loads(text)
            except Exception:
                return default

        def _get_entry_page_node(fdg_node):
            core = getattr(fdg_node, "core_logic", None)
            if isinstance(core, dict):
                entry = core.get("entry_page", None)
                if isinstance(entry, int) and 0 <= entry < len(self.page_nodes):
                    return self.page_nodes[entry]
            return self.page_nodes[0] if self.page_nodes else None

        def _core_logic_to_task(core: dict) -> str:
            logic_text = (core.get("logic") or "").strip()
            return logic_text or "Execute the core steps of this functional point."

        def _collect_widget_descriptions(fdg_node):
            descs = []
            for (pidx, eidx) in getattr(fdg_node, "action_refs", []):
                if not (0 <= pidx < len(self.page_nodes)):
                    continue
                src = self.page_nodes[pidx]
                edges = getattr(src, "edges", [])
                if not (0 <= eidx < len(edges)):
                    continue
                edge = edges[eidx]
                d = (edge.get("description") or edge.get("content") or "").strip()
                if d:
                    descs.append(d)
            return descs

        for fdg_idx, fdg_node in enumerate(self.FDG):
            core = getattr(fdg_node, "core_logic", None)
            if not isinstance(core, dict):
                continue

            entry_page_node = _get_entry_page_node(fdg_node)
            if entry_page_node is None:
                print(f"[TaskTest] Skip FDG[{fdg_idx}] (no entry page).")
                continue

            print(f"\n================= [TaskTest] FDG[{fdg_idx}] =================")
            print(f"[Function] {fdg_node.function_description}")

            main_task = _core_logic_to_task(core)
            print(f"\n[Core Logic Task]\n{main_task}")

            # try:
            #     self._replay_to_page(entry_page_node)
            #     path_record = self._test_function(task_description=main_task)
            #     self.detect_bug_from_path_record(path_record)
            #     pass
            # except Exception as e:
            #     print(f"[TaskTest] Core logic execution failed for FDG[{fdg_idx}]: {e}")

            widget_desc_list = _collect_widget_descriptions(fdg_node)
            mutation_content = [
                {"type": "input_text", "text": TASK_MUTATION_PROMPT},
                {"type": "input_text", "text": f"Functional point description:\n{fdg_node.function_description}"},
                {"type": "input_text", "text": f"Core logic summary:\n{main_task}"},
                {"type": "input_text", "text": "Widget descriptions:"},
                {"type": "input_text", "text": json.dumps(widget_desc_list, ensure_ascii=False)},
            ]

            raw = ""
            try:
                raw = ask_llm(content=mutation_content)
                j = _safe_json(raw, default={"variant_paths": []})
                variants = j.get("variant_paths", []) or []
                if not isinstance(variants, list):
                    variants = []
            except Exception as e:
                print(f"[TaskTest] Mutation generation failed for FDG[{fdg_idx}]: {e}")
                variants = []

            print(f"\n[Variant Paths] ({len(variants)})")
            for i, v in enumerate(variants, 1):
                print(f" - V{i}: {v}")

            # for i, task in enumerate(variants, 1):
            #     try:
            #         self._replay_to_page(entry_page_node)
            #         path_record = self._test_function(task_description=task)
            #         self.detect_bug_from_path_record(path_record)
            #         break
            #     except Exception as e:
            #         print(f"[TaskTest] Variant V{i} failed for FDG[{fdg_idx}]: {e}")


    def app_level_test(self):

        def _safe_json(raw: str, default):
            text = (raw or "").strip()
            text = re.sub(r"^\s*```[a-zA-Z]*\s*", "", text).strip()
            text = re.sub(r"\s*```\s*$", "", text).strip()
            try:
                return json.loads(text)
            except Exception:
                return default

        def _entry_page_node(fdg_node):
            core = getattr(fdg_node, "core_logic", None)
            if isinstance(core, dict):
                ep = core.get("entry_page", None)
                if isinstance(ep, int) and 0 <= ep < len(self.page_nodes):
                    return self.page_nodes[ep]
            pns = getattr(fdg_node, "page_nodes", None)
            if isinstance(pns, list) and len(pns) > 0:
                return pns[0]
            return self.page_nodes[0] if self.page_nodes else None

        def _core_to_task(fdg_node):
            core = getattr(fdg_node, "core_logic", None)
            if isinstance(core, dict):
                return (core.get("logic") or "").strip()
            return (fdg_node.function_description or "").strip()

        def _collect_dependency_pairs():
            """
            Collect all (producer_idx, consumer_idx) dependency pairs.
            We assume: consumer.data_dependencies means consumer -> [producers].
            """
            pairs = []
            seen = set()

            for producer_idx, node in enumerate(self.FDG):
                consumers = getattr(node, "data_dependencies", None)
                if not consumers or not isinstance(consumers, list):
                    continue

                for consumer_idx in consumers:
                    if not isinstance(consumer_idx, int):
                        continue
                    if not (0 <= consumer_idx < len(self.FDG)):
                        continue

                    key = (producer_idx, consumer_idx)
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append(key)

            return pairs


        # -------------------------
        # 0) collect all dependency pairs
        # -------------------------
        pairs = _collect_dependency_pairs()
        if not pairs:
            print("[AppTest] No dependency pairs found in FDG.data_dependencies.")
            return

        print(f"[AppTest] Found {len(pairs)} dependency pairs to test.")

        # -------------------------
        # 1) LLM plan prompt (same)
        # -------------------------
        APP_LEVEL_PLAN_PROMPT = """
You are an expert in cross-functional testing for mobile apps.

You are given two functional points with a data dependency: Producer -> Consumer.
Producer produces/updates some abstract data_out that may be consumed by Consumer via data_in.

Your task:
Generate a LIST of cross-functional test cases to validate the dependency across these two functions.

Requirements:
- Each test case MUST execute the Producer first to create/update the required data, then execute the Consumer to use/view/edit that data.
- Provide clear natural-language task instructions that a UI testing agent can follow.
- Focus on verifying the dependency: the Consumer should reflect/use the data produced by the Producer.
- Do NOT invent steps that are not supported by the provided core_logic steps; you may rephrase core_logic into tasks.
- Output no more than 2 test cases.

Output STRICT JSON ONLY (and nothing else):
[
  {
    "producer_task": "<natural-language task for producer core logic (concise; you may include 1–3 sub-steps)>",
    "consumer_task": "<natural-language task for consumer core logic (concise; you may include 1–3 sub-steps; explicitly reference the data produced)>"
  }
]
"""



        # -------------------------
        # 2) test each dependency pair
        # -------------------------
        for pair_idx, (producer_idx, consumer_idx) in enumerate(pairs, 1):
            fdg_node_before = self.FDG[producer_idx]  # producer
            fdg_node_after = self.FDG[consumer_idx]   # consumer

            print("\n==============================================")
            print(f"[AppTest] Pair {pair_idx}/{len(pairs)}: {producer_idx} -> {consumer_idx}")
            print(f"Producer: {fdg_node_before.function_description}")
            print(f"Consumer: {fdg_node_after.function_description}")

            # ---------- 2.1 LLM plan ----------
            llm_content = [
                {"type": "input_text", "text": APP_LEVEL_PLAN_PROMPT},

                {"type": "input_text", "text": f"Producer index: {producer_idx}"},
                {"type": "input_text", "text": f"Producer description: {fdg_node_before.function_description}"},
                {"type": "input_text", "text": f"Producer data_out: {json.dumps(getattr(fdg_node_before, 'data_out', []), ensure_ascii=False)}"},
                {"type": "input_text", "text": f"Producer core_logic (logic field): {_core_to_task(fdg_node_before)}"},

                {"type": "input_text", "text": f"Consumer index: {consumer_idx}"},
                {"type": "input_text", "text": f"Consumer description: {fdg_node_after.function_description}"},
                {"type": "input_text", "text": f"Consumer data_in: {json.dumps(getattr(fdg_node_after, 'data_in', []), ensure_ascii=False)}"},
                {"type": "input_text", "text": f"Consumer core_logic (logic field): {_core_to_task(fdg_node_after)}"},
            ]

            raw_plan = ""
            try:
                raw_plan = ask_llm(content=llm_content)
                plan_obj = _safe_json(raw_plan, default=[])
            except Exception as e:
                print(f"[AppTest] LLM plan failed for pair {producer_idx}->{consumer_idx}: {e}")
                plan_obj = []

            # -----------------------------
            # Normalize: allow both list and dict (backward compatible)
            # -----------------------------
            if isinstance(plan_obj, dict):
                # old format: {"producer_task": "...", "consumer_task": "..."}
                plan_cases = [plan_obj]
            elif isinstance(plan_obj, list):
                # new format: [{"producer_task": "...", "consumer_task": "..."}, ...]
                plan_cases = [x for x in plan_obj if isinstance(x, dict)]
            else:
                plan_cases = []

            # limit to <= 5 cases as prompt requires (double safety)
            plan_cases = plan_cases[:5]

            # Fallback base tasks (when LLM output missing/empty)
            fallback_producer = _core_to_task(fdg_node_before)
            fallback_consumer = _core_to_task(fdg_node_after)

            print("\n[Plan]")
            if not plan_cases:
                # if nothing parsed, still produce one case using fallback
                print("Case 1/1")
                print("Producer task:\n", fallback_producer)
                print("Consumer task:\n", fallback_consumer)
                plan_cases = [{"producer_task": fallback_producer, "consumer_task": fallback_consumer}]
            else:
                for i, case in enumerate(plan_cases, 1):
                    producer_task = (case.get("producer_task") or "").strip() or fallback_producer
                    consumer_task = (case.get("consumer_task") or "").strip() or fallback_consumer

                    # (optional) write back normalized tasks, helpful if later code uses plan_cases
                    case["producer_task"] = producer_task
                    case["consumer_task"] = consumer_task

                    print(f"Case {i}/{len(plan_cases)}")
                    print("Producer task:\n", producer_task)
                    print("Consumer task:\n", consumer_task)
                    print("-" * 30)


            # # ---------- 2.2 Execute producer ----------
            # path_record = []

            # producer_entry = _entry_page_node(fdg_node_before)
            # if producer_entry is None:
            #     print("[AppTest] Producer entry page not found. Skip this pair.")
            #     continue

            # try:
            #     self._replay_to_page(producer_entry)
            #     pr = self._test_function(task_description=producer_task)
            #     if isinstance(pr, list):
            #         path_record.extend(pr)
            #     else:
            #         path_record.append(pr)
            # except Exception as e:
            #     print(f"[AppTest] Producer execution failed for {producer_idx}->{consumer_idx}: {e}")
            #     continue

            # # ---------- 2.3 Execute consumer ----------
            # consumer_entry = _entry_page_node(fdg_node_after)
            # if consumer_entry is None:
            #     print("[AppTest] Consumer entry page not found. Skip this pair.")
            #     continue

            # try:
            #     self._replay_to_page(consumer_entry)
            #     cr = self._test_function(task_description=consumer_task)
            #     if isinstance(cr, list):
            #         path_record.extend(cr)
            #     else:
            #         path_record.append(cr)
            # except Exception as e:
            #     print(f"[AppTest] Consumer execution failed for {producer_idx}->{consumer_idx}: {e}")
            #     pass

            # # ---------- 2.4 Final check for this pair ----------
            # try:
            #     self.detect_bug_from_path_record(path_record)
            # except Exception as e:
            #     print(f"[AppTest] detect_bug_from_path_record failed for {producer_idx}->{consumer_idx}: {e}")


    def _replay_to_page(self, page_node):
            root = self.page_nodes[0]
            target = page_node

            if target is root:
                # self.device.restart_app(self.app)
                self.device.restart_app_by_bundle(self.app_bundle)
                time.sleep(10)
                return True
            
            q = deque()
            visited = set()

            q.append((root, []))
            visited.add(root)

            shortest_path_edges = None 

            while q:
                current, path_edges = q.popleft()

                if current is target:
                    shortest_path_edges = path_edges
                    break

                for edge in getattr(current, "edges", []):
                    next_node = edge.get("page_node")
                    if next_node is None or getattr(next_node, "type", None) != "page":
                        continue
                    if next_node in visited:
                        continue
                    visited.add(next_node)
                    q.append((next_node, path_edges + [edge]))

            print(
                f"[widget_level_test] Found shortest path with {len(shortest_path_edges)} steps "
                f"to page {getattr(target, 'index', '?')}"
            )

            # self.device.restart_app(self.app)
            self.device.restart_app_by_bundle(self.app_bundle)
            time.sleep(10)

            for edge in shortest_path_edges:
                self._excute_action(edge)

            return True

     
    def _test_function(self, task_description: str):
        page = self.device.dump_page(refresh=True)
        path_record = []  

        print("=== 测试计划 ===")
        print(task_description)
        print("===============\n")

        conversation_history = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": test_function_prompt.format(
                            language="English",
                            instruction=task_description
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{page.encoded_img}"
                    }
                ]
            }
        ]

        parsed_output = None
        MAX_PARSE_RETRIES = 3
        parse_retry_count = 0
        max_operations = 20  

        while True:
            response = ask_uitars_messages(conversation_history)
            logger.debug(f"LLM response: {response}")

            conversation_history.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "input_text",
                            "text": response
                        }
                    ]
                }
            )

            try:
                parsed_output = action_parser.parse_action_output(
                    response,
                    page.img.shape[1],
                    page.img.shape[0]
                )
                if not (
                    parsed_output
                    and parsed_output.get("action") in
                    ["click", "input", "scroll", "press_back", "finished", "long_click"]
                ):
                    raise ValueError(f"无效或不支持的操作: {parsed_output.get('action')}")
                parse_retry_count = 0
            except Exception as e:
                logger.error(
                    f"解析或验证LLM响应失败 (尝试 {parse_retry_count + 1}/{MAX_PARSE_RETRIES})，错误: {e}"
                )
                parse_retry_count += 1
                if parse_retry_count >= MAX_PARSE_RETRIES:
                    logger.error("已达到最大重试次数，任务失败。")
                    break
                time.sleep(1)
                continue

            action_type = parsed_output.get("action")

            path_record.append((page.img, parsed_output.get("description", "")))

            if action_type == "finished":
                logger.info("Task finished as per LLM instruction.")
                break

            if action_type == "click" and parsed_output.get("point"):
                center_pos = parsed_output["point"]
                logger.debug(f"Executing click at coordinates: {center_pos}")
                self.device.click(center_pos[0], center_pos[1])

            elif action_type == "long_click" and parsed_output.get("point"):
                center_pos = parsed_output["point"]
                logger.debug(f"Executing long click at coordinates: {center_pos}")
                self.device.long_click(center_pos[0], center_pos[1])

            elif action_type == "input" and parsed_output.get("content"):
                try:
                    self.device.input(parsed_output["content"])
                except Exception as e:
                    logger.error(f"Input action failed: {e}")

            elif action_type == "press_back":
                self.device.back()

            time.sleep(3)

            page = self.device.dump_page(refresh=True)
            self._is_page_exist(page, llm_open=False)

            max_operations -= 1
            if max_operations <= 0:
                return path_record

            conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{page.encoded_img}"
                        }
                    ]
                }
            )

            new_history = []
            image_count = 0

            for message in reversed(conversation_history):
                is_image_message = False

                if message.get("role") == "user":
                    content = message.get("content", [])
                    for part in content:
                        if part.get("type") == "input_image":
                            is_image_message = True
                            break

                if is_image_message:
                    if image_count < 5:
                        new_history.append(message)
                        image_count += 1
                else:
                    new_history.append(message)

            conversation_history = list(reversed(new_history))

        return path_record


    def detect_bug_from_path_record(self, path_record, logger=None):        
        if not path_record:
            return {"has_bug": False, "bug_type": "none", "bug_description": ""}

        llm_content = [
            {"type": "input_text", "text": path_bug_detection_prompt},
            {"type": "input_text", "text": "\nHere is the execution path record (Sequence of steps):\n"}
        ]

        for idx, (img_data, desc) in enumerate(path_record):
            step_info = f"\n--- Step {idx + 1} ---"
            if desc:
                step_info += f"\nDescription: {desc}"
            
            llm_content.append({"type": "input_text", "text": step_info})

            if img_data is not None:
                try:
                    _, buffer = cv2.imencode('.jpg', img_data)
                    img_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    llm_content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img_b64}"
                    })
                except Exception as e:
                    if logger:
                        logger.warning(f"[PathDetect] Failed to encode image at step {idx}: {e}")
                    llm_content.append({"type": "input_text", "text": "[Image missing due to encoding error]"})
            else:
                llm_content.append({"type": "input_text", "text": "[No Image available for this step]"})

        llm_content.append({"type": "input_text", "text": "\nBased on the sequence above, please output the JSON analysis."})
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = ask_llm(llm_content)
                
                cleaned_response = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw_response.strip(), flags=re.DOTALL).strip()
                
                result = json.loads(cleaned_response)
                
                if "has_bug" in result:
                    bug_info = {
                        "has_bug": bool(result.get("has_bug", False)),
                        "bug_type": result.get("bug_type", "none"),
                        "bug_description": (result.get("bug_description", "") or "").strip()
                    }
                    break
                    
            except json.JSONDecodeError:
                if logger:
                    logger.error(f"[PathDetect] JSON parsing failed on attempt {attempt}")
            except Exception as e:
                if logger:
                    logger.error(f"[PathDetect] LLM call error on attempt {attempt}: {e}")
                
        if bug_info.get("has_bug", False):
            try:
                output_dir = getattr(self, "output_dir", "output")
                os.makedirs(output_dir, exist_ok=True)

                if not hasattr(self, "bug_counter"):
                    self.bug_counter = 0
                self.bug_counter += 1

                bug_dir = os.path.join(output_dir, f"bug{self.bug_counter}")
                os.makedirs(bug_dir, exist_ok=True)

                for idx, (img_data, desc) in enumerate(path_record):
                    if img_data is None:
                        continue
                    img_path = os.path.join(bug_dir, f"step_{idx + 1:02d}.png")
                    try:
                        cv2.imwrite(img_path, img_data)
                    except Exception as e_img:
                        if logger:
                            logger.warning(f"[BUG-DETECT] failed to save step image {img_path}: {e_img}")

                bug_json = {
                    "bug_info": bug_info,
                    "path_record": [
                        {
                            "step": idx + 1,
                            "description": desc or ""
                        }
                        for idx, (img_data, desc) in enumerate(path_record)
                    ]
                }

                json_path = os.path.join(bug_dir, "bug.json")
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(bug_json, f, ensure_ascii=False, indent=2)
                except Exception as e_json:
                    if logger:
                        logger.warning(f"[BUG-DETECT] failed to save bug.json for {bug_dir}: {e_json}")

                if logger:
                    logger.info(f"[BUG-DETECT] bug saved to {bug_dir}")
            except Exception as e_outer:
                if logger:
                    logger.warning(f"[BUG-DETECT] unexpected error when saving bug record: {e_outer}")

        return bug_info


    def _is_page_exist(self, page: Page, llm_open=True) -> int:
        if page.info is None or not page.info.ability:
            return len(self.page_nodes)
        
        current_ability = page.info.ability

        if current_ability not in self.explored_abilities:
            self.explored_abilities.append(current_ability)
            return len(self.page_nodes)

        found_indices = [
            p.index for p in self.page_nodes
            if p.page and p.page.info and p.page.info.ability == current_ability and p.type == "page"
        ]
        # print(found_indices)

        if not found_indices:
            return len(self.page_nodes)

        if page.vht_hash:
            for index in found_indices:
                candidate_page = self.page_nodes[index].page
                distance = page.img_hash - candidate_page.img_hash
                if page.vht_hash == candidate_page.vht_hash:
                    if distance <= 4:
                        return index
                if distance <= 2:
                    return index

                
        similarities = []
        for index in found_indices:
            candidate_page = self.page_nodes[index].page
            sim = self._page_similarity(page, candidate_page)
            similarities.append((sim, index))
        # print(similarities)   

        similarities.sort(key=lambda x: x[0], reverse=True)

        best_sim, best_index = similarities[0]

        if best_sim > 0.90:
            return best_index
        # elif best_sim < 0.50:
        #     return len(self.page_nodes)
        else:
            if not similarities:
                return len(self.page_nodes)
            
            candidate_node = self.page_nodes[best_index]

            content = [
                {"type": "input_text", "text": page_exist_prompt},
                {"type": "input_text", "text": "--- \n## New Screenshot"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{page.encoded_img}",
                },
                {"type": "input_text", "text": "\n## Most Similar Candidate Page"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{candidate_node.page.encoded_img}"
                }
            ]

            try:
                response_text = ask_llm(content).strip()

                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    response_text = json_match.group(1)
                response_json = json.loads(response_text)

                if response_json.get("is_same_page", False):
                    return best_index  
                else:
                    return len(self.page_nodes) 
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.error(f"处理LLM单图对比响应时出错: {e}。将默认视为新页面。")
                return len(self.page_nodes)


    def _page_similarity(self, page1, page2) -> float:

        def _calculate_vht_similarity(page1, page2):
            if not all([page1, page2, page1.vht, page2.vht, page1.vht._root, page2.vht._root]):
                return 0.0

            if page1.vht_hash and page1.vht_hash == page2.vht_hash:
                return 1.0

            features1 = page1.feature_set
            features2 = page2.feature_set

            if not features1 and not features2:
                return 1.0 
            if not features1 or not features2:
                return 0.0  

            intersection = len(features1.intersection(features2)) 
            union = len(features1.union(features2))

            return intersection / union if union > 0 else 0.0
        
        vht_sim = _calculate_vht_similarity(page1, page2)
        # print(f"VHT similarity: {vht_sim}")

        img_sim = 0.0
        if page1.img_hash and page2.img_hash:
            distance = page1.img_hash - page2.img_hash
            # print(f"Image hash distance: {distance}")
            img_sim = 1.0 - (distance / 64.0)

        # print(f"Image hash similarity: {img_sim}")

        return vht_sim * VHT_WEIGHT + img_sim * IMG_WEIGHT  


    def dectect_bug(self):
        while self._bug_detector_running:
            task = self.bug_queue.get()   
            try:
                self._detect_bug_once(
                    page_before=task["page_before"],
                    page_after=task["page_after"],
                    action=task["action"],
                    expected_state=task["expected_state"],
                )
            except Exception as e:
                logger.error(f"[BUG-DETECT] detect bug task failed: {e}")
            finally:
                self.bug_queue.task_done()


    def _detect_bug_once(
        self,
        page_before: Page | None,
        page_after: Page,
        action: str,
        expected_state: str,
    ) -> dict | None:
        if page_before is None:
            return None
        human_content = [
            {"type": "input_text", "text": bug_detection_prompt},
            {"type": "input_text", "text": "\nBefore action screenshot:"},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{page_before.encoded_img}"
            },
            {"type": "input_text", "text": "\nAfter action screenshot:"},
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{page_after.encoded_img}"
            },
            {"type": "input_text", "text": f"\nUser action: {action}"},
        ]

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                raw = ask_llm(human_content)
                # print("=====================dectect_bug======================")
                # print("LLM raw output:", raw)
                # print("======================================================")
            except Exception as e:
                logger.error(f"[BUG-DETECT] LLM call failed on attempt {attempt}/{max_retries}: {e}")
                if attempt == max_retries:
                    return None
                continue

            cleaned = clean_llm_json(raw)

            try:
                result = json.loads(cleaned)
                has_bug = bool(result.get("has_bug", False))
                bug_type = result.get("bug_type", "none")
                bug_desc = result.get("bug_description", "").strip()

                bug_info = {
                    "has_bug": has_bug,
                    "bug_type": bug_type,
                    "bug_description": bug_desc,
                    "action": action,
                    "expected_state": expected_state,
                }

                if has_bug:
                    # # KB Recheck for Functional Bugs
                    # if bug_type == "functional":
                    #     kb_query = f"""User action:\n{action}\nExpected state:\n{expected_state}""".strip()

                    #     kb_cases = self.kb_retriever.retrieve(kb_query, topk=2)

                    #     if kb_cases:
                    #         recheck_content = [
                    #             {
                    #                 "type": "input_text",
                    #                 "text": bug_recheck_prompt
                    #             },

                    #             {"type": "input_text", "text": "\n=== CURRENT CASE ==="},
                    #             {"type": "input_text", "text": "\nBefore action screenshot:"},
                    #             {
                    #                 "type": "input_image",
                    #                 "image_url": f"data:image/jpeg;base64,{page_before.encoded_img}"
                    #             },
                    #             {"type": "input_text", "text": "\nAfter action screenshot:"},
                    #             {
                    #                 "type": "input_image",
                    #                 "image_url": f"data:image/jpeg;base64,{page_after.encoded_img}"
                    #             },
                    #             {"type": "input_text", "text": f"\nUser action: {action}"},
                    #             {"type": "input_text", "text": f"\nExpected state: {expected_state}"},
                    #         ]

                    #         for idx, case in enumerate(kb_cases, start=1):
                    #             case_dir = case["path"]

                    #             recheck_content.append({
                    #                 "type": "input_text",
                    #                 "text": f"\n=== NON-BUG REFERENCE EXAMPLE {idx} ==="
                    #             })
                    #             recheck_content.append({
                    #                 "type": "input_text",
                    #                 "text": (
                    #                     f"Action: {case['action']}\n"
                    #                     f"Function: {case['function']}"
                    #                 )
                    #             })

                    #             before_img = case_dir / "before.png"
                    #             after_img = case_dir / "after.png"

                    #             if before_img.exists():
                    #                 recheck_content.append({
                    #                     "type": "input_image",
                    #                     "image_url": f"data:image/png;base64,{encode_image(before_img)}"
                    #                 })

                    #             if after_img.exists():
                    #                 recheck_content.append({
                    #                     "type": "input_image",
                    #                     "image_url": f"data:image/png;base64,{encode_image(after_img)}"
                    #                 })

                    #         raw = ask_llm(recheck_content)
                    #         cleaned = clean_llm_json(raw)
                    #         recheck_result = json.loads(cleaned)

                    #         has_bug = bool(recheck_result.get("has_bug", False))
                    #         bug_desc = recheck_result.get("bug_description", "").strip()

                    #         bug_info = {
                    #             "has_bug": has_bug,
                    #             "bug_type": "functional",
                    #             "bug_description": bug_desc,
                    #             "action": action,
                    #             "expected_state": expected_state,
                    #         }

                    if has_bug:
                        output_dir = getattr(self, "output_dir", "output")
                        os.makedirs(output_dir, exist_ok=True)

                        self.bug_counter += 1
                        bug_dir = os.path.join(output_dir, f"bug{self.bug_counter}")
                        os.makedirs(bug_dir, exist_ok=True)

                        try:
                            if page_before is not None and page_before.img is not None:
                                before_path = os.path.join(bug_dir, "before.png")
                                cv2.imwrite(before_path, page_before.img)
                            if page_after is not None and page_after.img is not None:
                                after_path = os.path.join(bug_dir, "after.png")
                                cv2.imwrite(after_path, page_after.img)
                        except Exception as e_img:
                            logger.warning(f"[BUG-DETECT] failed to save screenshots for {bug_dir}: {e_img}")

                        try:
                            json_path = os.path.join(bug_dir, "bug.json")
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(bug_info, f, ensure_ascii=False, indent=2)
                        except Exception as e_json:
                            logger.warning(f"[BUG-DETECT] failed to save bug.json for {bug_dir}: {e_json}")

                        # logger.info(f"[BUG-DETECT] bug saved to {bug_dir}")
                
                return bug_info

            except Exception as e:
                logger.error(f"[BUG-DETECT] JSON parse failed on attempt {attempt}/{max_retries}: {e}")
                logger.error(f"[BUG-DETECT] raw cleaned text: {cleaned}")
                if attempt == max_retries:
                    return None

        return None


    def save_PTG(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        pages_root_dir = os.path.join(out_dir, f"pages_{timestamp}")
        os.makedirs(pages_root_dir, exist_ok=True)

        file_path = os.path.join(out_dir, f"ptg_report_{timestamp}.json")

        ptg_report = {
            "app_bundle": self.app_bundle,
            "explored_abilities": self.explored_abilities,
            "nodes": []
        }

        for node in self.page_nodes:
            page_details = None

            if node.page is not None and node.type == "page":
                page_subdir = os.path.join(pages_root_dir, str(node.index))
                os.makedirs(page_subdir, exist_ok=True)

                screenshot_rel_path = None
                vht_rel_path = None

                if getattr(node.page, "img", None) is not None:
                    try:
                        screenshot_path = os.path.join(page_subdir, "screenshot.png")
                        cv2.imwrite(screenshot_path, node.page.img)
                        screenshot_rel_path = os.path.relpath(screenshot_path, out_dir)
                    except Exception as e:
                        logger.error(f"Failed to save screenshot for node {node.index}: {e}")

                if getattr(node.page, "vht", None) is not None:
                    try:
                        vht_path = os.path.join(page_subdir, "vht.json")
                        VHTParser.dump(node.page.vht, vht_path, indent=2)
                        vht_rel_path = os.path.relpath(vht_path, out_dir)
                    except Exception as e:
                        logger.error(f"Failed to save VHT for node {node.index}: {e}")

                page_details = {
                    "bundle": node.page.info.bundle if node.page.info else None,
                    "ability": node.page.info.ability if node.page.info else None,
                    "screenshot_path": screenshot_rel_path,
                    "vht_path": vht_rel_path,
                }

            processed_edges = []
            for edge in getattr(node, "edges", []):
                edge_copy = edge.copy()
                if edge_copy.get("page_node") is not None:
                    edge_copy["page_node"] = edge_copy["page_node"].index
                processed_edges.append(edge_copy)

            node_info = {
                "index": node.index,
                "type": getattr(node, "type", None),
                "function_description": getattr(node, "function_description", ""),
                "page_details": page_details,
                "edges": processed_edges,
                "is_visited": getattr(node, "is_visited", False),
            }

            ptg_report["nodes"].append(node_info)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(ptg_report, f, indent=4, ensure_ascii=False)
            logger.info(f"PTG report successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save PTG report JSON: {e}")


    def read_PTG(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PTG report file not found: {file_path}")

        output_dir = os.path.dirname(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            ptg_report = json.load(f)

        self.app_bundle = ptg_report.get("app_bundle")
        self.explored_abilities = ptg_report.get("explored_abilities", [])

        node_map: dict[int, PageNode] = {}

        for node_info in ptg_report["nodes"]:
            index = node_info["index"]
            page_details = node_info.get("page_details")
            page = None

            if page_details and (page_details.get("screenshot_path") or page_details.get("vht_path")):
                img = None
                vht = None
                info = None

                screenshot_rel = page_details.get("screenshot_path")
                if screenshot_rel:
                    screenshot_path = os.path.join(output_dir, screenshot_rel)
                    if os.path.exists(screenshot_path):
                        img = cv2.imread(screenshot_path)

                vht_rel = page_details.get("vht_path")
                if vht_rel:
                    vht_path = os.path.join(output_dir, vht_rel)
                    if os.path.exists(vht_path):
                        try:
                            with open(vht_path, 'r', encoding='utf-8') as vf:
                                vht_json = json.load(vf)
                            vht = VHTParser._parse_hdc_json(vht_json, device=None)
                        except Exception as e:
                            logger.error(f"Failed to parse VHT for node {index}: {e}")

                bundle = page_details.get("bundle")
                ability = page_details.get("ability")
                if bundle or ability:
                    info = PageInfo(bundle, ability, ability)

                if vht is not None or img is not None:
                    try:
                        page = Page(vht=vht, img=img, info=info)
                    except Exception as e:
                        logger.error(f"Failed to reconstruct Page for node {index}: {e}")
                        page = None

            node = PageNode(index=index, page=page)
            node.type = node_info.get("type", "page")
            node.function_description = node_info.get("function_description", "")
            node.is_visited = node_info.get("is_visited", True)
            node_map[index] = node

        for node_info in ptg_report["nodes"]:
            index = node_info["index"]
            node = node_map[index]
            edges = node_info.get("edges", [])
            restored_edges = []

            for edge in edges:
                edge_copy = edge.copy()
                if edge_copy.get("page_node") is not None:
                    ref_index = edge_copy["page_node"]
                    edge_copy["page_node"] = node_map.get(ref_index)
                restored_edges.append(edge_copy)

            node.edges = restored_edges

        self.page_nodes = list(node_map.values())

        logger.info(f"PTG successfully loaded from {file_path} with {len(self.page_nodes)} nodes.")


    def save_FDG(self, filepath: str):
        fdg_data = []

        for i, node in enumerate(self.FDG):
            fdg_data.append({
                "index": i,
                "function_description": node.function_description,

                "action_refs": [[pidx, eidx] for (pidx, eidx) in getattr(node, "action_refs", [])],

                "data_in": node.data_in,
                "data_out": node.data_out,
                "data_dependencies": node.data_dependencies,

                "to_test": node.to_test,

                "core_logic": getattr(node, "core_logic", None),
            })

        result = {"FDG": fdg_data}

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"FDG saved successfully to: {filepath}")
        except Exception as e:
            print(f"Failed to save FDG: {e}")


    def read_FDG(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load FDG file: {e}")
            return

        fdg_nodes = data.get("FDG", [])
        self.FDG = []

        def _as_list(x):
            return x if isinstance(x, list) else []

        for _, item in enumerate(fdg_nodes):
            # 1) function_description
            fdg_node = FDGNode(function_description=item.get("function_description", "") or "")

            # 2) action_refs: [[pidx,eidx], ...] -> [(pidx,eidx), ...]
            action_refs_raw = _as_list(item.get("action_refs", []))
            action_refs = []
            for ref in action_refs_raw:
                if isinstance(ref, (list, tuple)) and len(ref) == 2:
                    try:
                        pidx = int(ref[0])
                        eidx = int(ref[1])
                        action_refs.append((pidx, eidx))
                    except Exception:
                        continue
            fdg_node.action_refs = action_refs

            # 3) data fields
            fdg_node.data_in = _as_list(item.get("data_in", []))
            fdg_node.data_out = _as_list(item.get("data_out", []))
            fdg_node.data_dependencies = _as_list(item.get("data_dependencies", []))

            # 4) to_test
            fdg_node.to_test = bool(item.get("to_test", False))

            # 5) core_logic
            fdg_node.core_logic = item.get("core_logic", None)

            self.FDG.append(fdg_node)

        print(f"Loaded {len(self.FDG)} functional nodes from {filepath}")


    def test(self, ptg_file_path: str, fdg_file_path: str):
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        class Tee:
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()

            def flush(self):
                for s in self.streams:
                    s.flush()

        base, ext = os.path.splitext(fdg_file_path)
        if not ext:
            ext = ".json"
            fdg_file_path = base + ext 

        new_fdg_file_path = f"{base}_with_data_dep{ext}"

        fdg_dir = os.path.dirname(os.path.abspath(new_fdg_file_path))
        os.makedirs(fdg_dir, exist_ok=True)

        log_path = os.path.join(fdg_dir, "test.log")  
        self.read_PTG(ptg_file_path)
        self.read_FDG(fdg_file_path) 

        with open(log_path, "a", encoding="utf-8") as f:
            tee = Tee(sys.stdout, f)

            with redirect_stdout(tee), redirect_stderr(tee):
                self.task_level_test()
                print("\n")
                self.app_level_test()



    def calculate_activity_coverage(self, file_path):
        self.read_PTG(file_path)
        explored_abilities = set()
        for ability in self.explored_abilities:
            if ability in self.app_abilities:
                explored_abilities.add(ability)
        
        try:
            while True:
                page = self.device.dump_page(refresh=True)
                print(f"Current ability: {page.info.ability if page.info else 'Unknown'}")
                if page.info and page.info.ability in self.app_abilities and page.info.ability not in explored_abilities:
                    explored_abilities.add(page.info.ability)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            activity_coverage = len(explored_abilities) / len(self.app_abilities)
            print("Explored Abilities:")
            for ability in explored_abilities:
                print(f"{ability}")
            print(f"Activity Coverage: {len(explored_abilities)}/{len(self.app_abilities)} ({activity_coverage:.2f})")


    def full_explorer(self, output_dir: str, ptg_file_path: str = None):
        self.output_dir = output_dir
        self.explore(output_dir=output_dir)

        ptg_file_path = os.path.join(output_dir, "ptg.json")
        fdg_file_path = os.path.join(output_dir, "fdg.json")

        self.build_FDG(ptg_file_path)
        self.build_FDG_with_dependency(ptg_file_path, fdg_file_path)

        # Task-level testing
        self.task_level_test()

        # App-level testing
        self.app_level_test()
        


    def _maybe_dump_activity_coverage(self):
        """call frequently; dump ACTIVITY coverage at most once per minute"""
        if not self.output_dir:
            return
        now = time.time()
        if now - self._last_act_cov_ts < self._act_cov_interval_sec:
            return
        self._last_act_cov_ts = now
        self._dump_activity_coverage_once()

    def _dump_activity_coverage_once(self):
        if not self._act_cov_path:
            return

        declared = sorted(self._declared_activities)
        visited = sorted(self._visited_activities)
        declared_set = set(declared)

        hit = [a for a in visited if a in declared_set]
        cov = (len(hit) / len(declared)) if declared else 0.0

        snap = {
            "ts": time.time(),
            "declared_count": len(declared),
            "visited_count": len(visited),
            "hit_count": len(hit),
            "activity_coverage": round(cov, 6),
            "hit_activities": hit,         
            "visited_activities": visited,  
        }

        Path(self._act_cov_path).write_text(
            json.dumps(snap, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        if self._act_cov_hist_path:
            with open(self._act_cov_hist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False) + "\n")

