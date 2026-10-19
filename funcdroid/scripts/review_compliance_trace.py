"""Review saved GUI evidence with the current detector without operating a device."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

from workflow.compliance_detection import analyze_execution_for_compliance
from workflow.llm import create_llm
from workflow.llm_usage import configure_llm_usage, save_llm_usage_summary


def load_trace(path: Path, rules: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = []
    for raw in data["steps"]:
        page = dict(raw.get("page") or {})
        reference = page.get("screenshot")
        if reference:
            screenshot = (path.parent / reference).resolve()
            if not screenshot.is_relative_to(path.parent.resolve()):
                raise ValueError("Trace screenshot reference escapes the evidence directory")
            page["screenshot"] = base64.b64encode(screenshot.read_bytes()).decode("ascii")
        operation = raw.get("operation") or {}
        steps.append({**raw, **(operation.get("parameters") or {}),
                      "action_name": operation.get("action_name", ""), "current_page": page})
    return {"current_plan": data["plan"], "current_test_scenario": data["scenario"],
            "execution_result": data.get("execution_result", {}),
            "navigation_execution_trace": [], "scenario_execution_trace": steps,
            "compliance_rules_path": str(rules.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    state = load_trace(args.trace.resolve(), args.rules)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configure_llm_usage(str(args.output_dir))
    result = analyze_execution_for_compliance(create_llm(timeout_seconds=90), state)
    result["source_trace"] = str(args.trace.resolve())
    result["source_trace_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    result["scope"] = "Saved-evidence model review; no new device execution or independent reproduction"
    (args.output_dir / "review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_llm_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
