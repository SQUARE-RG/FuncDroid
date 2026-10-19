"""Resolve independent detection modes before opening devices or output files."""

import argparse
import json
from pathlib import Path


def resolve_run_options(args):
    mode = getattr(args, "mode", None)
    bug = getattr(args, "bug_detection", None)
    compliance = getattr(args, "compliance", None)
    mode = mode or ("compliance" if compliance is True else "defect")
    if (mode == "defect" and compliance is True) or (
        mode == "compliance" and (bug is True or compliance is False)
    ):
        raise ValueError("Run functional bug detection and compliance testing separately with --mode defect or --mode compliance")
    args.mode = mode
    args.bug_detection = mode == "defect" and bug is not False
    args.compliance = mode == "compliance"
    if getattr(args, "compliance_targeted", False) and mode != "compliance":
        raise ValueError("--compliance-targeted requires --mode compliance")
    reset = getattr(args, "reset_app_data", None)
    resume = bool(getattr(args, "resume_output", None))
    seed = bool(getattr(args, "seed_ffg", None))
    if seed and (resume or reset is True or not getattr(args, "package_name", None)):
        raise ValueError("FFG seeding requires an explicit package, preserved app data, and no simultaneous resume")
    if resume and reset is True:
        raise ValueError("Resuming must preserve app data; --reset-app-data is not allowed")
    args.reset_app_data = False if resume or seed else (mode == "compliance" if reset is None else reset)
    if args.reset_app_data and not getattr(args, "package_name", None):
        raise ValueError("Clearing app data requires --package-name; use --no-reset-app-data to preserve state")
    # Functional testing pre-grants permissions so dialogs do not consume scenario steps.
    # Compliance testing examines permission/consent requests and does not pre-grant.
    # An omitted package means testing the foreground app; the caller skips granting
    # when the target package is not explicit rather than rejecting the run here.
    grant = getattr(args, "grant_permissions", None)
    args.grant_permissions = (mode == "defect") if grant is None else bool(grant)
    if getattr(args, "rules", None) and mode != "compliance":
        raise ValueError("--rules requires --mode compliance")
    return args


class RunArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        try:
            return resolve_run_options(parsed)
        except ValueError as exc:
            self.error(str(exc))


def validate_resume_options(args):
    """Validate identity before logging, restarting an app, or modifying artifacts."""
    if not getattr(args, "resume_output", None):
        return
    root = Path(args.resume_output)
    path = root / "run_result.json"
    if not path.is_file():
        raise ValueError("Resume requires run_result.json to verify the app and detection mode")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not result.get("package_name"):
        raise ValueError("Resume metadata has no package_name to identify the target app")
    previous_mode = result.get("mode")
    if not previous_mode:
        # Older compliance runs did not persist a mode; their checklist is the marker.
        previous_mode = "compliance" if (root / "compliance_checklist.json").is_file() else "defect"
        if previous_mode == "compliance" and result.get("bug_detection_enabled", False):
            raise ValueError("Legacy output enabled both modes; create a new directory for an independent run")
    if previous_mode != args.mode:
        raise ValueError(f"Resume mode mismatch: saved mode={previous_mode}, requested mode={args.mode}")
    if args.package_name and args.package_name != result["package_name"]:
        raise ValueError("Resume package does not match the original run")
    args.package_name = result["package_name"]
    if not getattr(args, "app_name", None):
        args.app_name = result.get("app_name") or args.package_name
    snapshot = root.resolve() / "compliance_rules.json"
    if args.mode == "compliance" and snapshot.is_file():
        if getattr(args, "rules", None):
            # The snapshot contains normalized load_rules() output, without a version wrapper
            # or entries lacking rule_id/clause. Comparing raw input JSON would therefore
            # report a mismatch for equivalent rule sets. Normalize both inputs identically
            # before comparing the effective rules used by the resumed run.
            from compliance.rule_library import load_rules

            original = load_rules(str(snapshot))
            requested = load_rules(str(Path(args.rules)))
            if original != requested:
                raise ValueError("Cannot replace the compliance rule library on resume; create a new output directory")
        args.rules = str(snapshot)
