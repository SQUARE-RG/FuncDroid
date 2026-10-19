"""Validate functionality-testing options before touching devices or artifacts."""

import argparse
import json
from pathlib import Path


def resolve_run_options(args):
    mode = getattr(args, "mode", None)
    if mode not in {None, "defect"}:
        raise ValueError("This distribution supports only functional bug detection")
    args.mode = "defect"
    args.bug_detection = getattr(args, "bug_detection", None) is not False
    reset = getattr(args, "reset_app_data", None)
    resume = bool(getattr(args, "resume_output", None))
    seed = bool(getattr(args, "seed_ffg", None))
    if seed and (resume or reset is True or not getattr(args, "package_name", None)):
        raise ValueError("FFG seeding requires an explicit package, preserved app data, and no simultaneous resume")
    if resume and reset is True:
        raise ValueError("Resuming must preserve app data; --reset-app-data is not allowed")
    args.reset_app_data = False if resume or seed else bool(reset)
    if args.reset_app_data and not getattr(args, "package_name", None):
        raise ValueError("Clearing app data requires --package-name; use --no-reset-app-data to preserve state")
    grant = getattr(args, "grant_permissions", None)
    args.grant_permissions = True if grant is None else bool(grant)
    return args


class RunArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        try:
            return resolve_run_options(parsed)
        except ValueError as exc:
            self.error(str(exc))


def validate_resume_options(args):
    """Validate app identity and mode before changing files or application state."""
    if not getattr(args, "resume_output", None):
        return
    root = Path(args.resume_output)
    path = root / "run_result.json"
    if not path.is_file():
        raise ValueError("Resume requires run_result.json to verify the app and detection mode")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not result.get("package_name"):
        raise ValueError("Resume metadata has no package_name to identify the target app")
    # A missing mode cannot establish the identity of an older run safely.
    if result.get("mode") != args.mode:
        raise ValueError(f"Resume mode mismatch: saved mode={result.get('mode')}, requested mode={args.mode}")
    if args.package_name and args.package_name != result["package_name"]:
        raise ValueError("Resume package does not match the original run")
    args.package_name = result["package_name"]
    if not getattr(args, "app_name", None):
        args.app_name = result.get("app_name") or args.package_name
