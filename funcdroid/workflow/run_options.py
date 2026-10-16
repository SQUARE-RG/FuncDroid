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
        raise ValueError("缺陷与合规检测必须分开运行，请选择 --mode defect 或 --mode compliance")
    args.mode = mode
    args.bug_detection = mode == "defect" and bug is not False
    args.compliance = mode == "compliance"
    if getattr(args, "compliance_targeted", False) and mode != "compliance":
        raise ValueError("--compliance-targeted 仅适用于 --mode compliance")
    reset = getattr(args, "reset_app_data", None)
    resume = bool(getattr(args, "resume_output", None))
    seed = bool(getattr(args, "seed_ffg", None))
    if seed and (resume or reset is True or not getattr(args, "package_name", None)):
        raise ValueError("FFG 种子要求明确包名、保留应用数据且不能同时续跑")
    if resume and reset is True:
        raise ValueError("续跑必须保留应用数据，不能使用 --reset-app-data")
    args.reset_app_data = False if resume or seed else (mode == "compliance" if reset is None else reset)
    if args.reset_app_data and not getattr(args, "package_name", None):
        raise ValueError("清除应用数据必须显式指定 --package-name；保留现有状态请使用 --no-reset-app-data")
    # 缺陷检测关注功能而非授权流程，默认在启动前补齐运行期权限，避免逐条权限
    # 弹窗消耗场景步数；合规检测的检测对象正是权限申请与同意流程，默认不预授予。
    # --package-name 仍可省略（此时测试前台应用），那种情况下无法确定目标包名，
    # 由调用方在启动阶段跳过授予，不在这里报错。
    grant = getattr(args, "grant_permissions", None)
    args.grant_permissions = (mode == "defect") if grant is None else bool(grant)
    if getattr(args, "rules", None) and mode != "compliance":
        raise ValueError("--rules 仅适用于 --mode compliance")
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
        raise ValueError("续跑缺少 run_result.json，无法验证应用和检测模式")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not result.get("package_name"):
        raise ValueError("续跑记录缺少 package_name，无法验证目标应用")
    previous_mode = result.get("mode")
    if not previous_mode:
        # Older compliance runs did not persist a mode; their checklist is the marker.
        previous_mode = "compliance" if (root / "compliance_checklist.json").is_file() else "defect"
        if previous_mode == "compliance" and result.get("bug_detection_enabled", False):
            raise ValueError("旧目录同时启用了两种检测，请为独立模式创建新输出目录")
    if previous_mode != args.mode:
        raise ValueError(f"续跑模式不匹配：目录为 {previous_mode}，请求为 {args.mode}")
    if args.package_name and args.package_name != result["package_name"]:
        raise ValueError("续跑包名与原运行不一致")
    args.package_name = result["package_name"]
    if not getattr(args, "app_name", None):
        args.app_name = result.get("app_name") or args.package_name
    snapshot = root.resolve() / "compliance_rules.json"
    if args.mode == "compliance" and snapshot.is_file():
        if getattr(args, "rules", None):
            # 快照文件是运行开始时 load_rules() 归一化后的结果（无 "version" 包裹，
            # 且已过滤掉无 rule_id/clause 的条目）；直接比较原始 JSON 结构必然不等
            # （一个带 version 字段、一个不带，且未经过滤）。改为让两边都经过同一个
            # 归一化函数再比较，只关心"生效的规则集合"是否一致。
            from compliance.rule_library import load_rules

            original = load_rules(str(snapshot))
            requested = load_rules(str(Path(args.rules)))
            if original != requested:
                raise ValueError("续跑不能更换合规规则库，请创建新输出目录")
        args.rules = str(snapshot)
