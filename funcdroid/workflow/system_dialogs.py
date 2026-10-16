"""System permission windows are executable observations, not app-launch failures."""

PERMISSION_DIALOG_PACKAGES = frozenset({
    "com.android.permissioncontroller",
    "com.google.android.permissioncontroller",
})


def is_permission_dialog(package_name: str) -> bool:
    return str(package_name or "").strip() in PERMISSION_DIALOG_PACKAGES
