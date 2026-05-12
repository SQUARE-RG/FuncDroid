import subprocess


def disable_input_methods():
        """禁用所有输入法，防止输入法弹出干扰"""
        cmd = [
            "adb",
            "shell",
            "settings", "put", "secure",
            "default_input_method",
            "com.android.inputmethod.none/.NullIME"
        ]
        subprocess.run(cmd, capture_output=True, text=True)


def grant_all_permissions(package_name: str):
    dangerous_perms = [
        "android.permission.READ_CALENDAR",
        "android.permission.WRITE_CALENDAR",
        "android.permission.CAMERA",
        "android.permission.READ_CONTACTS",
        "android.permission.WRITE_CONTACTS",
        "android.permission.GET_ACCOUNTS",
        "android.permission.ACCESS_FINE_LOCATION",
        "android.permission.ACCESS_COARSE_LOCATION",
        "android.permission.RECORD_AUDIO",
        "android.permission.READ_PHONE_STATE",
        "android.permission.CALL_PHONE",
        "android.permission.READ_CALL_LOG",
        "android.permission.WRITE_CALL_LOG",
        "android.permission.ADD_VOICEMAIL",
        "android.permission.USE_SIP",
        "android.permission.PROCESS_OUTGOING_CALLS",
        "android.permission.BODY_SENSORS",
        "android.permission.SEND_SMS",
        "android.permission.RECEIVE_SMS",
        "android.permission.READ_SMS",
        "android.permission.RECEIVE_WAP_PUSH",
        "android.permission.RECEIVE_MMS",
        "android.permission.READ_EXTERNAL_STORAGE",
        "android.permission.WRITE_EXTERNAL_STORAGE",
    ]

    for perm in dangerous_perms:
        cmd = ["adb", "shell", "pm", "grant", package_name, perm]
        subprocess.run(cmd, capture_output=True, text=True)


def clean_llm_json(raw: str) -> str:
    """尽量从 LLM 返回里提取干净的 JSON 字符串."""
    if not isinstance(raw, str):
        raw = str(raw)
    text = raw.strip()

    # 1. 去掉 ```json ... ``` 或 ``` ... ``` 这种代码块包裹
    if text.startswith("```"):
        lines = text.splitlines()
        # 去掉第一行 ``` 或 ```json
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        # 去掉最后一行 ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # 2. 尝试截取第一个 '{' 到最后一个 '}' 之间的内容
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        text = text[start:end + 1].strip()

    return text