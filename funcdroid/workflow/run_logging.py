"""Run-level stdout/stderr and device logcat logging."""

from __future__ import annotations

import atexit
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from typing import BinaryIO, Callable, TextIO


RUNTIME_LOG_TERM_REPLACEMENTS = (
    ("基线", "起点"),
    ("批测", "应用队列"),
    ("验收", "运行"),
    ("试跑", "运行"),
    ("实验", "运行"),
    ("测试", "执行"),
)


# 术语替换只作用于框架自身输出的固定日志行：以下列 [标签] 开头的行由框架
# 代码生成、内容可控。其它行（LLM 原始响应 [原始内容]、规划场景 [计划]、
# 执行/判定详情、异常堆栈等）可能逐字包含模型输出或场景文本，全局改写会
# 篡改证据，使日志与实际发送给模型/设备的内容对不上，因此一律原样保留。
RUNTIME_LOG_NORMALIZED_TAGS = frozenset({
    "日志",
    "启动",
    "启动前清理",
    "启动前重置",
    "启动前授权",
    "运行快照",
    "输出",
    "输出修复",
    "产物校验",
    "完成",
    "运行中断",
    "运行异常",
    "路由",
    "轮次",
    "轮次信息",
    "长时运行",
    "持久化",
    "LLM统计",
    "logcat",
})
_RUNTIME_LOG_TAG_PATTERN = re.compile(r"^\s*\[([^\[\]\n]{1,32})\]")
# 启动横幅/轮次横幅/结束摘要中无标签的固定行（框架常量拼接，不含模型输出）。
RUNTIME_LOG_NORMALIZED_PREFIXES = (
    "  第 ",
    "  规划模式:",
    "  探索范围:",
    "  执行轮次:",
)


def _normalize_runtime_log_line(line: str) -> str:
    match = _RUNTIME_LOG_TAG_PATTERN.match(line)
    framework_line = (
        match.group(1) in RUNTIME_LOG_NORMALIZED_TAGS
        if match is not None
        else line.startswith(RUNTIME_LOG_NORMALIZED_PREFIXES)
    )
    if not framework_line:
        return line
    for source, target in RUNTIME_LOG_TERM_REPLACEMENTS:
        line = line.replace(source, target)
    return line


def normalize_runtime_log_text(text: str) -> str:
    """Keep framework runtime messages in product terminology.

    逐行处理，只改写以框架标签开头的行；模型输出、场景文本等动态内容不变。
    """
    text = str(text)
    if not any(source in text for source, _target in RUNTIME_LOG_TERM_REPLACEMENTS):
        return text
    return "".join(
        _normalize_runtime_log_line(line)
        for line in text.splitlines(keepends=True)
    )


class _TeeStream:
    def __init__(
        self,
        primary: TextIO,
        secondary: TextIO,
        lock: threading.RLock | None = None,
    ):
        self.primary = primary
        self.secondary = secondary
        self._secondary_enabled = True
        self._secondary_error_reported = False
        self._lock = lock or threading.RLock()
        self._pending_by_thread: dict[int, str] = {}

    def _disable_secondary(self, exc: BaseException) -> None:
        """Stop mirroring after a file-sink failure without breaking stdout/stderr.

        Logging is an observability aid.  A full disk, a transient filesystem
        error, or a failed rotation must not turn an otherwise healthy test run
        into a workflow exception.  Write the warning directly to the original
        console stream so reporting the logging failure cannot recurse through
        this tee.
        """
        with self._lock:
            self._secondary_enabled = False
            if self._secondary_error_reported:
                return
            self._secondary_error_reported = True
        try:
            self.primary.write(
                "[日志] run.log 写入失败，已停用本次文件镜像；"
                f"控制台输出和测试流程将继续: {exc}\n"
            )
            self.primary.flush()
        except Exception:
            pass

    def _write_complete_text(self, normalized: str) -> None:
        result = self.primary.write(normalized)
        if self._secondary_enabled:
            try:
                self.secondary.write(normalized)
                self.secondary.flush()
            except (OSError, ValueError) as exc:
                self._disable_secondary(exc)
        self.primary.flush()
        return result

    def write(self, text: str):
        # 先按行缓冲再规范化：标签与正文可能分多次 write 到达，按完整行判断标签。
        thread_id = threading.get_ident()
        with self._lock:
            buffered = self._pending_by_thread.get(thread_id, "") + str(text)
            last_newline = buffered.rfind("\n")
            if last_newline < 0:
                self._pending_by_thread[thread_id] = buffered
                return len(str(text))
            complete_text = buffered[:last_newline + 1]
            remainder = buffered[last_newline + 1:]
            if remainder:
                self._pending_by_thread[thread_id] = remainder
            else:
                self._pending_by_thread.pop(thread_id, None)
            self._write_complete_text(normalize_runtime_log_text(complete_text))
        return len(str(text))

    def flush(self):
        with self._lock:
            pending = "".join(self._pending_by_thread.values())
            self._pending_by_thread.clear()
            if pending:
                self._write_complete_text(normalize_runtime_log_text(pending))
            self.primary.flush()
            secondary_enabled = self._secondary_enabled
        if secondary_enabled:
            try:
                self.secondary.flush()
            except (OSError, ValueError) as exc:
                self._disable_secondary(exc)

    def isatty(self):
        return self.primary.isatty()

    @property
    def encoding(self):
        return getattr(self.primary, "encoding", "utf-8")


class _RotatingTextSink:
    """UTF-8 text facade over the shared size-based binary rotator."""

    def __init__(self, path: str, max_bytes: int, backup_count: int, *, append: bool = True):
        self.name = path
        self._sink = _RotatingBinarySink(
            path,
            max_bytes,
            backup_count,
            append=append,
            # 续写头的时间戳在每次滚动时生成，而不是固定为 sink 创建时刻。
            continuation_header=lambda: (
                "===== FuncDroid run log continued: "
                f"{datetime.now().isoformat()} =====\n"
            ).encode("utf-8"),
        )

    def write(self, text: str) -> int:
        encoded = str(text).encode("utf-8", errors="replace")
        self._sink.write(encoded)
        return len(str(text))

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()


class RunLogHandle:
    def __init__(self, log_file: TextIO, stdout: TextIO, stderr: TextIO):
        self._log_file = log_file
        self._stdout = stdout
        self._stderr = stderr
        self._closed = False

    @property
    def path(self) -> str:
        return self._log_file.name

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            try:
                sys.stderr.flush()
            except Exception:
                pass
        finally:
            sys.stdout = self._stdout
            sys.stderr = self._stderr
        try:
            self._log_file.flush()
        except Exception as exc:
            try:
                self._stderr.write(f"[日志] run.log 刷盘失败，关闭流程继续: {exc}\n")
            except Exception:
                pass
        try:
            self._log_file.close()
        except Exception as exc:
            try:
                self._stderr.write(f"[日志] run.log 关闭失败，退出流程继续: {exc}\n")
            except Exception:
                pass


def setup_run_logging(output_dir: str, filename: str = "run.log") -> RunLogHandle:
    """Mirror process stdout/stderr to a log file inside output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)
    rotate_bytes = max(
        0,
        int(os.getenv("RUN_LOG_ROTATE_BYTES", str(64 * 1024 * 1024))),
    )
    backup_count = max(1, int(os.getenv("RUN_LOG_BACKUP_COUNT", "3")))
    log_file = _RotatingTextSink(
        log_path,
        rotate_bytes,
        backup_count,
        append=True,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    handle = RunLogHandle(log_file, original_stdout, original_stderr)

    stream_lock = threading.RLock()
    sys.stdout = _TeeStream(original_stdout, log_file, stream_lock)
    sys.stderr = _TeeStream(original_stderr, log_file, stream_lock)

    print(f"[日志] 命令行输出同步保存到: {log_path}")
    rotation_label = (
        f"单文件 {rotate_bytes} bytes，保留 {backup_count} 个历史文件"
        if rotate_bytes > 0
        else "不滚动"
    )
    print(f"[日志] run.log 滚动策略: {rotation_label}")
    print(f"[日志] 会话开始: {datetime.now().isoformat()}")
    atexit.register(handle.close)
    return handle


class _RotatingBinarySink:
    def __init__(
        self,
        path: str,
        max_bytes: int,
        backup_count: int,
        *,
        append: bool = False,
        continuation_header: bytes | Callable[[], bytes] | None = None,
    ):
        self.path = path
        self.max_bytes = max(0, max_bytes)
        self.backup_count = max(1, backup_count)
        # 可传固定字节串或生成函数；默认在每次滚动时生成带当前时间的续写头。
        self.continuation_header = continuation_header or (
            lambda: (
                "===== FuncDroid logcat capture continued: "
                f"{datetime.now().isoformat()} =====\n"
            ).encode()
        )
        self._lock = threading.RLock()
        self._file: BinaryIO = open(path, "ab" if append else "wb", buffering=0)

    def write(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if (
                self.max_bytes > 0
                and self._file.tell() > 0
                and self._file.tell() + len(data) > self.max_bytes
            ):
                self._rotate()
            self._file.write(data)

    def _rotate(self) -> None:
        self._file.flush()
        self._file.close()
        oldest = f"{self.path}.{self.backup_count}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for index in range(self.backup_count - 1, 0, -1):
            source = f"{self.path}.{index}"
            if os.path.exists(source):
                os.replace(source, f"{self.path}.{index + 1}")
        if os.path.exists(self.path):
            os.replace(self.path, f"{self.path}.1")
        self._file = open(self.path, "wb", buffering=0)
        header = self.continuation_header
        self._file.write(header() if callable(header) else header)

    def flush(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file.closed:
                return
            self._file.flush()
            self._file.close()


class LogcatCaptureHandle:
    def __init__(self, process: subprocess.Popen, sink: _RotatingBinarySink, path: str):
        self._process = process
        self._sink = sink
        self._path = path
        self._closed = False
        self._pump_thread = threading.Thread(
            target=self._pump_output,
            name="funcdroid-logcat",
            daemon=True,
        )
        self._pump_thread.start()

    def _pump_output(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                self._sink.write(chunk)
        except (OSError, ValueError):
            pass

    @property
    def path(self) -> str:
        return self._path

    def close(self):
        if self._closed:
            return
        self._closed = True

        try:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        finally:
            try:
                if self._process.stdout is not None:
                    self._process.stdout.close()
            finally:
                self._pump_thread.join(timeout=5)
                self._sink.close()


def setup_logcat_capture(
    output_dir: str,
    device,
    filename: str = "logcat.log",
    clear: bool = True,
    append: bool = False,
) -> LogcatCaptureHandle | None:
    """Continuously save Android logcat output for the whole run."""
    if os.getenv("LOGCAT_CAPTURE", "1").strip().lower() in {"0", "false", "off", "no"}:
        print("[logcat] LOGCAT_CAPTURE 已关闭，跳过完整 logcat 采集。")
        return None

    operating_system = str(getattr(device, "operating_system", "") or "").strip().lower()
    if operating_system not in {"adb", "android"}:
        print("[logcat] 非 Android/ADB 设备，跳过完整 logcat 采集。")
        return None

    device_serial = str(getattr(device, "serial", "") or "").strip()
    if not device_serial:
        print("[logcat] 未获取到设备序列号，跳过完整 logcat 采集。")
        return None

    os.makedirs(output_dir, exist_ok=True)
    logcat_path = os.path.join(output_dir, filename)

    if clear:
        try:
            subprocess.run(
                ["adb", "-s", device_serial, "logcat", "-c"],
                timeout=5,
                capture_output=True,
            )
            print(f"[logcat] 已清空历史 logcat: device={device_serial}")
        except Exception as exc:
            print(f"[logcat] 清空历史 logcat 失败，继续采集: {exc}")

    rotate_bytes = max(0, int(os.getenv("LOGCAT_ROTATE_BYTES", str(256 * 1024 * 1024))))
    backup_count = max(1, int(os.getenv("LOGCAT_BACKUP_COUNT", "3")))
    sink = _RotatingBinarySink(logcat_path, rotate_bytes, backup_count, append=append)
    header = (
        f"===== FuncDroid logcat capture started: {datetime.now().isoformat()} =====\n"
        f"device={device_serial}\n"
    )
    sink.write(header.encode("utf-8", errors="replace"))

    try:
        process = subprocess.Popen(
            ["adb", "-s", device_serial, "logcat", "-v", "threadtime"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except Exception as exc:
        sink.close()
        print(f"[logcat] 启动完整 logcat 采集失败: {exc}")
        return None

    handle = LogcatCaptureHandle(process, sink, logcat_path)
    rotation_label = (
        f"每卷 {rotate_bytes} bytes，保留 {backup_count} 个历史卷"
        if rotate_bytes > 0
        else "不滚动"
    )
    print(f"[logcat] 完整 logcat 正在保存到: {logcat_path}（{rotation_label}）")
    atexit.register(handle.close)
    return handle
