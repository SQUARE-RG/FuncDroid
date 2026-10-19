"""Run-level stdout/stderr and device logcat logging."""

# Unicode escapes preserve literal UI labels and legacy evidence matching.
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
    ("\u57fa\u7ebf", "\u8d77\u70b9"),
    ("\u6279\u6d4b", "\u5e94\u7528\u961f\u5217"),
    ("\u9a8c\u6536", "\u8fd0\u884c"),
    ("\u8bd5\u8dd1", "\u8fd0\u884c"),
    ("\u5b9e\u9a8c", "\u8fd0\u884c"),
    ("\u6d4b\u8bd5", "\u6267\u884c"),
)


# Normalize only controlled framework messages; preserve model responses, scenarios, traces, and errors as evidence.
RUNTIME_LOG_NORMALIZED_TAGS = frozenset({
    "\u65e5\u5fd7",
    "\u542f\u52a8",
    "\u542f\u52a8\u524d\u6e05\u7406",
    "\u542f\u52a8\u524d\u91cd\u7f6e",
    "\u542f\u52a8\u524d\u6388\u6743",
    "\u8fd0\u884c\u5feb\u7167",
    "\u8f93\u51fa",
    "\u8f93\u51fa\u4fee\u590d",
    "\u4ea7\u7269\u6821\u9a8c",
    "\u5b8c\u6210",
    "\u8fd0\u884c\u4e2d\u65ad",
    "\u8fd0\u884c\u5f02\u5e38",
    "\u8def\u7531",
    "\u8f6e\u6b21",
    "\u8f6e\u6b21\u4fe1\u606f",
    "\u957f\u65f6\u8fd0\u884c",
    "\u6301\u4e45\u5316",
    "LLM\u7edf\u8ba1",
    "logcat",
})
_RUNTIME_LOG_TAG_PATTERN = re.compile(r"^\s*\[([^\[\]\n]{1,32})\]")
# Fixed untagged banner and summary prefixes contain no model-authored output.
RUNTIME_LOG_NORMALIZED_PREFIXES = (
    "  \u7b2c ",
    "  \u89c4\u5212\u6a21\u5f0f:",
    "  \u63a2\u7d22\u8303\u56f4:",
    "  \u6267\u884c\u8f6e\u6b21:",
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
    """Normalize fixed framework messages one line at a time.

    Only recognized framework prefixes are rewritten; preserve model output, scenario text, and other dynamic evidence verbatim."""
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
                "[Logging] run.log write failed; disabled file mirroring. "
                f"Console output and execution will continue: {exc}\n"
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
        # Buffer complete lines before identifying framework prefixes.
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
            # Generate rollover timestamps at rotation time, not at sink creation.
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
                self._stderr.write(f"[Logging] run.log flush failed; shutdown will continue: {exc}\n")
            except Exception:
                pass
        try:
            self._log_file.close()
        except Exception as exc:
            try:
                self._stderr.write(f"[Logging] run.log close failed; exit will continue: {exc}\n")
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

    print(f"[Logging] Console output is mirrored to: {log_path}")
    rotation_label = (
        f"{rotate_bytes} bytes per file; retain {backup_count} backups"
        if rotate_bytes > 0
        else "rotation disabled"
    )
    print(f"[Logging] run.log rotation: {rotation_label}")
    print(f"[Logging] Session started: {datetime.now().isoformat()}")
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
        # Accept fixed bytes or a factory; generate a fresh continuation timestamp by default.
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
        print("[logcat] LOGCAT_CAPTURE is disabled; skipping full logcat capture.")
        return None

    operating_system = str(getattr(device, "operating_system", "") or "").strip().lower()
    if operating_system not in {"adb", "android"}:
        print("[logcat] Device is not Android/ADB; skipping full logcat capture.")
        return None

    device_serial = str(getattr(device, "serial", "") or "").strip()
    if not device_serial:
        print("[logcat] Device serial is unavailable; skipping full logcat capture.")
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
            print(f"[logcat] Cleared historical logcat: device={device_serial}")
        except Exception as exc:
            print(f"[logcat] Could not clear logcat; continuing capture: {exc}")

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
        print(f"[logcat] Could not start full logcat capture: {exc}")
        return None

    handle = LogcatCaptureHandle(process, sink, logcat_path)
    rotation_label = (
        f"{rotate_bytes} bytes per segment; retain {backup_count} backups"
        if rotate_bytes > 0
        else "rotation disabled"
    )
    print(f"[logcat] Full logcat is being saved to: {logcat_path}（{rotation_label}）")
    atexit.register(handle.close)
    return handle
