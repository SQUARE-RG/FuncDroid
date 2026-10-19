# Multilingual UI and model-response fixtures use Unicode escapes to preserve their exact text.
import io
import threading

from workflow.run_logging import (
    _RotatingBinarySink,
    _RotatingTextSink,
    _TeeStream,
    normalize_runtime_log_text,
)


def test_runtime_log_terms_are_productized_for_framework_messages_only():
    # Terminology replacement applies only to fixed framework-prefixed log lines.
    framework = "[\u542f\u52a8] \u57fa\u7ebf\u8bd5\u8dd1：\u6309\u7167\u6d4b\u8bd5\u4efb\u52a1\u5b8c\u6210\u6d4b\u8bd5\u9a8c\u8bc1，\u7b49\u5f85\u9a8c\u6536。\n"
    assert normalize_runtime_log_text(framework) == (
        "[\u542f\u52a8] \u8d77\u70b9\u8fd0\u884c：\u6309\u7167\u6267\u884c\u4efb\u52a1\u5b8c\u6210\u6267\u884c\u9a8c\u8bc1，\u7b49\u5f85\u8fd0\u884c。\n"
    )
    # Preserve raw model responses and dynamic scenario text to avoid altering evidence.
    for dynamic in (
        "[\u539f\u59cb\u5185\u5bb9]: \u8bf7\u6d4b\u8bd5\u767b\u5f55\u529f\u80fd\n",
        "[\u8ba1\u5212] \u573a\u666f: \u6d4b\u8bd5\u5220\u9664\u7b14\u8bb0\u540e\u5217\u8868\u5237\u65b0\n",
        "\u6a21\u578b\u8f93\u51fa：\u6d4b\u8bd5\u4efb\u52a1\n",
    ):
        assert normalize_runtime_log_text(dynamic) == dynamic


def test_tee_stream_normalizes_framework_lines_and_keeps_dynamic_text():
    console = io.StringIO()
    run_log = io.StringIO()
    stream = _TeeStream(console, run_log)

    # When prefix and body arrive in separate writes, classify the complete line.
    stream.write("[\u8fd0\u884c\u5feb\u7167] ")
    stream.write("\u4e3b\u6d4b\u8bd5\u5faa\u73af\u7ee7\u7eed\n")
    stream.write("\u6a21\u578b\u8f93\u51fa：\u6d4b\u8bd5\u4efb\u52a1\n")

    assert console.getvalue() == "[\u8fd0\u884c\u5feb\u7167] \u4e3b\u6267\u884c\u5faa\u73af\u7ee7\u7eed\n\u6a21\u578b\u8f93\u51fa：\u6d4b\u8bd5\u4efb\u52a1\n"
    assert run_log.getvalue() == console.getvalue()


def test_tee_stream_keeps_console_alive_when_log_sink_fails():
    class BrokenLog:
        def write(self, _text):
            raise OSError("disk full")

        def flush(self):
            raise OSError("disk full")

    console = io.StringIO()
    stream = _TeeStream(console, BrokenLog())

    stream.write("first\n")
    stream.write("second\n")

    output = console.getvalue()
    assert "first\n" in output
    assert "second\n" in output
    assert output.count("disabled file mirroring") == 1


def test_tee_stream_writes_concurrent_messages_as_complete_lines():
    console = io.StringIO()
    run_log = io.StringIO()
    stream = _TeeStream(console, run_log)
    first_message_buffered = threading.Event()
    second_message_written = threading.Event()

    def write_first_message():
        stream.write("[worker-a] pending")
        first_message_buffered.set()
        second_message_written.wait(timeout=1)
        stream.write("\n")

    def write_second_message():
        first_message_buffered.wait(timeout=1)
        stream.write("[worker-b] complete\n")
        second_message_written.set()

    first = threading.Thread(target=write_first_message)
    second = threading.Thread(target=write_second_message)
    first.start()
    second.start()
    first.join(timeout=1)
    second.join(timeout=1)

    assert console.getvalue().splitlines() == [
        "[worker-b] complete",
        "[worker-a] pending",
    ]
    assert run_log.getvalue() == console.getvalue()


def test_logcat_sink_rotates_without_growing_unbounded(tmp_path):
    path = tmp_path / "logcat.log"
    sink = _RotatingBinarySink(str(path), max_bytes=8, backup_count=2)
    sink.write(b"123456")
    sink.write(b"abcdef")
    sink.close()

    assert path.read_bytes().endswith(b"abcdef")
    assert (tmp_path / "logcat.log.1").read_bytes() == b"123456"


def test_run_log_text_sink_rotates_and_keeps_history(tmp_path):
    path = tmp_path / "run.log"
    sink = _RotatingTextSink(str(path), max_bytes=128, backup_count=2, append=True)
    sink.write("a" * 100)
    sink.write("b" * 100)
    sink.close()

    assert path.read_text(encoding="utf-8").endswith("b" * 100)
    assert (tmp_path / "run.log.1").read_text(encoding="utf-8") == "a" * 100
