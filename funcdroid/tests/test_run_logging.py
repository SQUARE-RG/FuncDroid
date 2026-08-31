import io
import threading

from workflow.run_logging import (
    _RotatingBinarySink,
    _RotatingTextSink,
    _TeeStream,
    normalize_runtime_log_text,
)


def test_runtime_log_terms_are_productized_for_static_and_dynamic_text():
    text = "基线试跑：按照测试任务完成测试验证，等待验收。"

    normalized = normalize_runtime_log_text(text)

    assert normalized == "起点运行：按照执行任务完成执行验证，等待运行。"
    assert all(term not in normalized for term in ("基线", "试跑", "测试", "验收"))


def test_tee_stream_writes_the_same_normalized_text_to_console_and_log():
    console = io.StringIO()
    run_log = io.StringIO()
    stream = _TeeStream(console, run_log)

    stream.write("模型输出：测试任务\n")

    assert console.getvalue() == "模型输出：执行任务\n"
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
    assert output.count("已停用本次文件镜像") == 1


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
