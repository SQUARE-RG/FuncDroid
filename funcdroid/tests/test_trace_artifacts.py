from __future__ import annotations

import base64
import json

import pytest

from model.flow_graph import Page
from workflow.nodes.scenario_exec import _persist_unwritten_trace_screenshots
from workflow.trace_artifacts import (
    configure_trace_artifacts,
    disable_trace_artifacts,
    save_trace_screenshot,
)


@pytest.fixture(autouse=True)
def _disable_after_test():
    yield
    disable_trace_artifacts()


def test_trace_screenshot_is_written_with_jsonl_index(tmp_path):
    image_bytes = b"diagnostic-image-bytes"
    page = Page(
        id="page_2",
        name="detail",
        activity="DetailActivity",
        representative_screenshot=base64.b64encode(image_bytes).decode("ascii"),
    )
    configure_trace_artifacts(str(tmp_path), reset=True)

    relative = save_trace_screenshot(
        page,
        phase="test",
        step_index=4,
        moment="before_action",
        metadata={"previous_action_name": "click"},
    )

    assert relative == "trace_screenshots/00001_test_step_004_before_action.jpg"
    assert (tmp_path / relative).read_bytes() == image_bytes
    records = [
        json.loads(line)
        for line in (tmp_path / "trace_screenshots" / "index.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["page_id"] == "page_2"
    assert records[0]["activity"] == "DetailActivity"
    assert records[0]["metadata"]["previous_action_name"] == "click"


def test_trace_screenshot_resume_appends_sequence(tmp_path):
    screenshot = base64.b64encode(b"image").decode("ascii")
    page = Page(representative_screenshot=screenshot)
    configure_trace_artifacts(str(tmp_path), reset=True)
    save_trace_screenshot(
        page,
        phase="navigation",
        step_index=1,
        moment="before_action",
    )
    configure_trace_artifacts(str(tmp_path), reset=False)

    relative = save_trace_screenshot(
        page,
        phase="test",
        step_index=1,
        moment="before_action",
    )

    assert relative.startswith("trace_screenshots/00002_")


def test_recovery_and_terminal_trace_observations_are_also_persisted(tmp_path):
    screenshot = base64.b64encode(b"terminal-image").decode("ascii")
    trace = [{
        "action_name": "capture_screenshot",
        "action_description": "terminal observation",
        "current_page": Page(
            id="page_9",
            activity="ResultActivity",
            representative_screenshot=screenshot,
        ),
        "trace_phase": "test",
        "observation_only": True,
        "terminal_observation": True,
    }]
    configure_trace_artifacts(str(tmp_path), reset=True)

    saved = _persist_unwritten_trace_screenshots(trace, "test")

    assert saved == 1
    assert trace[0]["trace_screenshot_before"].endswith(
        "_test_step_001_terminal_observation.jpg"
    )
