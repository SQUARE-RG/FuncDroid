"""Shared offline-test configuration."""

import sys
from pathlib import Path

# Make the source tree importable when pytest is launched from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


@pytest.fixture(autouse=True)
def _disable_tap_refine(monkeypatch):
    """Disable tap refinement by default because its extra model request disrupts scripted fake responses. Dedicated tests enable it explicitly."""
    from workflow.nodes import scenario_exec

    monkeypatch.setattr(scenario_exec, "TAP_REFINE_ENABLED", False)
