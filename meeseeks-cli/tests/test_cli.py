# ruff: noqa: I001
import os
import sys

test_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(test_dir, ".."))
sys.path.append(os.path.join(test_dir, "..", ".."))

from core.session_store import SessionStore  # noqa: E402

from cli_master import _format_steps, _parse_command, _resolve_session_id  # noqa: E402


class DummyStep:
    def __init__(self, tool: str, action: str, argument: str) -> None:
        self.action_consumer = tool
        self.action_type = action
        self.action_argument = argument


def test_parse_command():
    command, args = _parse_command("/tag primary")
    assert command == "/tag"
    assert args == ["primary"]


def test_format_steps():
    steps = [DummyStep("tool_a", "get", "status")]
    rows = _format_steps(steps)
    assert rows == [("tool_a", "get", "status")]


def test_resolve_session_id(tmp_path):
    store = SessionStore(root_dir=str(tmp_path))
    session_id = _resolve_session_id(store, None, "primary", None)
    assert session_id
    assert store.resolve_tag("primary") == session_id

    forked_id = _resolve_session_id(store, None, None, session_id)
    assert forked_id != session_id
