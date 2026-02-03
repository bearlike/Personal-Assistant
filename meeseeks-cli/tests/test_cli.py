# ruff: noqa: I001
import os
import sys

test_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(test_dir, ".."))
sys.path.append(os.path.join(test_dir, "..", ".."))

from core.session_store import SessionStore  # noqa: E402
from core.tool_registry import load_registry  # noqa: E402
from rich.console import Console  # noqa: E402

from cli_commands import get_registry  # noqa: E402
from cli_context import CliState, CommandContext  # noqa: E402
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


def test_handle_new_session_command(tmp_path):
    store = SessionStore(root_dir=str(tmp_path))
    console = Console(record=True)
    tool_registry = load_registry()
    registry = get_registry()
    session_id = store.create_session()
    state = CliState(session_id=session_id, show_plan=True)
    context = CommandContext(
        console=console,
        store=store,
        state=state,
        tool_registry=tool_registry,
        prompt_func=None,
    )

    registry.execute("/new", context, [])

    assert state.session_id != session_id
