"""Tests for Aider-based local file tools."""

from __future__ import annotations

from meeseeks_core.classes import ActionStep
from meeseeks_tools.integration.aider_file_tools import AiderListDirTool, AiderReadFileTool


def test_aider_read_file_tool_reads(tmp_path):
    """Read a file using the Aider read tool."""
    target = tmp_path / "hello.txt"
    target.write_text("hello\n", encoding="utf-8")

    tool = AiderReadFileTool()
    step = ActionStep(
        action_consumer="aider_read_file_tool",
        action_type="get",
        action_argument={"path": "hello.txt", "root": str(tmp_path)},
    )
    result = tool.get_state(step)

    payload = result.content
    assert isinstance(payload, dict)
    assert payload.get("kind") == "file"
    assert payload.get("path") == "hello.txt"
    assert payload.get("text") == "hello\n"


def test_aider_list_dir_tool_lists(tmp_path):
    """List a directory using the Aider list tool."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file.txt").write_text("data", encoding="utf-8")

    tool = AiderListDirTool()
    step = ActionStep(
        action_consumer="aider_list_dir_tool",
        action_type="get",
        action_argument={"path": "a", "root": str(tmp_path)},
    )
    result = tool.get_state(step)

    payload = result.content
    assert isinstance(payload, dict)
    assert payload.get("kind") == "dir"
    assert payload.get("path") == "a"
    entries = payload.get("entries")
    assert isinstance(entries, list)
    assert "a/file.txt" in entries


def test_aider_read_file_blocks_escape(tmp_path):
    """Reject path traversal attempts."""
    tool = AiderReadFileTool()
    step = ActionStep(
        action_consumer="aider_read_file_tool",
        action_type="get",
        action_argument={"path": "../oops.txt", "root": str(tmp_path)},
    )
    result = tool.get_state(step)
    assert isinstance(result.content, str)
    assert "outside the project root" in result.content
