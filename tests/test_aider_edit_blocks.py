"""Tests for Aider-style SEARCH/REPLACE block handling."""

from __future__ import annotations

import textwrap

import pytest
from meeseeks_core.classes import ActionStep
from meeseeks_core.errors import ToolInputError
from meeseeks_tools.aider_bridge import EditBlockApplyError, apply_search_replace_blocks
from meeseeks_tools.integration.aider_edit_blocks import AiderEditBlockTool


def _block(path: str, search: str, replace: str) -> str:
    return (
        f"{path}\n"
        "```text\n"
        "<<<<<<< SEARCH\n"
        f"{search}=======\n"
        f"{replace}>>>>>>> REPLACE\n"
        "```\n"
    )


def test_apply_search_replace_block(tmp_path):
    """Apply a simple SEARCH/REPLACE block to an existing file."""
    target = tmp_path / "hello.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    content = _block("hello.txt", "world\n", "there\n")
    results = apply_search_replace_blocks(content, root=str(tmp_path), write=True)

    assert target.read_text(encoding="utf-8") == "hello\nthere\n"
    assert results[0].applied is True


def test_create_new_file(tmp_path):
    """Create a new file from an empty SEARCH block."""
    content = _block("new.txt", "", "first line\n")
    results = apply_search_replace_blocks(content, root=str(tmp_path), write=True)

    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "first line\n"
    assert results[0].created is True


def test_shell_blocks_are_rejected(tmp_path):
    """Reject shell command blocks in edit content."""
    content = textwrap.dedent(
        """
        ```bash
        echo "hello"
        ```
        """
    ).lstrip()
    with pytest.raises(EditBlockApplyError):
        apply_search_replace_blocks(content, root=str(tmp_path), write=False)


def test_path_traversal_is_blocked(tmp_path):
    """Block edits that escape the project root."""
    content = _block("../oops.txt", "", "nope\n")
    with pytest.raises(EditBlockApplyError):
        apply_search_replace_blocks(content, root=str(tmp_path), write=False)


def test_append_when_search_empty(tmp_path):
    """Append content when SEARCH block is empty and file exists."""
    target = tmp_path / "append.txt"
    target.write_text("hello\n", encoding="utf-8")

    content = _block("append.txt", "", "world\n")
    results = apply_search_replace_blocks(content, root=str(tmp_path), write=True)

    assert target.read_text(encoding="utf-8") == "hello\nworld\n"
    assert results[0].applied is True


def test_search_miss_includes_hint(tmp_path):
    """Include a hint when SEARCH block fails to match."""
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")

    content = _block("hello.txt", "alpha\nbeta\ngamaa\ndelta\n", "there\n")
    with pytest.raises(EditBlockApplyError) as exc:
        apply_search_replace_blocks(content, root=str(tmp_path), write=False)

    assert "Did you mean" in str(exc.value)


def test_edit_block_tool_requires_blocks(tmp_path):
    """Raise a tool input error when no blocks are provided."""
    tool = AiderEditBlockTool()
    step = ActionStep(
        action_consumer="aider_edit_block_tool",
        action_type="set",
        action_argument={"content": "no edits here", "root": str(tmp_path)},
    )
    with pytest.raises(ToolInputError) as exc:
        tool.set_state(step)
    assert "SEARCH/REPLACE blocks" in str(exc.value)


def test_edit_block_tool_wraps_apply_errors(tmp_path):
    """Wrap apply errors with format guidance."""
    target = tmp_path / "hello.txt"
    target.write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
    tool = AiderEditBlockTool()
    step = ActionStep(
        action_consumer="aider_edit_block_tool",
        action_type="set",
        action_argument={
            "content": _block("hello.txt", "alpha\nbeta\ngamaa\ndelta\n", "there\n"),
            "root": str(tmp_path),
        },
    )
    with pytest.raises(ToolInputError) as exc:
        tool.set_state(step)
    message = str(exc.value)
    assert "Did you mean" in message
    assert "Expected format" in message
