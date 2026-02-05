#!/usr/bin/env python3
"""Aider-style search/replace block application tool."""

from __future__ import annotations

import os
from dataclasses import dataclass

from meeseeks_core.classes import AbstractTool, ActionStep
from meeseeks_core.common import MockSpeaker, get_mock_speaker

from meeseeks_tools.aider_bridge import EditBlockApplyError, apply_search_replace_blocks


@dataclass(frozen=True)
class EditBlockRequest:
    """Parsed request payload for edit-block application."""

    content: str
    root: str
    files: list[str] | None


class AiderEditBlockTool(AbstractTool):
    """Apply Aider SEARCH/REPLACE blocks to local files."""

    def __init__(self) -> None:
        """Initialize the edit block tool."""
        super().__init__(
            name="Aider Edit Blocks",
            description="Apply Aider-style SEARCH/REPLACE blocks to files.",
            use_llm=False,
        )

    def set_state(self, action_step: ActionStep | None = None) -> MockSpeaker:
        """Apply search/replace blocks to files."""
        request = _parse_request(action_step)
        results = apply_search_replace_blocks(
            request.content,
            root=request.root,
            valid_fnames=request.files,
            write=True,
        )
        message = _format_summary(results, dry_run=False)
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content=message)

    def get_state(self, action_step: ActionStep | None = None) -> MockSpeaker:
        """Validate search/replace blocks without writing changes."""
        request = _parse_request(action_step)
        results = apply_search_replace_blocks(
            request.content,
            root=request.root,
            valid_fnames=request.files,
            write=False,
        )
        message = _format_summary(results, dry_run=True)
        MockSpeaker = get_mock_speaker()
        return MockSpeaker(content=message)


def _parse_request(action_step: ActionStep | None) -> EditBlockRequest:
    if action_step is None:
        raise EditBlockApplyError("Action step is required for edit block operations.")

    argument = action_step.action_argument
    if isinstance(argument, str):
        return EditBlockRequest(content=argument, root=os.getcwd(), files=None)

    if isinstance(argument, dict):
        content = argument.get("content") or argument.get("blocks")
        if not isinstance(content, str) or not content.strip():
            raise EditBlockApplyError("Edit block content is required.")
        root = argument.get("root") or os.getcwd()
        if not isinstance(root, str) or not root.strip():
            raise EditBlockApplyError("Root path must be a non-empty string.")
        files = argument.get("files")
        if files is not None:
            if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
                raise EditBlockApplyError("files must be a list of strings.")
        return EditBlockRequest(content=content, root=root, files=files)

    raise EditBlockApplyError("Action argument must be a string or object payload.")


def _format_summary(results, *, dry_run: bool) -> str:
    if not results:
        return "No SEARCH/REPLACE blocks found."

    created = [result.path for result in results if result.created]
    applied = [result.path for result in results]

    mode = "Validated" if dry_run else "Applied"
    summary = f"{mode} {len(applied)} SEARCH/REPLACE block(s) across {len(set(applied))} file(s)."
    if created:
        summary += f" Created {len(created)} file(s)."
    return summary


__all__ = ["AiderEditBlockTool", "EditBlockRequest"]
