#!/usr/bin/env python3
"""Hook manager for orchestration lifecycle events."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from core.classes import ActionStep
from core.common import MockSpeaker
from core.permissions import PermissionDecision
from core.types import EventRecord


@dataclass
class HookManager:
    """Container for hook callbacks used during orchestration."""
    pre_tool_use: list[Callable[[ActionStep], ActionStep]] = field(
        default_factory=list
    )
    post_tool_use: list[Callable[[ActionStep, MockSpeaker], MockSpeaker]] = field(
        default_factory=list
    )
    permission_request: list[
        Callable[[ActionStep, PermissionDecision], PermissionDecision]
    ] = field(default_factory=list)
    pre_compact: list[Callable[[list[EventRecord]], list[EventRecord]]] = field(
        default_factory=list
    )

    def run_pre_tool_use(self, action_step: ActionStep) -> ActionStep:
        """Apply pre-tool hooks to an action step."""
        for hook in self.pre_tool_use:
            action_step = hook(action_step)
        return action_step

    def run_post_tool_use(
        self, action_step: ActionStep, result: MockSpeaker
    ) -> MockSpeaker:
        """Apply post-tool hooks to a tool result."""
        for hook in self.post_tool_use:
            result = hook(action_step, result)
        return result

    def run_permission_request(
        self, action_step: ActionStep, decision: PermissionDecision
    ) -> PermissionDecision:
        """Apply permission hooks to a decision outcome."""
        for hook in self.permission_request:
            decision = hook(action_step, decision)
        return decision

    def run_pre_compact(self, events: Iterable[EventRecord]) -> list[EventRecord]:
        """Apply compaction hooks to events prior to summarization."""
        event_list: list[EventRecord] = list(events)
        for hook in self.pre_compact:
            event_list = hook(event_list)
        return event_list


def default_hook_manager() -> HookManager:
    """Create a hook manager with no custom hooks registered."""
    return HookManager()


__all__ = [
    "HookManager",
    "default_hook_manager",
]
