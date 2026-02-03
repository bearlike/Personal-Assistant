#!/usr/bin/env python3
"""Hook manager for orchestration lifecycle events."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from core.classes import ActionStep
from core.permissions import PermissionDecision


@dataclass
class HookManager:
    pre_tool_use: list[Callable[[ActionStep], ActionStep]] = field(
        default_factory=list
    )
    post_tool_use: list[Callable[[ActionStep, object | None], object | None]] = field(
        default_factory=list
    )
    permission_request: list[
        Callable[[ActionStep, PermissionDecision], PermissionDecision]
    ] = field(default_factory=list)
    pre_compact: list[Callable[[list[dict]], list[dict]]] = field(default_factory=list)

    def run_pre_tool_use(self, action_step: ActionStep) -> ActionStep:
        for hook in self.pre_tool_use:
            action_step = hook(action_step)
        return action_step

    def run_post_tool_use(self, action_step: ActionStep, result: object | None) -> object | None:
        for hook in self.post_tool_use:
            result = hook(action_step, result)
        return result

    def run_permission_request(
        self, action_step: ActionStep, decision: PermissionDecision
    ) -> PermissionDecision:
        for hook in self.permission_request:
            decision = hook(action_step, decision)
        return decision

    def run_pre_compact(self, events: Iterable[dict]) -> list[dict]:
        event_list = list(events)
        for hook in self.pre_compact:
            event_list = hook(event_list)
        return event_list


def default_hook_manager() -> HookManager:
    return HookManager()


__all__ = [
    "HookManager",
    "default_hook_manager",
]
