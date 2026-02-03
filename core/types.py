#!/usr/bin/env python3
"""Shared type definitions for core components."""
from __future__ import annotations

from typing import Any, TypedDict

from typing_extensions import NotRequired


class ActionStepPayload(TypedDict):
    action_consumer: str
    action_type: str
    action_argument: str


class ActionPlanPayload(TypedDict):
    steps: list[ActionStepPayload]


class PermissionPayload(TypedDict):
    action_consumer: str
    action_type: str
    action_argument: str
    decision: str


class ToolResultPayload(TypedDict):
    action_consumer: str
    action_type: str
    action_argument: str
    result: str | None
    error: NotRequired[str]


class UserPayload(TypedDict):
    text: str


class CompletionPayload(TypedDict):
    done: bool
    done_reason: str | None
    task_result: str | None


EventPayload = (
    ActionPlanPayload
    | PermissionPayload
    | ToolResultPayload
    | UserPayload
    | CompletionPayload
    | dict[str, Any]
)


class Event(TypedDict):
    type: str
    payload: EventPayload


class EventRecord(Event):
    ts: str


__all__ = [
    "ActionPlanPayload",
    "ActionStepPayload",
    "CompletionPayload",
    "Event",
    "EventPayload",
    "EventRecord",
    "PermissionPayload",
    "ToolResultPayload",
    "UserPayload",
]
