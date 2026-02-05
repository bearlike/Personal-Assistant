#!/usr/bin/env python3
"""Task planning and orchestration loop for Meeseeks."""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import cast

from dotenv import load_dotenv
from langchain_core._api.beta_decorator import LangChainBetaWarning

from meeseeks_core.action_runner import ActionPlanRunner
from meeseeks_core.classes import ActionStep, OrchestrationState, TaskQueue
from meeseeks_core.common import get_logger
from meeseeks_core.context import ContextSnapshot
from meeseeks_core.hooks import HookManager, default_hook_manager
from meeseeks_core.orchestrator import Orchestrator
from meeseeks_core.permissions import (
    PermissionPolicy,
    approval_callback_from_env,
    load_permission_policy,
)
from meeseeks_core.planning import Planner
from meeseeks_core.reflection import StepReflector
from meeseeks_core.session_store import SessionStore
from meeseeks_core.token_budget import get_token_budget
from meeseeks_core.tool_registry import ToolRegistry, load_registry
from meeseeks_core.types import Event, EventRecord

logging = get_logger(name="core.task_master")

warnings.simplefilter("ignore", LangChainBetaWarning)
load_dotenv()


def _build_context_snapshot(
    session_summary: str | None,
    recent_events: list[EventRecord] | None,
    selected_events: list[EventRecord] | None,
    model_name: str | None,
) -> ContextSnapshot:
    summary = session_summary
    recent = recent_events or []
    budget = get_token_budget([], summary, model_name)
    return ContextSnapshot(
        summary=summary,
        recent_events=recent,
        selected_events=selected_events,
        events=[],
        budget=budget,
    )


def generate_action_plan(
    user_query: str,
    model_name: str | None = None,
    tool_registry: ToolRegistry | None = None,
    session_summary: str | None = None,
    recent_events: list[EventRecord] | None = None,
    selected_events: list[EventRecord] | None = None,
) -> TaskQueue:
    if tool_registry is None:
        tool_registry = load_registry()
    resolved_model = cast(
        str,
        model_name
        or os.getenv("ACTION_PLAN_MODEL")
        or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"),
    )
    context = _build_context_snapshot(
        session_summary,
        recent_events,
        selected_events,
        resolved_model,
    )
    planner = Planner(tool_registry)
    return planner.generate(user_query, resolved_model, context=context)


def run_action_plan(
    task_queue: TaskQueue,
    tool_registry: ToolRegistry | None = None,
    event_logger: Callable[[Event], None] | None = None,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: Callable[[ActionStep], bool] | None = None,
    hook_manager: HookManager | None = None,
    model_name: str | None = None,
) -> TaskQueue:
    if tool_registry is None:
        tool_registry = load_registry()
    if permission_policy is None:
        permission_policy = load_permission_policy()
    if approval_callback is None:
        approval_callback = approval_callback_from_env()
    if hook_manager is None:
        hook_manager = default_hook_manager()
    reflector = StepReflector(model_name)
    runner = ActionPlanRunner(
        tool_registry=tool_registry,
        permission_policy=permission_policy,
        approval_callback=approval_callback,
        hook_manager=hook_manager,
        event_logger=event_logger,
        reflector=reflector,
    )
    return runner.run(task_queue)


def orchestrate_session(
    user_query: str,
    model_name: str | None = None,
    max_iters: int = 3,
    initial_task_queue: TaskQueue | None = None,
    return_state: bool = False,
    session_id: str | None = None,
    session_store: SessionStore | None = None,
    tool_registry: ToolRegistry | None = None,
    permission_policy: PermissionPolicy | None = None,
    approval_callback: Callable[[ActionStep], bool] | None = None,
    hook_manager: HookManager | None = None,
) -> TaskQueue | tuple[TaskQueue, OrchestrationState]:
    orchestrator = Orchestrator(
        model_name=model_name,
        session_store=session_store,
        tool_registry=tool_registry,
        permission_policy=permission_policy,
        approval_callback=approval_callback,
        hook_manager=hook_manager,
    )
    return orchestrator.run(
        user_query,
        max_iters=max_iters,
        initial_task_queue=initial_task_queue,
        return_state=return_state,
        session_id=session_id,
    )


__all__ = ["generate_action_plan", "orchestrate_session", "run_action_plan"]
