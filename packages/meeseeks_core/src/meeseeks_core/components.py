#!/usr/bin/env python3
"""Helpers for optional components and observability integration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from meeseeks_core.common import get_logger
from meeseeks_core.config import get_config
from meeseeks_core.types import JsonValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

logging = get_logger(name="core.components")


@dataclass(frozen=True)
class ComponentStatus:
    """Describe whether a component is enabled and why."""

    name: str
    enabled: bool
    reason: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


def resolve_langfuse_status() -> ComponentStatus:
    """Determine whether Langfuse callbacks are available and configured."""
    enabled, reason, metadata = get_config().langfuse.evaluate()
    return ComponentStatus(name="langfuse", enabled=enabled, reason=reason, metadata=metadata)


def build_langfuse_handler(
    *,
    user_id: str,
    session_id: str,
    trace_name: str,
    version: str,
    release: str,
) -> LangfuseCallbackHandler | None:
    """Create a Langfuse callback handler when configured."""
    status = resolve_langfuse_status()
    if not status.enabled:
        logging.debug("Langfuse disabled: {}", status.reason)
        return None

    from langfuse.callback import CallbackHandler

    try:
        return CallbackHandler(
            user_id=user_id,
            session_id=session_id,
            trace_name=trace_name,
            version=version,
            release=release,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Langfuse initialization failed: {}", exc)
        return None


def resolve_home_assistant_status() -> ComponentStatus:
    """Determine whether the Home Assistant tool is configured."""
    enabled, reason, metadata = get_config().home_assistant.evaluate()
    return ComponentStatus(
        name="home_assistant_tool",
        enabled=enabled,
        reason=reason,
        metadata=metadata,
    )


def format_component_status(statuses: Iterable[ComponentStatus]) -> str:
    """Format component statuses for inclusion in prompts."""
    lines: list[str] = []
    for status in statuses:
        state = "enabled" if status.enabled else "disabled"
        reason = f" ({status.reason})" if status.reason else ""
        lines.append(f"- {status.name}: {state}{reason}")
    return "\n".join(lines)


__all__ = [
    "ComponentStatus",
    "build_langfuse_handler",
    "format_component_status",
    "resolve_home_assistant_status",
    "resolve_langfuse_status",
]
