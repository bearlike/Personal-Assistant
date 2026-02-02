#!/usr/bin/env python3
"""Transcript compaction utilities."""
from __future__ import annotations

from collections.abc import Iterable

from core.common import get_logger

logging = get_logger(name="core.compaction")


def summarize_events(events: Iterable[dict], max_items: int = 20) -> str:
    """Generate a lightweight summary of recent events.

    This is intentionally simple to keep dependencies small. It can be replaced by
    a model-backed summarizer later.
    """
    snippets: list[str] = []
    for event in list(events)[-max_items:]:
        event_type = event.get("type", "event")
        payload = event.get("payload", "")
        if isinstance(payload, dict):
            payload = payload.get("text") or payload.get("message") or str(payload)
        if payload:
            snippets.append(f"{event_type}: {payload}")
        else:
            snippets.append(f"{event_type}.")
    return " | ".join(snippets).strip()


def should_compact(events: Iterable[dict], threshold: int = 50) -> bool:
    return len(list(events)) >= threshold

