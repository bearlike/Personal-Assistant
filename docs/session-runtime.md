# Session Runtime

The session runtime is a small shared facade that powers both the CLI and the REST API. It centralizes session lifecycle, async run tracking, cancellation, and event polling so interfaces stay thin and behavior stays consistent.

## What it does
- Resolves sessions by id, tag, or fork.
- Runs orchestration synchronously or in a background thread.
- Tracks active runs per session and supports cancellation.
- Filters session events for polling (`after` timestamp).
- Summarizes a session (title, status, done reason, context).

## Core commands
These commands are supported across interfaces:
- `/compact`: compact the session transcript and write a summary.
- `/terminate`: request cancellation for the active run.
- `/status`: return the current session summary.

The runtime only recognizes these core commands. Interface-specific commands remain in each UI layer.

## Event polling model
Events are stored as JSONL records by `SessionStore`. The runtime exposes `load_events(session_id, after)` which filters by the ISO-8601 timestamp (`ts`) on each event.

Typical polling flow:
1. Create a session.
2. Start an async run.
3. Poll `/events` with `after` to receive only new records.

## Minimal usage (Python)
```python
from meeseeks_core.session_runtime import SessionRuntime
from meeseeks_core.session_store import SessionStore

runtime = SessionRuntime(session_store=SessionStore())
session_id = runtime.resolve_session(session_tag="primary")

# synchronous run
result = runtime.run_sync(user_query="Hello", session_id=session_id)

# async run + polling
runtime.start_async(session_id=session_id, user_query="Do the task")
events = runtime.load_events(session_id, after=None)
```

## Design goals
- Keep the core orchestration engine centralized.
- Make interface layers thin and easy to extend.
- Avoid duplicate session lifecycle logic.
