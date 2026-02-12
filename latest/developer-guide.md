# Developer Guide

This page highlights the core abstractions and a minimal path to build a new client.

## Core abstractions
- `ActionStep`, `Plan`, `TaskQueue` (`meeseeks_core.classes`): planning and tool-execution payloads.
- `AbstractTool` (`meeseeks_core.classes`): base class for local tools; implement `get_state` and `set_state`.
- `ToolRegistry` / `ToolSpec` (`meeseeks_core.tool_registry`): tool metadata, schema, and routing.
- `PermissionPolicy` (`meeseeks_core.permissions`): allow/deny/ask rules for tool execution.
- `HookManager` (`meeseeks_core.hooks`): pre/post hooks and compaction transforms.
- `SessionStore` / `SessionRuntime` (`meeseeks_core.session_store`, `meeseeks_core.session_runtime`): transcripts and shared runtime.
- Event payloads (`meeseeks_core.types`): typed records for plans, tool results, and completions.

## Build a new client (short walkthrough)
1. Load config and tool registry (`load_registry`).
2. Create a `SessionStore` and `SessionRuntime`.
3. Resolve a session id, then run `run_sync` or `start_async`.
4. Handle core commands (`/compact`, `/status`, `/terminate`) using `parse_core_command`.
5. For async runs, poll `load_events(session_id, after)` for new JSONL events.

Minimal sync example:
```python
from meeseeks_core.permissions import approval_callback_from_config, load_permission_policy
from meeseeks_core.session_runtime import SessionRuntime, parse_core_command
from meeseeks_core.session_store import SessionStore
from meeseeks_core.tool_registry import load_registry

session_store = SessionStore()
tool_registry = load_registry()
runtime = SessionRuntime(session_store=session_store)

session_id = runtime.resolve_session(session_tag="client")

command = parse_core_command(user_text)
if command:
    # Handle /status, /terminate, or /compact at the client layer.
    pass

result = runtime.run_sync(
    session_id=session_id,
    user_query=user_text,
    tool_registry=tool_registry,
    permission_policy=load_permission_policy(),
    approval_callback=approval_callback_from_config(),
)
```
