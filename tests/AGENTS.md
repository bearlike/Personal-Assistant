# Tests - Project Guidance

Scope: this file applies to the root `tests/` suite and shared test patterns.

## What we test
- Orchestration loop (`core.task_master.orchestrate_session`): completion, replanning, max-iter behavior.
- Tool registry behavior and tool disabling when init fails.
- CLI integration flows using lightweight stubs.

## Hidden dependencies / assumptions
- Many tests rely on `monkeypatch` for env vars (LLM config, MCP config, log levels).
- LLM calls are mocked at the orchestration boundary (not at HTTP layer).
- Avoid pulling in real MCP servers or external HTTP.

## Pitfalls / gotchas
- Over-mocking hides real behavior. Mock only the LLM call or tool execution boundary.
- Ensure tool schema mismatches are covered (JSON args vs string args).
- Keep tests deterministic: fixed timestamps, fixed session IDs when needed.

## Preferred patterns
- Use fake tools that implement the same `ToolSpec` interface.
- Validate tool-call reflection: tool output must be passed to response synthesis.
- Use integration-style tests for CLI or API that cover end-to-end behavior with minimal stubs.

## Cross-project insights (for test design)
- Validate tool outputs before final response (detect false positives).
- Test turn boundaries (user -> plan -> tool -> response) and approval flows.
- Track context changes and summarization triggers without making network calls.
